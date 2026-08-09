# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the vision encode AHEAD of the text pipeline (report 5.2.3's DEP).

The report: "The ViT forward passes of the first PP micro-batches are executed
synchronously upfront, the remaining forward passes are scheduled into pipeline
bubbles, and the backward passes are handled analogously."

Measured earlier, three ways, that this cannot be done by making the ViT a pipeline
STAGE: the bubble count is identical with and without that (2019 either way), every
IR reorder is rejected by the lowering as unschedulable, and the reason both hold is
that a stage sits in the dependency chain -- at one vision stage it IS the pipeline
head, so the bubbles are downstream of the work that would fill them. Overlap
therefore has to come from concurrency, not from placement.

## Why the hook is on the STEP and not the stage

PP hands the first stage one micro-batch's inputs at a time, so at micro-batch m the
stage cannot see m+k's pixels. But ``kwarg_mbs`` -- the per-micro-batch kwargs list --
is passed to ``schedule.step()`` whole. Capturing it at step entry makes every
micro-batch's vision input available from the first action onward, with no change to
core: the hook goes on the schedule instance from our own ``pipelining_fn``.

## The two constraints that are not negotiable

**Same Python thread.** The AttnRes adapter keys its per-micro-batch cache in a
``threading.local``, and its ``forward`` reads a missing key as "this call is PP's
shape inference" and diverts WITHOUT raising. Driving any model code from a worker
thread would silently return shape-inference outputs. Concurrency comes from a CUDA
stream; the prefetch is issued from the same thread that runs the schedule.

**Collectives gated on the mesh, never on the data.** The vision encode issues
collectives (dynamic CP's gather-KV, the feature all-gather, FSDP's tower
all-gather). Every rank in those groups must reach every one of them in the same
order, or two communicators deadlock on a cyclic wait without either one's ordering
being violated. So the prefetch schedule is a function of the micro-batch COUNT --
which every rank agrees on before the step -- and never of what a rank's own batch
contains.

## Memory is the bound on the lookahead

Holding k micro-batches of vision features live is the cost, and it is what decides
whether "most" of the encoder can be hidden or only some. ``depth`` defaults to 1 for
that reason: one micro-batch of slack is enough to overlap an encode with a text
forward, and every extra unit is paid for in resident activations.
"""

from __future__ import annotations

import os

import torch

from torchtitan.tools.logging import logger


def prefetch_depth() -> int:
    """How many micro-batches ahead to encode. 0 disables the prefetch."""
    return max(0, int(os.environ.get("KIMI_VIT_PREFETCH", "0")))


class VisionPrefetcher:
    """Per-step cache of vision features, encoded ahead on a side stream.

    Keyed by micro-batch index, populated by ``ensure(m)`` and drained by ``take(m)``.
    ``take`` removes the entry: a feature tensor held past its micro-batch is resident
    memory for nothing, and the lookahead's whole cost is residency.
    """

    def __init__(self, owner) -> None:
        self._owner = owner
        self._features: dict[int, list[torch.Tensor]] = {}
        self._kwargs: list[dict] | None = None
        self._num_mbs = 0
        # Hit/miss counters, reported once per step. Installing the hook proves the
        # patch is in place; only a HIT proves a micro-batch's encode was already
        # done when its forward asked for it, which is the whole claim.
        self._hits = 0
        self._misses = 0

    def begin_step(self, kwarg_mbs) -> None:
        """Record the step's per-micro-batch kwargs and reset the cache.

        Called at ``schedule.step`` entry. The count is taken from the list, so it is
        identical on every rank -- which is what lets the prefetch order be a mesh
        property rather than a data one.
        """
        if self._hits or self._misses:
            logger.info(
                "DEP vision prefetch: %d hit(s), %d miss(es) in the previous step",
                self._hits,
                self._misses,
            )
            self._hits = self._misses = 0
        self._features.clear()
        if kwarg_mbs is None:
            self._kwargs, self._num_mbs = None, 0
            return
        self._kwargs = list(kwarg_mbs)
        self._num_mbs = len(self._kwargs)

    def _inputs_for(self, mb: int):
        if self._kwargs is None or not (0 <= mb < self._num_mbs):
            return None
        kw = self._kwargs[mb] or {}
        pixel_values = kw.get("pixel_values")
        grid_thw = kw.get("grid_thw")
        if pixel_values is None or grid_thw is None:
            return None
        return pixel_values, grid_thw

    def ensure(self, mb: int) -> None:
        """Encode micro-batch ``mb`` now if it is not cached yet.

        Issued on the owner's vision stream, so it can proceed while the default
        stream runs text compute. The encode itself is the owner's existing
        ``encode_images``, which already carries the dynamic-CP and replicated paths.
        """
        if mb in self._features:
            return
        inputs = self._inputs_for(mb)
        if inputs is None:
            return
        pixel_values, grid_thw = inputs
        feats = self._owner._run_on_vision_stream(
            lambda: self._owner.encode_images(pixel_values, grid_thw),
            pixel_values if isinstance(pixel_values, torch.Tensor) else None,
        )
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        self._features[mb] = feats

    def take(self, mb: int):
        """Features for ``mb`` if prefetched, else None. Removes the entry."""
        feats = self._features.pop(mb, None)
        if feats is None:
            self._misses += 1
        else:
            self._hits += 1
        return feats

    def advance(self, mb: int, depth: int) -> None:
        """After serving ``mb``, start the encodes for the next ``depth``.

        Driven by the micro-batch index the schedule is on, not by wall clock or by a
        queue depth measured at runtime, so every rank issues the same encodes in the
        same order.
        """
        for ahead in range(1, depth + 1):
            self.ensure(mb + ahead)


class VisionStepInputs:
    """Per-step ``grid_thw`` lookup for tower shares that never see the batch.

    When the tower spans PP stages (report 5.2.3's "balances vision forward and
    backward passes across PP stages"), every share needs ``grid_thw`` to recompute its
    RoPE frequencies and segment bounds. Only the first stage receives the batch, and
    the value cannot be sent down the pipe: PP's metadata inference pushes dummy values
    through pipe tensors, and these are used as indices and bounds where a dummy
    asserts out of bounds.

    ``kwarg_mbs`` is handed to ``schedule.step`` whole, so capturing it at step entry
    makes every micro-batch's grid available to every stage on this rank, with no change
    to core and nothing extra on the wire.
    """

    def __init__(self) -> None:
        self._kwargs: list[dict] | None = None

    def begin_step(self, kwarg_mbs) -> None:
        self._kwargs = None if kwarg_mbs is None else list(kwarg_mbs)

    def grid_for(self, mb: int):
        """This micro-batch's ``grid_thw``, or None when it carries no images."""
        if self._kwargs is None or not (0 <= mb < len(self._kwargs)):
            return None
        return (self._kwargs[mb] or {}).get("grid_thw")


def install_step_hook(schedule, observer) -> None:
    """Make ``schedule.step`` hand its ``kwarg_mbs`` to ``observer.begin_step`` first.

    Takes any object with ``begin_step`` -- :class:`VisionPrefetcher` for the run-ahead
    and :class:`VisionStepInputs` for a split tower -- so both can be installed on the
    same schedule without either knowing about the other.

    Bound on the instance rather than the class: two schedules in one process (a
    validator alongside a trainer) must not share one.
    """
    original = schedule.step

    def step(*args, **kwargs):
        observer.begin_step(kwargs.get("kwarg_mbs"))
        return original(*args, **kwargs)

    schedule.step = step
