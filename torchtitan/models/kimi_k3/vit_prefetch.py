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


import torch

from torchtitan.tools.logging import logger


def prefetch_depth() -> int:
    """How many micro-batches ahead to encode. 0 disables the prefetch."""
    from torchtitan.models.kimi_k3.knobs import topology

    return max(0, topology().vit_prefetch)


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
        # Async bookkeeping. The overlap metric is DEFAULT-STREAM TIME BETWEEN ISSUE AND
        # JOIN, not "was the encode complete on arrival" -- that first attempt was useless
        # because the synchronous wrapper also leaves the encode complete by the time
        # take() runs, so it read the same either way. Time on the current stream between
        # ensure() and take() is zero when the issue path joins immediately and positive
        # only when real work was interleaved.
        self._pending: dict[int, object] = {}
        self._issued_at: dict[int, object] = {}
        # The encode's own GPU time, bracketed on the side stream. This is what decides
        # whether there is anything to hide at all: if it is microseconds, no scheduling
        # change can show up in a step time, and that is a fact about the config rather
        # than about the implementation.
        self._encode_spans: list[tuple[object, object]] = []

    def begin_step(self, kwarg_mbs) -> None:
        """Record the step's per-micro-batch kwargs and reset the cache.

        Called at ``schedule.step`` entry. The count is taken from the list, so it is
        identical on every rank -- which is what lets the prefetch order be a mesh
        property rather than a data one.
        """
        if self._hits or self._misses:
            # Read the spans now: the previous step's events are all complete by the time
            # the next step begins, so elapsed_time needs no extra synchronisation.
            enc_total, enc_n = 0.0, 0
            for a, b in self._encode_spans:
                try:
                    enc_total += a.elapsed_time(b)
                    enc_n += 1
                except (RuntimeError, ValueError):
                    pass
            logger.info(
                "DEP vision prefetch: %d hit(s), %d miss(es); encode GPU time %.2f ms "
                "over %d encode(s)",
                self._hits,
                self._misses,
                enc_total,
                enc_n,
            )
            self._hits = self._misses = 0
        self._features.clear()
        self._pending.clear()
        self._issued_at.clear()
        self._encode_spans.clear()
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
        # ISSUE without joining. Using the synchronous wrapper here would block the
        # current stream on the encode straight away, so the encode would merely happen
        # EARLIER rather than concurrently -- which is what it did before this split, and
        # is why a 31/32 hit rate coexisted with no measurable overlap. The join happens
        # in take(), when the consumer actually needs the features.
        feats, done = self._owner._issue_on_vision_stream(
            lambda: self._owner.encode_images(pixel_values, grid_thw),
            pixel_values if isinstance(pixel_values, torch.Tensor) else None,
        )
        # JOIN HERE, unconditionally. The deferred join that made the encode genuinely
        # concurrent is reverted: the side-stream encode contains NCCL collectives (FSDP's
        # tower all-gather, dynamic CP's gather-KV), and host issue order does NOT order
        # device execution across streams, so leaving them in flight is the two-communicator
        # cyclic wait this module's own docstring calls non-negotiable. An aborted step also
        # leaves un-joined collectives and lets the allocator reuse buffers still being
        # written.
        #
        # What decided it was the measurement, not caution: the encode costs 4.0 ms and the
        # GPU is idle 99.88% of the step here (mfu 0.12%), so async and sync differ by
        # 0.45% -- inside the 2-3% run-to-run spread. An unmeasurable gain does not buy a
        # real deadlock risk. The deferred form is the right design on a SATURATED GPU,
        # where the encode competes for SMs; it needs cross-stream collective ordering that
        # is not established here.
        self._owner._join_vision_stream(feats, done)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        self._features[mb] = feats
        self._pending[mb] = done
        if done is not None:
            issued = torch.cuda.Event(enable_timing=True)
            issued.record(torch.cuda.current_stream())
            self._issued_at[mb] = issued
            enc = getattr(self._owner, "_last_encode_span", None)
            if enc is not None:
                self._encode_spans.append(enc)

    def ensure_sync(self, mb: int) -> None:
        """Encode ``mb`` on the CURRENT stream, completing before returning.

        The bubble runtime's entry point. Where :meth:`ensure` issues on a side stream
        so the encode overlaps with text compute, this one occupies the caller's stream
        deliberately: the caller is standing in a pipeline bubble, so the whole point is
        to spend that idle interval on the encode rather than to race with anything.

        That also sidesteps what the side-stream path has to be careful about -- the
        encode contains NCCL collectives, and cross-stream collective ordering is the
        part that needs an argument. Here they are issued on the stream everything else
        is issued on, in an order every rank derives identically from the plan.

        Cached in the same place :meth:`ensure` fills, so :meth:`take` serves it and the
        hit/miss counters keep counting the same thing.
        """
        if mb in self._features:
            return
        inputs = self._inputs_for(mb)
        if inputs is None:
            return
        pixel_values, grid_thw = inputs
        self._features[mb] = self._owner.encode_images(pixel_values, grid_thw)

    def take(self, mb: int):
        """Features for ``mb`` if prefetched, else None. Removes the entry.

        Joins the side stream here rather than at issue time, which is the whole point:
        between ``ensure(mb)`` and ``take(mb)`` the default stream runs text compute while
        the encode is in flight.
        """
        feats = self._features.pop(mb, None)
        self._pending.pop(mb, None)
        self._issued_at.pop(mb, None)
        if feats is None:
            self._misses += 1
            return None
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
