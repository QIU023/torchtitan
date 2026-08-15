# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the vision encodes inside the schedule's idle time, on the main stream.

The companion to :mod:`dep_bubble_plan`, which decides WHERE each encode goes. This
puts it there.

## Why the hook is on ``fwd_recv_ops.pop`` and not on the forward

A bubble is the rank waiting for a P2P receive. In
``_PipelineScheduleRuntime._step_microbatches`` the FORWARD branch does::

    _wait_batch_p2p(self.fwd_recv_ops.pop((stage_idx, mb_index)))
    output = stage.forward_one_chunk(...)

so the wait -- the bubble -- is consumed BEFORE ``forward_one_chunk``. Hooking the
forward would run the encode after the rank had already finished waiting, which
serialises the two instead of overlapping them: the encode would delay the forward by
its full duration and fill nothing. Checked in the source rather than assumed, because
the difference between the two is the entire point of the design.

``fwd_recv_ops.pop`` is called immediately before that wait and carries the
``(stage_index, microbatch_index)`` the plan anchors on, so replacing that dict with
one that fires on ``pop`` puts the encode exactly at the start of the idle interval.
The receive is already in flight by then, so the GPU runs the encode while it lands.

``fwd_recv_ops`` is assigned once in the runtime's ``__init__`` and never rebound, so
replacing the instance attribute after construction holds for the whole run.

## Main stream, not the side stream

The prefetch path issues encodes on a side stream so they overlap with whatever the
main stream is doing. This path is the opposite by design: the bubble IS main-stream
idle time, so the encode belongs on the main stream, where it occupies the gap rather
than competing for SMs with text compute.

## What is not here yet

Backward. The report handles the backward passes analogously, and the same hook exists
for it (``bwd_recv_ops``), but the vision backward is triggered by the spliced
features' gradient rather than by a schedule action, so it needs the adapter's gradient
path involved. Forward first, with the occupancy criterion, then backward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torchtitan.tools.logging import logger

from torchtitan.models.kimi_k3.dep_bubble_plan import BubblePlan


class _AnchoredRecvOps(dict):
    """``fwd_recv_ops`` that runs planned encodes when a wait is about to happen.

    Subclasses ``dict`` rather than wrapping it because the runtime uses ``in``,
    ``__setitem__`` and ``pop`` on the attribute directly; anything less transparent
    than a dict subclass would have to reimplement those and would drift.
    """

    def __init__(self, on_anchor: Callable[[Sequence[int]], None]) -> None:
        super().__init__()
        self._on_anchor = on_anchor
        self._by_anchor: dict[tuple[int, int], list[int]] = {}
        self.fired = 0

    def arm(self, plan: BubblePlan) -> None:
        """Load this step's placements. Called once per step, before the loop."""
        self._by_anchor = {}
        for placement in plan.placed:
            kind, stage_index, mb_index = placement.anchor
            if "FORWARD" not in kind:
                # Backward anchors need bwd_recv_ops and the adapter's gradient path.
                continue
            self._by_anchor.setdefault((stage_index, mb_index), []).append(
                placement.microbatch
            )

    def pop(self, key, *default):  # type: ignore[override]
        queued = self._by_anchor.pop(key, None)
        if queued:
            # In the bubble: the receive for `key` is in flight and has not been waited
            # on yet, so this compute occupies time the rank would otherwise idle.
            self._on_anchor(queued)
            self.fired += len(queued)
        return super().pop(key, *default)


def install_bubble_runtime(
    pp_schedule,
    *,
    plan_for_step: Callable[[], BubblePlan | None],
    encode_now: Callable[[Sequence[int]], None],
    upfront_encode: Callable[[Sequence[int]], None],
) -> None:
    """Make ``pp_schedule`` run planned vision encodes in its idle intervals.

    ``plan_for_step`` returns this rank's plan, or None to leave the schedule alone --
    which is how a step with no visual items, or a rank owning no vision work, opts out
    without a second code path.

    ``encode_now`` runs the encodes on the current (main) stream. ``upfront_encode``
    runs the report's synchronous prefix before the action loop.

    Patches the instance, not the class: torchtitan chooses which schedule class to
    build, and the same reasoning already applies to the cross-stage adapter's own
    ``step`` patch next door.
    """
    if getattr(pp_schedule, "_kimi_bubble_runtime", False):
        return
    recv_ops = _AnchoredRecvOps(encode_now)
    if not hasattr(pp_schedule, "fwd_recv_ops"):
        raise AttributeError(
            "pp_schedule has no fwd_recv_ops: the bubble runtime needs "
            "_PipelineScheduleRuntime's action loop, so a schedule that does not use "
            "it cannot host this. Disable KIMI_VIT_BUBBLE for that schedule."
        )
    recv_ops.update(pp_schedule.fwd_recv_ops)
    pp_schedule.fwd_recv_ops = recv_ops
    orig_step = pp_schedule.step

    def patched_step(*args, **kwargs):
        plan = plan_for_step()
        if plan is None:
            return orig_step(*args, **kwargs)
        recv_ops.arm(plan)
        before = recv_ops.fired
        if plan.upfront:
            # The report's own design: the first micro-batches' encodes cannot be
            # placed, because nothing precedes them.
            upfront_encode(plan.upfront)
        try:
            return orig_step(*args, **kwargs)
        finally:
            placed = len(plan.placed)
            fired = recv_ops.fired - before
            # Placed-but-never-fired means the anchor action did not run on this rank
            # this step, i.e. the plan and the schedule disagree. Silence there would
            # let the encode fall back to its synchronous path and still look correct.
            level = logger.info if fired == placed else logger.warning
            level(
                "DEP bubble runtime: %d/%d planned encode(s) ran in a bubble, "
                "%d upfront, %d left synchronous",
                fired,
                placed,
                len(plan.upfront),
                len(plan.synchronous),
            )

    pp_schedule.step = patched_step  # type: ignore[method-assign]
    pp_schedule._kimi_bubble_runtime = True
