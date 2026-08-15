# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bubble planner's invariants, and that the runtime fires before the wait.

No GPU and no model: both pieces are scheduling logic, and the property that matters
for the runtime -- that the encode happens BEFORE the rank waits on its receive -- is
an ordering fact that a fake schedule can check exactly.
"""

from __future__ import annotations

import unittest

from torchtitan.models.kimi_k3.dep_bubble_plan import build_plans, plan_for_rank
from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime


class _FakeAction:
    def __init__(self, kind: str, stage: int, mb: int | None) -> None:
        self.computation_type = kind
        self.stage_index = stage
        self.microbatch_index = mb


class TestBubblePlan(unittest.TestCase):
    def test_no_encode_is_placed_after_its_own_consumer(self):
        """The constraint that halved the first version's claimed placements.

        A bubble after micro-batch j's features are consumed cannot pay for encoding
        them, however much budget has accumulated.
        """
        for vp in (1, 2, 4):
            plans = build_plans(
                pp_size=8, vp=vp, n_microbatches=32, cost_ratio=0.493
            )
            for rank, plan in plans.items():
                for p in plan.placed:
                    kind, stage, anchor_mb = p.anchor
                    if "FORWARD" in kind and stage == 0 and anchor_mb >= 0:
                        # The anchor is stage 0's forward of anchor_mb, which runs at
                        # anchor_mb's consumption point, so the placed micro-batch must
                        # not be earlier than it.
                        self.assertGreaterEqual(
                            p.microbatch,
                            anchor_mb,
                            f"vp={vp} rank={rank}: encode for mb {p.microbatch} placed "
                            f"at mb {anchor_mb}'s consumption point",
                        )

    def test_every_microbatch_is_accounted_for_exactly_once(self):
        plans = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        for plan in plans.values():
            seen = (
                list(plan.upfront)
                + [p.microbatch for p in plan.placed]
                + list(plan.synchronous)
            )
            self.assertEqual(sorted(seen), list(range(32)))
            self.assertEqual(len(seen), len(set(seen)))

    def test_all_ranks_derive_the_same_plan_shape(self):
        """Consistency is what makes the vision collectives safe to issue here.

        Ranks own different stages so their action lists differ, but the plan must be a
        function of values every rank agrees on -- so recomputing it must be
        deterministic, and the per-rank counts must not depend on call order.
        """
        a = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        b = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        self.assertEqual(
            {r: (p.upfront, p.placed, p.synchronous) for r, p in a.items()},
            {r: (p.upfront, p.placed, p.synchronous) for r, p in b.items()},
        )

    def test_a_bubble_run_too_short_to_pay_places_nothing(self):
        actions = [_FakeAction("FORWARD", 0, 0), None, _FakeAction("FORWARD", 0, 1)]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=5.0, upfront=0
        )
        self.assertEqual(plan.placed, ())
        self.assertEqual(sorted(plan.synchronous), [0, 1])

    def test_trailing_bubbles_cannot_be_anchored(self):
        """Idle time at the end has no following action, so it helps nothing."""
        actions = [_FakeAction("FORWARD", 0, 0), None, None, None]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertEqual(plan.placed, ())
        self.assertIn(1, plan.synchronous)


class _FakeSchedule:
    """Enough of _PipelineScheduleRuntime to test the ordering property."""

    def __init__(self, order: list) -> None:
        self.fwd_recv_ops: dict = {}
        self._order = order
        self.trace: list[str] = []

    def step(self, *args, **kwargs):
        for action in self._order:
            if action is None:
                continue
            key = (action.stage_index, action.microbatch_index)
            if "FORWARD" in action.computation_type:
                self.fwd_recv_ops[key] = ["work"]
                # The runtime's own order: pop, then wait, then forward.
                self.fwd_recv_ops.pop(key, None)
                self.trace.append(f"wait{key}")
                self.trace.append(f"fwd{key}")
        return "stepped"


class TestBubbleRuntime(unittest.TestCase):
    def _install(self, order, plan):
        sched = _FakeSchedule(order)
        install_bubble_runtime(
            sched,
            plan_for_step=lambda: plan,
            encode_now=lambda mbs: sched.trace.append(f"encode{list(mbs)}"),
            upfront_encode=lambda mbs: sched.trace.append(f"upfront{list(mbs)}"),
        )
        return sched

    def test_the_encode_runs_before_the_wait_it_hides_behind(self):
        """The whole design in one assertion.

        If the encode landed after the wait it would serialise with the forward instead
        of occupying the idle interval, which is what hooking forward_one_chunk would
        have done.
        """
        order = [_FakeAction("FORWARD", 0, 0), None, _FakeAction("FORWARD", 0, 1)]
        plan = plan_for_rank(
            order, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertTrue(plan.placed, "fixture must place at least one encode")
        sched = self._install(order, plan)
        sched.step()
        enc = next(i for i, t in enumerate(sched.trace) if t.startswith("encode"))
        wait = next(
            i
            for i, t in enumerate(sched.trace)
            if t.startswith("wait") and i > enc - 2
        )
        self.assertLess(enc, wait, f"trace={sched.trace}")

    def test_no_plan_leaves_the_schedule_untouched(self):
        order = [_FakeAction("FORWARD", 0, 0)]
        sched = self._install(order, None)
        self.assertEqual(sched.step(), "stepped")
        self.assertEqual([t for t in sched.trace if "encode" in t], [])

    def test_installing_twice_is_a_no_op(self):
        order = [_FakeAction("FORWARD", 0, 0)]
        plan = plan_for_rank(
            order, rank=0, vision_microbatches=1, cost_ratio=1.0, upfront=1
        )
        sched = self._install(order, plan)
        first = sched.step
        install_bubble_runtime(
            sched,
            plan_for_step=lambda: plan,
            encode_now=lambda mbs: None,
            upfront_encode=lambda mbs: None,
        )
        self.assertIs(sched.step, first)


if __name__ == "__main__":
    unittest.main()
