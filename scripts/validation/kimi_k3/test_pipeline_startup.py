# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from pipeline_startup import PipelineStartupTimeout


class PipelineStartupTest(unittest.TestCase):
    def test_lazily_created_directed_groups_receive_startup_timeout(self):
        events = []
        parent, forward, backward = object(), object(), object()
        stages = [SimpleNamespace(group=parent, _p2p_edge_groups={})]

        def initialize(stages, has_backward, initialize_p2p):
            events.append(("initialize", has_backward, initialize_p2p))
            stages[0]._p2p_edge_groups = {(0, 1): forward, (1, 0): backward}

        schedule = SimpleNamespace(_initialize_pipeline_distributed_state=initialize)
        control = PipelineStartupTimeout(schedule, 1800, 600)
        with patch(
            "pipeline_startup.dist.set_timeout",
            side_effect=lambda t, g: events.append(("timeout", g, t)),
        ):
            schedule._initialize_pipeline_distributed_state(stages, True, True)
        self.assertEqual(events[0], ("initialize", True, True))
        self.assertEqual(
            set(events[1:]),
            {
                ("timeout", g, timedelta(seconds=1800))
                for g in (parent, forward, backward)
            },
        )
        with patch(
            "pipeline_startup.dist.barrier",
            side_effect=lambda: events.append("barrier"),
        ), patch(
            "pipeline_startup.torch.cuda.synchronize",
            side_effect=lambda: events.append("sync"),
        ), patch(
            "pipeline_startup.dist.set_timeout",
            side_effect=lambda t, g: events.append(("timeout", g, t)),
        ):
            control.finish()
        self.assertEqual(events[4:6], ["barrier", "sync"])
        self.assertEqual(
            set(events[6:]),
            {
                ("timeout", g, timedelta(seconds=600))
                for g in (parent, forward, backward)
            },
        )
        self.assertIs(schedule._initialize_pipeline_distributed_state, initialize)

    def test_unsupported_runtime_fails_before_training(self):
        with self.assertRaisesRegex(RuntimeError, "pipeline initialization"):
            PipelineStartupTimeout(SimpleNamespace(), 1800, 600)

    def test_invalid_budget_rejected(self):
        for startup, steady in [(0, 600), (600, 0), (600, 1800)]:
            with self.subTest(startup=startup, steady=steady), self.assertRaises(
                ValueError
            ):
                PipelineStartupTimeout(SimpleNamespace(), startup, steady)


if __name__ == "__main__":
    unittest.main()
