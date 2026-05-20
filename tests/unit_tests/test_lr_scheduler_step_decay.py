# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torch.optim import Adam

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ConfigManager


class TestStepDecayLRScheduler(unittest.TestCase):
    def setUp(self):
        # Create a simple model with parameters
        self.model = torch.nn.Linear(10, 10)
        # base_lr=1.0 so the multiplicative factor equals the observed lr
        self.optimizer = Adam(self.model.parameters(), lr=1.0)

        # We don't actually call `optimizer.step()` which will cause a warning
        # from PyTorch. Avoid the warnings that may confuse people.
        self.optimizer._opt_called = True

        # Create an optimizer container
        self.optimizer_container = MagicMock(spec=OptimizersContainer)
        self.optimizer_container.__iter__.return_value = iter([self.optimizer])
        self.optimizer_container.__len__.return_value = 1

    def _build(self, **overrides):
        args = [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--training.steps",
            str(overrides.pop("training_steps", 100)),
            "--lr_scheduler.decay_type",
            "step",
        ]
        for k, v in overrides.items():
            args += [f"--lr_scheduler.{k}", str(v)]

        config = ConfigManager().parse_args(args)
        return config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

    def test_warmup_then_step_decay(self):
        """Warmup ramps linearly to 1.0, then lr *= decay_factor every decay_freq."""
        lr_scheduler = self._build(
            training_steps=100,
            warmup_steps=10,
            decay_freq=20,
            decay_factor=0.9,
        )

        expected = {
            # warmup: factor at step k is (k+1)/warmup_steps
            0: 1 / 10,
            4: 5 / 10,
            5: 6 / 10,
            9: 10 / 10,  # end of warmup
            # post-warmup: intervals = (step - warmup) // decay_freq
            10: 1.0,  # 0 intervals
            29: 1.0,  # 19/20 = 0 intervals
            30: 0.9,  # 1 interval
            49: 0.9,  # 39/20 = 1 interval
            50: 0.9 * 0.9,  # 2 intervals = 0.81
            70: 0.9**3,  # 3 intervals = 0.729
            90: 0.9**4,  # 4 intervals = 0.6561
        }

        for step in range(100):
            lr = self.optimizer.param_groups[0]["lr"]
            if step in expected:
                self.assertAlmostEqual(
                    lr,
                    expected[step],
                    places=6,
                    msg=f"step {step}: expected lr {expected[step]}, got {lr}",
                )
            lr_scheduler.step()

    def test_min_lr_factor_floor(self):
        """min_lr_factor acts as a floor after warmup."""
        lr_scheduler = self._build(
            training_steps=200,
            warmup_steps=2,
            decay_freq=1,
            decay_factor=0.5,
            min_lr_factor=0.1,
        )

        # After warmup at step 2, factor = 0.5^0 = 1.0
        # Step 3: 0.5, step 4: 0.25, step 5: 0.125, step 6: would be 0.0625 but
        # floored to 0.1.
        for _ in range(2):  # finish warmup
            lr_scheduler.step()
        # now at step 2: factor = 1.0
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 1.0, places=6)
        lr_scheduler.step()  # step 3
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.5, places=6)
        lr_scheduler.step()  # step 4
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.25, places=6)
        lr_scheduler.step()  # step 5
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.125, places=6)
        lr_scheduler.step()  # step 6 - floored
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.1, places=6)
        # subsequent steps stay at the floor
        for _ in range(10):
            lr_scheduler.step()
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"], 0.1, places=6
            )

    def test_decay_ratio_is_ignored(self):
        """decay_ratio is documented as ignored under step decay; build should not raise."""
        lr_scheduler = self._build(
            training_steps=50,
            warmup_steps=5,
            decay_freq=10,
            decay_factor=0.9,
            decay_ratio=0.3,  # should be silently ignored (warning logged)
        )

        # Step 5 (end warmup) -> factor 1.0
        # Step 15 -> 1 interval -> 0.9
        # Step 25 -> 2 intervals -> 0.81
        # Step 35 -> 3 intervals -> 0.729
        for _ in range(15):
            lr_scheduler.step()
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], 0.9, places=6
        )

    def test_invalid_decay_factor(self):
        with self.assertRaises(ValueError):
            self._build(
                training_steps=20,
                warmup_steps=2,
                decay_freq=5,
                decay_factor=1.5,
            )

    def test_invalid_decay_freq(self):
        with self.assertRaises(ValueError):
            self._build(
                training_steps=20,
                warmup_steps=2,
                decay_freq=0,
                decay_factor=0.9,
            )


if __name__ == "__main__":
    unittest.main()
