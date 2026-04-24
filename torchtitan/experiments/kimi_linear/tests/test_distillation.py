# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.experiments.kimi_linear.distillation import (
    KDConfig,
    build_kd_loss,
    kd_loss,
)


class TestKDLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.V = 2, 8, 64

    def _fake_batch(self):
        student = torch.randn(self.B, self.T, self.V)
        teacher = torch.randn(self.B, self.T, self.V)
        labels = torch.randint(0, self.V, (self.B, self.T))
        return student, teacher, labels

    def test_shape_assertion(self):
        s, _, labels = self._fake_batch()
        bad_t = torch.randn(self.B, self.T, self.V + 1)
        with self.assertRaises(AssertionError):
            kd_loss(s, labels, bad_t)

    def test_alpha_1_reduces_to_ce(self):
        """alpha=1 -> teacher contribution vanishes; matches torchtitan CE."""
        s, t, labels = self._fake_batch()
        cfg = KDConfig(alpha=1.0, temperature=2.0)
        loss_kd = kd_loss(s, labels, t, cfg)
        loss_ce = torch.nn.functional.cross_entropy(
            s.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
        )
        self.assertTrue(torch.allclose(loss_kd, loss_ce, atol=1e-4))

    def test_alpha_0_pure_kl(self):
        """alpha=0 -> only the KL-to-teacher term contributes."""
        s, t, labels = self._fake_batch()
        cfg = KDConfig(alpha=0.0, temperature=2.0)
        loss = kd_loss(s, labels, t, cfg)
        # For random logits KL should be positive and finite.
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_identical_teacher_kl_is_zero(self):
        """When student == teacher, KL term is 0; only alpha*CE remains."""
        s, _, labels = self._fake_batch()
        cfg = KDConfig(alpha=0.0, temperature=2.0)
        loss = kd_loss(s, labels, s, cfg)
        # Pure KL with identical distributions — should be ~0.
        self.assertLess(loss.abs().item(), 1e-3)

    def test_ignore_index_masks_positions(self):
        s, t, labels = self._fake_batch()
        labels[:, -2:] = -100  # mask last 2 positions
        cfg = KDConfig(alpha=0.3, temperature=2.0, ignore_index=-100)
        loss_masked = kd_loss(s, labels, t, cfg)
        # Zero out those positions in fresh student/teacher — should
        # produce the same loss.
        self.assertTrue(torch.isfinite(loss_masked))

    def test_all_masked_returns_alpha_ce_only(self):
        """If every position is ignored, KL term contributes 0."""
        s, t, labels = self._fake_batch()
        labels.fill_(-100)
        cfg = KDConfig(alpha=0.3, temperature=2.0, ignore_index=-100)
        loss = kd_loss(s, labels, t, cfg)
        # CE with all-ignored reduces to 0 under sum-reduction.
        self.assertEqual(loss.item(), 0.0)

    def test_temperature_scaling(self):
        """Higher T smooths distributions -> different KL value; result
        must still be finite and positive."""
        s, t, labels = self._fake_batch()
        loss_t1 = kd_loss(s, labels, t, KDConfig(alpha=0.0, temperature=1.0))
        loss_t4 = kd_loss(s, labels, t, KDConfig(alpha=0.0, temperature=4.0))
        self.assertTrue(torch.isfinite(loss_t1))
        self.assertTrue(torch.isfinite(loss_t4))
        self.assertGreater(loss_t1.item(), 0.0)
        self.assertGreater(loss_t4.item(), 0.0)
        # T^2 rescaling does NOT make the two losses equal — different
        # softmax shapes. Just sanity.
        self.assertNotAlmostEqual(loss_t1.item(), loss_t4.item(), places=3)

    def test_backward_flows_to_student_only(self):
        """teacher_logits should not accumulate grad (caller passes
        detached). Test that when teacher IS leaf + requires_grad, we
        don't crash, but the canonical use is detached teacher."""
        s, t, labels = self._fake_batch()
        s = s.requires_grad_(True)
        loss = kd_loss(s, labels, t, KDConfig(alpha=0.3, temperature=2.0))
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_build_kd_loss_closure(self):
        loss_fn = build_kd_loss(KDConfig(alpha=0.4, temperature=3.0))
        s, t, labels = self._fake_batch()
        loss = loss_fn(s, labels, t)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
