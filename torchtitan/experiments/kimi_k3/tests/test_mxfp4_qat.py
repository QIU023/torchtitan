# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4/MXFP8 fake-quant QAT wrapper tests (K3-faithful quant path).

CUDA-only (torchao MX primitives). Locks: wrapping, forward finiteness,
straight-through gradient to the bf16 master, and that quantization
actually perturbs the output (a no-op wrapper would silently disable
QAT).
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3 import config_registry
from torchtitan.experiments.kimi_k3.model import KimiLinearSpec


@unittest.skipIf(not torch.cuda.is_available(), "torchao MX needs CUDA")
class TestMXFP4QAT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.kimi_config = (
            config_registry.kimi_linear_debugmodel().model_spec.model.kimi_config
        )

    def _build(self):
        spec = KimiLinearSpec(kimi_config=self.kimi_config, num_blocks=None)
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        return m.to(torch.bfloat16)

    def test_wrap_forward_and_ste_grad(self):
        from torchtitan.experiments.kimi_k3.mxfp4_qat import apply_mxfp4_qat

        m = self._build()
        n = apply_mxfp4_qat(m, quantize_act=True)
        self.assertGreater(n, 0)
        tok = torch.randint(0, 2016, (2, 128), device="cuda")
        out = m(tok)
        self.assertTrue(torch.isfinite(out).all())
        out.float().sum().backward()
        base_keys = [
            k for k, _ in m.named_parameters() if k.endswith(".base.weight")
        ]
        self.assertTrue(base_keys)
        named = dict(m.named_parameters())
        for k in base_keys:
            self.assertIsNotNone(named[k].grad, k)
            self.assertTrue(torch.isfinite(named[k].grad).all(), k)

    def test_quantization_actually_perturbs(self):
        from torchtitan.experiments.kimi_k3.mxfp4_qat import apply_mxfp4_qat

        m_ref = self._build()
        tok = torch.randint(0, 2016, (2, 128), device="cuda")
        with torch.no_grad():
            ref = m_ref(tok).float()

        m_q = self._build()
        m_q.load_state_dict(m_ref.state_dict())
        apply_mxfp4_qat(m_q, quantize_act=True)
        with torch.no_grad():
            q = m_q(tok).float()
        # MXFP4 weights + MXFP8 acts must measurably change the logits;
        # a silent no-op (e.g. wrong elem dtype) would make these equal.
        self.assertGreater((ref - q).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
