# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA merge (checkpoint export) tests.

merge_lora_state_dict folds adapters into base weights so a trained
LoRA can be saved back to HF (the raw adapter drops lora_* keys). Tested
at the tensor level -- W_merged == W_base + scaling*(B@A) -- which is the
merge contract; full-model forward fidelity is subject to bf16 (the
standard LoRA merge-and-unload property).
"""

import unittest

import torch


@unittest.skipIf(not torch.cuda.is_available(), "build needs CUDA (fla)")
class TestLoRAMerge(unittest.TestCase):
    def _lora_model(self, quantize=None):
        # Real QLoRA order: build+init a plain backbone, THEN wrap/quantize
        # (quantizing a loaded weight, not init'ing over an NF4 tensor).
        from torchtitan.experiments.kimi_k3.lora import apply_lora

        spec = self._spec_plain()
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        m = m.to(torch.bfloat16)
        apply_lora(m, rank=8, alpha=16, quantize_base=quantize)
        for n, p in m.named_parameters():
            if n.endswith("lora_b"):
                p.data.normal_(0, 0.02)  # trained-like adapter
        return m

    def _spec_plain(self):
        from torchtitan.experiments.kimi_k3 import config_registry
        from torchtitan.experiments.kimi_k3.model import KimiLinearSpec

        kc = config_registry.kimi_linear_debugmodel().model_spec.model.kimi_config
        return KimiLinearSpec(kimi_config=kc, num_blocks=None)

    def test_merge_tensor_math_and_key_space(self):
        from torchtitan.experiments.kimi_k3.lora import (
            KimiLoRALinear,
            merge_lora_state_dict,
        )

        m = self._lora_model()
        merged = merge_lora_state_dict(m)
        self.assertFalse(any("lora_a" in k or "lora_b" in k for k in merged))
        self.assertFalse(any(".base." in k for k in merged))
        checked = 0
        for mod_name, module in m.named_modules():
            if isinstance(module, KimiLoRALinear):
                expect = (
                    module.base.weight.float()
                    + module._lora_scaling
                    * (module.lora_b.float() @ module.lora_a.float())
                ).to(module.base.weight.dtype)
                got = merged[f"{mod_name}.weight"]
                self.assertEqual(got.shape, module.base.weight.shape)
                self.assertLess(
                    (got.float() - expect.float()).abs().max().item(), 1e-2
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_merge_exports_to_hf(self):
        from torchtitan.experiments.kimi_k3.lora import merge_lora_state_dict
        from torchtitan.experiments.kimi_k3.state_dict_adapter import (
            KimiLinearStateDictAdapter,
        )

        spec = self._spec_plain()
        m = self._lora_model()
        merged = merge_lora_state_dict(m)
        adapter = KimiLinearStateDictAdapter(spec, hf_assets_path=None)
        hf = adapter.to_hf(merged)
        self.assertTrue(hf)
        self.assertFalse(any("lora" in k or ".base." in k for k in hf))

    def test_merge_dequantizes_nf4_base(self):
        from torchao.dtypes.nf4tensor import NF4Tensor

        from torchtitan.experiments.kimi_k3.lora import merge_lora_state_dict

        m = self._lora_model(quantize="nf4")
        merged = merge_lora_state_dict(m)
        self.assertFalse(
            any(isinstance(v, NF4Tensor) for v in merged.values())
        )


if __name__ == "__main__":
    unittest.main()
