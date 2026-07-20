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
    def _lora_model(self, quantize=None, mla_only=False):
        # Real QLoRA order: build+init a plain backbone, THEN wrap/quantize
        # (quantizing a loaded weight, not init'ing over an NF4 tensor).
        from torchtitan.experiments.kimi_k3.lora import apply_lora

        spec = self._spec_plain(mla_only=mla_only)
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        m = m.to(torch.bfloat16)
        apply_lora(m, rank=8, alpha=16, quantize_base=quantize)
        for n, p in m.named_parameters():
            if n.endswith("lora_b"):
                p.data.normal_(0, 0.02)  # trained-like adapter
        return m

    def _spec_plain(self, mla_only=False):
        import dataclasses

        from torchtitan.experiments.kimi_k3 import config_registry
        from torchtitan.experiments.kimi_k3.model import KimiLinearSpec

        kc = config_registry.kimi_linear_debugmodel().model_spec.model.kimi_config
        if mla_only:
            # All-MLA: KDA kernels are nondeterministic at debug scale and
            # can NaN under accumulated cross-test GPU state; forward-
            # executing tests use the deterministic MLA path.
            n = kc.num_hidden_layers
            kc = dataclasses.replace(
                kc, kda_layers=[], full_attn_layers=list(range(1, n + 1))
            )
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

    def test_post_load_quantize_hook(self):
        # The trainer order: build+load bf16, THEN quantize (not at
        # build over init noise / meta storage).
        from torchao.dtypes.nf4tensor import NF4Tensor

        from torchtitan.experiments.kimi_k3.lora import (
            KimiLoRALinear,
            quantize_lora_bases,
        )

        # all-MLA: this test runs a forward (deterministic MLA path)
        m = self._lora_model(mla_only=True)  # bf16 bases, loaded-like
        # a reference: one alignable base weight, pre-quantization
        ref = None
        for module in m.modules():
            if (
                isinstance(module, KimiLoRALinear)
                and module.base.weight.numel() % 16384 == 0
            ):
                ref = (module, module.base.weight.detach().float().clone())
                break
        self.assertIsNotNone(ref, "need >=1 NF4-alignable base for this test")

        packed = quantize_lora_bases(m, experts=False)
        self.assertGreater(packed, 0)
        module, ref_w = ref
        self.assertIsInstance(module.base.weight, NF4Tensor)
        # dequant tracks the loaded weight within NF4 error (not init noise)
        deq = module.base.weight.get_original_weight().float()
        self.assertLess(
            (deq - ref_w).norm().item() / ref_w.norm().item(), 0.15
        )
        # idempotent: second call packs nothing new, no error
        self.assertEqual(quantize_lora_bases(m, experts=False), packed)
        # forward still runs through the NF4 base path
        tok = torch.randint(0, 2016, (1, 96), device="cuda")
        with torch.no_grad():
            m(tok)


if __name__ == "__main__":
    unittest.main()
