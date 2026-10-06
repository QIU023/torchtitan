# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchao.prototype.qat import MXFakeQuantizeConfig
from torchao.quantization.quantize_.common import KernelPreference
from torchtitan.config.loader import ConfigLoader
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel_mx_qat

from tests.unit_tests.cpu.mx_qat_test_utils import write_mixed_checkpoint_metadata


class MXQATRecipeTest(unittest.TestCase):
    def test_standard_model_entrypoint(self):
        config = ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.kimi_k3",
                "--config",
                "kimi_k3_debugmodel_mx_qat",
            ]
        )
        self.assertIsInstance(config.model, KimiK3Model.Config)
        self.assertFalse(config.checkpointer.initial_load_in_hf_quantized)
        self.assertTrue(type(config.model.layers[1].moe.routed_up)._owner._mx_qat)

    def test_custom_recipe_uses_existing_torchao_configs(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint_path = directory.name
        write_mixed_checkpoint_metadata(
            Path(checkpoint_path),
            KimiK3StateDictAdapter(build_model_config("debugmodel"), None),
        )

        def qat():
            return kimi_k3_debugmodel_mx_qat(
                seq_len=16,
                checkpoint_path=checkpoint_path,
                weight_fake_quant_config=MXFakeQuantizeConfig(
                    dtype=torch.float4_e2m1fn_x2,
                    kernel_preference=KernelPreference.AUTO,
                ),
                activation_fake_quant_config=MXFakeQuantizeConfig(
                    dtype=torch.float8_e4m3fn,
                    kernel_preference=KernelPreference.AUTO,
                ),
            )

        module = types.ModuleType("my_kimi_runs")
        module.qat = qat
        with patch.dict(sys.modules, {module.__name__: module}):
            config = ConfigLoader().load(
                ["--module", module.__name__, "--config", "qat"]
            )
        self.assertEqual(config.checkpointer.initial_load_path, checkpoint_path)
        self.assertTrue(config.checkpointer.initial_load_in_hf)
        self.assertTrue(config.checkpointer.initial_load_in_hf_quantized)
        experts = list(config.model.traverse(GroupedLinear.Config))
        self.assertTrue(experts)
        for _, expert, _, _ in experts:
            self.assertEqual(
                expert.weight_fake_quant_config.kernel_preference, KernelPreference.AUTO
            )
            self.assertEqual(
                expert.activation_fake_quant_config.kernel_preference,
                KernelPreference.AUTO,
            )
        self.assertFalse(
            getattr(type(config.model.layers[1].moe.routed_up)._owner, "_mx_qat", False)
        )
        adapter = KimiK3StateDictAdapter(config.model, None)
        reader = adapter.get_hf_storage_reader(checkpoint_path, from_quantized=True)
        self.assertEqual(
            reader.spec.target_fqns, adapter.mxfp4_policy(checkpoint_path).weight_fqns
        )


if __name__ == "__main__":
    unittest.main()
