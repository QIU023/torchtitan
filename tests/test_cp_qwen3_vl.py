# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CPU-only smoke tests for Qwen3-VL Context Parallel wiring (Agent C).

Scope:
  * ``parallelize_qwen3_vl`` raises ``NotImplementedError`` for the
    unsupported PP+CP combination.
  * When CP is enabled (and PP is not), the CP code path is reached and
    ``apply_cp_to_attention_module`` is invoked with the attention modules
    and the CP mesh — verified via mock without spinning up a process
    group, so the test stays CPU-only.
  * The CP-only call site mirrors ``torchtitan/models/qwen3/parallelize.py``
    so the wiring is consistent with the text-only Qwen3 reference.

Full numerical parity (cp_size=1 vs cp_size=2) requires multi-rank
torchrun and is left to Agent D's combined smoke runs.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer


class _DummyAttention(torch.nn.Module):
    """Stand-in for ``GQAttention`` carrying an ``inner_attention`` submodule."""

    def __init__(self):
        super().__init__()
        # ScaledDotProductAttention is the simplest concrete inner attention
        # for the test; the actual class instance is what
        # apply_cp_to_attention_module dispatches on.
        from torchtitan.models.common.attention import (
            ScaledDotProductAttention,
        )
        self.inner_attention = ScaledDotProductAttention.Config().build()


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _DummyAttention()


class _DummyModel(torch.nn.Module):
    """Minimal stand-in for ``Qwen3VLModel`` exposing only what the CP
    wiring touches: ``model.layers[*].attention.inner_attention``."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {str(i): _DummyBlock() for i in range(n_layers)}
        )
        # parallelize_qwen3_vl also touches ``model.vision_encoder`` —
        # set to None so the vision-encoder branches are skipped.
        self.vision_encoder = None


def _make_parallel_dims(
    *, cp: int, pp: int, tp: int = 1, dp_shard: int = 1, ep: int = 1
):
    """Build a mock ``ParallelDims``-like object.

    Real ``ParallelDims.get_mesh("cp")`` would require a process group;
    we return a ``MagicMock`` instead. The CP wiring code only uses the
    mesh as an opaque argument to ``apply_cp_to_attention_module``.
    """
    pd = MagicMock(name="parallel_dims")
    pd.cp_enabled = cp > 1
    pd.pp_enabled = pp > 1
    pd.tp_enabled = tp > 1
    pd.ep_enabled = ep > 1
    pd.dp_replicate_enabled = False
    pd.dp_shard_enabled = dp_shard > 1
    pd.cp = cp
    pd.pp = pp
    pd.tp = tp
    pd.ep = ep

    cp_mesh = MagicMock(name="cp_mesh")
    cp_mesh.size.return_value = cp
    cp_mesh.device_type = "cpu"

    fsdp_mesh = MagicMock(name="fsdp_mesh")

    def _get_mesh(name):
        if name == "cp":
            return cp_mesh
        return fsdp_mesh

    def _get_optional_mesh(name):
        return None

    pd.get_mesh.side_effect = _get_mesh
    pd.get_optional_mesh.side_effect = _get_optional_mesh
    return pd, cp_mesh


def _make_configs():
    """Build the keyword-only config bundle that ``parallelize_qwen3_vl``
    expects. Most fields are unused on the CP-only code path."""
    training = MagicMock(spec=TrainingConfig)
    training.mixed_precision_param = "bfloat16"
    training.mixed_precision_reduce = "float32"
    training.enable_cpu_offload = False

    parallelism = MagicMock(spec=ParallelismConfig)
    parallelism.enable_async_tensor_parallel = False
    parallelism.disable_loss_parallel = False
    parallelism.fsdp_reshard_after_forward = "default"

    compile_config = MagicMock(spec=CompileConfig)
    compile_config.enable = False
    compile_config.components = []

    ac_config = MagicMock(spec=ActivationCheckpointConfig)
    ac_config.mode = "none"

    model_converters = MagicMock(spec=ModelConvertersContainer.Config)
    model_converters.converters = []

    return training, parallelism, compile_config, ac_config, model_converters


class TestCpQwen3VlWiring(unittest.TestCase):
    """CPU-only checks for the CP code path in ``parallelize_qwen3_vl``."""

    def test_pp_plus_cp_raises_not_implemented(self):
        """PP+CP combined must raise; mirrors Megatron's hard assertion."""
        from torchtitan.models.qwen3_vl.parallelize import parallelize_qwen3_vl

        model = _DummyModel(n_layers=2)
        pd, _ = _make_parallel_dims(cp=2, pp=2)
        training, parallelism, compile_config, ac_config, model_converters = (
            _make_configs()
        )

        with self.assertRaises(NotImplementedError) as ctx:
            parallelize_qwen3_vl(
                model,
                parallel_dims=pd,
                training=training,
                model_converters=model_converters,
                parallelism=parallelism,
                compile_config=compile_config,
                ac_config=ac_config,
                dump_folder=".",
            )
        self.assertIn("PP+CP", str(ctx.exception))

    def test_cp_only_path_calls_apply_cp(self):
        """When CP>1 and PP=1, ``apply_cp_to_attention_module`` is called
        with all attention modules and the CP mesh."""
        from torchtitan.models.qwen3_vl import parallelize as p

        model = _DummyModel(n_layers=3)
        pd, cp_mesh = _make_parallel_dims(cp=2, pp=1)
        training, parallelism, compile_config, ac_config, model_converters = (
            _make_configs()
        )

        # Mock the heavyweight bits so we don't need a real mesh or
        # process group on CPU.
        with patch.object(p, "apply_cp_to_attention_module") as mock_cp, patch.object(
            p, "apply_fsdp"
        ), patch.object(p, "apply_moe_ep_tp"):
            p.parallelize_qwen3_vl(
                model,
                parallel_dims=pd,
                training=training,
                model_converters=model_converters,
                parallelism=parallelism,
                compile_config=compile_config,
                ac_config=ac_config,
                dump_folder=".",
            )

        self.assertEqual(mock_cp.call_count, 1)
        args, kwargs = mock_cp.call_args
        passed_modules, passed_mesh = args
        # Same set of inner_attention modules, in layer order.
        expected = [
            model.layers[k].attention.inner_attention for k in model.layers.keys()
        ]
        self.assertEqual(len(passed_modules), len(expected))
        for got, exp in zip(passed_modules, expected):
            self.assertIs(got, exp)
        self.assertIs(passed_mesh, cp_mesh)

    def test_cp_disabled_no_apply_cp(self):
        """When CP=1, ``apply_cp_to_attention_module`` must not be invoked."""
        from torchtitan.models.qwen3_vl import parallelize as p

        model = _DummyModel(n_layers=2)
        pd, _ = _make_parallel_dims(cp=1, pp=1)
        training, parallelism, compile_config, ac_config, model_converters = (
            _make_configs()
        )

        with patch.object(p, "apply_cp_to_attention_module") as mock_cp, patch.object(
            p, "apply_fsdp"
        ), patch.object(p, "apply_moe_ep_tp"):
            p.parallelize_qwen3_vl(
                model,
                parallel_dims=pd,
                training=training,
                model_converters=model_converters,
                parallelism=parallelism,
                compile_config=compile_config,
                ac_config=ac_config,
                dump_folder=".",
            )

        mock_cp.assert_not_called()

    def test_apply_cp_to_attention_module_is_importable(self):
        """The symbol used by parallelize.py must be importable from
        the distributed context_parallel module — guards against silent
        rename/regression at the import site."""
        from torchtitan.distributed.context_parallel import (
            apply_cp_to_attention_module,
            cp_shard,
            prepare_context_parallel_input,
        )
        self.assertTrue(callable(apply_cp_to_attention_module))
        self.assertTrue(callable(prepare_context_parallel_input))
        self.assertTrue(callable(cp_shard))

    def test_cp_call_site_mirrors_qwen3_text_only(self):
        """Sanity: qwen3_vl uses the same ``[block.attention.inner_attention
        for block in model.layers.values()]`` pattern as qwen3 text-only."""
        import inspect
        from torchtitan.models.qwen3_vl import parallelize as p

        src = inspect.getsource(p.parallelize_qwen3_vl)
        self.assertIn("apply_cp_to_attention_module", src)
        self.assertIn("block.attention.inner_attention", src)
        self.assertIn("parallel_dims.get_mesh(\"cp\")", src)


if __name__ == "__main__":
    unittest.main()
