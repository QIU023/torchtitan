# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.models.kimi_k3 import model as k3_model, model_registry
from torchtitan.models.kimi_k3.model import AttentionResidual
from torchtitan.protocols.module import Module, ModuleDict

_TOKENS = 8
# The attention kernels are CUDA-only; these CPU stand-ins keep the call signatures.


class _CpuKernel(Module):
    """A CPU stand-in installed where the model expects a kernel module."""

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


def _mla_inner_attention(q_THK, k_THK, v_THV, *, attention_masks=None, scale=None):
    del attention_masks
    return torch.nn.functional.scaled_dot_product_attention(
        q_THK.transpose(0, 1),
        k_THK.transpose(0, 1),
        v_THV.transpose(0, 1),
        is_causal=True,
        scale=scale,
    ).transpose(0, 1)


def _kda_inner(
    query_TC, key_TC, value_TC, raw_gate_THK, raw_beta_TH, *params, cu_seqlens=None
):
    del cu_seqlens
    num_tokens, num_heads = raw_beta_TH.shape
    q_THK, k_THK, v_THV = (
        t.view(num_tokens, num_heads, -1) for t in (query_TC, key_TC, value_TC)
    )
    gate_THK = torch.sigmoid(raw_gate_THK) * torch.sigmoid(raw_beta_TH).unsqueeze(-1)
    scale = 1 + sum(p.float().mean() for p in params)
    return (q_THK * k_THK).sum(-1, keepdim=True).tanh() * v_THV * gate_THK * scale


class _TwoBlocks(Module):
    """A KDA block that opens the attention-residual block, then an MLA block."""

    def __init__(self):
        super().__init__()
        layers = model_registry("debugmodel").layers
        kda_config = layers[0]
        assert kda_config.delta_attention is not None and kda_config.moe is None
        mla_config = dataclasses.replace(
            next(layer for layer in layers if layer.attention is not None),
            layer_id=1,
            moe=None,
            feed_forward=kda_config.feed_forward,
        )
        torch.manual_seed(0)
        blocks = [kda_config.build(), mla_config.build()]
        for block in blocks:
            block.init_states()
        blocks[0].delta_attention.inner_kda = _CpuKernel(_kda_inner)
        blocks[1].attention.inner_attention = _CpuKernel(_mla_inner_attention)
        self.layers = ModuleDict({str(i): block for i, block in enumerate(blocks)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        block_residual_TND = x_TD.unsqueeze(1)[:, :0]
        for block in self.layers.values():
            x_TD, block_residual_TND = block(x_TD, block_residual_TND)
        return x_TD.pow(2).mean()


def _run_forward_backward(model: Module, x_TD: torch.Tensor):
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    output = model(input_TD)
    output.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return output.detach(), input_TD.grad, grads


def _unwrapped_residual():
    """The residual math without its checkpoint: the reference and the control."""
    return patch.object(k3_model.remat, "checkpoint", lambda **_: (lambda fn: fn))


def _saved_shapes(model: Module, x_TD: torch.Tensor) -> list[tuple[int, ...]]:
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(x_TD.detach().clone().requires_grad_(True))
    return shapes


class TestKimiK3AttentionResidual(unittest.TestCase):
    def test_the_aggregation_is_a_config_node(self):
        # The override mechanism swaps Config nodes, so every aggregation sits
        # in the model's config tree while the weights stay on the block.
        config = model_registry("debugmodel")
        for node in (
            config.layers[0].ffn_res_fn,
            config.layers[1].attention_res_fn,
            config.output_res_fn,
        ):
            self.assertIsInstance(node, AttentionResidual.Config)
        model = config.build()
        self.assertEqual(
            [name for name, _ in model.named_parameters() if "_res_fn" in name], []
        )

    def test_residual_math_reruns_in_backward_bitwise(self):
        model = _TwoBlocks()
        reference = deepcopy(model)
        x_TD = torch.randn(_TOKENS, 1024)
        with _unwrapped_residual():
            expected = _run_forward_backward(reference, x_TD)
        calls = []
        original = AttentionResidual.__call__

        def counting(self, *args):
            calls.append(None)
            return original(self, *args)

        with patch.object(AttentionResidual, "__call__", counting):
            actual = _run_forward_backward(model, x_TD)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(actual[2].keys(), expected[2].keys())
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, rtol=0, atol=0, msg=name)
        # Three residual computations in forward, each run again in backward.
        self.assertEqual(len(calls), 6)

    def test_covered_block_runs_the_residual_once(self):
        model = _TwoBlocks()
        for block in model.layers.values():
            assert isinstance(block, k3_model.KimiK3TransformerBlock)
            block.checkpoint_residual = False
        reference = deepcopy(model)
        x_TD = torch.randn(_TOKENS, 1024)
        with _unwrapped_residual():
            expected = _run_forward_backward(reference, x_TD)
        calls = []
        original = AttentionResidual.__call__

        def counting(self, *args):
            calls.append(None)
            return original(self, *args)

        with patch.object(AttentionResidual, "__call__", counting):
            actual = _run_forward_backward(model, x_TD)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, rtol=0, atol=0, msg=name)
        # The block's own activation checkpoint is the recompute; no second one here.
        self.assertEqual(len(calls), 3)

    def test_no_stack_shaped_activation_is_saved(self):
        model = _TwoBlocks()
        x_TD = torch.randn(_TOKENS, 1024)
        # Every residual here reads a one-entry stack plus the prefix sum.
        stack_shape = (_TOKENS, 2, 1024)
        with _unwrapped_residual():
            self.assertIn(stack_shape, _saved_shapes(model, x_TD))
        self.assertNotIn(stack_shape, _saved_shapes(model, x_TD))

    def test_region_ac_runs_the_residual_as_a_plain_call(self):
        # RegionAC makes each block a torch_remat checkpoint, and torch_remat
        # rejects a nested one, so a covered block must not open its own.
        x_TD = torch.randn(_TOKENS, 1024)
        model = _TwoBlocks()
        reference = deepcopy(model)
        with _unwrapped_residual():
            expected = _run_forward_backward(reference, x_TD)
        uncovered = deepcopy(model)
        RegionAC.Config(save_regions=[]).build().apply(uncovered)
        with self.assertRaisesRegex(NotImplementedError, "nested torch_remat"):
            _run_forward_backward(uncovered, x_TD)
        for block in model.layers.values():
            assert isinstance(block, k3_model.KimiK3TransformerBlock)
            block.checkpoint_residual = False
        RegionAC.Config(save_regions=[]).build().apply(model)
        actual = _run_forward_backward(model, x_TD)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, rtol=0, atol=0, msg=name)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_parallelize_marks_blocks_an_ac_policy_wraps(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)
        config = model_registry("debugmodel")
        for ac_config, checkpoint_residual in (
            (None, True),
            (SelectiveAC.Config(), False),
            (FullAC.Config(), False),
            (RegionAC.Config(save_regions=[]), False),
        ):
            with torch.device("meta"):
                model = config.build()
            model.parallelize(
                parallel_dims=ParallelDims(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=1,
                    pp=1,
                    ep=1,
                    world_size=1,
                    enable_sequence_parallel=False,
                ),
                training=TrainingConfig(),
                parallelism=ParallelismConfig(),
                compile_config=None,
                ac_config=ac_config,
                dump_folder="",
                skip_dp=True,
            )
            for layer in model.layers.values():
                block = getattr(layer, "_checkpoint_wrapped_module", layer)
                assert isinstance(block, k3_model.KimiK3TransformerBlock)
                self.assertIs(
                    block.checkpoint_residual, checkpoint_residual, msg=ac_config
                )


if __name__ == "__main__":
    unittest.main()
