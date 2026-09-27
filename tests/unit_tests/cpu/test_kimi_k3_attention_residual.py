# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3's attention residual: its block list and its recompute in backward."""

import dataclasses
import unittest

import torch

from torchtitan.models.kimi_k3 import model as k3_model, model_registry
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
        config = model_registry("debugmodel", enable_sp=False)
        assert isinstance(config, k3_model.KimiK3Model.Config)
        layers = config.layers
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
        blocks_TD: list[torch.Tensor] = []
        for block in self.layers.values():
            x_TD, blocks_TD = block(x_TD, blocks_TD)
        return x_TD.pow(2).mean()


def _dim() -> int:
    return model_registry("debugmodel", enable_sp=False).dim


class TestKimiK3AttentionResidualBlocks(unittest.TestCase):
    def test_a_block_opening_appends_the_blocks_it_was_given(self):
        opener = _TwoBlocks().layers["0"]
        assert isinstance(opener, k3_model.KimiK3TransformerBlock)
        assert opener.first_layer_in_block
        x_TD = torch.randn(_TOKENS, _dim())
        given = [torch.randn(_TOKENS, _dim())]
        _, blocks_TD = opener(x_TD, given)
        self.assertEqual(len(blocks_TD), 2)
        self.assertIs(blocks_TD[0], given[0])
        self.assertIs(blocks_TD[1], x_TD)


if __name__ == "__main__":
    unittest.main()
