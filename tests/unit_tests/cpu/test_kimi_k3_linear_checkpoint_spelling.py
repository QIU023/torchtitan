# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The text-only checkpoint spelling of a Kimi-Linear-shaped model.

A config without a vision tower exports the released Kimi-Linear-48B spelling
(``model.*`` / ``lm_head.weight``, the low-rank KDA output gate, the
uncompressed MLA query, the plain routed MoE) and nothing the released
architecture lacks: the attention-residual reads and the graft alphas stay
out, so an official checkpoint loads into a graft flavor and the reads keep
their initialisation. The export reads back unchanged.
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import _apply_graft_gate, _kimi_linear_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

_NUM_LAYERS = 5
_FULL_ATTENTION_LAYERS = {3, 4}
_NUM_EXPERTS = 4


def _config() -> KimiK3Model.Config:
    """KDA and MLA layers, the dense layer 0 and the routed MoE, gated."""
    return _apply_graft_gate(
        _kimi_linear_config(
            dim=64,
            vocab_size=64,
            num_layers=_NUM_LAYERS,
            full_attention_layers=_FULL_ATTENTION_LAYERS,
            attn_res_block_size=2,
            num_heads=2,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            kda_head_dim=128,
            conv_kernel_size=4,
            dense_hidden_dim=128,
            expert_hidden_dim=32,
            num_experts=_NUM_EXPERTS,
            top_k=2,
            num_shared_experts=1,
            route_scale=2.446,
            attn_backend="flex",
        )
    )


_KDA_LEAVES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
    "A_log",
    "dt_bias",
)
_MLA_LEAVES = (
    "q_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
)


def _released_spelling() -> set[str]:
    """Every key the released text-only spelling has for ``_config()``."""
    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for i in range(_NUM_LAYERS):
        layer = f"model.layers.{i}."
        keys |= {
            layer + "input_layernorm.weight",
            layer + "post_attention_layernorm.weight",
        }
        leaves = _MLA_LEAVES if i in _FULL_ATTENTION_LAYERS else _KDA_LEAVES
        keys |= {layer + "self_attn." + leaf for leaf in leaves}
        if i == 0:
            keys |= {
                layer + f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")
            }
        else:
            moe = layer + "block_sparse_moe."
            keys |= {moe + "gate.weight", moe + "gate.e_score_correction_bias"}
            keys |= {
                moe + f"shared_experts.{p}.weight"
                for p in ("gate_proj", "up_proj", "down_proj")
            }
            keys |= {
                moe + f"experts.{e}.{w}.weight"
                for e in range(_NUM_EXPERTS)
                for w in ("w1", "w2", "w3")
            }
    return keys


class TestKimiLinearCheckpointSpelling(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        torch.manual_seed(0)
        with torch.device("cpu"):
            self.model = self.config.build()
        self.model.init_states(buffer_device=torch.device("cpu"))
        self.adapter = KimiK3StateDictAdapter(self.config, hf_assets_path=None)

    def test_to_hf_writes_the_released_text_only_spelling(self):
        hf_state_dict = self.adapter.to_hf(self.model.state_dict())
        self.assertEqual(set(hf_state_dict), _released_spelling())
        # The KDA decay bias is flat per the release, [H * K].
        self.assertEqual(hf_state_dict["model.layers.0.self_attn.dt_bias"].ndim, 1)

    def test_from_hf_reads_the_export_back_unchanged(self):
        state_dict = self.model.state_dict()
        extras = {
            k
            for k in state_dict
            if "attention_res" in k or "ffn_res" in k or k.startswith("output_res")
        }
        # Every read and alpha of the gated model, and nothing else, stays out.
        self.assertEqual(len(extras), 4 * _NUM_LAYERS - 2 + 2 + 2 * _NUM_LAYERS + 1)
        restored = self.adapter.from_hf(self.adapter.to_hf(state_dict))
        self.assertEqual(set(restored), set(state_dict) - extras)
        for key, value in restored.items():
            self.assertTrue(torch.equal(value, state_dict[key]), key)

    def test_a_released_checkpoint_loads_into_the_graft_and_leaves_its_reads_alone(
        self,
    ):
        """The graft path: base weights without reads or alphas load into a
        gated model; the reads keep their initialisation and the alphas their
        zero (the checkpointer loads with strict=False, as the released K3 does)."""
        exported = self.adapter.to_hf(self.model.state_dict())
        torch.manual_seed(1)
        with torch.device("cpu"):
            graft = self.config.build()
        graft.init_states(buffer_device=torch.device("cpu"))
        before = {k: v.clone() for k, v in graft.state_dict().items()}
        missing, unexpected = graft.load_state_dict(
            self.adapter.from_hf(exported), strict=False
        )
        self.assertEqual(unexpected, [])
        self.assertEqual(
            set(missing),
            {
                k
                for k in before
                if "attention_res" in k or "ffn_res" in k or k.startswith("output_res")
            },
        )
        after = graft.state_dict()
        for key in missing:
            self.assertTrue(torch.equal(after[key], before[key]), key)
            if key.endswith("alpha"):
                self.assertEqual(after[key].item(), 0.0)
        # Everything the checkpoint carries took the checkpoint's values.
        source = self.model.state_dict()
        for key in set(after) - set(missing):
            self.assertTrue(torch.equal(after[key], source[key]), key)


if __name__ == "__main__":
    unittest.main()
