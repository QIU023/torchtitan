# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Kimi K3 state dict adapter against a checkpoint in the released format.

A debug-size checkpoint in the released layout (``KIMI_K3_RELEASED_DEBUG_DIR``,
a directory with ``config.json`` and one safetensors shard) is loaded into the
``report_arch`` flavor, whose topology it matches; every key must land on a
model parameter with the parameter's shape, and the export must reproduce the
released tensors bitwise. Skipped when the checkpoint is not on the machine.
"""

import json
import os
import pathlib

from typing import cast

import pytest
import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

_RELEASED_DIR = pathlib.Path(
    os.environ.get("KIMI_K3_RELEASED_DEBUG_DIR", "/workspace/k3qat_mm_hf")
)


def _released_dir() -> pathlib.Path:
    if not (_RELEASED_DIR / "config.json").exists():
        pytest.skip(f"no released-format checkpoint at {_RELEASED_DIR}")
    return _RELEASED_DIR


def _twin() -> tuple[KimiK3Model.Config, KimiK3Model]:
    spec = model_registry("report_arch", seq_len=256)
    model_config = cast(KimiK3Model.Config, spec.model)
    with torch.device("meta"):
        model = model_config.build()
    return model_config, model


def test_report_arch_matches_the_released_config():
    config = json.load(open(_released_dir() / "config.json"))
    text = config.get("text_config", config)
    model_config, _ = _twin()
    layers = model_config.layers
    assert len(layers) == text["num_hidden_layers"]
    full_attention = [
        i + 1 for i, layer in enumerate(layers) if layer.attention is not None
    ]
    assert full_attention == text["linear_attn_config"]["full_attn_layers"]
    assert layers[0].attn_res_block_size == text["attn_res_block_size"]
    mla = layers[full_attention[0] - 1].attention
    assert mla is not None
    assert (mla.qk_nope_head_dim, mla.qk_rope_head_dim, mla.v_head_dim) == (
        text["qk_nope_head_dim"],
        text["qk_rope_head_dim"],
        text["v_head_dim"],
    )
    assert mla.kv_lora_rank == text["kv_lora_rank"]
    kda = layers[0].delta_attention
    assert kda is not None
    assert (kda.num_heads, kda.head_dim) == (
        text["linear_attn_config"]["num_heads"],
        text["linear_attn_config"]["head_dim"],
    )
    moe = layers[1].moe
    assert moe is not None
    assert moe.num_experts == text["num_experts"]
    assert moe.router.top_k == text["num_experts_per_token"]
    vision = config["vision_config"]
    tower = model_config.vision_encoder
    assert tower is not None
    assert tower.num_layers == vision["vt_num_hidden_layers"]
    assert tower.dim == vision["vt_hidden_size"]
    assert tower.init_pos_emb_height == vision["init_pos_emb_height"]


def test_released_checkpoint_round_trips_through_the_adapter():
    from safetensors.torch import load_file

    released = _released_dir()
    shards = sorted(released.glob("*.safetensors"))
    hf_state_dict = {}
    for shard in shards:
        hf_state_dict.update(load_file(str(shard)))
    model_config, model = _twin()
    expected = {name: tuple(param.shape) for name, param in model.state_dict().items()}
    adapter = KimiK3StateDictAdapter(model_config, None)

    converted = adapter.from_hf(hf_state_dict)
    assert set(converted) == set(expected)
    mismatched = {
        key: (tuple(value.shape), expected[key])
        for key, value in converted.items()
        if tuple(value.shape) != expected[key]
    }
    assert mismatched == {}

    exported = adapter.to_hf(converted)
    assert set(exported) == set(hf_state_dict)
    for key, value in hf_state_dict.items():
        assert exported[key].shape == value.shape, key
        assert torch.equal(exported[key], value), key


def test_packed_tensors_are_refused():
    model_config, _ = _twin()
    adapter = KimiK3StateDictAdapter(model_config, None)
    packed = {
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": (
            torch.zeros(4, 4, dtype=torch.uint8)
        )
    }
    with pytest.raises(NotImplementedError, match="packed"):
        adapter.from_hf(packed)
