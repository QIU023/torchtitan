# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""to_hf on a pipeline stage's partial state dict (the HF initial load under PP)."""

import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

_NORM0 = "language_model.model.layers.0.self_attention_res_norm.weight"
_PROJ0 = "language_model.model.layers.0.self_attention_res_proj.weight"


def _adapter():
    return KimiK3StateDictAdapter(model_registry("debugmodel").model, hf_assets_path=None)


def test_a_stage_without_layer_0_places_nothing():
    out = _adapter().to_hf({"layers.2.attention_norm.weight": torch.ones(1024)})
    assert _NORM0 not in out and _PROJ0 not in out


def test_layer_0_without_layer_1_is_shaped_from_the_config():
    out = _adapter().to_hf({"layers.0.attention_norm.weight": torch.ones(1024, dtype=torch.bfloat16)})
    assert out[_NORM0].shape == (1024,) and out[_NORM0].dtype == torch.bfloat16
    assert out[_PROJ0].shape == (1, 1024) and not out[_PROJ0].any()


def test_layer_0_with_layer_1_follows_the_template():
    out = _adapter().to_hf(
        {
            "layers.0.attention_norm.weight": torch.ones(1024),
            "layers.1.attention_res_norm.weight": torch.ones(1024),
            "layers.1.attention_res_proj.weight": torch.zeros(1, 1024),
        }
    )
    assert out[_NORM0].shape == (1024,) and out[_PROJ0].shape == (1, 1024)
