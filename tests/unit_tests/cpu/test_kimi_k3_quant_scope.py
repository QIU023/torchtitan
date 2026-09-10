# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MXFP4 QAT scope against the released checkpoint's quantized keys.

The released Kimi K3 index stores every quantized weight as ``.weight_packed``
plus ``.weight_scale``; the QAT converter's scope is structural (every
``GroupedExperts.Config``). This checks the two agree: the packed keys are
exactly the routed experts, and the converter reaches exactly the routed-expert
modules. Skipped when the released index (``KIMI_K3_RELEASED_INDEX``) is not
on the machine.
"""

import json
import os
import pathlib
import re
from typing import cast

import pytest
import torch

from torchtitan.components.quantization.mx_qat import MXFP4QATConverter
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import KimiK3Model

_INDEX = pathlib.Path(
    os.environ.get(
        "KIMI_K3_RELEASED_INDEX",
        "/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/"
        "official_k3/reference/model.safetensors.index.json",
    )
)
_EXPERT_KEY = re.compile(
    r"language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\."
)


def test_released_index_quantizes_exactly_the_routed_experts():
    if not _INDEX.exists():
        pytest.skip(f"no released index at {_INDEX}")
    keys = list(json.load(open(_INDEX))["weight_map"])
    packed = {k for k in keys if k.endswith((".weight_packed", ".weight_scale"))}
    assert packed, "the released index carries no packed weights"
    not_experts = sorted(k for k in packed if _EXPERT_KEY.match(k) is None)
    assert not_experts == []
    plain_experts = sorted(
        k for k in keys if _EXPERT_KEY.match(k) is not None and k.endswith(".weight")
    )
    assert plain_experts == []


def test_converter_reaches_exactly_the_routed_expert_modules():
    spec = model_registry("report_arch", seq_len=256, num_mtp_layers=0)
    model_config = cast(KimiK3Model.Config, spec.model)
    with torch.device("meta"):
        model = model_config.build()
    expert_modules = {
        name
        for name, module in model.named_modules()
        if isinstance(module, GroupedExperts)
    }
    assert expert_modules
    converted = MXFP4QATConverter(MXFP4QATConverter.Config()).convert(model_config)
    with torch.device("meta"):
        qat_model = converted.build()
    reached = {
        name
        for name, module in qat_model.named_modules()
        if isinstance(module, GroupedExperts) and type(module) is not GroupedExperts
    }
    assert reached == expert_modules
