# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The packed-weight scope of the released Kimi K3 checkpoint.

The released index stores every quantized weight as ``.weight_packed`` plus
``.weight_scale``, and the adapter's quantized storage reader dequantizes
exactly those on load. This checks the packed keys are the routed experts and
nothing else, so a load through the reader leaves no packed tensor behind for
``from_hf`` to refuse. Skipped when the released index
(``KIMI_K3_RELEASED_INDEX``) is not on the machine.
"""

import json
import os
import pathlib
import re

import pytest

_INDEX = os.environ.get("KIMI_K3_RELEASED_INDEX")
_EXPERT_KEY = re.compile(
    r"language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\."
)


def test_released_index_packs_exactly_the_routed_experts():
    if _INDEX is None or not pathlib.Path(_INDEX).exists():
        pytest.skip(f"no released index at KIMI_K3_RELEASED_INDEX={_INDEX}")
    keys = list(json.load(open(_INDEX))["weight_map"])
    packed = {k for k in keys if k.endswith((".weight_packed", ".weight_scale"))}
    assert packed, "the released index carries no packed weights"
    not_experts = sorted(k for k in packed if _EXPERT_KEY.match(k) is None)
    assert not_experts == []
    plain_experts = sorted(
        k for k in keys if _EXPERT_KEY.match(k) is not None and k.endswith(".weight")
    )
    assert plain_experts == []
