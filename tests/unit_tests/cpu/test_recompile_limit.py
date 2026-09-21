# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch._dynamo
import torch.nn as nn

from torchtitan.distributed.compile import raise_dynamo_recompile_limit
from torchtitan.models.gpt_oss.parallelize import _min_recompile_limit
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.parallelize import _kimi_k3_recompile_limit


def test_raise_dynamo_recompile_limit_only_raises(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    raise_dynamo_recompile_limit(12)
    assert torch._dynamo.config.recompile_limit == 12
    raise_dynamo_recompile_limit(10)
    assert torch._dynamo.config.recompile_limit == 12


def test_gpt_oss_recompile_limit():
    class _Attn(nn.Module):
        def __init__(self, window_size):
            super().__init__()
            self.window_size = window_size

    sliding = nn.Sequential(_Attn((128, 0)), _Attn((-1, -1)))
    full = nn.Sequential(_Attn((-1, -1)))
    assert _min_recompile_limit(sliding) == 12
    assert _min_recompile_limit(full) == 10


def test_kimi_k3_recompile_limit_counts_layer_variants():
    # 17 layers in blocks of 4: MLA or KDA, opening a block or not, stack width 0 / 1 / 2+
    # give eight layer variants, plus the tower's two.
    assert _kimi_k3_recompile_limit(model_registry("debugmodel").model) == 10
    assert _kimi_k3_recompile_limit(model_registry("Kimi-K3").model) == 9
