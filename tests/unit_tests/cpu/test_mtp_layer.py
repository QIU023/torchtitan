# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build-level checks for the MTP layer (forward needs GPU: KDA is triton)."""

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.mtp import KimiK3MTPLayer


def test_mtp_spec_builds_mirror_layer():
    spec = model_registry("debugmodel", num_mtp_layers=1)
    model = spec.model.build()
    model.init_states()
    assert model.mtp_layers is not None and len(model.mtp_layers) == 1
    layer = model.mtp_layers["0"]
    assert isinstance(layer, KimiK3MTPLayer)
    # KDA-typed mirror: no MLA (which would need per-depth mask rebuilds),
    # opens its own block (layer_id 0), dense FFN not MoE.
    assert layer.block.attention is None
    assert layer.block.delta_attention is not None
    assert layer.block.moe is None
    d = model.tok_embeddings.weight.shape[1]
    assert layer.eh_proj.weight.shape == (d, 2 * d)


def test_default_spec_has_no_mtp():
    spec = model_registry("debugmodel")
    model = spec.model.build()
    assert model.mtp_layers is None
