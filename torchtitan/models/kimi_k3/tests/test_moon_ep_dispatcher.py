# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-side checks for the MoonEP dispatcher wiring.

MoonEP itself needs NVLink hardware and its package; nothing here touches
either. What CAN be established on CPU: the spec selects the dispatcher, the
buffer is sized by the LATENT width, EP=1 never imports moon_ep, and the
import guard names the package.
"""

import pytest
import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.moon_ep_dispatcher import (
    _import_moon_ep,
    MoonEPTokenDispatcher,
)


def _find_dispatcher(model):
    for _, mod in model.named_modules():
        if hasattr(mod, "token_dispatcher"):
            return mod.token_dispatcher
    raise AssertionError("no routed experts found")


def test_moonep_spec_selects_dispatcher_sized_by_latent():
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    assert isinstance(dispatcher, MoonEPTokenDispatcher)
    # The routed experts consume routed_down's output, so the buffer width is
    # the latent width, not the model width.
    layer = next(m for _, m in model.named_modules() if hasattr(m, "routed_down"))
    assert dispatcher.hidden_dim == layer.routed_down.weight.shape[0]


def test_moonep_ep1_falls_back_to_local_dispatch():
    """With no EP mesh the local fallback runs and moon_ep is never imported:
    dispatch/combine round-trips on CPU like the standard dispatcher's EP=1
    path."""
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    dispatcher.wire_meshes(ep_mesh=None)

    num_tokens, num_experts, top_k = 8, dispatcher.num_experts, dispatcher.top_k
    dim = dispatcher.hidden_dim
    x_TD = torch.randn(num_tokens, dim)
    scores_TK, ids_TK = torch.rand(num_tokens, num_experts).topk(top_k, dim=-1)
    counts_E = torch.zeros(num_experts, dtype=torch.long).scatter_add_(
        0, ids_TK.reshape(-1), torch.ones(num_tokens * top_k, dtype=torch.long)
    )
    routed_RD, counts_e, metadata = dispatcher.dispatch(
        x_TD, scores_TK, ids_TK, counts_E
    )
    assert counts_e.sum().item() == num_tokens * top_k
    out_TD = dispatcher.combine(routed_RD, metadata, x_TD)
    assert out_TD.shape == x_TD.shape


def test_moonep_import_guard_names_the_package():
    try:
        import moon_ep  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("moon_ep installed; the guard has nothing to raise")
    with pytest.raises(ImportError, match="MoonshotAI/MoonEP"):
        _import_moon_ep()
