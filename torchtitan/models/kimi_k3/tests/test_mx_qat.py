# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for MXFP4/MXFP8 fake-quant QAT on the routed experts."""

import pytest
import torch

pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")

from torchtitan.components.quantization.mx_qat import _fake_quant_mx, MXQATExpertsBase
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_mx_qat


def test_fake_quant_is_ste():
    """Forward is the dequantized value; the gradient is identity."""
    torch.manual_seed(0)
    t = (torch.randn(4, 64) * 0.2).requires_grad_(True)
    q = _fake_quant_mx(t, torch.float4_e2m1fn_x2, 32)
    assert not torch.equal(q, t), "fake-quant changed nothing"
    q.sum().backward()
    torch.testing.assert_close(t.grad, torch.ones_like(t))


def test_fake_quant_leaves_unblockable_alone():
    t = torch.randn(4, 33)
    assert _fake_quant_mx(t, torch.float4_e2m1fn_x2, 32) is t


def test_qat_flavor_swaps_experts_and_masters_stay_params():
    torch.manual_seed(0)
    config = kimi_k3_debugmodel_mx_qat()
    model = config.model_spec.model.build()
    model.init_states()
    qat = [m for _, m in model.named_modules() if isinstance(m, MXQATExpertsBase)]
    assert qat
    m = qat[0]
    # Masters stay ordinary trainable params under their exact names -- the
    # state-dict contract and the EP/TP layout key off them.
    assert "w1_EFD" in m._parameters
    assert m.w1_EFD.requires_grad
    e = m.num_experts
    x = torch.randn(6, m.w1_EFD.shape[-1])
    counts = torch.zeros(e, dtype=torch.long)
    counts[0] = 6
    out = m(x, counts)
    assert out.shape == x.shape
    # The forward-window shadow is gone afterwards: getattr returns the
    # parameter itself, which is what FSDP2's reset_sharded_param needs.
    assert m.w1_EFD is m._parameters["w1_EFD"]


def test_qat_forward_differs_from_parent_and_grads_flow():
    torch.manual_seed(0)
    config = kimi_k3_debugmodel_mx_qat()
    model = config.model_spec.model.build()
    model.init_states()
    m = next(x for _, x in model.named_modules() if isinstance(x, MXQATExpertsBase))
    e = m.num_experts
    x = torch.randn(6, m.w1_EFD.shape[-1])
    counts = torch.zeros(e, dtype=torch.long)
    counts[0] = 6
    out = m(x, counts)
    m._qat_quantize_act = False
    with torch.no_grad():
        ref = type(m).__mro__[1].forward(m, x, counts)
    assert not torch.allclose(
        out, ref, atol=0, rtol=0
    ), "fake-quant forward equals the unquantized parent's"
    out.sum().backward()
    assert m.w1_EFD.grad is not None and m.w1_EFD.grad.abs().sum() > 0
