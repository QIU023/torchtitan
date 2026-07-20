# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 (weight) + MXFP8 (activation) fake-quant QAT for Kimi K3.

The K3-faithful quantization path (vs the NF4 QLoRA convenience in
``lora.py``): K3 is MXFP4-QAT from SFT (MXFP4 weights, MXFP8
activations, OCP microscaling, block 32). This module provides an
EMULATED fake-quant so QAT runs on any GPU (fake-quant is bf16 compute;
FP4 hardware only speeds deployment, not QAT).

Fidelity scope (honest, matches PLAN 3b):
- Emulated MX rounding targets the OCP spec but is NOT verified
  bit-identical to Moonshot's kernels -> "MX-deployable", not
  "K3-QAT-bit-parity".
- Continued QAT from K3's shipped packed MXFP4 starts from an
  already-degraded master (K3's bf16 master is not released).
- torchao provides the MX primitives (MXTensor.to_mx / dequantize).

The wrapper does straight-through fake-quant: forward uses
dequant(quant(w)) so the loss sees quantized weights, while the bf16
master trains (STE via detach trick).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_WEIGHT_ELEM = torch.float4_e2m1fn_x2  # MXFP4
_ACT_ELEM = torch.float8_e4m3fn        # MXFP8
_BLOCK = 32                        # OCP microscaling block


def _fake_quant_mx(t: torch.Tensor, elem_dtype, block_size: int) -> torch.Tensor:
    """Straight-through emulated MX fake-quant: value = dequant(quant(t)),
    gradient = identity (STE)."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    if t.shape[-1] % block_size != 0:
        return t  # non-blockable dim: leave in high precision (report elsewhere)
    q = MXTensor.to_mx(
        t.contiguous().to(torch.bfloat16), elem_dtype=elem_dtype, block_size=block_size
    ).dequantize()
    # STE: forward q, backward identity through t.
    return t + (q - t).detach()


class MXFP4QATLinear(nn.Module):
    """Fake-quant QAT wrapper over an nn.Linear.

    Weight is fake-quantized to MXFP4, activation to MXFP8, each forward.
    The underlying nn.Linear.weight stays the trainable bf16 master.
    """

    def __init__(self, base: nn.Linear, quantize_act: bool = True) -> None:
        super().__init__()
        self.base = base
        self.quantize_act = quantize_act

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _fake_quant_mx(self.base.weight, _WEIGHT_ELEM, _BLOCK)
        if self.quantize_act:
            x = _fake_quant_mx(x, _ACT_ELEM, _BLOCK)
        return F.linear(x, w, self.base.bias)


def apply_mxfp4_qat(
    model: nn.Module,
    *,
    targets: tuple[str, ...] = (
        "q_proj",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    quantize_act: bool = True,
) -> int:
    """Wrap target Linears with MXFP4/MXFP8 fake-quant QAT.

    KDA projections are excluded (fla reads .weight directly, bypassing
    the module forward -- a wrapper there would be silently skipped).
    GroupedExperts need a separate grouped path (not wrapped here; the
    expert bulk QAT is future work, mirroring the NF4 experts hack but
    with MX primitives).
    """
    from torchtitan.experiments.kimi_k3.model import KimiDeltaAttention

    n = 0
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            continue
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, MXFP4QATLinear(child, quantize_act=quantize_act))
                n += 1
    if n == 0:
        raise ValueError("apply_mxfp4_qat matched no target Linears")
    return n
