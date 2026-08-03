# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3 routed experts: SiTU-GLU instead of SwiGLU.

The released config sets ``hidden_act: "situ"`` globally, so the routed
experts -- which are the overwhelming majority of the model's FLOPs and
parameters -- use tech report Eq. 12, not SiLU. ``GroupedExperts`` in
``models/common`` hardcodes ``F.silu``, so this subclasses it the way
``GptOssGroupedExperts`` does for its clamped SwiGLU: same ``w1_EFD`` /
``w2_EDF`` / ``w3_EFD`` parameters (so the state-dict adapter, the expert
TP/EP layout, and the torchao MX/Float8 expert converters all keep working
unchanged), only the activation differs.

Shape suffixes follow ``models/common/moe.py``: R routed tokens on this
rank, D model dim, F expert hidden dim, E experts.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.moe import GroupedExperts

from .model import situ_and_mul


class KimiSiTUGroupedExperts(GroupedExperts):
    """Grouped routed experts with K3's SiTU-GLU activation (Eq. 12).

    ``situ_linear_beta=None`` leaves the linear branch unclipped; K3 ships
    ``beta1=4`` on the gate branch and ``beta2=25`` on the linear branch,
    bounding the product at 100.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        situ_beta: float = 4.0
        situ_linear_beta: float | None = 25.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.situ_beta = config.situ_beta
        self.situ_linear_beta = config.situ_linear_beta

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            # Plain tensors for the dynamic-shape EP path, as the base does.
            w1_EFD = self.w1_EFD.to_local()
            w2_EDF = self.w2_EDF.to_local()
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        gate_RF = torch._grouped_mm(
            x_RD.bfloat16(), w1_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        up_RF = torch._grouped_mm(
            x_RD.bfloat16(), w3_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        h_RF = situ_and_mul(
            gate_RF, up_RF, self.situ_beta, self.situ_linear_beta
        )
        return torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)
