# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-level LoRA for the plain-module Kimi Linear model.

Upstream's ``LoRAConverter`` (components/lora.py) operates on
``Linear.Config`` trees and cannot apply to this experiment's
directly-built modules -- same situation as Float8 (see
``KimiLinearFloat8Spec``). This is the module-level counterpart:
``apply_lora`` swaps target ``nn.Linear`` projections for
:class:`KimiLoRALinear` wrappers after build.

P0 semantics (HANDOFF LoRA trio):

* ``lora_b`` zero-init -> the wrapped model is BIT-IDENTICAL to the
  base model at step 0 (composes with the alpha graft gate: gated
  graft + LoRA both preserve the pretrained function exactly).
* Base-freeze walks the whole model, EXCEPT the AttnRes graft params
  (pseudo-queries, norms, alphas): those are new zero-init params with
  no pretrained value -- LoRA-ing them is meaningless, they must train
  full-param (the "alpha-fullparam exception").
* ``trainable_state_dict`` gives the LoRA-only checkpoint payload
  (adapters + AttnRes params), the unit veRL weight-sync ships.

TP-plan extension for LoRA (colwise/rowwise adapter placements per
``_lora_adapter_sharding``) is NOT wired yet: P0 targets the veRL
FSDP path first. Wrapped FQNs keep their public name (``q_proj`` ->
``q_proj.base`` + ``q_proj.lora_a/b``), so the TP plan must be
extended before combining LoRA with tensor_parallel_degree > 1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# KDA-internal projections are NOT targetable: KimiDeltaAttention reads
# ``linear.weight`` directly for the fla kernels (module forward is
# bypassed), so a wrapper there would be silently dead. apply_lora
# skips the KDA subtree structurally; the name set below only needs to
# cover MLA + dense/shared FFN.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Params that stay full-param trainable under base-freeze: the AttnRes
# graft set (new zero-init params; the "alpha-fullparam exception").
_FULLPARAM_EXCEPTION_MARKERS: tuple[str, ...] = (
    "attn_res",
    "mlp_res",
)


class KimiLoRALinear(nn.Module):
    """LoRA wrapper over an existing ``nn.Linear``.

    ``forward = base(x) + (alpha / rank) * lora_b(lora_a(x))`` with
    ``lora_a`` kaiming-init and ``lora_b`` zero-init (identity at
    step 0). Adapters are raw parameters (not nn.Linear children) so
    the model's generic init pass does not blindly re-init them;
    :meth:`reset_parameters` is dispatched from
    ``KimiLinearModel.init_weights`` by class name.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        assert rank > 0
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._lora_scaling = alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.lora_b = nn.Parameter(
            torch.empty(base.out_features, rank)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lora_a.device.type != "meta":
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_out + self._lora_scaling * lora_out


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    freeze_base: bool = True,
) -> int:
    """Swap target Linears for LoRA wrappers; optionally freeze the base.

    Returns the number of wrapped modules. Freezing covers every
    parameter except LoRA adapters and the AttnRes graft params
    (alpha-fullparam exception).
    """
    from torchtitan.experiments.kimi_k3.model import KimiDeltaAttention

    num_wrapped = 0
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            # Structural skip -- see DEFAULT_LORA_TARGETS note.
            continue
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    KimiLoRALinear(child, rank=rank, alpha=alpha),
                )
                num_wrapped += 1
    if num_wrapped == 0:
        raise ValueError(
            f"apply_lora matched no target Linears (targets={targets})."
        )

    if freeze_base:
        for name, p in model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                continue
            if any(m in name for m in _FULLPARAM_EXCEPTION_MARKERS):
                continue
            p.requires_grad_(False)
    return num_wrapped


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA-only checkpoint payload: adapters + AttnRes graft params.

    This is the unit a veRL trainer->rollout weight sync ships when the
    base is frozen (LoRA-only DCP leg of the P0 trio).
    """
    return {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad
    }
