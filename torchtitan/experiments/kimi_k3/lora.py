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
    # shared experts (common FeedForward leaf naming)
    "w1",
    "w2",
    "w3",
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

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        quantize_base: str | None = None,
    ) -> None:
        super().__init__()
        assert rank > 0
        self.base = base
        self.base.weight.requires_grad_(False)
        self._quantize_base = quantize_base
        if quantize_base == "nf4":
            # QLoRA: pack the frozen base to NF4 (torchao). Lossy by
            # design -- the step-0 identity anchor holds only for the
            # unquantized gated graft; QLoRA trades exactness for a ~4x
            # cut in memory AND (on comms-bound fabrics) in FSDP
            # all-gather traffic.
            from torchao.dtypes.nf4tensor import to_nf4

            self.base.weight = nn.Parameter(
                to_nf4(self.base.weight.data.to(torch.bfloat16)),
                requires_grad=False,
            )
        elif quantize_base is not None:
            raise ValueError(f"Unsupported quantize_base={quantize_base!r}")
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._lora_scaling = alpha / rank
        dev = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=dev)
        )
        self.lora_b = nn.Parameter(
            torch.empty(base.out_features, rank, device=dev)
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
        if self._quantize_base == "nf4":
            from torchao.dtypes.nf4tensor import linear_nf4

            base_out = linear_nf4(x, self.base.weight)
        else:
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
    quantize_base: str | None = None,
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
                    KimiLoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        quantize_base=quantize_base,
                    ),
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
            # Frozen params need no fp32 master copy: keep them bf16
            # resident. At 48B this is the difference between 12 GiB/card
            # sharded (fast, no offload) and 24.6 GiB fp32 shards that
            # force CPU offload (~5 min/step over PCIe). HF checkpoints
            # are bf16, so the load path is dtype-exact too.
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
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


_nf4_experts_cls_cache: dict[type, type] = {}


def _nf4_experts_subclass(cls: type) -> type:
    """Subclass with dequant properties over the NF4-packed expert params."""
    if cls in _nf4_experts_cls_cache:
        return _nf4_experts_cls_cache[cls]

    def _make_fget(name: str):
        def fget(self):
            from torch.distributed.tensor import DTensor
            from torchao.dtypes.nf4tensor import NF4Tensor

            t = self._parameters[name + "_nf4"]
            if isinstance(t, DTensor):
                # Pre-unshard access (outside FSDP's forward window):
                # gather explicitly. During forward FSDP2 exposes the
                # plain unsharded NF4.
                t = t.full_tensor()
            if isinstance(t, NF4Tensor):
                t = t.get_original_weight()
            return t.view(self._nf4_shapes[name])

        return fget

    sub = type(
        f"NF4{cls.__name__}",
        (cls,),
        {n: property(_make_fget(n)) for n in ("w1_EFD", "w2_EDF", "w3_EFD")},
    )
    _nf4_experts_cls_cache[cls] = sub
    return sub


def quantize_grouped_experts_nf4(model: nn.Module) -> int:
    """Pack every GroupedExperts weight to NF4 (the 48B memory/comms bulk).

    3-D [E, A, B] params pack as a 2-D (E*A, B) NF4 view; a dequant
    property restores the logical shape at forward time (GroupedExperts
    reads self.w1_EFD etc. and casts to bf16 anyway). Params stay
    registered (frozen) so FSDP can shard the packed bytes.
    """
    from torchao.dtypes.nf4tensor import to_nf4

    from torchtitan.models.common.moe import GroupedExperts

    num_quantized = 0
    for m in model.modules():
        if isinstance(m, GroupedExperts) and not hasattr(m, "_nf4_shapes"):
            shapes: dict[str, tuple[int, ...]] = {}
            for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
                p = m._parameters.get(name)
                if p is None:
                    continue
                shapes[name] = tuple(p.shape)
                packed = to_nf4(
                    p.data.reshape(-1, p.shape[-1]).to(torch.bfloat16)
                )
                # Store under a distinct name: the logical name becomes
                # a dequant property, and FSDP shards the packed param.
                del m._parameters[name]
                m.register_parameter(
                    name + "_nf4", nn.Parameter(packed, requires_grad=False)
                )
            m._nf4_shapes = shapes
            m.__class__ = _nf4_experts_subclass(type(m))
            num_quantized += 1
    return num_quantized
