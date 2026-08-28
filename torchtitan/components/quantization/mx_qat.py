# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 (weight) + MXFP8 (activation) fake-quant QAT for grouped experts.

The Kimi K3 quantization path (pytorch/torchtitan RFC on K3 MXFP4/MXFP8 QAT):
MXFP4 weights and MXFP8 activations, OCP microscaling, block 32, with bf16
master parameters training underneath -- fake-quant is bf16 compute, so QAT
runs on any GPU; FP4 hardware only speeds deployment.

Scope. K3's released ``quantization_config`` targets ``["Linear"]`` but its
ignore list removes attention, shared experts, dense FFN projections, lm_head
and the vision tower -- report sec 4.1.4 states the intent directly: quantize
the MoE expert weights, which dominate parameter memory, and keep every
non-expert component in higher precision. In this module tree the routed
experts are the ``GroupedExperts`` 3-D parameters and nothing else, so the
converter's isinstance check on ``GroupedExperts.Config`` IS the official
scope. (Name-based Linear target lists quantized precisely the set K3 keeps in
high precision and skipped the only set it quantizes.)

Fidelity, honestly: the emulated MX rounding targets the OCP spec but is NOT
verified bit-identical to Moonshot's kernels -- "MX-deployable", not
"K3-QAT-bit-parity". torchao provides the MX primitives.

This is fake-quant QAT (bf16 masters, quantize at import/export only), the
complement of ``components/lora.py``'s QLoRA (really-packed frozen bases,
trainable adapters). The two do not compose on the same weights.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

_WEIGHT_ELEM = torch.float4_e2m1fn_x2  # MXFP4
_ACT_ELEM = torch.float8_e4m3fn  # MXFP8
_BLOCK = 32  # OCP microscaling block

_EXPERT_WEIGHT_NAMES = ("w1_EFD", "w2_EDF", "w3_EFD")

_warned_shapes: set[tuple] = set()


def _warn_unquantized(shape: tuple, block_size: int) -> None:
    if shape in _warned_shapes:
        return
    _warned_shapes.add(shape)
    logger.warning(
        "MXFP4 QAT: tensor of shape %s left UNQUANTIZED -- last dim %d is not "
        "a multiple of the MX block size %d. Under TP this is what a shard of "
        "w2_EDF looks like, so the effective quantization scope is narrower "
        "than requested and depends on the parallel layout. Choose an "
        "intermediate size divisible by block_size * tensor_parallel_degree "
        "to quantize it.",
        shape,
        shape[-1],
        block_size,
    )


def _fake_quant_mx(t: torch.Tensor, elem_dtype, block_size: int) -> torch.Tensor:
    """Straight-through emulated MX fake-quant: value = dequant(quant(t)),
    gradient = identity (STE)."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    if t.shape[-1] % block_size != 0:
        # Not blockable: leave in high precision. Warn rather than skip in
        # silence -- for w2_EDF the last dim IS the expert-TP-sharded one, so
        # a tensor that is blockable whole becomes non-blockable per shard and
        # the run would quietly train a different quantization scope than
        # requested. Once per shape, not once per forward.
        _warn_unquantized(tuple(t.shape), block_size)
        return t
    q = MXTensor.to_mx(
        t.contiguous().to(torch.bfloat16), elem_dtype=elem_dtype, block_size=block_size
    ).dequantize()
    # Emulated MX can overflow E2M1/E4M3 range on out-of-distribution values
    # (real QAT weights train in-range; random-init or exploding activations
    # do not). Never emit non-finite: fall back to the high-precision value
    # elementwise where quant blew up.
    q = q.to(t.dtype)
    q = torch.where(torch.isfinite(q), q, t)
    # STE: forward q, backward identity through t.
    return t + (q - t).detach()


_qat_experts_cls_cache: dict[type, type] = {}


class MXQATExpertsBase:
    """Marker base of every QAT grouped-experts class."""


def _get_qat_experts_cls(parent_cls: type) -> type:
    """Get or create a fake-quant QAT subclass of a grouped-experts class.

    The fake-quantized weights are installed into ``self.__dict__`` for the
    duration of forward, which shadows ``_parameters`` for normal attribute
    lookup, and removed afterwards. A class-level property would be simpler
    but is wrong here: FSDP2's ``reset_sharded_param`` does ``getattr(module,
    name)`` OUTSIDE forward and requires the DTensor parameter back. Renaming
    the masters (as the QLoRA packing path does) would also work but would
    break the state-dict contract and the expert TP/EP layout, both of which
    key off these exact names -- QAT's masters must stay ordinary trainable
    params.
    """
    if parent_cls in _qat_experts_cls_cache:
        return _qat_experts_cls_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class MXQATExperts(parent_cls, MXQATExpertsBase):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            quantize_act: bool = True

        def __init__(self, config) -> None:
            super().__init__(config)
            self._qat_quantize_act = config.quantize_act

        def forward(self, x_RD, num_tokens_per_expert_E):
            if self._qat_quantize_act:
                x_RD = _fake_quant_mx(x_RD, _ACT_ELEM, _BLOCK)
            for name in _EXPERT_WEIGHT_NAMES:
                w = self._parameters.get(name)
                if w is None:
                    continue
                if isinstance(w, DTensor):
                    # Under EP/TP the master is a DTensor and the parent
                    # forward would call to_local() itself; localize here so
                    # MX quantization sees a plain tensor. Bare to_local
                    # mirrors the parent: the gradient keeps the parameter's
                    # own placement, correct because each rank quantizes
                    # exactly its own shard.
                    #
                    # Per-shard quantization is NOT equivalent to quantizing
                    # the whole tensor: MX block scales come from the max-abs
                    # within each block, so a shard boundary cutting the
                    # blocked dim changes the scales. w1_EFD/w3_EFD block on
                    # D, which expert TP does not shard; w2_EDF blocks on the
                    # intermediate size -- exactly what expert TP shards -- so
                    # w2 under TP quantizes per shard, and is skipped with a
                    # warning when the shard stops being block-divisible.
                    w = w.to_local()
                self.__dict__[name] = _fake_quant_mx(w, _WEIGHT_ELEM, _BLOCK)
            try:
                return super().forward(x_RD, num_tokens_per_expert_E)
            finally:
                for name in _EXPERT_WEIGHT_NAMES:
                    self.__dict__.pop(name, None)

    MXQATExperts.__name__ = f"MXQAT{parent_cls.__name__}"
    MXQATExperts.__qualname__ = f"MXQAT{parent_cls.__name__}"
    _qat_experts_cls_cache[parent_cls] = MXQATExperts
    return MXQATExperts


class MXFP4QATConverter(ModelConfigConverter):
    """MXFP4-weight / MXFP8-activation fake-quant QAT on the routed experts.

    Operates on the model config tree: every ``GroupedExperts.Config`` is
    replaced with a fake-quant subclass config. That isinstance check is K3's
    official quantization scope (module docstring); a model with no grouped
    experts has nothing in scope and the conversion raises rather than
    silently training unquantized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        quantize_act: bool = True
        """MXFP8 fake-quant on the expert input activations."""

    def __init__(self, config: Config, **kwargs):
        self.config = config

    def convert(self, model_config: Module.Config) -> Module.Config:
        converted = 0
        configs = list(model_config.traverse(Module.Config, recurse=True))
        for _fqn, cfg, parent, attr in reversed(configs):
            if not isinstance(cfg, GroupedExperts.Config):
                continue
            assert cfg._owner is not None
            qat_cls = _get_qat_experts_cls(cfg._owner)
            new_cfg = qat_cls.Config(
                **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
                quantize_act=self.config.quantize_act,
            )
            assert parent is not None and isinstance(attr, str)
            setattr(parent, attr, new_cfg)
            converted += 1
        if converted == 0:
            raise ValueError(
                "MXFP4QATConverter found no GroupedExperts in the model "
                "config; a dense model has nothing in K3's MXFP4 scope."
            )
        logger.info(
            "MXFP4 QAT (K3 official scope): %d routed-expert modules, "
            "MXFP8 activations %s",
            converted,
            "on" if self.config.quantize_act else "off",
        )
        return model_config
