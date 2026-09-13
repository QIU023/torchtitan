# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention using Attention Gym kernels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from attn_gym.linear.kda import bound_gate, chunk_kda
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
from attn_gym.linear.short_conv import causal_conv1d
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# W = convolution kernel width.


class KimiRMSNormGated(Module):
    """Per-head RMSNorm followed by a sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x_THV: torch.Tensor, gate_THV: torch.Tensor) -> torch.Tensor:
        input_dtype = x_THV.dtype
        normalized_THV = F.rms_norm(
            x_THV.float(),
            (x_THV.shape[-1],),
            self.weight.float(),
            self.eps,
        )
        return (normalized_THV * gate_THV.float().sigmoid()).to(input_dtype)



# LOCAL PROBE HACK (not committed): FP64_PROBE=1 runs KDA and its short convolution in float64 on
# Attention Gym's eager oracles (the fused kernels take fp16 / bf16 / fp32 only).
def _kda_fp64(q, k, v, raw_gate, raw_beta, A_log, dt_bias, lower_bound, cu_seqlens):
    import math
    import sys
    from functools import partial

    from attn_gym.linear.kda import chunk_kda as _ck
    from attn_gym.linear.kda.impl.reference import reference_kda
    from attn_gym.linear.kda.naive import l2norm_fwd_ref, naive_chunk_kda

    f = torch.float64
    q, k, v, raw_gate, raw_beta = (t.to(f) for t in (q, k, v, raw_gate, raw_beta))
    heads = raw_gate.shape[2]
    gate = lower_bound * torch.sigmoid(A_log.to(f).exp().view(1, 1, heads, 1) * (raw_gate + dt_bias.to(f)))
    chunk = sys.modules[_ck.__module__]._CHUNK_SIZE
    out, _ = reference_kda(
        partial(naive_chunk_kda, chunk_size=chunk),
        l2norm_fwd_ref(q), l2norm_fwd_ref(k), v, gate * math.log2(math.e), raw_beta.sigmoid(),
        None, cu_seqlens, 1.0 / math.sqrt(q.shape[-1]), False,
    )
    return out


def _causal_conv1d_fp64(x_1TC, weight_CW, cu_seqlens):
    """Depthwise causal conv + SiLU per document, taps in stored order (matches the kernel to 8.6e-6 in fp32)."""
    import torch.nn.functional as F

    x_1TC = x_1TC.to(torch.float64); weight_CW = weight_CW.to(torch.float64)
    channels, width = weight_CW.shape
    bounds = [0, x_1TC.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    pieces = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        doc = x_1TC[:, start:end].transpose(1, 2)
        y = F.conv1d(F.pad(doc, (width - 1, 0)), weight_CW.unsqueeze(1), groups=channels)
        pieces.append(F.silu(y).transpose(1, 2))
    tail = x_1TC.shape[1] - bounds[-1]
    if tail:
        pieces.append(x_1TC.new_zeros(1, tail, channels))
    return torch.cat(pieces, dim=1)


class KDAKernel(Module):
    """Apply KDA preprocessing and the Attention Gym kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float = -5.0

        def __post_init__(self):
            if not -5.0 <= self.lower_bound < 0.0:
                raise ValueError(
                    "KDA lower_bound must be in the safe range [-5, 0), "
                    f"got {self.lower_bound}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound

    def forward(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not q_1THK.is_cuda:
            raise RuntimeError("Attention Gym KDA requires CUDA tensors.")
        capability = torch.cuda.get_device_capability(q_1THK.device)
        if capability < (8, 0):  # LOCAL RUN HACK (not committed)
            raise RuntimeError(
                "Attention Gym KDA requires CUDA capability 8.0 or newer; "
                f"got {capability}."
            )

        if __import__("os").environ.get("FP64_PROBE") == "1":  # LOCAL PROBE HACK (not committed)
            return _kda_fp64(q_1THK, k_1THK, v_1THV, raw_gate_1THK, raw_beta_1TH, A_log_H, dt_bias_HK, self.lower_bound, cu_seqlens)
        gate_1THK = bound_gate(
            raw_gate_1THK,
            # TODO: The long-term solution is to specify mixed precision per FQN
            # instead of per layer. https://github.com/pytorch/pytorch/issues/156784
            A_log_H.float(),
            dt_bias_HK.float(),
            lower_bound=self.lower_bound,
            impl="reference" if __import__("os").environ.get("FP32_PROBE") == "1" else "fused",  # LOCAL PROBE HACK (not committed)
        )
        output_1THV, _ = chunk_kda(
            l2norm(q_1THK),
            l2norm(k_1THK),
            v_1THV,
            gate_1THK,
            raw_beta_1TH.float().sigmoid(),
            cu_seqlens=cu_seqlens,
            autotune=__import__("os").environ.get("KDA_NOAUTOTUNE") != "1",  # LOCAL PROBE HACK (not committed)
            **({"impl": "reference"} if __import__("os").environ.get("FP32_PROBE") == "1" else {}),  # LOCAL PROBE HACK (not committed)
        )
        return output_1THV


class InnerKDA(Module):
    """Run short convolution and KDA behind the vLLM replacement boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        kernel: KDAKernel.Config

        def __post_init__(self):
            if self.head_dim != 128:
                raise ValueError(
                    "Attention Gym KDA requires head_dim=128, " f"got {self.head_dim}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.kernel = config.kernel.build()

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        raw_gate_1THK = raw_gate_THK.unsqueeze(0)
        raw_beta_1TH = raw_beta_TH.unsqueeze(0)
        mixed_qkv_1TC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        if __import__("os").environ.get("FP64_PROBE") == "1":  # LOCAL PROBE HACK (not committed)
            conv_output_1TC = _causal_conv1d_fp64(mixed_qkv_1TC, conv_weight_C1W[:, 0], cu_seqlens)
        else:
            conv_output_1TC = causal_conv1d(
                mixed_qkv_1TC,
                conv_weight_C1W[:, 0],
                activation="silu",
                cu_seqlens=cu_seqlens,
            )
        assert isinstance(conv_output_1TC, torch.Tensor)

        q_1TC, k_1TC, v_1TC = conv_output_1TC.chunk(3, dim=-1)
        q_1THK, k_1THK, v_1THV = (
            tensor.unflatten(-1, (-1, self.head_dim))
            for tensor in (q_1TC, k_1TC, v_1TC)
        )
        output_1THV = self.kernel(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
        )
        return output_1THV.squeeze(0)


class KDA(Module):
    """Kimi Delta Attention with checkpoint-compatible Kimi K3 parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv: Conv1d.Config
        k_conv: Conv1d.Config
        v_conv: Conv1d.Config
        forget_a: Linear.Config
        forget_b: Linear.Config
        beta: Linear.Config
        output_gate: Linear.Config
        inner_kda: Module.Config
        output_norm: KimiRMSNormGated.Config
        output_proj: Linear.Config

        def __post_init__(self):
            if self.num_heads < 1:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            if self.head_dim != 128:
                raise ValueError(
                    "Attention Gym KDA requires head_dim=128, " f"got {self.head_dim}."
                )
            if self.conv_kernel_size < 1:
                raise ValueError(
                    f"conv_kernel_size must be positive, got {self.conv_kernel_size}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv = config.q_conv.build()
        self.k_conv = config.k_conv.build()
        self.v_conv = config.v_conv.build()
        self.forget_a = config.forget_a.build()
        self.forget_b = config.forget_b.build()
        self.beta = config.beta.build()
        self.output_gate = config.output_gate.build()
        self.inner_kda = config.inner_kda.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if x_TD.ndim != 2:
            raise ValueError(
                f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
            )

        if attention_masks is None:
            cu_seqlens = None
        elif isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
        else:
            raise ValueError(
                "KDA attention_masks must be VarlenMetadata or None, "
                f"got {type(attention_masks).__name__}."
            )
        num_tokens = x_TD.shape[0]
        raw_gate_THK = self.forget_b(self.forget_a(x_TD)).reshape(
            num_tokens, self.num_heads, self.head_dim
        )
        raw_beta_TH = self.beta(x_TD).reshape(num_tokens, self.num_heads)
        out_THV = self.inner_kda(
            self.q_proj(x_TD),
            self.k_proj(x_TD),
            self.v_proj(x_TD),
            raw_gate_THK,
            raw_beta_TH,
            self.q_conv.weight,
            self.k_conv.weight,
            self.v_conv.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens=cu_seqlens,
        )

        output_gate_THV = self.output_gate(x_TD).view_as(out_THV)
        return self.output_proj(self.output_norm(out_THV, output_gate_THV).flatten(-2))
