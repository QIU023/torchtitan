# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention using Attention Gym kernels."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from attn_gym.linear.context_parallel import ContextParallelRouting
from attn_gym.linear.kda import bound_gate, chunk_kda
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
from attn_gym.linear.short_conv import causal_conv1d
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    local_head_split,
    VarlenMetadata,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# W = convolution kernel width.


def cu_seqlens_from_positions(
    positions: torch.Tensor | None, num_tokens: int
) -> torch.Tensor | None:
    """Packed-stream offsets ``[0, ..., T]`` from the positions of a folded
    ``[T]`` stream: a document starts wherever the position restarts at 0.
    ``None`` when there are no positions or no interior restart. A helper for
    callers that pack documents (verl's engine); the model never derives
    boundaries on its own, since a padded single sample restarts too."""
    if positions is None:
        return None
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.ndim != 1 or positions.shape[0] != num_tokens:
        return None
    starts = torch.nonzero(positions == 0).flatten()
    if starts.numel() == 0 or starts[0].item() != 0:
        starts = torch.cat((starts.new_zeros(1), starts))
    if starts.numel() == 1:
        return None
    ends = starts.new_full((1,), num_tokens)
    return torch.cat((starts, ends)).to(torch.int32)


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
        cp_routing: ContextParallelRouting | None = None,
        cp_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        if not q_1THK.is_cuda:
            raise RuntimeError("Attention Gym KDA requires CUDA tensors.")
        gate_1THK = bound_gate(
            raw_gate_1THK,
            # TODO: The long-term solution is to specify mixed precision per FQN
            # instead of per layer. https://github.com/pytorch/pytorch/issues/156784
            A_log_H.float(),
            dt_bias_HK.float(),
            lower_bound=self.lower_bound,
            impl="fused",
        )
        if cp_routing is not None:
            # KCP: the sequence stays sharded; attn-gym exchanges per-fragment
            # affine state summaries so each rank scans from its true entry state.
            from attn_gym.linear.kda import context_parallel_kda

            assert cp_group is not None
            output_1THV, _ = context_parallel_kda(
                l2norm(q_1THK),
                l2norm(k_1THK),
                v_1THV,
                gate_1THK,
                raw_beta_1TH.float().sigmoid(),
                routing=cp_routing,
                group=cp_group,
            )
            return output_1THV
        output_1THV, _ = chunk_kda(
            l2norm(q_1THK),
            l2norm(k_1THK),
            v_1THV,
            gate_1THK,
            raw_beta_1TH.float().sigmoid(),
            cu_seqlens=cu_seqlens,
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
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        mixed_qkv_1TC, conv_weight_C1W = self._pack_inputs(
            query_TC,
            key_TC,
            value_TC,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
        )
        return self._conv_and_scan(
            mixed_qkv_1TC,
            conv_weight_C1W,
            raw_gate_THK,
            raw_beta_TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
        )

    @staticmethod
    def _pack_inputs(
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse q/k/v and their conv weights so one convolution serves all three."""
        mixed_qkv_1TC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        return mixed_qkv_1TC, conv_weight_C1W

    def _conv_and_scan(
        self,
        mixed_qkv_1TC: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        conv_state: torch.Tensor | None = None,
        cp_routing: ContextParallelRouting | None = None,
        cp_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Causal conv then the delta-rule scan; the CP kernel passes its routing."""
        conv_output_1TC = causal_conv1d(
            mixed_qkv_1TC,
            conv_weight_C1W[:, 0],
            activation="silu",
            cu_seqlens=cu_seqlens,
            initial_state=conv_state,
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
            raw_gate_THK.unsqueeze(0),
            raw_beta_TH.unsqueeze(0),
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            cp_routing=cp_routing,
            cp_group=cp_group,
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
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if x_TD.ndim != 2:
            raise ValueError(
                f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
            )

        if isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
        elif attention_masks is not None:
            raise ValueError(
                "KDA attention_masks must be VarlenMetadata or None, "
                f"got {type(attention_masks).__name__}."
            )
        # Otherwise ``cu_seqlens`` is the caller's: a packed stream under flex
        # attention (verl's rmpad micro-batches) hands the document offsets in
        # explicitly, so the recurrent state and the short convolution reset
        # at every document; an unpacked run passes nothing and keeps the
        # single-sequence kernels.
        # The projections hand back the TP-local head slice; the head split
        # runs as core's local_head_split, typed head-sharded on TP.
        raw_gate_THK = local_head_split(
            self.forget_b(self.forget_a(x_TD)), self.head_dim
        )
        raw_beta_TH = self.beta(x_TD)
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

        output_gate_THV = local_head_split(self.output_gate(x_TD), self.head_dim)
        return self.output_proj(self.output_norm(out_THV, output_gate_THV).flatten(-2))
