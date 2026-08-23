# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention (KDA), the linear-attention half of the K3 alternation.

Split out of model.py to match the reference tree's file layout. It reaches
back into that module for nothing: the flat config is a type annotation only,
and the four helpers it shares with MLA live in sharding.py and
dtensor_ops.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.tensor import DTensor

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k3.dtensor_ops import to_local_if_dtensor
from torchtitan.models.kimi_k3.sharding import (
    contract_for_mode,
    cp_all_to_all_headseq,
    tp_replicate,
    ULYSSES,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Config

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:  # pragma: no cover - import-time guard
    raise ImportError(
        "Kimi Linear KDA path requires fla-core. Run `pip install fla-core`."
    ) from err


def _local_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply ``linear`` with both weight and (optional) bias unwrapped to local.

    Used by :class:`KimiDeltaAttention.forward` so each projection can
    operate in plain-Tensor land alongside the fla-core triton kernels,
    even when the parent NoParallel(self_attn) wrap makes ``linear.weight``
    a DTensor(Replicate) on tp_mesh.
    """
    weight = to_local_if_dtensor(linear.weight)
    bias = to_local_if_dtensor(linear.bias) if linear.bias is not None else None
    return F.linear(x, weight, bias)


class KimiDeltaAttention(Module):
    """Kimi Delta Attention — linear-attention variant using
    fla-core's gated delta rule kernel.

    Faithful port of ``reference:KimiDeltaAttention`` minus the
    HF ``Cache`` / ``cu_seqlens`` / padding-aware fast-path (training
    fixed-seqlen doesn't exercise those).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config-driven KDA.

        The scalar fields deliberately carry the SAME names as the flat model
        config's, so the constructor body reads them unchanged. fla's
        ``ShortConvolution`` and ``FusedRMSNormGated`` are not Configurable, so
        they stay constructed from scalars here rather than from child configs --
        upstream avoids that by using core's Conv1d and its own gated norm, and we
        keep fla's fused kernels deliberately (they are on every KDA layer's
        critical path).
        """

        layer_idx: int
        hidden_size: int
        kda_short_conv_kernel_size: int
        kda_head_dim: int
        kda_num_heads: int
        kda_use_full_rank_gate: bool
        kda_gate_lower_bound: float
        kda_cp_mode: str
        rms_norm_eps: float
        q_proj: "Linear.Config"
        k_proj: "Linear.Config"
        v_proj: "Linear.Config"
        f_a_proj: "Linear.Config"
        f_b_proj: "Linear.Config"
        b_proj: "Linear.Config"
        o_proj: "Linear.Config"
        g_proj: "Linear.Config | None" = None
        g_a_proj: "Linear.Config | None" = None
        g_b_proj: "Linear.Config | None" = None

    @staticmethod
    def make_config(
        config: KimiK3Config, layer_idx: int
    ) -> "KimiDeltaAttention.Config":
        """The one place that reads the flat config for KDA."""
        projection_size = config.kda_head_dim * config.kda_num_heads

        def _lin(fan_in, fan_out, *, replicate=True):
            # Replicate throughout, matching what NoParallel gave these. Their
            # outputs feed the fla kernels, which KDA unwraps at the call site
            # (to_local_if_dtensor), so the kernels see plain tensors either way.
            return Linear.Config(
                in_features=fan_in,
                out_features=fan_out,
                bias=False,
                sharding_config=tp_replicate() if replicate else None,
            )

        cfg = KimiDeltaAttention.Config(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            kda_short_conv_kernel_size=config.kda_short_conv_kernel_size,
            kda_head_dim=config.kda_head_dim,
            kda_num_heads=config.kda_num_heads,
            kda_use_full_rank_gate=config.kda_use_full_rank_gate,
            kda_gate_lower_bound=config.kda_gate_lower_bound,
            kda_cp_mode=config.kda_cp_mode,
            rms_norm_eps=config.rms_norm_eps,
            q_proj=_lin(config.hidden_size, projection_size),
            k_proj=_lin(config.hidden_size, projection_size),
            v_proj=_lin(config.hidden_size, projection_size),
            f_a_proj=_lin(config.hidden_size, config.kda_head_dim),
            f_b_proj=_lin(config.kda_head_dim, projection_size),
            b_proj=_lin(config.hidden_size, config.kda_num_heads),
            o_proj=_lin(projection_size, config.hidden_size),
        )
        # K3 (report Eq. 6) makes the output gate full rank; Kimi Linear factored
        # it through head_dim. Both feed the same FusedRMSNormGated.
        if config.kda_use_full_rank_gate:
            cfg.g_proj = _lin(config.hidden_size, projection_size)
        else:
            cfg.g_a_proj = _lin(
                config.hidden_size, config.kda_head_dim, replicate=False
            )
            cfg.g_b_proj = _lin(config.kda_head_dim, projection_size, replicate=False)
        return cfg

    def __init__(self, config: "KimiDeltaAttention.Config") -> None:
        super().__init__()
        self.layer_idx = config.layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.kda_short_conv_kernel_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads

        projection_size = self.head_dim * self.num_heads
        projection_k_size = projection_size  # k heads == v heads for Kimi

        # Replicate, matching what NoParallel gave them. Their outputs feed the
        # fla kernels, which KDA unwraps at the call site
        # (to_local_if_dtensor), so the kernels still see plain tensors.
        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()

        # Short causal convolutions with silu activation on q/k/v
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        # A_log: per-head log-decay parameter, init uniform in log([1, 16])
        # fla-core 0.5.0 expects shape [H]; HF reference had [1, 1, H, 1]
        # but it's fed through fused_kda_gate which reshapes internally.
        # Drawn and log'd in fp32 for the init math, then stored at the default
        # dtype like every other parameter. Keeping the parameter itself fp32
        # (which is what dtype= on the empty() used to do) makes the module's
        # dtypes non-uniform under training.dtype=bfloat16, and FSDP2 rejects
        # that outright: "FSDP expects uniform original parameter dtype".
        # No-op when the default dtype is fp32.
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).to(torch.get_default_dtype())
        )

        # dt_bias: per-(head, head_dim) bias, shape [H * K]. Applied
        # inside fused_kda_gate as softplus(g + dt_bias). Kept zero-init
        # to reproduce HF reference's default init behavior.
        self.dt_bias = nn.Parameter(torch.zeros(projection_size))

        # Declared here rather than driven by ``plan["self_attn"] = NoParallel(...)``:
        # A_log and dt_bias are this module's OWN parameters, so only a module-level
        # declaration can reach them. tp-Replicate matches what NoParallel does, and
        # keeps every parameter on one mesh for clip_grad_norm_'s stack.
        #
        # ``param_init`` is not optional once this class is a Module:
        # ``_init_self_parameters`` RAISES for own parameters when neither a param_init
        # map nor ``reset_parameters`` exists, and both of these are initialized above --
        # so the map re-applies exactly that, rather than leaving a trap for the first
        # caller that reaches init_states from the root.
        self._sharding_config = ShardingConfig(
            state_shardings={
                "A_log": dense_param_placement(tp=spmd.R),
                "dt_bias": dense_param_placement(tp=spmd.R),
            }
        )
        self._param_init = {
            "A_log": lambda t: t.copy_(
                torch.log(
                    torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
                ).to(t.dtype)
            ),
            "dt_bias": lambda t: t.zero_(),
        }

        # Low-rank forget-gate and output-gate projections
        self.f_a_proj = config.f_a_proj.build()
        self.f_b_proj = config.f_b_proj.build()
        # Output gate. K3 (report Eq. 6) makes W_g full rank; Kimi Linear
        # factored it through head_dim. Both feed the same
        # FusedRMSNormGated(o, g) = Sigmoid(g) (.) RMSNorm(o~) below.
        self.use_full_rank_gate = config.kda_use_full_rank_gate
        if self.use_full_rank_gate:
            self.g_proj = config.g_proj.build()
        else:
            self.g_a_proj = config.g_a_proj.build()
            self.g_b_proj = config.g_b_proj.build()
        self.gate_lower_bound = config.kda_gate_lower_bound
        self.cp_mode = config.kda_cp_mode
        # Validate against the CP contracts so the accepted modes are declared
        # in one place rather than restated here.
        contract_for_mode(self.cp_mode)

        # Beta: per-head, per-token scalar (delta-rule learning rate)
        self.b_proj = config.b_proj.build()

        # Output RMSNorm with sigmoid-gated modulation from g, then o_proj
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation="sigmoid",
        )
        # Replicate, unlike MLA's o_proj: KDA's core runs on plain tensors, so
        # this projection's input is not head-sharded and has nothing to reduce.
        self.o_proj = config.o_proj.build()

    def _output_gate_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid output-gate logits, flat ``[..., H * head_dim]``.

        Full rank is K3's (report Eq. 6); the low-rank pair is Kimi Linear's.
        The sigmoid itself lives in FusedRMSNormGated.
        """
        if self.use_full_rank_gate:
            return _local_linear(self.g_proj, x)
        return _local_linear(self.g_b_proj, _local_linear(self.g_a_proj, x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without KV cache, fixed seq_len.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` KDA output.
        """
        # Under TP, the parent KimiK3TransformerBlock's NoParallel(self_attn)
        # wraps this forward: x arrives as DTensor(Replicate) on tp_mesh,
        # and all child params (q/k/v projections, conv1d weights,
        # A_log, dt_bias, FusedRMSNormGated) are DTensors on the same
        # mesh. The standard nn.Linear ops (DTensor x × DTensor weight)
        # dispatch correctly through DTensor's op set; the fla-core
        # triton kernels (causal_conv1d in ShortConvolution,
        # fused_kda_gate, chunk_kda, FusedRMSNormGated) do not. We
        # stash the input's DTensor metadata, run the body in plain-
        # tensor land, and re-DTensor at the end so the parent
        # NoParallel hook's prepare_output sees a DTensor.
        in_mesh = None
        in_placements = None
        if isinstance(x, DTensor):
            in_mesh = x.device_mesh
            in_placements = x.placements
        x = to_local_if_dtensor(x)
        # Context parallel: Ulysses path (seq-local projections,
        # all-to-all seq<->head, full-seq conv + scan on this rank's head
        # subset). chunk_kda is bit-exactly per-head independent
        # (kda_ulysses_cp_probe), so head-sharding the scan is exact.
        # MLA layers get the same treatment in KimiMLAAttention.
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out = (
                self._forward_kcp(x, cp_group)
                if self.cp_mode == "kcp"
                else self._forward_cp(x, cp_group)
            )
            if in_mesh is not None and in_placements is not None:
                out = DTensor.from_local(
                    out,
                    in_mesh,
                    in_placements,
                    run_check=False,
                )
            return out
        _, T, _ = x.shape
        # mode selection matches reference: chunk for long, recurrent for short
        # training gate: chunk required (ref asserts this)
        mode = "fused_recurrent" if T <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 1) Q/K/V projection + short causal conv with silu.
        # _local_linear unwraps DTensor weight to local before F.linear.
        # ShortConvolution.forward is patched at TP-init time to handle
        # DTensor input/weight by to_local + re-DTensor; we feed plain
        # x here so the patch is a no-op when x is already plain.
        q, _ = self.q_conv1d(
            x=_local_linear(self.q_proj, x),
            cache=None,
            output_final_state=False,
        )
        k, _ = self.k_conv1d(
            x=_local_linear(self.k_proj, x),
            cache=None,
            output_final_state=False,
        )
        v, _ = self.v_conv1d(
            x=_local_linear(self.v_proj, x),
            cache=None,
            output_final_state=False,
        )

        # 2) Forget-gate g: (B,T,D) low-rank via f_a/f_b, reshape to
        #    (B, T, H, K) for fla-core 0.5.0's fused_kda_gate API:
        #      fused_kda_gate(g: [..., H, K], A_log: [H], dt_bias: [H*K])
        #      → [..., H, K] log-decay
        g_raw = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(
            g_raw,
            to_local_if_dtensor(self.A_log),
            dt_bias=to_local_if_dtensor(self.dt_bias),
            lower_bound=self.gate_lower_bound,
        )

        # 3) Beta: per-head, per-token learning-rate (delta-rule)
        beta = _local_linear(self.b_proj, x).float().sigmoid()

        # 4) Reshape to (..., H, D) for KDA kernel
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # 6) Output gate (computed before the head-shard so the slice below
        # covers it too).
        g_out = self._output_gate_raw(x)
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)

        # 5) Run KDA op
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o, _ = kda_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        # FusedRMSNormGated.forward is patched at TP-init time too, so
        # it handles DTensor weight transparently. We pass plain o + g_out
        # here (both are plain after the to_local+linear chain).
        o = self.o_norm(o, g_out)  # o * sigmoid(g_out), normed

        # 7) Reshape back and project
        o = rearrange(o, "b t h d -> b t (h d)")
        out = _local_linear(self.o_proj, o)

        # Re-wrap the output as DTensor so the parent NoParallel hook
        # gets the type it expects. Replicate placement matches the
        # incoming x's placement (input_layernorm output).
        if in_mesh is not None and in_placements is not None:
            out = DTensor.from_local(
                out,
                in_mesh,
                in_placements,
                run_check=False,
            )
        return out

    def _forward_kcp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """KCP forward: the sequence stays sharded (report sec 5.1.2).

        Unlike the Ulysses path, no rank ever holds the full sequence. The two
        cross-rank dependencies are handled separately because they have
        different structure:

        * the short convolutions need only the previous rank's tail, since their
          support is finite -- one fixed-size halo, no scan (see kcp.py);
        * the delta-rule recurrence needs the true incoming state, which does
          NOT decompose by summation, so fla's cp_context does a prefix scan
          over (cumulative transition, zero-started state) fragments.

        Constraints this path inherits from fla: ``output_final_state`` is
        unsupported under cp_context, which is fine for training (the final
        state is only needed for decoding), and the sequence must divide evenly
        across the CP ranks.

        A batch axis is handled by looping, because fla's ``causal_conv1d_cp``
        asserts ``[1, T, D]``: its CP path is built around a single packed
        sequence. Flattening ``[B, L, D]`` into one packed sequence instead would
        be cheaper in launches but wrong -- ``build_cp_context`` derives each
        rank's slice by cutting the GLOBAL packed sequence into contiguous
        rank-ordered pieces, while what this rank actually holds is piece ``r`` of
        every sequence, so the two layouts only coincide at B = 1. The loop is
        also what the recurrence wants: sequences in a batch are independent, and
        the delta-rule state must not carry from one into the next.

        The cost is B prefix-scan all-gathers instead of one. Each is fixed size
        (state fragments, not activations) and independent of sequence length, and
        B is identical on every rank, so the collective counts match and cannot
        deadlock. K3's own regime is the cheap end of this: local batch 1 with a
        long sequence, the batch coming from DP.
        """
        B = x.shape[0]
        if B > 1:
            return torch.cat(
                [self._forward_kcp_one(x[b : b + 1], cp_group) for b in range(B)],
                dim=0,
            )
        return self._forward_kcp_one(x, cp_group)

    def _forward_kcp_one(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """One sequence's KCP forward. ``x`` is this rank's ``[1, L, D]`` shard."""
        from torchtitan.models.kimi_k3.kcp import build_kcp_context, conv_with_halo

        t_loc = x.shape[1]

        # One context serves both the conv halo and the recurrence; the conv
        # needs the kernel width, the recurrence ignores it.
        ctx = build_kcp_context(
            t_loc, cp_group, x.device, conv1d_kernel_size=self.q_conv1d.kernel_size[0]
        )

        # Projections are seq-local: nothing to exchange yet.
        q = conv_with_halo(self.q_conv1d, _local_linear(self.q_proj, x), ctx)
        k = conv_with_halo(self.k_conv1d, _local_linear(self.k_proj, x), ctx)
        v = conv_with_halo(self.v_conv1d, _local_linear(self.v_proj, x), ctx)

        g_raw = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(
            g_raw,
            to_local_if_dtensor(self.A_log),
            dt_bias=to_local_if_dtensor(self.dt_bias),
            lower_bound=self.gate_lower_bound,
        )
        beta = _local_linear(self.b_proj, x).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
        g_out = rearrange(
            self._output_gate_raw(x), "... (h d) -> ... h d", d=self.head_dim
        )

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            # fla asserts this is unsupported under cp_context.
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=ctx.cu_seqlens,
            cp_context=ctx,
        )
        o = self.o_norm(o, g_out)
        o = rearrange(o, "b t h d -> b t (h d)")
        return _local_linear(self.o_proj, o)

    def _forward_cp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward for KDA.

        Tensor-name legend (shape suffixes): B batch, L local seq (T/cp),
        T full seq, H head count (KDA is never tp-sharded), G CP-local
        head count (H/cp), K head_dim, C flattened head-subset channels
        (G*K).

        Input x is the plain local ``[B, L, D]`` shard (caller already
        stripped DTensor). Projections run seq-local at L; one fused
        all-to-all moves (q, k, v, g_raw, g_out, beta) to full-seq
        head-subset layout; the causal short conv, fused_kda_gate, and
        chunk_kda then run on the full sequence for this rank's G heads
        (conv weights channel-sliced -- depthwise conv, exact; validated
        bit-exact vs ShortConvolution). No rank materializes the full
        sequence at hidden dim D.

        Gradient note: each rank's param-grad contribution covers its
        (seq shard x head subset) sector with zeros elsewhere; FSDP's
        dp_shard_cp mesh reduces over cp, reconstructing full grads --
        the same contract the previous all-gather-SP path relied on.
        """
        from fla.modules.conv.causal_conv1d import causal_conv1d

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        B, t_loc, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        if num_heads % cp_size != 0:
            raise ValueError(
                f"KDA CP: num_heads {num_heads} is not divisible by " f"cp={cp_size}"
            )
        h_cp = num_heads // cp_size
        h0 = cp_rank * h_cp

        # 1) Seq-local projections at L (no cross-seq ops here).
        q_BLHK = _local_linear(self.q_proj, x).view(B, t_loc, num_heads, head_dim)
        k_BLHK = _local_linear(self.k_proj, x).view(B, t_loc, num_heads, head_dim)
        v_BLHK = _local_linear(self.v_proj, x).view(B, t_loc, num_heads, head_dim)
        g_raw_BLHK = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x)).view(
            B, t_loc, num_heads, head_dim
        )
        g_out_BLHK = self._output_gate_raw(x).view(B, t_loc, num_heads, head_dim)
        beta_BLH1 = _local_linear(self.b_proj, x).unsqueeze(-1)

        # 2) One fused all-to-all: seq-shard -> full-seq head-subset.
        packed_BLHW = torch.cat(
            [q_BLHK, k_BLHK, v_BLHK, g_raw_BLHK, g_out_BLHK, beta_BLH1],
            dim=-1,
        )
        src_dim, dst_dim = ULYSSES.in_dims()
        packed_BTGW = cp_all_to_all_headseq(
            packed_BLHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
        )
        q_BTGK, k_BTGK, v_BTGK, g_raw_BTGK, g_out_BTGK, beta_BTG1 = torch.split(
            packed_BTGW,
            [head_dim, head_dim, head_dim, head_dim, head_dim, 1],
            dim=-1,
        )
        t_full = t_loc * cp_size

        mode = "fused_recurrent" if t_full <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 3) Short causal conv on the full sequence, weights sliced to
        # this rank's head-subset channels (depthwise conv -> exact).
        def conv_subset(conv: ShortConvolution, x_BTGK: torch.Tensor):
            w_CW = to_local_if_dtensor(conv.weight).squeeze(1)[
                h0 * head_dim : (h0 + h_cp) * head_dim
            ]
            b_C = (
                to_local_if_dtensor(conv.bias)[h0 * head_dim : (h0 + h_cp) * head_dim]
                if conv.bias is not None
                else None
            )
            y_BTC, _ = causal_conv1d(
                x_BTGK.reshape(B, t_full, h_cp * head_dim),
                weight=w_CW,
                bias=b_C,
                activation=conv.activation,
                backend=conv.backend,
            )
            return y_BTC.view(B, t_full, h_cp, head_dim)

        q_BTGK = conv_subset(self.q_conv1d, q_BTGK)
        k_BTGK = conv_subset(self.k_conv1d, k_BTGK)
        v_BTGK = conv_subset(self.v_conv1d, v_BTGK)

        # 4) Forget gate + beta on the head subset (A_log/dt_bias sliced).
        g_BTGK = fused_kda_gate(
            g_raw_BTGK,
            to_local_if_dtensor(self.A_log)[h0 : h0 + h_cp],
            dt_bias=to_local_if_dtensor(self.dt_bias)
            .view(num_heads, head_dim)[h0 : h0 + h_cp]
            .reshape(-1),
            lower_bound=self.gate_lower_bound,
        )
        beta_BTG = beta_BTG1.squeeze(-1).float().sigmoid()

        # 5) KDA scan on this rank's heads over the full sequence.
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o_BTGK, _ = kda_fn(
            q=q_BTGK,
            k=k_BTGK,
            v=v_BTGK,
            g=g_BTGK,
            beta=beta_BTG,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        o_BTGK = self.o_norm(o_BTGK, g_out_BTGK)

        # 6) All-to-all back to seq-shard full-head layout, then o_proj.
        out_src_dim, out_dst_dim = ULYSSES.out_dims()
        o_BLHK = cp_all_to_all_headseq(
            o_BTGK, cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
        )
        out = _local_linear(self.o_proj, o_BLHK.reshape(B, t_loc, num_heads * head_dim))
        return out
