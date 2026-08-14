# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KCP: KDA context parallelism (report sec 5.1.2), on the upstream K3 model.

Ulysses handles MLA by giving every rank the full sequence for a head subset
(see ``cp_ulysses``). KDA cannot use it as the only mechanism: the whole point
of the sequence-sharded path is that no rank ever holds ``[B, T, D]``, which is
what a 1M-token context needs, and the delta rule is a recurrence rather than a
per-head-independent attention.

Two pieces, both already solved in fla-core and already adapted once by the
imperative implementation. ``build_kcp_context`` and ``conv_with_halo`` are
imported from ``kimi_k3.kcp`` rather than rewritten; what is new here is only
the attachment to the upstream module shapes.

The recurrence. K3 does not use LASP-style state summation -- plain summation is
wrong because the delta rule applies a token-dependent transition to the
incoming state. KCP decomposes each rank's segment into a cumulative transition
and a zero-started state, which compose associatively, so a prefix scan over
them recovers each rank's true incoming state in one fixed-size all-gather,
independent of sequence length. ``chunk_kda(cp_context=...)`` implements it.

The short convolution. KDA runs a causal depthwise conv of width ``W`` on q, k
and v. Shard the sequence and each rank's first ``W - 1`` outputs get computed
against zero padding instead of the previous rank's tail.
``causal_conv1d_cp`` is a real ``autograd.Function`` that exchanges the tail in
the forward and the matching ``dx`` in the backward -- a hand-rolled halo built
on ``dist.all_gather`` is NOT autograd-aware and silently drops the gradient
owed to the left neighbour while the forward stays bit-exact.

Their model applies SiLU outside the conv and pads manually; fla takes the
activation as an argument, so the wrapper passes ``"silu"`` and drops both.

Attaches by replacing ``forward``, so the vendored model file stays
byte-identical to upstream.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.models.common.attention import AttentionMasksType

# The imperative implementation already solved both halves and validated them;
# reuse it rather than growing a second copy. When kimi_k3 is retired at the end
# of the migration this module moves here, it does not get rewritten.
from torchtitan.models.kimi_k3.kcp import build_kcp_context, conv_with_halo


def _make_kcp_forward(module, cp_group):
    """Build the KCP forward for one ``KimiDeltaAttention``.

    Mirrors the module's own forward, with the three convolutions taking the
    CP-aware path and the recurrence receiving the same context.
    """

    def kcp_forward(
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference KDA does not support packed-document masks."
            )
        self = module
        B, L, _ = x_BLD.shape
        # Rebuilt per forward: the context encodes this rank's slice of the
        # GLOBAL sequence, so it depends on the local length, which varies with
        # the batch. Cheap -- it is a two-element cu_seqlens plus group metadata.
        ctx = build_kcp_context(L, cp_group, x_BLD.device, self.conv_kernel_size)
        # The kernel module reads this; see attach_kcp on why it is passed by
        # attribute rather than argument.
        self.kernel._kcp_context = ctx

        q_BLHK = conv_with_halo(self.q_conv, self.q_proj(x_BLD), ctx, activation="silu").view(
            B, L, self.num_heads, self.head_dim
        )
        k_BLHK = conv_with_halo(self.k_conv, self.k_proj(x_BLD), ctx, activation="silu").view(
            B, L, self.num_heads, self.head_dim
        )
        v_BLHV = conv_with_halo(self.v_conv, self.v_proj(x_BLD), ctx, activation="silu").view(
            B, L, self.num_heads, self.head_dim
        )
        forget_BLHK = self.forget_b(self.forget_a(x_BLD)).view(
            B, L, self.num_heads, self.head_dim
        )
        beta_BLH = self.beta(x_BLD).float()

        out_BLHV = self.kernel(
            q_BLHK, k_BLHK, v_BLHV, forget_BLHK, beta_BLH, self.A_log, self.dt_bias
        )
        output_gate_BLHV = self.output_gate(x_BLD).view(
            B, L, self.num_heads, self.head_dim
        )
        out_BLHV = self.output_norm(out_BLHV, output_gate_BLHV)
        return self.output_proj(out_BLHV.reshape(B, L, -1))

    return kcp_forward


def _make_kcp_kernel_forward(module):
    """``KimiKDAKernel.forward`` with the CP context threaded into chunk_kda."""

    def kernel_forward(
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        from fla.ops.kda import chunk_kda

        self = module
        out_BLHV, _ = chunk_kda(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
            cp_context=self._kcp_context,
        )
        return out_BLHV

    return kernel_forward


def apply_kcp(kda_modules, cp_mesh: DeviceMesh) -> None:
    """Replace each KDA's forward, and its kernel's, with the KCP variants.

    The context is passed from the attention module to its kernel by attribute
    rather than as an argument because the kernel's call signature is upstream's
    and threading a new parameter through it would mean editing their model
    file, which this migration keeps byte-identical on purpose.
    """
    cp_group = cp_mesh.get_group()
    for mod in kda_modules:
        mod.kernel.forward = _make_kcp_kernel_forward(mod.kernel)
        mod.forward = _make_kcp_forward(mod, cp_group)
