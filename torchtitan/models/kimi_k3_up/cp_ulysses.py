# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses context parallelism for Kimi K3 MLA.

Core ships two CP strategies (``apply_cp_to_forward``): all-gather K/V for
FlexAttention, and sequence-sharding plus the CP dispatcher -- ring attention --
for SDPA. K3's MLA builds an SDPA inner attention, so it lands on the ring path,
and measured there it fails inside the dispatcher's accumulation with
"aten.add.Tensor got mixed torch.Tensor and DTensor" while llama3 passes the same
cell on the same tree. Ring is therefore not available for this attention today.

Ulysses is a different decomposition, not a fallback:

  ring     keeps the sequence sharded end to end and rotates K/V blocks. Scales
           past the head count, needs the attention kernel to support the
           accumulation.
  Ulysses  all-to-alls the head axis so each rank holds the FULL sequence for its
           head subset. The attention kernel sees an ordinary unsharded problem
           and needs to support nothing. Bounded by cp <= num_heads.

For K3 the bound bites -- the released topology has few attention heads -- so the
two are complements, and both are kept. Nothing in upstream torchtitan implements
Ulysses; this is our increment.

This module attaches by replacing ``forward``, the same way core's helper does,
so the vendored model file stays byte-identical to upstream and rebases stay
cheap.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.models.common.attention import AttentionMasksType


def cp_all_to_all_headseq(
    x_BABK: torch.Tensor, cp_group, seq_to_head: bool
) -> torch.Tensor:
    """Differentiable all-to-all swapping which of (seq, head) is CP-sharded.

    ``seq_to_head=True``:  ``[B, T/cp, H, K]`` -> ``[B, T, H/cp, K]``.
    ``seq_to_head=False``: ``[B, T, H/cp, K]`` -> ``[B, T/cp, H, K]``.

    Backward is the transposed all-to-all, via
    ``torch.distributed.nn.functional``. Round-trip and per-head parity were
    validated bit-exact against a single-rank reference.
    """
    import torch.distributed.nn.functional as dist_nn

    cp = dist.get_world_size(cp_group)
    B, d1, d2, K = x_BABK.shape
    if seq_to_head:
        t_loc, num_heads = d1, d2
        x_split = (
            x_BABK.reshape(B, t_loc, cp, num_heads // cp, K)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split), x_split, group=cp_group
        )
        # recv[s] holds src s's T/cp for THIS rank's head subset; stack on seq.
        return (
            out.permute(1, 0, 2, 3, 4)
            .reshape(B, cp * t_loc, num_heads // cp, K)
            .contiguous()
        )
    t_full, h_loc = d1, d2
    t_loc = t_full // cp
    x_split = (
        x_BABK.reshape(B, cp, t_loc, h_loc, K).permute(1, 0, 2, 3, 4).contiguous()
    )
    out = dist_nn.all_to_all_single(torch.empty_like(x_split), x_split, group=cp_group)
    # out[s] = src s's head subset for THIS rank's seq shard; put T/cp before the
    # src axis so the reshape stacks heads in ascending order.
    return out.permute(1, 2, 0, 3, 4).reshape(B, t_loc, cp * h_loc, K).contiguous()


def _make_ulysses_forward(module, cp_group):
    """Build the CP forward for one ``KimiK3MLAAttention``.

    Mirrors the module's own forward exactly, with the sequence-local
    projections feeding one fused all-to-all before attention and one after.
    No rank ever materializes ``[B, T, D]`` hidden states.
    """

    def ulysses_forward(
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch.distributed.nn.functional as dist_nn

        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference MLA does not support packed-document masks."
            )
        self = module
        cp = dist.get_world_size(cp_group)
        B, t_loc, _ = x_BLD.shape
        if self.n_heads % cp != 0:
            raise ValueError(
                f"Ulysses CP needs num_heads ({self.n_heads}) divisible by "
                f"cp ({cp}); this is the head-count bound ring does not have."
            )
        g_loc = self.n_heads // cp
        t_full = t_loc * cp

        q_BLNQ = self.wq_b(self.q_norm(self.wq_a(x_BLD))).view(
            B, t_loc, self.n_heads, self.q_head_dim
        )
        compressed_kv = self.wkv_a(x_BLD)
        kv_latent, k_rope_BLR = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_BLNF = self.wkv_b(self.kv_norm(kv_latent)).view(
            B, t_loc, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        # One fused all-to-all for q and kv, concatenated on the feature axis.
        qkv_BTGW = cp_all_to_all_headseq(
            torch.cat([q_BLNQ, kv_BLNF], dim=-1), cp_group, seq_to_head=True
        )
        q_BTGQ, k_nope_BTGN, v_BTGV = torch.split(
            qkv_BTGW,
            [self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        # k_rope carries no head axis, so it cannot ride the head all-to-all.
        # All-gather its sequence shards instead (differentiable: the backward
        # is a reduce-scatter) and expand onto this rank's head subset. It is
        # qk_rope_head_dim per token, so the extra traffic is small.
        k_rope_BTR = torch.cat(
            dist_nn.all_gather(k_rope_BLR.contiguous(), group=cp_group), dim=1
        )
        k_BTGQ = torch.cat(
            [
                k_nope_BTGN,
                k_rope_BTR.view(B, t_full, 1, self.qk_rope_head_dim).expand(
                    B, t_full, g_loc, self.qk_rope_head_dim
                ),
            ],
            dim=-1,
        )

        out_BTGV = self.inner_attention(q_BTGQ, k_BTGQ, v_BTGV, scale=self.scale)
        out_BLNV = cp_all_to_all_headseq(
            out_BTGV.contiguous(), cp_group, seq_to_head=False
        )
        out_BLD = out_BLNV.reshape(B, t_loc, self.n_heads * self.v_head_dim)
        # The gate is pointwise on the sequence-local x, so it applies after the
        # heads have come back sequence-local.
        out_BLD = out_BLD * torch.sigmoid(self.gate(x_BLD))
        return self.wo(out_BLD)

    return ulysses_forward


def apply_ulysses_cp(mla_modules, cp_mesh: DeviceMesh) -> None:
    """Replace each MLA's forward with the Ulysses CP variant.

    Call before ``Module.parallelize()``, matching the contract of core's
    ``apply_cp_to_forward``.
    """
    cp_group = cp_mesh.get_group()
    for mod in mla_modules:
        mod.forward = _make_ulysses_forward(mod, cp_group)
