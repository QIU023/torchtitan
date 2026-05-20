# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Q-Former projector for Qwen3-VL: token-budget compressor.

This module implements a BLIP-2-style Querying Transformer (Q-Former) as a
SWITCHABLE alternative to the linear PatchMerger projector used in stock
Qwen3-VL. Given a variable-length sequence of vision encoder features
``(B, N_vision, vit_dim)`` (here ``vit_dim`` is the ViT hidden dimension
*before* the spatial merger or the LM-dim output *after* the merger;
configurable via ``in_features``), it produces a FIXED-length sequence of
``num_queries`` (default 64) tokens at the LM hidden dimension.

Why this exists
---------------
For 3-cam x 4-frame nuScenes planning the stock Qwen3-VL pipeline emits
~1680 visual tokens per sample before the text. That is fine for HBM-only
training but is expensive for TRT-deploy inference (KV-cache scales with
context length). A 64-query Q-Former compresses the visual token budget
by ~26x, which directly translates to lower TTFT and a smaller KV-cache on
the edge device. This is a **deployment-efficiency** trade, NOT a
fusion-mechanism research comparison — from-scratch Q-Former on 24K
samples will lose to the pretrained Qwen3-VL alignment at the same
token budget, and that is expected and acceptable.

Architecture (simplified vs BLIP-2)
-----------------------------------
* ``num_queries`` learnable parameters of shape ``(num_queries, internal_dim)``
  where ``internal_dim`` (default 1024) is the Q-Former's internal width.
  This is DECOUPLED from the LM hidden dim ``lm_dim`` (e.g. 4096 for
  Qwen3-VL-8B) — keeping ``internal_dim`` small (BLIP-2-base uses 768)
  is what bounds the parameter count to ~80M instead of ~1.2B.
* ``num_layers`` cross-attention blocks. Each block operates entirely at
  ``internal_dim``::

      q_norm   = LayerNorm(queries)           # queries: (B, Nq, internal_dim)
      kv_norm  = LayerNorm(kv)                # kv: (B, Nkv, in_features)
      attn_out = MultiheadAttention(query=q_norm, key=kv_norm, value=kv_norm)
      queries  = queries + attn_out           # still (B, Nq, internal_dim)
      ffn_out  = FFN(LayerNorm(queries))
      queries  = queries + ffn_out

  No self-attention between queries (BLIP-2 has it; we don't need it for
  downstream tasks at this token budget). The FFN follows the standard
  ViT pattern: Linear(internal_dim, ffn_mult * internal_dim) -> GELU ->
  Linear(ffn_mult * internal_dim, internal_dim).
* Final ``LayerNorm`` on the queries followed by a Linear projection
  ``out_proj: internal_dim -> lm_dim`` that lifts the compressed tokens
  back to LM-compatible width.

Shape contract
--------------
* Input  ``vision_features``: ``(B, N_vision, in_features)`` where ``N_vision``
  may vary per batch (padded). Optional ``key_padding_mask``:
  ``(B, N_vision)`` bool, ``True`` at padded positions.
* Output ``compressed``: ``(B, num_queries, lm_dim)`` — a fixed-length set
  of LM-dim tokens, ready to be scattered into the LLM input at the
  ``<|image_pad|>`` placeholder positions.

Note on input dimension
-----------------------
``in_features`` controls the cross-attn KV projection's input dim. If the
caller feeds ViT pre-merger features the dim is ``vit_dim`` (e.g. 1152
for Qwen3-VL-8B); if they feed post-merger features it is ``lm_dim``
(e.g. 4096). The output is always ``lm_dim`` (via ``out_proj`` from
``internal_dim`` to ``lm_dim``).

Wiring this into ``Qwen3VLModel`` is done by ``__init__.py`` when the model
config carries ``projector_type="qformer"``. The model then *bypasses*
the in-encoder ``PatchMerger`` and feeds raw ViT features into the
Q-Former. See ``model.py`` for the switch and the open issue about
double-compression (in-encoder spatial-merge 2x2 + Q-Former 26x).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common import Linear
from torchtitan.protocols.module import Module, ModuleList

LayerNorm = Module.from_nn_module(nn.LayerNorm)
GELU = Module.from_nn_module(nn.GELU)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _QFormerCrossAttention(Module):
    """Multi-head cross-attention from queries (internal_dim) to kv (in_features).

    All q/k/v/o projections operate at ``internal_dim`` (not ``lm_dim``);
    the lift to ``lm_dim`` happens once at the projector boundary.

    Uses plain torch ops (no FlexAttention) because:
      * ``num_queries`` is tiny (64) — the FlexAttention block-mask machinery
        is not worth its setup cost.
      * The KV side carries a per-batch padding mask, which is easier to
        express as a standard boolean ``key_padding_mask`` than as a
        BlockMask mod_fn.
    """

    def __init__(
        self,
        internal_dim: int,
        kv_in_features: int,
        n_heads: int,
        *,
        q_proj: Linear.Config,
        k_proj: Linear.Config,
        v_proj: Linear.Config,
        o_proj: Linear.Config,
    ):
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.internal_dim = internal_dim
        self.kv_in_features = kv_in_features
        self.n_heads = n_heads
        self.head_dim = internal_dim // n_heads

        self.q_proj = q_proj.build()
        self.k_proj = k_proj.build()
        self.v_proj = v_proj.build()
        self.o_proj = o_proj.build()

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention with padded KV.

        Args:
            queries: (B, Nq, internal_dim)
            kv: (B, Nkv, kv_in_features)
            key_padding_mask: (B, Nkv) bool, ``True`` at PAD positions.
                Padded positions get a -inf attention score so softmax weight
                drops to 0.

        Returns:
            (B, Nq, internal_dim)
        """
        B, Nq, _ = queries.shape
        Nkv = kv.shape[1]
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(queries).view(B, Nq, H, D).transpose(1, 2)  # (B,H,Nq,D)
        k = self.k_proj(kv).view(B, Nkv, H, D).transpose(1, 2)  # (B,H,Nkv,D)
        v = self.v_proj(kv).view(B, Nkv, H, D).transpose(1, 2)  # (B,H,Nkv,D)

        if key_padding_mask is not None:
            # Convert (B, Nkv) bool → (B, 1, 1, Nkv) additive mask.
            # SDPA expects an ADDITIVE mask (not a bool mask of "valid"),
            # so we pass it via attn_mask.
            attn_bias = torch.zeros(
                B, 1, 1, Nkv, dtype=queries.dtype, device=queries.device
            )
            attn_bias = attn_bias.masked_fill(
                key_padding_mask.view(B, 1, 1, Nkv), float("-inf")
            )
        else:
            attn_bias = None

        # scaled_dot_product_attention is CPU-safe and avoids the FlexAttention
        # compile path (Q-Former is tiny so we don't need it).
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias
        )  # (B, H, Nq, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, Nq, self.internal_dim)
        return self.o_proj(attn_out)


class _QFormerFFN(Module):
    """Position-wise feed-forward network with GELU.

    Matches the ViT-style FFN pattern used elsewhere in this codebase
    (see ``vision_encoder.VisionMLP``).
    """

    def __init__(self, *, fc1: Linear.Config, fc2: Linear.Config):
        super().__init__()
        self.linear_fc1 = fc1.build()
        self.linear_fc2 = fc2.build()
        self.act_fn = GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class _QFormerBlock(Module):
    """One Q-Former block: cross-attn + FFN with pre-LN residuals.

    All internal tensors are at ``internal_dim`` width; the final lift to
    ``lm_dim`` is done by the top-level projector's ``out_proj``.
    """

    def __init__(
        self,
        internal_dim: int,
        kv_in_features: int,
        n_heads: int,
        layer_norm_eps: float,
        *,
        q_proj: Linear.Config,
        k_proj: Linear.Config,
        v_proj: Linear.Config,
        o_proj: Linear.Config,
        ffn_fc1: Linear.Config,
        ffn_fc2: Linear.Config,
    ):
        super().__init__()
        self.norm_q = LayerNorm(internal_dim, eps=layer_norm_eps)
        self.norm_kv = LayerNorm(kv_in_features, eps=layer_norm_eps)
        self.norm_ffn = LayerNorm(internal_dim, eps=layer_norm_eps)

        self.cross_attn = _QFormerCrossAttention(
            internal_dim=internal_dim,
            kv_in_features=kv_in_features,
            n_heads=n_heads,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )
        self.ffn = _QFormerFFN(fc1=ffn_fc1, fc2=ffn_fc2)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN cross-attention (residual on raw queries).
        queries = queries + self.cross_attn(
            self.norm_q(queries),
            self.norm_kv(kv),
            key_padding_mask=key_padding_mask,
        )
        # Pre-LN FFN.
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


# ---------------------------------------------------------------------------
# Top-level projector
# ---------------------------------------------------------------------------


class Qwen3VLQFormerProjector(Module):
    """BLIP-2-style Q-Former projector for Qwen3-VL.

    Duck-types the same call signature as the in-encoder ``PatchMerger`` for
    the wiring layer's convenience:

        out = projector(vision_features)

    but returns a fixed ``num_queries`` tokens regardless of input length.

    Param-count design
    ------------------
    The whole point of decoupling ``internal_dim`` from ``lm_dim`` is to
    keep the Q-Former at BLIP-2 scale (~80-120M params) instead of having
    its width be dragged up to ``lm_dim=4096`` by the LM. At the default
    config (``internal_dim=1024``, ``num_layers=6``, ``n_heads=8``,
    ``ffn_mult=4``)::

        per_block (4 attn projections + 2 FFN linears + 3 LNs):
            q_proj  : 1024 * 1024 + 1024  ≈ 1.05M
            k_proj  : kv_in * 1024 + 1024 ≈ 1.18M (kv_in=1152) / 4.20M (kv_in=4096)
            v_proj  : kv_in * 1024 + 1024 ≈ 1.18M / 4.20M
            o_proj  : 1024 * 1024 + 1024  ≈ 1.05M
            ffn_fc1 : 1024 * 4096 + 4096  ≈ 4.20M
            ffn_fc2 : 4096 * 1024 + 1024  ≈ 4.20M
            subtotal ≈ 12.86M (kv_in=1152) / 18.90M (kv_in=4096)
        x 6 layers ≈ 77M / 113M
        + queries  : 64 * 1024 = 65K
        + out_proj : 1024 * 4096 + 4096 ≈ 4.20M
        + norm_out LN: 2K
        Total ≈ 81M (kv_in=1152) / 118M (kv_in=4096)

    See ``tests/unit_tests/test_qformer_projector.py`` for the source-of-
    truth measured count.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configuration for ``Qwen3VLQFormerProjector``."""

        # Cross-attn KV input dimension. Set to ViT hidden dim when the
        # caller is bypassing the in-encoder merger and feeding raw ViT
        # features (e.g. 1152 for Qwen3-VL-8B). Set to LM hidden dim
        # (4096) when the caller has already routed features through the
        # in-encoder merger.
        in_features: int

        # LM hidden dim (final output dim of the projector — the lift from
        # ``internal_dim`` to ``lm_dim`` is done by ``out_proj``).
        lm_dim: int

        # Q-Former internal width. Decoupled from ``lm_dim`` so the param
        # count stays bounded (~80-120M at default) instead of growing
        # geometrically with ``lm_dim``. BLIP-2-base uses 768; we default
        # to 1024 to give a bit of headroom on driving scene complexity.
        internal_dim: int = 1024

        # Q-Former depth and width.
        num_queries: int = 64
        num_layers: int = 6
        n_heads: int = 8
        ffn_mult: int = 4
        layer_norm_eps: float = 1e-6

        # Per-sub-module Linear configs. These must be filled in by the
        # ``__init__.py`` factory because Linear.Config requires concrete
        # in/out features. Note: all q/k/v/o/ffn linears operate at
        # ``internal_dim``; only ``out_proj`` lifts to ``lm_dim``.
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        o_proj: Linear.Config
        ffn_fc1: Linear.Config
        ffn_fc2: Linear.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_queries = config.num_queries
        self.lm_dim = config.lm_dim
        self.internal_dim = config.internal_dim
        self.in_features = config.in_features

        # Learnable query tokens at internal_dim — initialized by
        # ``_param_init`` via the standard ``Module.init_states`` path.
        self.queries = nn.Parameter(
            torch.empty(config.num_queries, config.internal_dim)
        )

        self.layers = ModuleList(
            [
                _QFormerBlock(
                    internal_dim=config.internal_dim,
                    kv_in_features=config.in_features,
                    n_heads=config.n_heads,
                    layer_norm_eps=config.layer_norm_eps,
                    q_proj=config.q_proj,
                    k_proj=config.k_proj,
                    v_proj=config.v_proj,
                    o_proj=config.o_proj,
                    ffn_fc1=config.ffn_fc1,
                    ffn_fc2=config.ffn_fc2,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = LayerNorm(config.internal_dim, eps=config.layer_norm_eps)
        # Final boundary projection: internal_dim -> lm_dim. This is the
        # ONLY place lm_dim shows up; the rest of the Q-Former is sized
        # by internal_dim.
        self.out_proj = config.out_proj.build()

    def forward(
        self,
        vision_features: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress a variable-length vision feature sequence to ``num_queries``.

        Args:
            vision_features: (B, N_vision, in_features) — ViT (pre- or
                post-merger) features.
            key_padding_mask: (B, N_vision) bool, ``True`` at PAD positions.
                Optional; pass when the input is padded.

        Returns:
            (B, num_queries, lm_dim) — compressed visual tokens, ready to
            be scattered into the LLM input at the ``<|image_pad|>``
            placeholder positions.
        """
        B = vision_features.shape[0]
        # Broadcast the learnable queries across batch.
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()

        for layer in self.layers:
            queries = layer(
                queries,
                vision_features,
                key_padding_mask=key_padding_mask,
            )

        # LayerNorm at internal_dim, then lift to lm_dim.
        return self.out_proj(self.norm_out(queries))
