# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PixelShuffle + Linear projector for Qwen3-VL.

This module implements a LLaVA-NeXT / InternVL-style space-to-depth ("pixel
unshuffle") compressor followed by a single Linear projection, as a
SWITCHABLE alternative to the in-encoder PatchMerger (linear) projector used
in stock Qwen3-VL and to the BLIP-2-style Q-Former projector. The goal is
*deterministic* spatial token compression: no learnable query pool, no
cross-attention, just a fixed-shape rearrange + a single Linear.

Why this exists
---------------
For 3-cam x 4-frame nuScenes planning the stock Qwen3-VL pipeline emits
~1680 visual tokens / sample at the LM input boundary. PixelShuffle 2x
deterministically merges every 2x2 spatial block into one token (with
4 x channel dim) before a single Linear projection — a 4x reduction in
visual token budget at the LM boundary, putting this between the linear
baseline (~1680 tokens) and Q-Former-64 (64 tokens) in the
fusion-mechanism school comparison (A.0 / A.1 / A.2 / A.3).

Key properties vs Q-Former:
  * Zero learnable query parameters; only one Linear (in_dim*r^2 -> lm_dim).
  * Deterministic compression — every output token is a fixed linear combo
    of a known spatial neighbourhood. No cross-attention free parameters.
  * Cheap forward: pure reshape + matmul; no softmax, no per-layer LN.

Architecture
------------
Inputs are the post-encoder visual features at one of two staging points:

  (a) post-merger features  (in_features = out_hidden_size = 4096 for 8B):
      The in-encoder ``PatchMerger`` already collapsed each 2x2 patch group
      into one (lm_dim) token. We then run PixelShuffle 2x on the merged
      grid, giving a TOTAL spatial compression of (merge 2x2) * (shuffle
      2x2) = 16x raw ViT patches. Param count: Linear(4*lm_dim, lm_dim) =
      4 * 4096 * 4096 ~ 67M (FP32 bias-included).

  (b) pre-merger features   (in_features = ViT dim = 1152 for 8B):
      We bypass the in-encoder PatchMerger and consume raw ViT features.
      PixelShuffle 2x then gives a TOTAL spatial compression of 4x raw
      ViT patches. Param count: Linear(4*vit_dim, lm_dim) = 4 * 1152 *
      4096 ~ 19M (the ~17M target stated in the agent spec). Plumbing
      this path requires bypassing the in-encoder PatchMerger inside the
      vision encoder, which is a follow-up plumbing change (see
      docs/upstream_prs/006_torchtitan_qwen3_vl_pixelshuffle.md).

This file is path-(a) ready and is path-(b) capable as soon as the caller
flips the in-encoder PatchMerger off and routes raw ViT features through
this projector with ``in_features=1152``.

Shape contract
--------------
Inputs::

    vision_features : (B, N_vision, in_features)
    grid_thw        : (B, 3) — [t, h, w] patches in the **current** grid
                      (post-merger when consuming merged features, or raw
                      ViT grid when consuming pre-merger features).

Output::

    compressed      : (B, N_vision // (shuffle_ratio ** 2), lm_dim)

Note on multi-cam / multi-frame samples
---------------------------------------
``grid_thw`` carries the temporal dimension ``t``. PixelShuffle here is
**spatial only** (h, w) and we leave the temporal dim alone — each frame is
shuffled independently. For 3-cam x 4-frame the encoder emits 3 separate
visual items (each with its own ``grid_thw`` row), so the temporal axis is
typically t=1 per item; for true video inputs (t > 1) we still only shuffle
along the (h, w) plane.

Padding handling
----------------
The shape contract above is the *per-item* contract. The vision encoder
runs on a padded batch ``(num_items, max_num_patch, dim)``; the wiring
layer ``Qwen3VLModel._get_vision_embeds`` extracts valid tokens via a
``valid_mask`` AFTER the projector runs. For PixelShuffle this means we
must run the projector on the padded batch in shape ``(num_items,
max_num_patch, dim)``, then re-derive the valid-mask using
``num_tokens_per_item // shuffle_unit`` instead of ``// spatial_merge_unit``.
See ``Qwen3VLModel._get_vision_embeds`` for the wiring.

Even / odd grid sizes
---------------------
PixelShuffle 2x requires both H and W to be even AFTER any prior merger
step. Qwen3-VL's collator already pads images to multiples of
(patch_size * spatial_merge_size) = 32, so the post-merger grid sides are
always even (multiples of 1). For the pre-merger path, the collator's
patch_size=16 padding ensures raw ViT grid sides are at least multiples of
2. We assert this at forward time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common import Linear
from torchtitan.protocols.module import Module


class Qwen3VLPixelShufflePlusLinearProjector(Module):
    """PixelShuffle + Linear vision-to-LM projector.

    Duck-types the same call signature as the in-encoder ``PatchMerger`` so
    the wiring layer can swap projectors with ``model.projector_type``::

        out = projector(vision_features, grid_thw=grid_thw)

    Output is a deterministic ``shuffle_ratio ** 2`` compression of the
    input visual token sequence, projected to ``lm_dim``.

    Parameters
    ----------
    in_features : int
        Channel dim of incoming visual features. Set to ``out_hidden_size``
        (e.g. 4096 for Qwen3-VL-8B) when consuming the in-encoder
        PatchMerger output; set to ``vision_encoder.dim`` (e.g. 1152 for
        Qwen3-VL-8B) when consuming raw ViT features (pre-merger path).
    lm_dim : int
        LM hidden dim (output channel dim of the projector). 4096 for
        Qwen3-VL-8B.
    shuffle_ratio : int
        Spatial compression ratio. Must be >= 1. Each output token pools a
        ``shuffle_ratio x shuffle_ratio`` spatial block from the input grid.
        Default 2 (LLaVA-NeXT default).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configuration for ``Qwen3VLPixelShufflePlusLinearProjector``."""

        in_features: int
        lm_dim: int
        shuffle_ratio: int = 2
        # Optional ``proj`` Linear.Config; filled in by ``__init__.py`` so
        # this Config stays declarative.
        proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if config.shuffle_ratio < 1:
            raise ValueError(
                f"shuffle_ratio must be >= 1, got {config.shuffle_ratio}"
            )
        self.config = config
        self.in_features = config.in_features
        self.lm_dim = config.lm_dim
        self.shuffle_ratio = config.shuffle_ratio
        self.shuffle_unit = config.shuffle_ratio ** 2

        # Sanity: the Linear.Config wired in must match (in_features *
        # shuffle_unit) -> lm_dim. We don't enforce here (Linear is opaque)
        # but the factory in __init__.py constructs the right Linear.
        self.proj = config.proj.build()

    @staticmethod
    def _pixel_unshuffle_2d(
        x: torch.Tensor, shuffle_ratio: int
    ) -> torch.Tensor:
        """Space-to-depth on a (B, H, W, C) tensor.

        Reorders so that each output token covers a ``r x r`` spatial block,
        with the four channel-direction slots packed in row-major (top-left,
        top-right, bottom-left, bottom-right) order.

        Args:
            x: ``(B, H, W, C)`` with ``H`` and ``W`` divisible by
                ``shuffle_ratio``.
            shuffle_ratio: ``r``.

        Returns:
            ``(B, H // r, W // r, C * r * r)``.

        Note: ``torch.nn.PixelUnshuffle`` expects ``(B, C, H, W)``. We use a
        manual reshape so we can keep the channel-last layout that matches
        the rest of the vision pipeline and avoid two extra permutes per
        forward.
        """
        B, H, W, C = x.shape
        r = shuffle_ratio
        if H % r != 0 or W % r != 0:
            raise ValueError(
                f"PixelShuffle requires H ({H}) and W ({W}) divisible by "
                f"shuffle_ratio ({r}). Pad upstream (collator) if needed."
            )
        # Group every r x r spatial block. The intra-block layout is
        # (row_in_block, col_in_block) and gets packed into the channel
        # dim in row-major order; this is the LLaVA-NeXT / InternVL
        # convention.
        x = x.view(B, H // r, r, W // r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/r, W/r, r, r, C)
        x = x.view(B, H // r, W // r, r * r * C)
        return x

    def forward(
        self,
        vision_features: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PixelShuffle + Linear.

        Args:
            vision_features: ``(B, N_vision, in_features)`` — padded visual
                features at the **current** grid (post-merger if the
                in-encoder PatchMerger ran, pre-merger otherwise).
            grid_thw: ``(B, 3)`` per-item ``[t, h, w]`` patch counts at the
                CURRENT grid. The (h, w) here is the same grid as
                ``vision_features`` (NOT the raw collator grid divided by
                ``spatial_merge_size``). The wiring layer is responsible
                for passing the correct grid_thw — if the in-encoder
                merger ran, divide raw (h, w) by ``spatial_merge_size``
                before calling this projector.

        Returns:
            ``(B, N_vision // shuffle_unit, lm_dim)`` — compressed and
            LM-dim-projected visual features.

        Notes
        -----
        Padding: vision_features may have trailing PAD positions (the
        encoder runs in a padded batch). We rearrange the *padded* tensor
        as-is; downstream ``_get_vision_embeds`` re-masks valid tokens
        using ``num_tokens_per_item // shuffle_unit``.

        Multi-frame (t > 1): we flatten the temporal axis into the batch
        dimension so PixelShuffle stays spatial-only. Conceptually each
        frame is shuffled independently.
        """
        B, N, C = vision_features.shape
        if C != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, got channel dim "
                f"{C} in vision_features of shape {tuple(vision_features.shape)}"
            )

        # Recover the 2D (h, w) grid from grid_thw. We require all items in
        # the padded batch to share the same (t, h, w) — which is the case
        # for the nuScenes 3-cam / 1-cam dataloader (all frames at the
        # same resolution). For mixed-resolution batches the caller must
        # group by shape and call this projector per group.
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"grid_thw must have shape (B, 3); got {tuple(grid_thw.shape)}"
            )
        if grid_thw.shape[0] != B:
            raise ValueError(
                f"grid_thw batch dim ({grid_thw.shape[0]}) does not match "
                f"vision_features batch dim ({B})"
            )

        # All items must share the same (t, h, w) for the rearrange to be
        # valid on the padded batch. This is the standard case for nuScenes
        # planning (3 cams x 4 frames at fixed resolution).
        first = grid_thw[0]
        if not torch.equal(grid_thw, first.unsqueeze(0).expand_as(grid_thw)):
            raise ValueError(
                "PixelShuffle projector requires all items in the batch to "
                "share the same (t, h, w). For mixed-resolution batches, "
                "group by shape and call this projector per group. "
                f"grid_thw = {grid_thw.tolist()}"
            )
        t = int(first[0].item())
        h = int(first[1].item())
        w = int(first[2].item())

        if t * h * w > N:
            raise ValueError(
                f"grid_thw implies {t * h * w} tokens but only {N} are "
                f"present in vision_features. PixelShuffle projector expects "
                f"the padded length to be >= t*h*w."
            )

        r = self.shuffle_ratio
        if h % r != 0 or w % r != 0:
            raise ValueError(
                f"PixelShuffle ratio {r} requires h ({h}) and w ({w}) to be "
                f"divisible by {r}. Got grid_thw[0]={[t, h, w]}. The Qwen3-VL "
                f"collator pads images to multiples of patch_size * "
                f"spatial_merge_size = 32, so post-merger (h, w) should "
                f"already be even; check the wiring layer's grid_thw "
                f"computation."
            )

        # Split the padded length into [valid t*h*w block] and [pad tail].
        # We rearrange the valid block, then pad-zero the tail to match the
        # output's new (compressed) padded length so the downstream mask
        # ``num_tokens_per_item // shuffle_unit`` lines up.
        valid_len = t * h * w
        valid = vision_features[:, :valid_len, :]  # (B, t*h*w, C)
        # Move temporal into batch so PixelShuffle is spatial-only.
        valid = valid.view(B * t, h, w, C)
        valid = self._pixel_unshuffle_2d(valid, r)  # (B*t, h/r, w/r, C*r*r)
        valid = valid.view(B, t * (h // r) * (w // r), C * r * r)

        # Project to LM dim.
        out = self.proj(valid)  # (B, t*(h/r)*(w/r), lm_dim)

        # Reconstitute the padded shape: the compressed padded length is
        # N // shuffle_unit. Pad with zeros along the sequence dim. The
        # downstream valid_mask will drop these slots.
        compressed_padded_len = N // self.shuffle_unit
        new_valid_len = out.shape[1]
        if new_valid_len > compressed_padded_len:
            # Should be impossible (valid_len <= N and r^2 divides both),
            # but guard anyway.
            raise RuntimeError(
                f"Compressed valid length {new_valid_len} exceeds compressed "
                f"padded length {compressed_padded_len}. Internal invariant "
                f"violated."
            )
        if new_valid_len < compressed_padded_len:
            pad_amt = compressed_padded_len - new_valid_len
            pad_tail = out.new_zeros(B, pad_amt, self.lm_dim)
            out = torch.cat([out, pad_tail], dim=1)

        return out
