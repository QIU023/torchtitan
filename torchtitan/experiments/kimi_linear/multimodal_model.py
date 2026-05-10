# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLaVA-style multimodal wrapper around Kimi Linear.

Phase 4e scope: architecture scaffolding only. No training recipe,
no image data pipeline, no vision tower pretrained weights loading
— those are Phase 5 deliverables. This module lands the module
layout + forward signature so a CPU smoke can walk through the
image→projector→LLM hidden-state interleaving without crashing.

Why this matters: Kimi Linear's KDA-heavy backbone should handle
interleaved (vision_patch, text_token) sequences well because
linear attention's O(T) complexity doesn't blow up at long visual
contexts. The MLA layers (every 4th) provide full-attention
capacity where cross-modal global dependencies need it. This
scaffolding lets us experiment with that hypothesis.

Reference pattern (LLaVA 1.5 / LLaVA-NeXT):

  1. Vision tower: pretrained ViT (CLIP-ViT-L/14 or SigLIP), frozen
     by default. Patches → [B, N_vision, D_vision] features.
  2. Projector: 2-layer MLP that maps D_vision → D_llm. Trained.
  3. LLM: Kimi Linear model. Input is interleaved
     (text_embed, vision_embed, text_embed, ...) along the sequence
     axis. Loss computed on text tokens only (image tokens masked).

This class implements steps 2 and 3 + the interleaving logic.
Step 1 (ViT) is pluggable — constructor takes a pre-built
vision module, leaving HF-download / preprocessing policy to the
caller. Phase 5 adds a concrete SigLIP integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.kimi_linear.attn_res_model import (
    KimiLinearAttnResModel,
)
from torchtitan.experiments.kimi_linear.model import (
    KimiLinearConfig,
    KimiLinearModel,
)


@dataclass(kw_only=True, slots=True)
class KimiMultimodalConfig:
    """Config for the LLaVA-style multimodal wrapper.

    Attributes:
        kimi_config: The underlying Kimi Linear model's config.
        num_blocks: Optional AttnRes block count (None = plain
            KimiLinearModel backbone; int N = KimiLinearAttnResModel).
        vision_hidden_size: Output dim of the vision tower
            (e.g. CLIP-ViT-L/14 = 1024, SigLIP-400M = 1152).
        projector_hidden_size: Intermediate dim of the 2-layer
            projector. Common choice: 4 × vision_hidden_size.
        vision_token_id: Sentinel token id in the LLM vocab that
            marks "substitute this position with a vision feature".
            The caller's tokenizer + image preprocessor must agree
            on this value; we use it in ``forward`` to locate
            vision-insertion positions.
    """

    kimi_config: KimiLinearConfig
    num_blocks: int | None = None
    vision_hidden_size: int = 1024
    projector_hidden_size: int = 4096
    vision_token_id: int = -200  # LLaVA convention: negative sentinel


class KimiVisionProjector(nn.Module):
    """2-layer MLP projector: vision feature dim → LLM hidden dim.

    Standard LLaVA-1.5 recipe (GELU between layers). No bias on the
    linear layers so parameter shapes are minimal.

    Frozen vision tower is assumed; projector is the primary trained
    module during Stage-1 alignment training.
    """

    def __init__(
        self, *, vision_hidden_size: int, projector_hidden_size: int,
        llm_hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(projector_hidden_size, llm_hidden_size, bias=False)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project [B, N_vision, vision_d] → [B, N_vision, llm_d]."""
        return self.fc2(F.gelu(self.fc1(vision_features)))


class KimiLinearMultimodalModel(nn.Module):
    """Multimodal wrapper around Kimi Linear (LLaVA-style).

    Layout (top-level parameters):

      - ``vision_tower``: pretrained frozen ViT (SigLIP or CLIP).
        Passed in at construction so users can pick HF / local path.
        None-able; when None this class degenerates to a text-only
        path (useful for tests / ablations).
      - ``projector``: :class:`KimiVisionProjector` — trained.
      - ``llm``: :class:`KimiLinearModel` OR :class:`KimiLinearAttnResModel`.

    Forward accepts:
      - ``input_ids``: ``[B, T]`` token sequence where positions
        matching ``vision_token_id`` are sentinels to be replaced with
        projected vision features.
      - ``pixel_values``: ``[B, num_images, C, H, W]`` or ``None``.
        ``None`` takes the text-only path.

    Output: LLM logits ``[B, T, vocab_size]``.

    Loss is standard cross-entropy on text tokens. Typical training
    masks out labels at vision-token positions (ignore_index=-100);
    that's the caller's responsibility.

    No KV cache / generation path: this is training-time scaffolding.
    """

    def __init__(
        self, config: KimiMultimodalConfig, *,
        vision_tower: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower  # may be None for text-only

        self.projector = KimiVisionProjector(
            vision_hidden_size=config.vision_hidden_size,
            projector_hidden_size=config.projector_hidden_size,
            llm_hidden_size=config.kimi_config.hidden_size,
        )

        if config.num_blocks is None:
            self.llm = KimiLinearModel(config.kimi_config)
        else:
            self.llm = KimiLinearAttnResModel(
                config.kimi_config, num_blocks=config.num_blocks
            )

        # Freeze vision tower by default (LLaVA-1.5 stage-1 recipe).
        if self.vision_tower is not None:
            for p in self.vision_tower.parameters():
                p.requires_grad = False

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run frozen vision tower → projector. Returns projected
        features of shape ``[B, num_images, N_vision, llm_hidden_size]``.

        ``pixel_values``: ``[B, num_images, C, H, W]``. We flatten the
        ``(B, num_images)`` batch for the vision tower call, then
        re-nest.
        """
        assert self.vision_tower is not None
        B, num_images, C, H, W = pixel_values.shape
        flat = pixel_values.view(B * num_images, C, H, W)
        vision_features = self.vision_tower(flat)
        # Expect vision_tower output shape [B*num_images, N_vision, D_vision].
        # Different encoders return different shapes; we coerce to that here.
        if vision_features.dim() != 3:
            raise RuntimeError(
                f"vision_tower returned shape {vision_features.shape}; "
                "expected [B*num_images, N_vision, D_vision]."
            )
        projected = self.projector(vision_features)
        _, N_vision, D_llm = projected.shape
        return projected.view(B, num_images, N_vision, D_llm)

    def _inject_vision_features(
        self, input_ids: torch.Tensor, vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """Build the interleaved embedding sequence.

        For every position in ``input_ids`` matching
        ``config.vision_token_id``, substitute the corresponding
        projected vision feature into that embedding slot.

        This is the LLaVA approach: the tokenizer emits a special
        ``<image>`` token (mapped to ``vision_token_id``) at each
        image-insertion site, and each such token is expanded to
        ``N_vision`` feature vectors at embed time.

        Args:
            input_ids: ``[B, T]`` (raw ids; ``vision_token_id`` marks
                image-insertion positions, one token per image).
            vision_features: ``[B, num_images, N_vision, D_llm]``.
        Returns:
            ``[B, T_expanded, D_llm]`` where each ``vision_token_id``
            has been replaced with ``N_vision`` projected-feature
            vectors (so the sequence length grows accordingly).
        """
        B, T = input_ids.shape
        _, num_images, N_vision, D_llm = vision_features.shape

        # Text embeddings for the whole sequence from the LLM's embed_tokens.
        # Note: at vision-token positions the embed output will be
        # meaningless (embedding of the sentinel id) and gets replaced.
        # We still call embed_tokens so non-vision positions get their
        # real embeddings; for vision positions the result is discarded.
        text_embeds = self.llm.embed_tokens(
            torch.where(
                input_ids == self.config.vision_token_id,
                # Replace sentinel with token 0 for safe embedding lookup;
                # we'll overwrite those positions anyway.
                torch.zeros_like(input_ids),
                input_ids,
            )
        )  # [B, T, D_llm]

        # Build the expanded sequence. For each sample, walk tokens:
        #   non-vision → keep text_embed[b, t]
        #   vision     → emit N_vision slots filled with
        #                vision_features[b, img_counter] (advance counter)
        out_per_batch = []
        for b in range(B):
            img_counter = 0
            pieces: list[torch.Tensor] = []
            for t in range(T):
                tok = input_ids[b, t].item()
                if tok == self.config.vision_token_id:
                    if img_counter >= num_images:
                        raise RuntimeError(
                            f"Sample {b} has more vision tokens than images "
                            f"({img_counter + 1} > {num_images})."
                        )
                    pieces.append(vision_features[b, img_counter])
                    img_counter += 1
                else:
                    pieces.append(text_embeds[b, t : t + 1])
            out_per_batch.append(torch.cat(pieces, dim=0))

        # Pad to the longest expanded length in the batch. Usually all
        # samples have the same number of image tokens so this is a
        # no-op; handles mixed batches gracefully.
        max_len = max(x.size(0) for x in out_per_batch)
        padded = torch.zeros(
            (B, max_len, D_llm),
            device=vision_features.device,
            dtype=vision_features.dtype,
        )
        for b, seq in enumerate(out_per_batch):
            padded[b, : seq.size(0)] = seq
        return padded

    def forward(
        self, input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multimodal forward.

        Args:
            input_ids: ``[B, T]``. Contains ``vision_token_id`` sentinels
                where vision features should be spliced in.
            pixel_values: ``[B, num_images, C, H, W]`` or ``None``. If
                ``None`` OR contains no ``vision_token_id``, takes the
                text-only path through the LLM.

        Returns:
            Logits ``[B, T_expanded, vocab_size]``.
        """
        has_vision = (
            pixel_values is not None
            and (input_ids == self.config.vision_token_id).any()
        )

        if has_vision:
            assert self.vision_tower is not None, (
                "pixel_values supplied but vision_tower is None; "
                "construct with a vision module or drop pixel_values."
            )
            vision_features = self._encode_images(pixel_values)
            inputs_embeds = self._inject_vision_features(
                input_ids, vision_features
            )
            # LLM was designed to take token ids via embed_tokens.
            # Bypass the embedding by passing hidden states directly.
            # This works because forward(tokens) dispatches based on
            # whether embed_tokens is present — if we feed a pre-embedded
            # [B, T, D] tensor, the if-branch skips embedding. BUT for
            # stage 0 the embed is present; we'd double-embed. Workaround:
            # send a sentinel-rich ids buffer through the embed anyway
            # and SUBSTITUTE at the output — which is what
            # _inject_vision_features already did. So here we need to
            # feed the LLM pre-embedded inputs.
            #
            # Directly call the LLM's internal layers to bypass embed:
            return self._llm_forward_from_embeds(inputs_embeds)

        # Text-only path: plain LLM.
        return self.llm(input_ids)

    def _llm_forward_from_embeds(self, h: torch.Tensor) -> torch.Tensor:
        """Run the LLM's decoder stack starting from pre-embedded
        hidden states. Needed because ``_inject_vision_features``
        already did the embedding lookup — we must not re-embed.

        Kimi Linear's forward is signature-based on stage detection
        (tokens int64 → embed, tokens float → pass-through). We
        temporarily detach ``embed_tokens`` so the model takes the
        pass-through branch, then restore.
        """
        saved_embed = self.llm.embed_tokens
        try:
            self.llm.embed_tokens = None
            return self.llm(h)
        finally:
            self.llm.embed_tokens = saved_embed
