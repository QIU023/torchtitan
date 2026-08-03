# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLaVA-style multimodal wrapper around Kimi Linear.

Scaffolding scope: architecture only. No training recipe, no image
data pipeline, no vision tower pretrained weights loading -- those are
follow-up work. This module lands the module
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
caller. A concrete SigLIP integration is follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.kimi_k3.attn_res_model import (
    KimiLinearAttnResModel,
)
from torchtitan.models.kimi_k3.moonvit import MoonViTConfig  # noqa: F401
from torchtitan.models.kimi_k3.model import (
    KimiLinearSpec,
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


# ----- K3's own vision path ---------------------------------------------- #


@dataclass(kw_only=True, slots=True)
class KimiK3MultimodalConfig:
    """Config for K3's native vision path.

    Differs from :class:`KimiMultimodalConfig` in three ways that follow from
    the release rather than from the LLaVA recipe:

    * the projector belongs to the tower (``mm_projector`` is a MoonViT child in
      the checkpoint), so there is no separate projector here;
    * the tower is NOT frozen -- report sec 2.4 trains MoonViT-V2 from scratch
      with next-token prediction, and the whole point of that choice was joint
      stability, so freezing it reproduces the opposite recipe;
    * vision features are variable length per sample (native resolution), so
      they arrive as a list rather than a padded ``[B, num_images, N, D]``.
    """

    kimi_config: KimiLinearConfig
    vision_config: "MoonViTConfig"
    num_blocks: int | None = None
    vision_token_id: int = -200


class KimiK3MultimodalModel(nn.Module):
    """MoonViT-V2 + Kimi Linear backbone, wired as the release has it.

    Submodule names mirror the checkpoint (``vision_tower``, ``language_model``)
    so ``hf_key_map`` is a prefix rename.
    """

    def __init__(self, config: KimiK3MultimodalConfig) -> None:
        super().__init__()
        from torchtitan.models.kimi_k3.moonvit import MoonViT

        self.config = config
        self.vision_tower = MoonViT(config.vision_config)
        if config.num_blocks is None:
            self.language_model = KimiLinearModel(config.kimi_config)
        else:
            self.language_model = KimiLinearAttnResModel(
                config.kimi_config, num_blocks=config.num_blocks
            )
        if config.vision_config.text_hidden_size != config.kimi_config.hidden_size:
            raise ValueError(
                "the projector's output width must equal the LLM's hidden size: "
                f"{config.vision_config.text_hidden_size} != "
                f"{config.kimi_config.hidden_size}"
            )

    def encode_images(
        self, patches: torch.Tensor, grid_thws: torch.Tensor
    ) -> list[torch.Tensor]:
        """Packed patches -> one ``[N_i, D_llm]`` feature block per sample."""
        return self.vision_tower(patches, grid_thws)

    def _splice(
        self,
        input_ids: torch.Tensor,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Replace each vision sentinel with that sample's feature block.

        One sentinel per image, expanded in place to its own token count, so the
        sequence grows by a different amount per sample. Rows are right-padded
        with the embedding of token 0 to a common length; the caller masks the
        padding in the loss, exactly as it must for the sentinel positions.
        """
        B, T = input_ids.shape
        sentinel = self.config.vision_token_id
        embed = self.language_model.embed_tokens
        safe_ids = torch.where(
            input_ids == sentinel, torch.zeros_like(input_ids), input_ids
        )
        text = embed(safe_ids)

        rows, feat_iter = [], iter(features)
        for b in range(B):
            positions = (input_ids[b] == sentinel).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                rows.append(text[b])
                continue
            pieces, cursor = [], 0
            for pos in positions.tolist():
                pieces.append(text[b, cursor:pos])
                pieces.append(next(feat_iter).to(text.dtype))
                cursor = pos + 1
            pieces.append(text[b, cursor:])
            rows.append(torch.cat(pieces, dim=0))

        width = max(r.size(0) for r in rows)
        pad = embed(torch.zeros(1, dtype=input_ids.dtype, device=input_ids.device))
        return torch.stack(
            [
                r
                if r.size(0) == width
                else torch.cat([r, pad.expand(width - r.size(0), -1)], dim=0)
                for r in rows
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        patches: torch.Tensor | None = None,
        grid_thws: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """``[B, T]`` ids (+ packed patches) -> logits.

        ``**kwargs`` is ignored, mirroring KimiLinearModel: torchtitan's Trainer
        and Validator inject ``attention_masks=None`` and ``positions=...`` for
        the FlexAttention / CP paths, and K3 uses plain SDPA plus KDA Triton
        kernels which take neither.

        Text-only when no patches are supplied or no sentinel is present.
        """
        sentinel_present = bool(
            (input_ids == self.config.vision_token_id).any().item()
        )
        if patches is None or not sentinel_present:
            if patches is not None and not sentinel_present:
                raise ValueError(
                    "patches supplied but input_ids contains no "
                    f"vision_token_id ({self.config.vision_token_id}); the "
                    "images would be silently dropped"
                )
            return self.language_model(input_ids)
        if grid_thws is None:
            raise ValueError("grid_thws is required alongside patches")

        features = self.encode_images(patches, grid_thws)
        num_sentinels = int((input_ids == self.config.vision_token_id).sum().item())
        if len(features) != num_sentinels:
            raise ValueError(
                f"{len(features)} image(s) encoded but {num_sentinels} vision "
                "sentinel(s) in input_ids; these must correspond one to one"
            )
        embeds = self._splice(input_ids, features)
        # The backbone's forward embeds int ids; we already embedded, so detach
        # embed_tokens to take its pre-embedded branch. Same mechanism as
        # KimiLinearMultimodalModel._llm_forward_from_embeds.
        saved = self.language_model.embed_tokens
        try:
            self.language_model.embed_tokens = None
            return self.language_model(embeds)
        finally:
            self.language_model.embed_tokens = saved

    def init_weights(self, init_range: float | None = None, **kwargs) -> None:
        self.vision_tower.init_weights(init_range)
        self.language_model.init_weights(init_range, **kwargs)


def _mm_layers(self):
    """Expose the text stack where parallelize.py and the PP splitter look.

    Both walk ``model.layers`` (a ModuleDict keyed by layer id -- the PP
    adapter's layer_to_stage discovery depends on those string keys). The
    multimodal wrapper keeps the text model at ``self.language_model``, so
    without this the FSDP wrap fails with "no attribute 'layers'" before any
    step runs.
    """
    return self.language_model.layers


KimiK3MultimodalModel.layers = property(_mm_layers)


def _mm_verify_module_protocol(self) -> None:
    """No-op, delegating to the text model's reasoning.

    KimiLinearModel overrides this as a no-op because its internals are plain
    nn.Modules rather than Config-built ``Module`` instances -- it ports the HF
    reference layer by layer. The multimodal wrapper adds a MoonViT tower built
    the same way, so the same holds. The trainer calls this post-build; without
    it the multimodal flavor cannot be constructed at all.
    """
    return None


KimiK3MultimodalModel.verify_module_protocol = _mm_verify_module_protocol
for _name in ("get_attention_masks", "init_weights"):
    if not hasattr(KimiK3MultimodalModel, _name) and hasattr(
        KimiLinearModel, _name
    ):
        setattr(KimiK3MultimodalModel, _name, getattr(KimiLinearModel, _name))


@dataclass(kw_only=True, slots=True)
class KimiK3MultimodalSpec(KimiLinearSpec):
    """``BaseModel.Config``-compatible spec for the multimodal model.

    KimiLinearSpec exists because torchtitan's trainer calls
    ``update_from_config`` and the property accessors on whatever sits at
    ``model_spec.model``; a bare dataclass config fails there. This subclasses it
    so the multimodal flavor gets the same integration surface, and overrides
    only ``build`` to construct the vision-bearing model.
    """

    vision_config: "MoonViTConfig" = None  # type: ignore[assignment]

    def build(self, **kwargs):
        return KimiK3MultimodalModel(
            KimiK3MultimodalConfig(
                kimi_config=self.kimi_config,
                vision_config=self.vision_config,
                num_blocks=self.num_blocks,
            )
        )
