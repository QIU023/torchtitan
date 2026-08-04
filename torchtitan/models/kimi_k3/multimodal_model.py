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
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.distributed.fsdp import add_zero_valued_dependency
from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel
from torchtitan.models.kimi_k3.model import KimiK3Config, KimiK3Model, KimiK3Spec
from torchtitan.models.kimi_k3.moonvit import MoonViTConfig  # noqa: F401


@dataclass(kw_only=True, slots=True)
class KimiMultimodalConfig:
    """Config for the LLaVA-style multimodal wrapper.

    Attributes:
        kimi_config: The underlying Kimi Linear model's config.
        num_blocks: Optional AttnRes block count (None = plain
            KimiK3Model backbone; int N = KimiK3AttnResModel).
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

    kimi_config: KimiK3Config
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
        self,
        *,
        vision_hidden_size: int,
        projector_hidden_size: int,
        llm_hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(projector_hidden_size, llm_hidden_size, bias=False)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project [B, N_vision, vision_d] → [B, N_vision, llm_d]."""
        return self.fc2(F.gelu(self.fc1(vision_features)))


class KimiK3LlavaMultimodalModel(nn.Module):
    """Multimodal wrapper around Kimi Linear (LLaVA-style).

    Layout (top-level parameters):

      - ``vision_tower``: pretrained frozen ViT (SigLIP or CLIP).
        Passed in at construction so users can pick HF / local path.
        None-able; when None this class degenerates to a text-only
        path (useful for tests / ablations).
      - ``projector``: :class:`KimiVisionProjector` — trained.
      - ``llm``: :class:`KimiK3Model` OR :class:`KimiK3AttnResModel`.

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
        self,
        config: KimiMultimodalConfig,
        *,
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
            self.llm = KimiK3Model(config.kimi_config)
        else:
            self.llm = KimiK3AttnResModel(
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
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
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
        self,
        input_ids: torch.Tensor,
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
            inputs_embeds = self._inject_vision_features(input_ids, vision_features)
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

    kimi_config: KimiK3Config
    vision_config: "MoonViTConfig"
    num_blocks: int | None = None
    vision_token_id: int = -200


class KimiK3MultimodalModel(nn.Module):
    """MoonViT-V2 + Kimi Linear backbone, wired as the release has it.

    Submodule names mirror the checkpoint (``vision_tower``, ``language_model``)
    so ``hf_key_map`` is a prefix rename.
    """

    @classmethod
    def from_parts(
        cls,
        config: KimiK3MultimodalConfig,
        vision_tower: nn.Module,
        language_model: nn.Module,
    ) -> "KimiK3MultimodalModel":
        """Assemble from already-built parts, skipping ``__init__``'s construction.

        The PP split cannot see through this wrapper: core's ``_split_module``
        walks only top-level ``named_children()``, so neither the flat FQNs
        (``embed_tokens``, ``layers.N``) nor dotted ones
        (``language_model.layers.N``) match anything here, and every child is
        replaced by None -- the stage ends up with zero parameters. The adapter
        therefore splits the TEXT model and rebuilds this wrapper around the
        chunk that owns ``embed_tokens``, which is the only place vision
        features are consumed (they are spliced into the embeddings, so nothing
        vision-side ever crosses a stage boundary).
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = vision_tower
        self.language_model = language_model
        return self

    def __init__(self, config: KimiK3MultimodalConfig) -> None:
        super().__init__()
        from torchtitan.models.kimi_k3.moonvit import MoonViT

        self.config = config
        self.vision_tower = MoonViT(config.vision_config)
        if config.num_blocks is None:
            self.language_model = KimiK3Model(config.kimi_config)
        else:
            self.language_model = KimiK3AttnResModel(
                config.kimi_config, num_blocks=config.num_blocks
            )
        if config.vision_config.text_hidden_size != config.kimi_config.hidden_size:
            raise ValueError(
                "the projector's output width must equal the LLM's hidden size: "
                f"{config.vision_config.text_hidden_size} != "
                f"{config.kimi_config.hidden_size}"
            )

    def encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[torch.Tensor]:
        """Collator patches -> one ``[N_i, D_llm]`` feature block per sample.

        The two sides disagree on layout and the shapes do not collide loudly:
        ``MMCollator`` emits ``[num_images, max_patches, C*P*P]``, zero-PADDED
        to the largest image in the batch, while MoonViT's patch_embed is a
        ``Conv2d`` over ``[L, C, P, P]`` with the images CONCATENATED and no
        padding. Feeding the collator's tensor straight through reaches the
        conv as a 3-D input and fails there.

        ``grid_thw`` carries each image's ``(t, h, w)``, whose product is that
        image's real patch count, so the padding is dropped exactly rather than
        by scanning for zero rows -- a black patch is legitimately all zeros.
        """
        cfg = self.config.vision_config
        counts = grid_thw.prod(dim=-1).tolist()
        packed = torch.cat([pixel_values[i, :n] for i, n in enumerate(counts)], dim=0)
        packed = packed.reshape(-1, cfg.in_channels, cfg.patch_size, cfg.patch_size)
        # The collator emits float32; under FSDP's mixed precision the tower's
        # weights are bf16, and Conv2d refuses the mix rather than promoting.
        weight = self.vision_tower.patch_embed.proj.weight
        packed = packed.to(weight.dtype)

        # Under TP the tower's params are replicated DTensors (parallelize.py
        # distributes them so grad-norm clipping sees one mesh). Lift the input
        # in and drop the outputs back out here: every placement is Replicate,
        # so both conversions are local metadata changes, not collectives.
        # Keyed on the mesh parallelize recorded, NOT on whether the weight is
        # a DTensor -- under FSDP it is one either way, and lifting onto the
        # FSDP mesh meets the plain all-gathered weight inside the conv.
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            packed = DTensor.from_local(
                packed, tp_mesh, (Replicate(),), run_check=False
            )
        features = self.vision_tower(packed, grid_thw)
        if isinstance(features, torch.Tensor):
            return features.to_local() if isinstance(features, DTensor) else features
        return [f.to_local() if isinstance(f, DTensor) else f for f in features]

    def _cp_world_size(self) -> int:
        group = getattr(self, "_cp_group", None)
        return 1 if group is None else torch.distributed.get_world_size(group)

    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group.
        """
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_tower.parameters()
        )

    def _tower_placeholder(self) -> torch.Tensor:
        """Smallest input that still exercises every tower collective."""
        cfg = self.config.vision_config
        weight = self.vision_tower.patch_embed.proj.weight
        dev = weight.device
        dtype = weight.dtype if not isinstance(weight, DTensor) else weight.dtype
        merge = cfg.merge_kernel_size[0]
        side = merge  # one post-merge token
        patches = torch.zeros(
            side * side,
            cfg.in_channels,
            cfg.patch_size,
            cfg.patch_size,
            device=dev,
            dtype=dtype,
        )
        grid = torch.tensor([[1, side, side]], device=dev, dtype=torch.long)
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            patches = DTensor.from_local(
                patches, tp_mesh, (Replicate(),), run_check=False
            )
        feats = self.vision_tower(patches, grid)
        if isinstance(feats, torch.Tensor):
            return feats.to_local() if isinstance(feats, DTensor) else feats
        f0 = feats[0]
        return f0.to_local() if isinstance(f0, DTensor) else f0

    def _exchange_sentinel_counts(self, local: int) -> torch.Tensor:
        """Per-rank vision-sentinel counts across the CP group.

        Called unconditionally whenever CP is on, including on ranks with no
        images: the collective's participants are decided by the mesh, never by
        the batch.
        """
        group = self._cp_group
        counts = torch.zeros(
            torch.distributed.get_world_size(group),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        counts[torch.distributed.get_rank(group)] = local
        torch.distributed.all_reduce(counts, group=group)
        return counts

    def _select_cp_shard(
        self,
        features: list[torch.Tensor] | torch.Tensor,
        num_rows: int,
        counts: torch.Tensor | None,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Keep only the visual features belonging to this CP rank's shard.

        ``prepare_context_parallel_input`` shards inputs, labels and positions
        along the sequence but leaves ``pixel_values`` whole, so every CP rank
        encodes every image while holding only a slice of the sentinels. The
        features are ordered by sequence position and the shards are contiguous
        and equal -- the flavor pins ``context_parallel_load_balancer`` to None
        precisely because a permuting balancer would break that -- so this
        rank's slice starts after however many sentinels the lower ranks hold.

        This is correctness, not the report's sec 5.2.3 optimization: the
        encoder still runs redundantly on every CP rank. Dynamic CP would shard
        the encoder itself along the patch dimension and gather KV instead.
        """
        if counts is None:
            return features

        if int(counts.sum().item()) != num_rows:
            raise ValueError(
                f"CP ranks hold {int(counts.sum().item())} vision sentinel(s) "
                f"in total but {num_rows} visual token(s) were encoded; the "
                "sequence shard and the image batch disagree"
            )
        rank = torch.distributed.get_rank(self._cp_group)
        start = int(counts[:rank].sum().item())
        local = int(counts[rank].item())
        flat = (
            features
            if isinstance(features, torch.Tensor)
            else torch.cat(list(features), dim=0)
        )
        return flat[start : start + local]

    def _splice_per_token(
        self,
        input_ids: torch.Tensor,
        features: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Scatter visual features into pre-reserved sentinel positions.

        ``MMCollator`` reserves ONE sentinel per post-merge visual token, so the
        sequence length is already correct and the features drop straight in;
        this is the convention the release uses. :meth:`_splice` implements the
        other one -- a single sentinel per image, expanded in place -- which
        changes the sequence length and cannot be used with this collator.
        Which one applies is decided by counting, in :meth:`forward`.
        """
        sentinel = self.config.vision_token_id
        embed = self.language_model.embed_tokens
        safe_ids = torch.where(
            input_ids == sentinel, torch.zeros_like(input_ids), input_ids
        )
        text = embed(safe_ids)

        flat = (
            features
            if isinstance(features, torch.Tensor)
            else torch.cat(list(features), dim=0)
        )
        mask = (input_ids == sentinel).unsqueeze(-1).expand_as(text)
        return text.masked_scatter(mask, flat.to(text.dtype))

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
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """``[B, T]`` ids (+ packed patches) -> logits.

        ``**kwargs`` is ignored, mirroring KimiK3Model: torchtitan's Trainer
        and Validator inject ``attention_masks=None`` and ``positions=...`` for
        the FlexAttention / CP paths, and K3 uses plain SDPA plus KDA Triton
        kernels which take neither.

        Text-only when no images are supplied or no sentinel is present.

        The image parameters are named for the COLLATOR's output keys, not this
        model's internal vocabulary. torchtitan's trainer forwards a batch as
        ``model(inputs, **extra_kwargs)``, so a parameter spelled any other way
        is absorbed by ``**kwargs`` and the tower silently never runs -- which
        is exactly how a whole multimodal parallelism matrix once passed while
        validating nothing vision-side.
        """
        num_sentinels = int((input_ids == self.config.vision_token_id).sum().item())
        cp_active = self._cp_world_size() > 1

        # Under CP the per-rank sentinel counts have to be exchanged, and the
        # decision to exchange them must not depend on data. A batch carrying
        # no images at all is a normal occurrence, and letting that rank return
        # early leaves its CP peers waiting in the collective forever -- a
        # 100-second NCCL watchdog timeout, not an error. Gate on the mesh,
        # which every rank agrees on before looking at anything.
        cp_counts = self._exchange_sentinel_counts(num_sentinels) if cp_active else None

        if pixel_values is None:
            out = self.language_model(input_ids)
            if self._tower_needs_collectives():
                # FSDP2 issues the tower's all-gather from its pre-forward hook.
                # A rank that skips the tower on an image-free batch does not
                # issue it, and its peers wait in that collective until the
                # NCCL watchdog fires. Run the tower on a placeholder and keep
                # the graph edge with a zero-valued dependency, so every rank
                # issues the same collectives and the tower's contribution to
                # the data-parallel average is a correct zero.
                out = add_zero_valued_dependency(out, self._tower_placeholder())
            return out
        if num_sentinels == 0 and not cp_active:
            raise ValueError(
                "pixel_values supplied but input_ids contains no "
                f"vision_token_id ({self.config.vision_token_id}); the "
                "images would be silently dropped"
            )
        if grid_thw is None:
            raise ValueError("grid_thw is required alongside pixel_values")

        # Under CP a rank's sequence shard legitimately holds no sentinel at all
        # -- every image's tokens landed in another rank's half. It still has to
        # reach _select_cp_shard, whose all_reduce every CP rank participates in;
        # returning early here would hang the others. So encode and select
        # first, and only then decide whether there is anything to splice.
        features = self.encode_images(pixel_values, grid_thw)
        num_images = len(features)
        num_rows = (
            features.shape[0]
            if isinstance(features, torch.Tensor)
            else sum(int(f.shape[0]) for f in features)
        )
        features = self._select_cp_shard(features, num_rows, cp_counts)
        num_rows = (
            features.shape[0]
            if isinstance(features, torch.Tensor)
            else sum(int(f.shape[0]) for f in features)
        )
        if num_sentinels == 0:
            return self.language_model(input_ids)
        if num_sentinels == num_rows:
            # Collator convention: one sentinel per post-merge visual token.
            embeds = self._splice_per_token(input_ids, features)
        elif num_sentinels == num_images:
            # LLaVA convention: one sentinel per image, expanded in place.
            embeds = self._splice(input_ids, features)
        else:
            raise ValueError(
                f"{num_sentinels} vision sentinel(s) in input_ids match neither "
                f"the image count ({num_images}, one sentinel per image) nor the "
                f"visual-token count ({num_rows}, one sentinel per token)"
            )
        # The backbone's forward embeds int ids; we already embedded, so detach
        # embed_tokens to take its pre-embedded branch. Same mechanism as
        # KimiK3LlavaMultimodalModel._llm_forward_from_embeds.
        saved = self.language_model.embed_tokens
        try:
            self.language_model.embed_tokens = None
            return self.language_model(embeds)
        finally:
            self.language_model.embed_tokens = saved

    def init_weights(self, init_range: float | None = None, **kwargs) -> None:
        # Under PP the module is split into stages and the pieces a stage does
        # not own are set to None -- only the first stage keeps the tower, only
        # the last keeps lm_head. Guard both rather than assume a whole model.
        if self.vision_tower is not None:
            self.vision_tower.init_weights(init_range)
        if self.language_model is not None:
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

    KimiK3Model overrides this as a no-op because its internals are plain
    nn.Modules rather than Config-built ``Module`` instances -- it ports the HF
    reference layer by layer. The multimodal wrapper adds a MoonViT tower built
    the same way, so the same holds. The trainer calls this post-build; without
    it the multimodal flavor cannot be constructed at all.
    """
    return None


KimiK3MultimodalModel.verify_module_protocol = _mm_verify_module_protocol
for _name in ("get_attention_masks", "init_weights"):
    if not hasattr(KimiK3MultimodalModel, _name) and hasattr(KimiK3Model, _name):
        setattr(KimiK3MultimodalModel, _name, getattr(KimiK3Model, _name))


@dataclass(kw_only=True, slots=True)
class KimiK3MultimodalSpec(KimiK3Spec):
    """``BaseModel.Config``-compatible spec for the multimodal model.

    KimiK3Spec exists because torchtitan's trainer calls
    ``update_from_config`` and the property accessors on whatever sits at
    ``model_spec.model``; a bare dataclass config fails there. This subclasses it
    so the multimodal flavor gets the same integration surface, and overrides
    only ``build`` to construct the vision-bearing model.
    """

    vision_config: "MoonViTConfig" = None  # type: ignore[assignment]

    vision_token_id: int = -200
    """Sentinel id the splice scans for; must equal the tokenizer's image id.

    Defaulted to the LLaVA convention for the standalone/test path. A flavor
    driving a real collator has to override it: at a value the tokenizer never
    emits, the sentinel scan finds nothing and forward takes its text-only
    branch without complaint.
    """

    def build(self, **kwargs):
        return KimiK3MultimodalModel(
            KimiK3MultimodalConfig(
                kimi_config=self.kimi_config,
                vision_config=self.vision_config,
                num_blocks=self.num_blocks,
                vision_token_id=self.vision_token_id,
            )
        )
