# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AttnRes-woven Kimi Linear model.

``KimiLinearAttnResModel`` subclasses :class:`KimiLinearModel` and threads
Block Attention Residuals through the decoder layer stack, reusing the
paper's Figure 2 aggregation primitive
:func:`torchtitan.experiments.attention_residual.attn_res.block_attn_res`
(shared with the existing ``AttnResLlama3Model`` / ``AttnResModel`` so
that a single implementation is used across both experiments).

Per-layer AttnRes (matching ``AttnResTransformerBlock`` in the
``attn_res/`` experiment):

* Two AttnRes applications per decoder layer — one before attention,
  one before the FFN — each contributing an RMSNorm + zero-initialized
  pseudo-query ``w_l ∈ R^d``.
* At a block-start layer the running ``partial_block`` is committed
  into ``blocks`` and reset; subsequent layers accumulate into the new
  ``partial_block`` until the next block start.
* On the last stage the final aggregation (one extra pseudo-query +
  RMSNorm) runs before ``norm`` + ``lm_head``, mirroring the reference.

``_return_only_new_blocks`` flag is respected on forward so the
Phase-3 PP cache adapter can drive this model unchanged at multi-node
scale. Local FSDP-only training leaves the flag ``False`` and passes
the full accumulated block stack forward.

Paper (Kimi Linear tech report §5):

> "AttnRes introduces only one RMSNorm and one pseudo-query vector
> wl ∈ R^d per layer, amounting to a negligible fraction of the total
> parameter count. Crucially, all pseudo-query vectors must be
> initialized to zero."

We use the two-per-layer pattern from the ``attn_res/`` experiment to
stay consistent with the validated PP adapter + CPU tests. The
paper's "one per layer" count treats each (attention, FFN) pair as a
sub-layer pair; the two-per-layer count here is the sub-layer-level
view and is equivalent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchtitan.experiments.attention_residual.attn_res import (
    AttnResProjection,
    block_attn_res,
    stack_blocks,
    unstack_blocks,
)
from torchtitan.experiments.attention_residual.kimi_linear.model import (
    KimiDecoderLayer,
    KimiLinearConfig,
    KimiLinearModel,
    Linear,
)


# ----- Per-layer AttnRes wrapper ------------------------------------------ #

class KimiAttnResDecoderLayer(nn.Module):
    """Kimi decoder layer with AttnRes woven around attn and FFN.

    Structurally the same as :class:`KimiDecoderLayer` (per-layer KDA/MLA
    choice + MoE/MLP choice) but the forward is driven by the model's
    block-threading loop: takes ``(blocks, partial_block, is_block_start)``
    and returns the updated ``(blocks, partial_block)``.

    Four extra AttnRes params (per layer):
      * ``attn_res_proj`` — pseudo-query for pre-attention aggregation
      * ``attn_res_norm`` — RMSNorm for keys in that aggregation
      * ``mlp_res_proj``  — pseudo-query for pre-FFN aggregation
      * ``mlp_res_norm``  — RMSNorm for keys in that aggregation

    ``_*_proj`` are Linear(d, 1, bias=False). Their weight vector IS
    the per-layer pseudo-query ``w_l``. :meth:`init_weights` zero-inits
    these (paper mandates it: uniform initial attention weights → at
    t=0 training is equivalent to standard residuals).
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        # Reuse the base KimiDecoderLayer entirely — we just delegate
        # to its sub-modules rather than calling its forward.
        base = KimiDecoderLayer(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = base.self_attn
        self.ffn = base.ffn
        self.input_layernorm = base.input_layernorm
        self.post_attention_layernorm = base.post_attention_layernorm
        self.is_linear_attn = base.is_linear_attn
        self.is_moe = base.is_moe

        d = config.hidden_size
        # AttnRes params: two pseudo-queries + two RMSNorms per layer.
        # ``AttnResProjection`` is the shared Linear(d, 1, bias=False)
        # wrapper from attn_res/; its weight [1, d] is the pseudo-query
        # vector ``w_l``. Zero-init happens in ``init_weights`` below.
        proj_cfg = AttnResProjection.Config(dim=d)
        self.attn_res_proj = AttnResProjection(proj_cfg)
        self.mlp_res_proj = AttnResProjection(proj_cfg)
        self.attn_res_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.mlp_res_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        is_block_start: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Pre-attention aggregation (paper Figure 2, pre-attention step).
        h = block_attn_res(
            blocks, partial_block, self.attn_res_proj, self.attn_res_norm
        )

        # Block boundary: commit partial into blocks, start fresh accumulator.
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None

        # Attention sub-layer (KDA or MLA).
        attn_out = self.self_attn(self.input_layernorm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # Pre-FFN aggregation (paper Figure 2, pre-FFN step).
        h = block_attn_res(
            blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm
        )

        # FFN sub-layer (MoE or dense SwiGLU).
        ffn_out = self.ffn(self.post_attention_layernorm(h))
        partial_block = partial_block + ffn_out
        return blocks, partial_block


# ----- Top-level AttnRes-woven model -------------------------------------- #

class KimiLinearAttnResModel(KimiLinearModel):
    """Kimi Linear with Block Attention Residuals threaded through layers.

    Backbone identical to :class:`KimiLinearModel` (KDA/MLA alternation,
    MoE/MLP FFN per layer). AttnRes weaving adds:

      * per-layer :class:`KimiAttnResDecoderLayer` in place of
        :class:`KimiDecoderLayer`
      * one final aggregation (``final_attn_res_proj`` + norm) before
        ``norm`` + ``lm_head`` on the last stage
      * ``layers_per_block`` attribute so block-start detection is
        layout-table-compatible with the Phase-3 PP cache adapter.

    ``num_blocks`` chooses between Full AttnRes (``num_blocks == L``,
    1 layer per block → every layer is block-start) and Block AttnRes
    (``num_blocks < L``, multiple layers per block → only every k-th
    layer commits a block).

    Forward signature changes vs base:

      * First / non-PP stage: ``forward(input_ids)`` — blocks start empty,
        ``partial_block = tok_embeddings(tokens)``.
      * Middle / last PP stage: ``forward(partial_in, blocks_in)`` —
        threads (partial, blocks) through the layer stack. PP adapter
        (:mod:`torchtitan.experiments.attention_residual.pipeline_adapter`) handles
        the rebuild / delta.

    FSDP-only training (no PP) keeps ``_return_only_new_blocks=False``,
    layers receive the full accumulated block list every layer.
    """

    def __init__(self, config: KimiLinearConfig, *, num_blocks: int) -> None:
        # Skip KimiLinearModel.__init__'s layer build (it builds
        # KimiDecoderLayer); we need KimiAttnResDecoderLayer instead.
        # Call nn.Module's init, then build what we need ourselves.
        nn.Module.__init__(self)
        self.config = config

        n_layers = config.num_hidden_layers
        assert n_layers > 0
        assert 1 <= num_blocks <= n_layers, (
            f"num_blocks={num_blocks} out of range [1, {n_layers}]"
        )
        assert n_layers % num_blocks == 0, (
            f"n_layers={n_layers} must be divisible by num_blocks={num_blocks}; "
            "Block AttnRes requires a uniform block layout"
        )
        self.num_blocks = num_blocks
        self.layers_per_block = n_layers // num_blocks

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # ModuleDict for pipeline_module_split compatibility — see
        # KimiLinearModel.__init__ for the same pattern.
        self.layers = nn.ModuleDict(
            {str(i): KimiAttnResDecoderLayer(config, i) for i in range(n_layers)}
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Final AttnRes aggregation (one extra pseudo-query + RMSNorm
        # before lm_head). Same ``AttnResProjection`` shared with the
        # attn_res/ experiment.
        self.final_attn_res_proj = AttnResProjection(
            AttnResProjection.Config(dim=config.hidden_size)
        )
        self.final_attn_res_norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # PP cache adapter hook — FSDP-only training leaves this False.
        self._return_only_new_blocks: bool = False

    # Default sentinel token id used to mark image-token positions in input_ids
    # when ``image_mask`` is not supplied alongside ``vision_embeds``. Phase 5
    # multimodal pretraining picks 32000 (a Llama-3.1 reserved special token);
    # any caller can override by passing ``image_token_id`` as a kwarg.
    _DEFAULT_IMAGE_TOKEN_ID = 32_000

    def forward(
        self,
        tokens: torch.Tensor,
        blocks: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        image_token_id: int | None = None,
        **kwargs,
    ):
        """AttnRes forward with PP-split awareness + block threading.

        The dispatch mirrors ``attn_res/model.py:AttnResModel.forward`` so
        the Phase-3 ``CrossStageCacheAdapter`` can drive this class via
        duck-typing on ``self.embed_tokens`` / ``self.lm_head`` /
        ``self.norm`` presence (pipeline_module_split strips these off
        non-first / non-last stages).

        Args:
            tokens: On stage 0 / non-PP: ``[B, T]`` int64 token ids. On
                PP middle / last stages: ``[B, T, D]`` hidden state from
                upstream stage's ``partial_block``.
            blocks: ``[N, B, T, D]`` stacked AttnRes blocks from upstream
                PP stage. ``None`` on stage 0 / non-PP.

        Returns:
            * Non-last PP stage: ``(partial_block, stacked_blocks)`` —
              PipelineStage sends both over P2P.
            * Last stage / single-GPU: ``[B, T, vocab_size]`` logits.

        The PP cache adapter toggles ``_return_only_new_blocks`` so
        non-last middle stages emit only THIS stage's new block
        commits rather than the full accumulated stack (constant per-hop
        bytes regardless of depth).
        """
        # 1) Initial hidden: pre-computed embeds (multimodal), embed on stage 0,
        #    pass-through on middle/last PP stages.
        if inputs_embeds is not None:
            h = inputs_embeds
        elif self.embed_tokens is not None:
            h = self.embed_tokens(tokens)
            # Multimodal scatter: replace embed positions for image tokens
            # with externally-supplied vision_embeds. Done INSIDE this
            # forward so FSDP sees a single root call. Under PP, only stage 0
            # has ``embed_tokens``, so this branch fires there exclusively.
            # ``image_mask`` is recomputed from ``tokens`` when not supplied
            # so callers don't have to plumb a bool mask through PP P2P
            # (which would chunk it as a separate kwarg without semantic
            # benefit — the mask is a deterministic function of input_ids).
            #
            # Implementation note: ``masked_scatter`` is used instead of
            # ``h[image_mask] = vision_embeds.reshape(-1, D)`` so the
            # operation is safe under PP shape inference, where the
            # scheduler runs forward once with zero-filled token tensors
            # to determine activation shapes — image_mask is then all
            # False and advanced-indexing assignment would crash with
            # "shape mismatch". masked_scatter copies as many elements
            # as the mask requires (zero in shape-inference, B*N_vision
            # in regular forward) and is autograd-friendly so the
            # downstream PP backward path still reaches vision_embeds.
            if vision_embeds is not None:
                if image_mask is None:
                    sentinel = (
                        image_token_id
                        if image_token_id is not None
                        else self._DEFAULT_IMAGE_TOKEN_ID
                    )
                    image_mask = (tokens == sentinel)
                # Variable image count per row support (B1): the original
                # ``vision_embeds.reshape(-1, D)`` assumes every row has
                # exactly ``vision_embeds.size(1)`` image tokens — which
                # holds for LLaVA-Pretrain (1 img × 196 tokens) but breaks
                # the moment data has zero-image rows (text-only mixed in)
                # or multi-image rows. Filter ``vision_embeds`` to the
                # leading ``n_image_per_row[i]`` slots of each row before
                # flattening so ``masked_scatter`` consumes the correct
                # row-major sequence of embeds.
                #
                # Under PP shape inference (zero-filled tokens, image_mask
                # all False), n_image_per_row=0 → valid all False →
                # source has shape (0, D), masked_scatter is a no-op.
                # Under uniform-image data (all rows == n_vis_max), valid
                # is all True → source equals the original reshape, so
                # this is a strict superset of the previous behavior.
                n_per_row = image_mask.sum(dim=1)
                n_vis_max = vision_embeds.size(1)
                arange = torch.arange(n_vis_max, device=image_mask.device)
                valid = arange.unsqueeze(0) < n_per_row.unsqueeze(1)
                source = vision_embeds[valid].to(h.dtype)
                # Robustness fix (seq-KD SFT crash, 2026-05): the number of
                # scatter DESTINATIONS (True positions in ``image_mask``) must
                # exactly equal the number of SOURCE embeds, or
                # ``Tensor.masked_scatter`` trips the CUDA device-side assert
                # ``masked_scatter_size_check: totalElements <= srcSize``
                # (IndexKernel.cu:400). That assert surfaces ASYNC at a much
                # later kernel (a KDA/MLA/FFN linear → CUBLAS_STATUS_EXECUTION_FAILED
                # or an FSDP all-gather), making it look like an MoE/attention
                # bug when it is really an embed-scatter count mismatch.
                #
                # The source is clamped to at most ``n_vis_max`` embeds per row
                # (``valid`` above), but ``image_mask`` can have MORE than
                # ``n_vis_max`` True positions in a row when the *text* tokens
                # happen to contain the image-sentinel id (e.g. distilled SFT
                # answers tokenize to Llama-3.1 id 32000 == ``IMAGE_TOKEN_ID``,
                # the reserved token reused as ``<image>`` — it decodes to the
                # ordinary subword 'utility', so ~0.03% of teacher-rewritten
                # rows contain it). Cap the destinations to the leading
                # ``n_vis_max`` per row so destinations == source. This is a
                # no-op for well-formed rows (``n_per_row <= n_vis_max``); for
                # over-count rows the surplus sentinel positions simply keep
                # their text-embedding (correct: they are real text tokens that
                # collided with the sentinel id, not image slots).
                n_keep_per_row = torch.clamp(n_per_row, max=n_vis_max)
                if bool((n_per_row > n_vis_max).any()):
                    pos_rank = (
                        image_mask.long().cumsum(dim=1) - 1
                    )  # 0-based rank of each True within its row
                    scatter_mask = image_mask & (
                        pos_rank < n_keep_per_row.unsqueeze(1)
                    )
                else:
                    scatter_mask = image_mask
                h = h.masked_scatter(
                    scatter_mask.unsqueeze(-1).expand_as(h), source
                )
        else:
            h = tokens

        # 2) Unstack incoming blocks; empty list on stage 0 / non-PP.
        if blocks is None:
            block_list: list[torch.Tensor] = []
        else:
            block_list = unstack_blocks(blocks)
        initial_num_blocks = len(block_list)
        partial_block = h

        # 3) Thread blocks + partial through this stage's layer slice.
        # ModuleDict keys are original layer indices (preserved across
        # pipeline_module_split); int() them to drive block-start detection.
        for layer_key, layer in self.layers.items():
            layer_idx = int(layer_key)
            is_block_start = (layer_idx % self.layers_per_block == 0)
            block_list, partial_block = layer(
                block_list, partial_block, is_block_start
            )

        is_last_stage = self.lm_head is not None

        if not is_last_stage:
            # PP middle stage: ship (partial_block, stacked_blocks) downstream.
            if self._return_only_new_blocks:
                new_blocks = block_list[initial_num_blocks:]
                if not new_blocks:
                    # This stage span covers no block boundary — emit a
                    # zero-first-dim tensor so the adapter's P2P handoff
                    # preserves a static per-stage shape.
                    empty = partial_block.new_zeros((0, *partial_block.shape))
                    return partial_block, empty
                return partial_block, stack_blocks(new_blocks)
            return partial_block, stack_blocks(block_list)

        # Last stage / single-GPU: final aggregation + norm + lm_head.
        h_final = block_attn_res(
            block_list,
            partial_block,
            self.final_attn_res_proj,
            self.final_attn_res_norm,
        )
        if self.norm is not None:
            h_final = self.norm(h_final)
        return self.lm_head(h_final)

    def init_weights(
        self, init_range: float | None = None, **kwargs,
    ) -> None:
        """Normal init + mandatory zero-init of every pseudo-query.

        Paper §5 requires ``w_l`` zero-init so initial softmax weights
        are uniform (equivalent to standard residuals at t=0, avoids
        training volatility).

        ``**kwargs`` forwards trainer-supplied args (e.g. ``buffer_device``)
        to :meth:`KimiLinearModel.init_weights`.
        """
        super().init_weights(init_range, **kwargs)
        # Zero-init every AttnRes pseudo-query (paper requirement).
        # Guard against PP-split stages that dropped some modules
        # (pipeline_module_split replaces non-owned modules with None
        # or Identity).
        for layer in self.layers.values():
            if hasattr(layer, "attn_res_proj") and layer.attn_res_proj is not None:
                nn.init.zeros_(layer.attn_res_proj.weight)
            if hasattr(layer, "mlp_res_proj") and layer.mlp_res_proj is not None:
                nn.init.zeros_(layer.mlp_res_proj.weight)
        if self.final_attn_res_proj is not None:
            nn.init.zeros_(self.final_attn_res_proj.weight)
