# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals — native dense Transformer (no Llama3 subclass).

Standalone implementation of Block AttnRes (Kimi Team, 2026,
https://arxiv.org/abs/2603.15031) as a torchtitan ``Decoder`` variant.
Previously this experiment subclassed ``Llama3Model`` /
``Llama3TransformerBlock``; that coupled the AttnRes evolution to a
specific dense family and made it awkward to pivot to the MoE shape
Kimi's production models use. The refactor keeps AttnRes entirely
under ``torchtitan/experiments/attn_res/``, inheriting only from the
shared ``Decoder`` / ``TransformerBlock`` bases in
``torchtitan/models/common/decoder.py`` — no dependency on the Llama3
model classes.

The block follows paper Figure 2: each sub-layer (attention, MLP)
reads its input via ``block_attn_res`` over the current ``blocks``
plus ``partial_block``. At a block start ``partial_block`` is
committed to ``blocks`` and reset, so the new block accumulates only
the output of the sub-layers inside it.

The model:

- First / non-PP stage: ``blocks`` is ``None``, so we start with an
  empty list and ``partial_block = tok_embeddings(tokens)``.
- Middle / last PP stage: ``blocks`` arrives as a stacked
  ``[N, B, T, D]`` tensor from the previous stage; we unstack for
  per-layer access.
- Last stage (``self.output is not None``): apply a final
  cross-block aggregation, then ``norm`` and ``output`` to emit
  logits.
- Intermediate stage: return ``(partial_block, stacked_blocks)`` so
  ``PipelineStage`` P2P-sends both tensors.
- ``_return_only_new_blocks``: the cross-stage caching adapter
  toggles this to keep per-stage send size constant in ``stage_id``
  by shipping only this stage's committed blocks; the receiving
  adapter concatenates them with its cached prefix before handing
  the full list to the next stage's model.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.experiments.attn_res.attn_res import (
    AttnResConfig,
    AttnResProjection,
    block_attn_res,
    stack_blocks,
    unstack_blocks,
)
from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


class AttnResTransformerBlock(TransformerBlock):
    """Dense transformer block with Block AttnRes over its two sub-layers.

    Inherits from the shared ``TransformerBlock`` base (not Llama3-specific).
    Reuses the same attention / feed-forward / attention_norm / ffn_norm
    fields as any other dense block, so ``parallelize_llama`` and the
    torchtitan TP / FSDP / AC / compile passes apply unchanged (they
    duck-type on those four attribute names, not on the concrete class).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attn_res_proj: AttnResProjection.Config
        mlp_res_proj: AttnResProjection.Config
        attn_res_norm: RMSNorm.Config
        mlp_res_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None, (
            "AttnResTransformerBlock requires a dense feed_forward. "
            "MoE support will come in a follow-up experiment."
        )
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn_res_proj = config.attn_res_proj.build()
        self.mlp_res_proj = config.mlp_res_proj.build()
        self.attn_res_norm = config.attn_res_norm.build()
        self.mlp_res_norm = config.mlp_res_norm.build()

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        is_block_start: bool,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Per-layer AttnRes forward (paper Figure 2 pseudocode).

        Each sub-layer reads its input via ``block_attn_res`` over the
        current ``blocks`` plus ``partial_block``. At a block start the
        current ``partial_block`` is committed to ``blocks`` and reset,
        so the next block accumulates only the output of its own
        sub-layers.

        The AttnRes-native forward is the ONLY forward on this class:
        no fallback to a standard residual path. Dispatched via
        ``__call__`` from the model's layer loop so FSDP2's pre-forward
        ``all_gather`` hook fires and AttnRes sub-params unshard before
        ``rms_norm``'s mul.
        """
        # Pre-attention cross-block aggregation.
        h = block_attn_res(
            blocks, partial_block, self.attn_res_proj, self.attn_res_norm
        )

        # Block boundary: commit current partial and start a fresh block.
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None

        attn_out = self.attention(
            self.attention_norm(h), freqs_cis, attention_masks, positions
        )
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # Pre-MLP cross-block aggregation.
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        mlp_out = self.feed_forward(self.ffn_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class AttnResModel(Decoder):
    """Dense decoder-only LM with Block AttnRes threaded through its layers.

    Inherits from the shared ``Decoder`` base (not ``Llama3Model``).
    Overrides ``forward`` to thread ``blocks`` and ``partial_block``
    through the layer stack, applies a final cross-block aggregation
    before ``norm`` + ``output`` on the last stage, and returns
    ``(partial_block, stacked_blocks)`` on intermediate PP stages.

    Config fields are deliberately a superset of ``Decoder.Config`` plus
    the AttnRes wiring; we add ``enable_weight_tying`` here (not on the
    base) so a future MoE variant can introduce its own tying policy
    without affecting this class.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Weight-tying between tok_embeddings and output (LM head). Must be
        # False under Pipeline Parallel — ``parallelize_llama`` raises on
        # tying + PP, and the fix is out of scope for this experiment.
        enable_weight_tying: bool = False
        # AttnRes wiring (all three required; no standard-residual fallback).
        attn_res: AttnResConfig
        final_attn_res_proj: AttnResProjection.Config
        final_attn_res_norm: RMSNorm.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_kv_heads ({n_kv_heads})."
                    )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # Reuse the dense helper; AttnRes adds ~0.05 % params via the
            # pseudo-queries, negligible in the flop estimate.
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        assert (
            config.attn_res.enabled
        ), "AttnResModel requires attn_res.enabled=True in its Config"

        num_blocks = config.attn_res.num_blocks
        num_layers_total = len(config.layers)
        assert num_layers_total % num_blocks == 0, (
            f"num_layers ({num_layers_total}) must be divisible by "
            f"num_blocks ({num_blocks})"
        )
        self._layers_per_block = num_layers_total // num_blocks

        self.final_attn_res_proj = config.final_attn_res_proj.build()
        self.final_attn_res_norm = config.final_attn_res_norm.build()

        # When the cross-stage caching adapter is active it flips this to
        # True; the non-last-stage return then only contains the blocks
        # THIS stage committed (not the cached prefix from earlier stages).
        # Keeps per-stage P2P send size constant in stage id.
        self._return_only_new_blocks: bool = False

        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            # Tie input embedding with output projection.
            self.tok_embeddings.weight = self.output.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie weights before parameter init so tok_embeddings.weight
            # (skipped by skip_param_init) and output.weight point to the
            # same tensor after output is initialized.
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight
        super().init_states(buffer_device=buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        blocks: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """AttnRes forward. Threads block representations through layers.

        First / non-PP stage: ``blocks`` is None; start with an empty
        list and ``partial_block = h`` (the token embedding).

        Middle / last PP stage: ``blocks`` is a ``[N, B, T, D]`` stacked
        tensor received from the previous stage; unstack into a list
        for per-layer access.

        Last stage (``self.output is not None``): apply a final
        cross-block aggregation, then ``norm`` and ``output``.
        Non-last stages return ``(partial_block, stacked_blocks)`` so
        ``PipelineStage`` can P2P-send both tensors.
        """
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        if blocks is None:
            block_list: list[torch.Tensor] = []
        else:
            block_list = unstack_blocks(blocks)
        # Snapshot how many blocks arrived from earlier stages so that,
        # when _return_only_new_blocks is set, we can slice off only this
        # stage's contributions for the send.
        initial_num_blocks = len(block_list)
        partial_block = h

        for layer_key, layer in self.layers.items():
            layer_id = int(layer_key)
            is_block_start = layer_id % self._layers_per_block == 0
            # Route through layer.__call__ so FSDP's pre-forward all_gather
            # hook fires on this block unit and AttnRes sub-params are
            # unsharded for the mul in rms_norm.
            block_list, partial_block = layer(
                block_list,
                partial_block,
                is_block_start,
                self.freqs_cis,
                attention_masks,
                positions,
            )

        is_last_stage = self.output is not None
        if not is_last_stage:
            # PP intermediate stage: return partial + stacked blocks as a
            # tuple so PipelineStage can send both tensors via P2P.
            #
            # With _return_only_new_blocks True (set by the cross-stage
            # caching adapter), only blocks committed by THIS stage are
            # sent; the receiving adapter concats them with its cached
            # prefix before handing the next stage's model the full list.
            # This keeps the forward send size constant in stage id.
            if self._return_only_new_blocks:
                new_blocks = block_list[initial_num_blocks:]
                if not new_blocks:
                    # Stage spans no block boundary this pass: emit a
                    # zero-first-dim stacked tensor so the adapter's P2P
                    # hand-off keeps a static per-stage shape (receiver
                    # sees N=0 and skips its cache add). Happens when
                    # num_virtual_stages > num_blocks -- e.g. PP=8 with
                    # VP=2 (16 virtual stages) and num_blocks=8, where
                    # odd virtual stages span no is_block_start layer.
                    empty_blocks = partial_block.new_zeros((0, *partial_block.shape))
                    return partial_block, empty_blocks
                return partial_block, stack_blocks(new_blocks)
            return partial_block, stack_blocks(block_list)

        # Last stage / non-PP: final cross-block aggregation, then norm+output.
        h = block_attn_res(
            block_list,
            partial_block,
            self.final_attn_res_proj,
            self.final_attn_res_norm,
        )
        h = self.norm(h) if self.norm is not None else h
        return self.output(h)
