# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals Llama3 variant.

Subclasses ``Llama3TransformerBlock`` and ``Llama3Model`` to add the Block
AttnRes forward path (Kimi Team, 2026). Core torchtitan model files are
untouched so this experiment can be enabled or removed independently.

Per-block layer subclass threads ``(blocks, partial_block, is_block_start)``
through ``forward`` so FSDP's pre-forward all_gather hook fires on the block
unit and AttnRes sub-params unshard before rms_norm. The model subclass
overrides ``forward`` to always take the AttnRes path and returns
``(partial_block, stacked_blocks)`` at PP intermediate stages.
"""

from dataclasses import dataclass

import torch

from torchtitan.experiments.attn_res.attn_res import (
    AttnResConfig,
    AttnResProjection,
    block_attn_res,
    stack_blocks,
    unstack_blocks,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


class AttnResLlama3TransformerBlock(Llama3TransformerBlock):
    """Llama3 transformer block that reads sub-layer inputs via Block AttnRes.

    Adds per-layer pseudo-query projections and key RMSNorms. Falls back to
    the parent's standard forward when called without AttnRes kwargs, which
    keeps unit tests of the base forward path unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        attn_res_proj: AttnResProjection.Config
        mlp_res_proj: AttnResProjection.Config
        attn_res_norm: RMSNorm.Config
        mlp_res_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_res_proj = config.attn_res_proj.build()
        self.mlp_res_proj = config.mlp_res_proj.build()
        self.attn_res_norm = config.attn_res_norm.build()
        self.mlp_res_norm = config.mlp_res_norm.build()

    def forward(
        self,
        x: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        blocks: list[torch.Tensor] | None = None,
        partial_block: torch.Tensor | None = None,
        is_block_start: bool | None = None,
    ):
        # AttnRes path must be dispatched via __call__ so FSDP's pre-forward
        # all_gather hook fires and AttnRes norm/proj weights unshard.
        if is_block_start is not None:
            assert blocks is not None and partial_block is not None
            return self.forward_attn_res(
                blocks,
                partial_block,
                is_block_start,
                freqs_cis,
                attention_masks,
                positions,
            )
        return super().forward(x, freqs_cis, attention_masks, positions)

    def forward_attn_res(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        is_block_start: bool,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Block AttnRes per-layer forward. See paper Figure 2 pseudocode.

        Each sub-layer (attention, MLP) reads its input via ``block_attn_res``
        over the current ``blocks`` plus ``partial_block``. At a block start,
        the current ``partial_block`` is committed to ``blocks`` and reset, so
        the new block accumulates only the output of subsequent sub-layers.
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


class AttnResLlama3Model(Llama3Model):
    """Llama3 dense model that threads Block AttnRes through its layers.

    Replaces ``Llama3Model.forward`` with the AttnRes path (always on). Adds
    a final cross-block aggregation before ``norm`` and ``output`` on the
    last pipeline stage; returns ``(partial_block, stacked_blocks)`` on
    intermediate stages so PipelineStage can P2P-send both tensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        attn_res: AttnResConfig
        final_attn_res_proj: AttnResProjection.Config
        final_attn_res_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        assert config.attn_res.enabled, (
            "AttnResLlama3Model requires attn_res.enabled=True in its Config"
        )
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

    def forward(
        self,
        tokens: torch.Tensor,
        blocks: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """AttnRes forward path. Threads block representations through layers.

        First / non-PP stage: ``blocks`` is None; start with an empty list and
        ``partial_block = h`` (the token embedding).

        Middle / last PP stage: ``blocks`` is a [N, B, T, D] stacked tensor
        received from the previous stage; unstack into a list for per-layer
        access.

        The last stage (identified by ``self.output is not None``) applies a
        final cross-block aggregation, then ``norm`` and ``output``. Non-last
        stages return ``(partial_block, stacked_blocks)`` so PipelineStage can
        send both tensors to the next stage.
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
                None,
                self.freqs_cis,
                attention_masks,
                positions,
                blocks=block_list,
                partial_block=partial_block,
                is_block_start=is_block_start,
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
                assert new_blocks, (
                    "Stage committed no new blocks but _return_only_new_blocks "
                    "is on. Check that num_blocks >= num_stages and each "
                    "stage spans at least one block boundary."
                )
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
