# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AttnRes-woven Kimi Linear model.

``KimiK3AttnResModel`` subclasses :class:`KimiK3Model` and threads
Block Attention Residuals through the decoder layer stack, reusing the
paper's Figure 2 aggregation primitive
:func:`torchtitan.models.kimi_k3.attn_res.block_attn_res`
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
cross-stage cache adapter (``pipeline_adapter.py``) can drive this model
unchanged at multi-node scale. Local FSDP-only training leaves the flag ``False`` and passes
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
from torch.distributed.tensor import DTensor

from torchtitan.models.kimi_k3.attn_res import (
    AttnResProjection,
    block_attn_res,
    stack_blocks,
    unstack_blocks,
)
from torchtitan.models.kimi_k3.model import (
    Embedding,
    RMSNorm,
    _tp_replicate,
    _tp_shard,
    KimiDecoderLayer,
    KimiK3Config,
    KimiK3Model,
    Linear,
    splice_vision_embeds,
    UpstreamFSDPNames,
)


def _scalar_local(a: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Under TP the graft alphas are NoParallel DTensors (Replicate); the
    plain block stream is a plain Tensor, so ``alpha * (h - plain)`` mixes
    DTensor and Tensor. The alpha is a replicated scalar -- to_local gives
    the identical value on every rank and keeps the mul plain.

    Also cast to ``like``'s dtype: frozen-base LoRA keeps the trainable
    alpha as an fp32 master while the stream is bf16; without the cast
    the elementwise mix silently promotes the residual stream to fp32
    (matches FSDP mixed-precision compute when the cast is a no-op)."""
    a = a.to_local() if isinstance(a, DTensor) else a
    return a.to(like.dtype) if a.dtype != like.dtype else a


def _plain_stream(
    blocks: list[torch.Tensor], partial_block: torch.Tensor
) -> torch.Tensor:
    """Reconstruct the standard residual stream: sum of committed blocks
    plus the current partial. This is the exact input the plain
    (non-AttnRes) backbone would see at this point."""
    out = partial_block
    for b in blocks:
        out = out + b
    return out


# ----- Per-layer AttnRes wrapper ------------------------------------------ #


class KimiAttnResDecoderLayer(nn.Module, UpstreamFSDPNames):
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

    def __init__(
        self,
        config: KimiK3Config,
        layer_idx: int,
        gated: bool = False,
    ) -> None:
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
        # NoParallel in the imperative plan -- the output dim is 1, so there
        # is nothing to shard; declared here so the module carries its own
        # placement like every other linear after the migration.
        proj_cfg = AttnResProjection.Config(dim=d, sharding_config=_tp_replicate())
        self.attn_res_proj = AttnResProjection(proj_cfg)
        self.mlp_res_proj = AttnResProjection(proj_cfg)
        self.attn_res_norm = RMSNorm(d, eps=config.rms_norm_eps, sharding_config=_tp_replicate())
        self.mlp_res_norm = RMSNorm(d, eps=config.rms_norm_eps, sharding_config=_tp_replicate())
        # Graft gate: per-read scalar alpha, zero-init, so at step 0 the
        # model is exactly the plain backbone (adapter-correctness anchor).
        # h = partial + alpha * (mix - partial): alpha=0 makes the read the
        # plain residual stream, so a pretrained backbone's step-0 function
        # is EXACTLY preserved; alpha then trains away from identity.
        # Ungated (from-scratch pretraining) keeps the paper's uniform-mix
        # zero-init read, matching all historical numerics evidence.
        self.attn_res_gated = gated
        if gated:
            self.attn_res_alpha = nn.Parameter(torch.zeros(1))
            self.mlp_res_alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        is_block_start: bool,
        plain_stream: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
        # Pre-attention aggregation (paper Figure 2, pre-attention step).
        h = block_attn_res(
            blocks, partial_block, self.attn_res_proj, self.attn_res_norm
        )
        if self.attn_res_gated:
            # plain_stream is accumulated SEQUENTIALLY (same op order as
            # the plain backbone) so alpha=0 is bit-identical to it;
            # reconstructing sum(blocks)+partial would reorder additions.
            assert plain_stream is not None
            h = plain_stream + _scalar_local(self.attn_res_alpha, plain_stream) * (
                h - plain_stream
            )

        # Block boundary: commit partial into blocks, start fresh accumulator.
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None

        # Attention sub-layer (KDA or MLA).
        attn_out = self.self_attn(self.input_layernorm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        if self.attn_res_gated:
            plain_stream = plain_stream + attn_out

        # Pre-FFN aggregation (paper Figure 2, pre-FFN step).
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        if self.attn_res_gated:
            h = plain_stream + _scalar_local(self.mlp_res_alpha, plain_stream) * (
                h - plain_stream
            )

        # FFN sub-layer (MoE or dense SwiGLU).
        ffn_out = self.ffn(self.post_attention_layernorm(h))
        partial_block = partial_block + ffn_out
        if self.attn_res_gated:
            plain_stream = plain_stream + ffn_out
        return blocks, partial_block, plain_stream


# ----- Top-level AttnRes-woven model -------------------------------------- #


class _DenseGrad(torch.autograd.Function):
    """Identity forward; makes the gradient dense on the way back.

    torch's pipeline P2P rejects a non-dense tensor
    ("Tensors for P2P must be non-overlapping and dense"), and what it ships
    backwards is the raw ``grad_input`` autograd produced for a stage's PP
    inputs. Our last stage aggregates the block stack together with the
    partial block, so the gradient w.r.t. the partial block comes back as a
    slice of that wider buffer -- dense-looking shape, strided layout.

    Making the stage OUTPUTS contiguous does not help: the buffer in question
    belongs to the inputs. This barrier sits on the inputs instead, so
    whatever layout autograd picks, what crosses the wire is dense.

    Found by pp8 over 13 layers, where the grad arrived as [1, 256, 256] with
    stride [256, 768, 1] (768 = 3 x 256: two blocks plus the partial). pp2 and
    pp4 never produced a strided grad there, which is why this survived to
    degree 8.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad.contiguous()


def _dense_grad(x: torch.Tensor | None) -> torch.Tensor | None:
    """Apply :class:`_DenseGrad` where autograd can actually carry a gradient."""
    if x is None or not x.is_floating_point() or not x.requires_grad:
        return x
    return _DenseGrad.apply(x)


class KimiK3MTPLayer(nn.Module):
    """One multi-token-prediction layer, mirroring a backbone block.

    Report sec 3.3: "Kimi K3 is pre-trained with a multi-token-prediction (MTP)
    layer that mirrors the structure of a backbone block", and Table 1 lists
    one. The released config.json ships ``num_nextn_predict_layers: 0``, so the
    published artifact was exported without it -- which is why this is built
    only when the field is set, and why the default is 0.

    Structure follows the MTP formulation this family uses: the depth-k input
    fuses the backbone's final hidden state with the embedding of the token k
    positions ahead, each RMSNormed, concatenated and projected back to the
    model width, then run through a block with the same structure as a
    backbone layer. Embedding and output head are shared with the backbone, as
    the released weight contract expects.
    """

    def __init__(self, config, layer_idx: int, *, gated: bool) -> None:
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps, sharding_config=_tp_replicate())
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps, sharding_config=_tp_replicate())
        self.eh_proj = Linear(2 * d, d, bias=False, sharding_config=_tp_shard(0))
        self.gated = gated
        self.block = KimiAttnResDecoderLayer(config, layer_idx, gated=gated)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        fused = self.eh_proj(torch.cat([self.hnorm(h), self.enorm(emb)], dim=-1))
        # An MTP layer has no incoming block stack: it mirrors one block's
        # structure, not the AttnRes depth-mixing across the backbone.
        _, out, _ = self.block([], fused, True, fused if self.gated else None)
        return out


class KimiK3AttnResModel(KimiK3Model):
    """Kimi Linear with Block Attention Residuals threaded through layers.

    Backbone identical to :class:`KimiK3Model` (KDA/MLA alternation,
    MoE/MLP FFN per layer). AttnRes weaving adds:

      * per-layer :class:`KimiAttnResDecoderLayer` in place of
        :class:`KimiDecoderLayer`
      * one final aggregation (``final_attn_res_proj`` + norm) before
        ``norm`` + ``lm_head`` on the last stage
      * ``layers_per_block`` attribute so block-start detection is
        layout-table-compatible with the cross-stage cache adapter.

    ``num_blocks`` chooses between Full AttnRes (``num_blocks == L``,
    1 layer per block → every layer is block-start) and Block AttnRes
    (``num_blocks < L``, multiple layers per block → only every k-th
    layer commits a block).

    Forward signature changes vs base:

      * First / non-PP stage: ``forward(input_ids)`` — blocks start empty,
        ``partial_block = tok_embeddings(tokens)``.
      * Middle / last PP stage: ``forward(partial_in, blocks_in)`` —
        threads (partial, blocks) through the layer stack. PP adapter
        (:mod:`torchtitan.models.kimi_k3.pipeline_adapter`) handles
        the rebuild / delta.

    FSDP-only training (no PP) keeps ``_return_only_new_blocks=False``,
    layers receive the full accumulated block list every layer.
    """

    def __init__(
        self,
        config: KimiK3Config,
        *,
        num_blocks: int,
        layers_per_block: int | None = None,
        gated: bool = False,
    ) -> None:
        # Skip KimiK3Model.__init__'s layer build (it builds
        # KimiDecoderLayer); we need KimiAttnResDecoderLayer instead.
        # Call nn.Module's init, then build what we need ourselves.
        nn.Module.__init__(self)
        self.config = config

        n_layers = config.num_hidden_layers
        assert n_layers > 0
        assert (
            1 <= num_blocks <= n_layers
        ), f"num_blocks={num_blocks} out of range [1, {n_layers}]"
        # K3 partitions by BLOCK SIZE, not by an equal split: the official
        # config ships attn_res_block_size=12 over 93 layers, i.e. 7 full
        # blocks plus a 9-layer partial tail (report sec 2.2: "we partition
        # its layers into 8 blocks with 12-layer size, giving a partial final
        # block"). The last block is allowed to be short; the commit rule
        # (layer_idx % layers_per_block) simply never fires inside the partial
        # tail, matching the reference (its remainder layer does not commit).
        #
        # layers_per_block is the operative quantity, so take it directly when
        # the caller knows the block size. Deriving it from num_blocks instead
        # loses information and cannot be inverted: block size 12 over 21
        # layers is 2 blocks, but ceil(21 / 2) is 11, and no num_blocks
        # whatsoever satisfies ceil(21 / n) == 12. The ceil fallback below is
        # exact when num_blocks came from the config directly (the official
        # pair 93/8 gives 12) and is only lossy for a size-derived count.
        if layers_per_block is not None:
            if not 1 <= layers_per_block <= n_layers:
                raise ValueError(
                    f"layers_per_block={layers_per_block} out of range "
                    f"[1, {n_layers}]"
                )
            self.layers_per_block = layers_per_block
        else:
            self.layers_per_block = -(-n_layers // num_blocks)  # ceil
        self.num_blocks = num_blocks
        self.num_committed_blocks = -(-n_layers // self.layers_per_block)

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            sharding_config=_tp_shard(0),
        )
        # ModuleDict for pipeline_module_split compatibility — see
        # KimiK3Model.__init__ for the same pattern.
        self.attn_res_gated = gated
        self.layers = nn.ModuleDict(
            {
                str(i): KimiAttnResDecoderLayer(config, i, gated=gated)
                for i in range(n_layers)
            }
        )
        # Off unless the config asks for it; see KimiK3MTPLayer for why the
        # released artifact has none.
        num_mtp = getattr(config, "num_nextn_predict_layers", 0)
        self.mtp_layers = (
            nn.ModuleDict(
                {
                    str(i): KimiK3MTPLayer(config, n_layers + i, gated=gated)
                    for i in range(num_mtp)
                }
            )
            if num_mtp
            else None
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, sharding_config=_tp_replicate())
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            sharding_config=_tp_shard(0, input_name="input"),
        )

        # Final AttnRes aggregation (one extra pseudo-query + RMSNorm
        # before lm_head). Same ``AttnResProjection`` shared with the
        # attn_res/ experiment.
        self.final_attn_res_proj = AttnResProjection(
            AttnResProjection.Config(
                dim=config.hidden_size, sharding_config=_tp_replicate()
            )
        )
        self.final_attn_res_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            sharding_config=_tp_replicate(),
        )
        if gated:
            self.final_attn_res_alpha = nn.Parameter(torch.zeros(1))

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # PP cache adapter hook — FSDP-only training leaves this False.
        self._return_only_new_blocks: bool = False

    # Default sentinel token id used to mark image-token positions in input_ids
    # when ``image_mask`` is not supplied alongside ``vision_embeds``. The
    # multimodal path picks 32000 (a Llama-3.1 reserved special token);
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
        the ``CrossStageCacheAdapter`` can drive this class via
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
                    image_mask = tokens == sentinel
                h = splice_vision_embeds(h, vision_embeds, image_mask)
        else:
            h = tokens

        # PP inputs only: keep the gradient that crosses the wire dense.
        # See _DenseGrad -- the last stage's aggregation makes the grad for the
        # partial block a slice of a wider buffer, and P2P refuses it.
        blocks = _dense_grad(blocks)
        h = _dense_grad(h)
        partial_block_src = h

        # 2) Unstack incoming blocks; empty list on stage 0 / non-PP.
        if blocks is None:
            block_list: list[torch.Tensor] = []
        else:
            block_list = unstack_blocks(blocks)
        initial_num_blocks = len(block_list)
        partial_block = partial_block_src

        # 3) Thread blocks + partial through this stage's layer slice.
        # ModuleDict keys are original layer indices (preserved across
        # pipeline_module_split); int() them to drive block-start detection.
        # Gated graft: seed the sequential plain stream. First stage:
        # the embedding; PP mid-stage: reconstruct once at entry (the
        # only reorder point -- single-stage runs stay bit-exact).
        plain_stream = (
            _plain_stream(block_list, partial_block) if self.attn_res_gated else None
        )
        for layer_key, layer in self.layers.items():
            layer_idx = int(layer_key)
            is_block_start = layer_idx % self.layers_per_block == 0
            block_list, partial_block, plain_stream = layer(
                block_list, partial_block, is_block_start, plain_stream
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
            if not block_list:
                # Non-delta mode has the same hole delta mode guards above: a
                # stage span that has not yet crossed any block boundary has
                # nothing to stack. VP is what exposes it -- with 4 virtual
                # stages per rank the early ones sit entirely inside the first
                # block, where pp8 with one stage per rank never landed.
                # .contiguous(): P2P requires non-overlapping dense tensors,
                # and a zero-first-dim new_zeros is not guaranteed to satisfy
                # that. VP4 is what surfaced it -- VP1 never sends an empty
                # stack because every rank's single stage spans a boundary.
                empty = partial_block.new_zeros((0, *partial_block.shape)).contiguous()
                return partial_block, empty
            return partial_block, stack_blocks(block_list)

        # Last stage / single-GPU: final aggregation + norm + lm_head.
        h_final = block_attn_res(
            block_list,
            partial_block,
            self.final_attn_res_proj,
            self.final_attn_res_norm,
        )
        if self.attn_res_gated:
            h_final = plain_stream + _scalar_local(
                self.final_attn_res_alpha, plain_stream
            ) * (h_final - plain_stream)
        # Keep the PRE-norm hidden state for MTP. The reference feeds MTP's hnorm the
        # unnormalised state (hnorm(h_pre_norm)); passing the already-normalised one
        # applies two RMSNorms in series, which is not an identity and silently breaks
        # parity against official MTP weights. The backbone's own norm still applies to
        # the backbone logits below.
        h_pre_norm = h_final
        if self.norm is not None:
            h_final = self.norm(h_final)

        # Multi-token prediction (report sec 3.3). Runs only where both the
        # embedding table and the head are present -- which is the last stage
        # only when PP has not split them apart; see _mtp_logits for why that is
        # checked rather than assumed.
        if self.mtp_layers is not None and self.lm_head is not None:
            from torchtitan.models.kimi_k3.mtp_loss import put_mtp_logits

            if self._skip_lm_head:
                # finding 44. This branch used to sit ahead of the _skip_lm_head return,
                # so a chunked-loss run still materialised a full [B, L, V] logits tensor
                # PER MTP DEPTH -- exactly the allocation chunking exists to avoid (OOM at
                # seq 8192), retaining ~1.3 GiB per depth across steps, and handing the
                # loss chunk-misaligned labels.
                #
                # Raised rather than skipped. Skipping would leave take_mtp_logits()
                # returning None, the MTP loss component contributing nothing, and a run
                # that looks like it is training MTP while it is not. Making MTP work
                # under chunked loss means computing its logits per chunk too, which is a
                # change to mtp_loss rather than a guard here.
                raise ValueError(
                    "MTP and chunked loss cannot be combined yet: MTP needs full-vocab "
                    "logits and ChunkedLossWrapper exists so they are never "
                    "materialised. Use a non-chunked loss for MTP flavors, or extend "
                    "mtp_loss to consume per-chunk logits."
                )
            self._mtp_logits = self._compute_mtp_logits(tokens, h_pre_norm)
            put_mtp_logits(self._mtp_logits)

        # _skip_lm_head is an attribute rather than a forward kwarg because PP
        # backward calls .requires_grad on all stage inputs, which fails on bool
        # kwargs -- same reason core's decoder does it this way. Set by the
        # trainer when ChunkedLossWrapper is in use, which then applies lm_head
        # per sequence chunk so the [B, L, V] logits are never materialised whole.
        if self._skip_lm_head:
            return h_final
        return self.lm_head(h_final)

    # Set by forward when MTP is on: one logits tensor per MTP depth, for a loss
    # component to consume. Not returned from forward because the trainer's
    # loss_fn takes a single ``pred``, and changing that is a core change.
    _mtp_logits: list[torch.Tensor] | None = None

    def _compute_mtp_logits(
        self, tokens: torch.Tensor, h_final: torch.Tensor
    ) -> list[torch.Tensor]:
        """Logits for each MTP depth. Depth k predicts the token k+1 ahead.

        The depth-k input fuses the backbone's final hidden state with the
        embedding of the token k+1 positions ahead, so the last k+1 positions
        have no target and are dropped by the loss rather than padded here --
        padding would invent supervision.

        MTP needs the embedding table AND the head, and raises when it cannot
        have both rather than silently producing nothing: a multi-token objective
        that quietly degrades to single-token is worse than a failed run.

        Two distinct reasons ``embed_tokens`` can be absent, and the message says
        which, because they need different answers:

        * PP has put the embedding and the head on different stages.
        * The multimodal wrapper set it to None on purpose. That is how it selects
          the backbone's pre-embedded branch after splicing vision features, so
          MTP under a multimodal model is not a plumbing problem -- the spliced
          sequence is LONGER than ``input_ids`` (each sentinel expands to many
          visual tokens), so "the token k+1 ahead" is no longer a shift of
          ``input_ids`` and the depth-k target has to come from the spliced
          sequence. Handing the table back would produce a misaligned objective
          that still trains, which is the worst outcome.
        """
        if self.embed_tokens is None:
            raise RuntimeError(
                "MTP needs embed_tokens and lm_head together, and embed_tokens "
                "is None. Either PP split them across stages, or the multimodal "
                "wrapper cleared it to take the pre-embedded branch -- in which "
                "case MTP needs targets from the SPLICED sequence, not from "
                "input_ids, because the splice changes the sequence length."
            )
        out = []
        for k in range(len(self.mtp_layers)):
            shift = k + 1
            # Token k+1 ahead, aligned to position t: drop the first `shift`
            # tokens and let the loss ignore the tail that has no target.
            ahead = tokens[:, shift:]
            emb = self.embed_tokens(ahead)
            h = h_final[:, : ahead.size(1)]
            hidden = self.mtp_layers[str(k)](h, emb)
            if self.norm is not None:
                hidden = self.norm(hidden)
            out.append(self.lm_head(hidden))
        return out

    def init_weights(
        self,
        init_range: float | None = None,
        **kwargs,
    ) -> None:
        """Normal init + mandatory zero-init of every pseudo-query.

        Paper §5 requires ``w_l`` zero-init so initial softmax weights
        are uniform (equivalent to standard residuals at t=0, avoids
        training volatility).

        ``**kwargs`` forwards trainer-supplied args (e.g. ``buffer_device``)
        to :meth:`KimiK3Model.init_weights`.
        """
        super().init_weights(init_range, **kwargs)
        # Zero-init every AttnRes pseudo-query (paper requirement).
        # Guard against PP-split stages that dropped some modules
        # (pipeline_module_split replaces non-owned modules with None
        # or Identity).

        # One helper over every AttnRes-bearing block, so a block reachable by a new
        # route cannot be missed. That is what happened with MTP: this loop walked
        # self.layers only, while an MTP layer wraps its own KimiAttnResDecoderLayer,
        # leaving those pseudo-queries and gate alphas at raw torch.empty values after
        # meta -> to_empty -> init. Garbage alphas are a step-0 NaN, and garbage
        # pseudo-queries violate the paper's zero-init requirement silently.
        def _zero_attn_res(block) -> None:
            for name in ("attn_res_proj", "mlp_res_proj"):
                m = getattr(block, name, None)
                if m is not None:
                    nn.init.zeros_(m.weight)
            # Graft-gate alphas start at exact zero (identity anchor).
            for name in ("attn_res_alpha", "mlp_res_alpha"):
                a = getattr(block, name, None)
                if isinstance(a, nn.Parameter):
                    nn.init.zeros_(a)

        for layer in self.layers.values():
            _zero_attn_res(layer)
        if self.mtp_layers is not None:
            for mtp in self.mtp_layers.values():
                block = getattr(mtp, "block", None)
                if block is not None:
                    _zero_attn_res(block)
        if self.final_attn_res_proj is not None:
            nn.init.zeros_(self.final_attn_res_proj.weight)
        a = getattr(self, "final_attn_res_alpha", None)
        if isinstance(a, nn.Parameter):
            nn.init.zeros_(a)
