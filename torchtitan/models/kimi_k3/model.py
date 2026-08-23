# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.hf_datasets.multimodal.mm_datasets import MMSamplePackingConfig

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    BaseAttention,
    FlexAttention,
)
import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.protocols.sharding import ShardingConfig

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .kda import KimiDeltaAttention
from .sharding import cp_all_to_all_headseq, ULYSSES
from .moe import KimiFeedForward, KimiLatentMoE
from .vision_encoder import KimiK3VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension,
# N = attention-residual entries.


class KimiMLAAttention(BaseAttention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    # Set by apply_cp_kimi_k3; None means the layer runs without CP. MLA is
    # Ulysses under either KDA CP mode -- KCP describes a recurrence that MLA
    # does not have.
    _cp_group = None
    _cp_mask = None

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.gate = config.gate.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions

        num_tokens = x_TD.shape[0]
        q_THK = self.wq_b(self.q_norm(self.wq_a(x_TD))).view(
            num_tokens, self.n_heads, self.q_head_dim
        )

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = self.wkv_b(self.kv_norm(kv_latent_TC)).view(
            num_tokens,
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rope_THK = k_rope_TK.view(num_tokens, 1, self.qk_rope_head_dim).expand(
            -1, self.n_heads, -1
        )
        k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)

        cp_group = self._cp_group
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out_THV = self._ulysses_attention(
                q_THK, kv_THC, k_rope_TK, cp_group
            )
        else:
            out_THV = self.inner_attention(
                q_THK,
                k_THK,
                v_THV,
                attention_masks=attention_masks,
                scale=self.scale,
            )
        out_TD = out_THV.reshape(num_tokens, self.n_heads * self.v_head_dim)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)


    def _full_sequence_causal_mask(self, num_tokens: int, device):
        """Causal mask for the sequence Ulysses reassembles.

        The mask the layer is handed has been sharded for context parallel by
        ``cp_shard``, which cuts it the way ring attention wants: local queries
        against global keys. Ulysses reassembles the whole sequence on every
        rank instead, so it needs the whole causal mask. Rebuilding it is
        correct here only because this model rejects sample packing, so the
        sequence is one document and the mask carries no boundaries; a packed
        sequence would need the global boundaries threaded down instead.

        Cached per (length, device) because the shape is constant across layers
        and steps, and create_block_mask is compiled.
        """
        key = (num_tokens, device)
        if self._cp_mask is None or self._cp_mask[0] != key:
            mask = create_attention_mask(
                get_causal_mask_mod(), None, None, num_tokens, num_tokens, device=device
            )
            self._cp_mask = (key, mask)
        return self._cp_mask[1]

    def _ulysses_attention(
        self,
        q_LHQ: torch.Tensor,
        kv_LHC: torch.Tensor,
        k_rope_LR: torch.Tensor,
        cp_group,
    ) -> torch.Tensor:
        """Attention over the full sequence for this rank's head subset.

        One fused all-to-all trades the sharded axis, sequence for heads, then
        the backend runs unchanged, then a second trades back. The gate and the
        output projection stay sequence-local, so they are outside this.

        The rotary slice is deliberately not in the all-to-all. It is headless
        -- one vector per token, shared by every head -- so it is all-gathered
        along the sequence and expanded onto this rank's heads afterwards.
        Packing the already-expanded key instead sends the same values once per
        head and reassembles them against the wrong head subset, which shows up
        as a forward that diverges from the same layer run without CP.

        Shape suffixes beyond the file legend: L local sequence (T/cp), G this
        rank's head count (H/cp), W the packed per-head channel width, R the
        rotary width.
        """
        import torch.distributed.nn.functional as dist_nn

        cp_size = dist.get_world_size(cp_group)
        if self.n_heads % cp_size != 0:
            raise ValueError(
                f"MLA Ulysses CP: n_heads {self.n_heads} is not divisible by "
                f"cp={cp_size}"
            )
        t_loc = q_LHQ.shape[0]
        t_full = t_loc * cp_size
        h_cp = self.n_heads // cp_size

        packed_LHW = torch.cat([q_LHQ, kv_LHC], dim=-1)
        src_dim, dst_dim = ULYSSES.in_dims()
        packed_TGW = cp_all_to_all_headseq(
            packed_LHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
        )
        q_TGQ, k_nope_TGN, v_TGV = torch.split(
            packed_TGW,
            [self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        # Differentiable all-gather: the backward is a reduce-scatter, which is
        # what a value every rank consumed needs.
        k_rope_TR = torch.cat(
            dist_nn.all_gather(k_rope_LR.contiguous(), group=cp_group), dim=0
        )
        k_TGQ = torch.cat(
            [
                k_nope_TGN,
                k_rope_TR.view(t_full, 1, self.qk_rope_head_dim).expand(
                    t_full, h_cp, self.qk_rope_head_dim
                ),
            ],
            dim=-1,
        )

        out_TGV = self.inner_attention(
            q_TGQ,
            k_TGQ,
            v_TGV,
            attention_masks=self._full_sequence_causal_mask(t_full, q_TGQ.device),
            scale=self.scale,
        )
        out_src_dim, out_dst_dim = ULYSSES.out_dims()
        return cp_all_to_all_headseq(
            out_TGV.contiguous(), cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
        )


def _tp_replicate_config() -> ShardingConfig:
    """Weight replicated on the TP axis, with no activation boundary declared.

    Not core's ``vision_invariant_linear_config``: that one declares all four
    activation boundaries, which lifts the input to a DTensor, and
    ``common/linear.py``'s ``Linear.forward`` unwraps its own weight to local
    before the matmul -- so the two meet as "aten.mm.default got mixed
    torch.Tensor and DTensor". Core's own dense ``colwise_config`` and
    ``rowwise_config`` leave those boundaries None for the same reason; this is
    the replicated member of that family, which core does not have.
    """
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)}
    )


def _set_mla_sharding(attention_cfg) -> None:
    """Head-parallel TP for MLA.

    Not ``set_gqa_attention_sharding``: that one asserts a GQAttention.Config,
    and this attention has no fused qkv linear. The shape is the same though --
    the projections that produce or consume the head axis split on it, the two
    compressions stay whole because they are rank-sized rather than head-sized.
    """
    attention_cfg.wq_b.sharding_config = colwise_config()
    attention_cfg.wkv_b.sharding_config = colwise_config()
    attention_cfg.gate.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config()
    attention_cfg.wq_a.sharding_config = _tp_replicate_config()
    attention_cfg.wkv_a.sharding_config = _tp_replicate_config()
    attention_cfg.q_norm.sharding_config = norm_config(enable_sp=False)
    attention_cfg.kv_norm.sharding_config = norm_config(enable_sp=False)


def _set_kda_sharding(delta_attention_cfg) -> None:
    """KDA is invariant at TP, and has to be.

    Its kernels are fla triton and never see a DTensor, so nothing declared
    here could reach them. Declaring the module invariant says exactly that,
    and keeps every parameter on one mesh so clip_grad_norm_ can stack them.
    """
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_conv",
        "k_conv",
        "v_conv",
        "forget_a",
        "forget_b",
        "beta",
        "output_gate",
        "output_proj",
    ):
        getattr(delta_attention_cfg, name).sharding_config = (
            _tp_replicate_config()
        )
    delta_attention_cfg.output_norm.sharding_config = norm_config(enable_sp=False)
    # A_log and dt_bias are the module's OWN parameters, so only a
    # module-level declaration reaches them.
    delta_attention_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": SpmdLayout({MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.I}),
            "dt_bias": SpmdLayout({MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.I}),
        }
    )

def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32.

    TODO: Add TP Support. The current implementation assumes that the input tensors are on a single device.
    """
    assert norm.eps is not None

    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KimiDeltaAttention.Config | None
        feed_forward: KimiFeedForward.Config | None
        moe: KimiLatentMoE.Config | None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_res_norm: RMSNorm.Config | None
        attention_res_proj: Linear.Config | None
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if (config.attention is None) == (config.delta_attention is None):
            raise ValueError(
                "Exactly one of attention or delta_attention must be configured."
            )
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError("Exactly one of feed_forward or moe must be configured.")
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention_res_norm = (
            config.attention_res_norm.build()
            if config.attention_res_norm is not None
            else None
        )
        self.attention_res_proj = (
            config.attention_res_proj.build()
            if config.attention_res_proj is not None
            else None
        )
        self.ffn_res_norm = config.ffn_res_norm.build()
        self.ffn_res_proj = config.ffn_res_proj.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum_TD = x_TD

        if block_residual_TND.shape[1] > 0:
            assert self.attention_res_proj is not None
            assert self.attention_res_norm is not None
            x_TD = _apply_attention_residual(
                prefix_sum_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )

        opens_block = self.layer_id % self.attn_res_block_size == 0
        if opens_block:
            block_residual_TND = torch.cat(
                (
                    block_residual_TND,
                    prefix_sum_TD.unsqueeze(1),
                ),
                dim=1,
            )

        h_TD = self.attention_norm(x_TD)
        if self.attention is not None:
            h_TD = self.attention(h_TD, attention_masks, positions)
        else:
            assert self.delta_attention is not None
            h_TD = self.delta_attention(h_TD, None, positions)
        prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

        h_TD = _apply_attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TND


class KimiK3Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        # KDA runs on fla triton kernels, which do not dispatch through
        # DTensor, so no ShardingConfig can drive its context parallel -- the
        # layer implements both CP modes itself. The preconditions that
        # replaces the backend check with are enforced below.
        cp_via_sharding_config: bool = False

        def update_from_config(self, *, config, **kwargs) -> None:
            dataset = config.dataloader.dataset
            # TODO: Support sample packing by resetting the Q/K/V causal-convolution
            # and KDA recurrent states at document boundaries.
            if isinstance(dataset, MMSamplePackingConfig):
                raise ValueError("Kimi K3 does not yet support sample packing.")
            parallelism = config.parallelism
            if (
                parallelism.context_parallel_degree > 1
                and parallelism.context_parallel_load_balancer is not None
            ):
                # Both CP algorithms here read the sequence as rank-ordered
                # contiguous chunks: the Ulysses all-to-all reassembles it in
                # rank order, and KDA's recurrence passes state from rank r to
                # rank r+1. A load balancer permutes tokens across ranks, which
                # silently breaks both -- the shapes still line up.
                raise ValueError(
                    "Kimi K3 context parallel requires "
                    "parallelism.context_parallel_load_balancer=None; "
                    f"got {parallelism.context_parallel_load_balancer!r}."
                )
            self._set_sharding_config(
                enable_ep=parallelism.expert_parallel_degree > 1,
                enable_tp=parallelism.tensor_parallel_degree > 1,
            )
            Decoder.Config.update_from_config(self, config=config, **kwargs)

        def _set_sharding_config(self, *, enable_ep: bool, enable_tp: bool) -> None:
            """Declare the sharding the SPMD backends act on.

            Sequence parallel is not offered: the sequence is already the axis
            context parallel shards here, and this model's CP is not
            ShardingConfig-driven, so the two would be describing the same axis
            from two places.

            Only the parts that upstream already has a helper for. Everything
            else is left undeclared on purpose rather than filled in with a
            guess -- an undeclared module is inert, a wrongly declared one is a
            silent numerics change.
            """
            if not (enable_ep or enable_tp):
                return
            set_decoder_sharding_config(self, enable_sp=False)
            attn_x_layout = dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            for layer in self.layers:
                if enable_tp:
                    # Every norm and the residual projections stay whole. They
                    # sit on the block stream, which TP does not split, and
                    # leaving them undeclared makes them plain tensors meeting
                    # DTensor activations: "aten._fused_rms_norm.default got
                    # mixed torch.Tensor and DTensor".
                    for name in (
                        "attention_norm",
                        "ffn_norm",
                        "attention_res_norm",
                        "ffn_res_norm",
                    ):
                        cfg = getattr(layer, name, None)
                        if cfg is not None:
                            cfg.sharding_config = norm_config(enable_sp=False)
                    for name in ("attention_res_proj", "ffn_res_proj"):
                        cfg = getattr(layer, name, None)
                        if cfg is not None:
                            cfg.sharding_config = _tp_replicate_config()
                    if layer.attention is not None:
                        _set_mla_sharding(layer.attention)
                    if layer.delta_attention is not None:
                        _set_kda_sharding(layer.delta_attention)
                    if layer.feed_forward is not None:
                        set_dense_ffn_sharding(
                            layer.feed_forward,
                            attn_x_layout=attn_x_layout,
                            enable_sp=False,
                        )
                if layer.moe is not None:
                    if enable_tp:
                        # The latent pair is Kimi's addition to core's MoE, so
                        # set_moe_sharding_config below does not know about it
                        # and would leave it plain against DTensor activations.
                        # It stays whole: it compresses to a rank, not to
                        # heads or to the expert axis.
                        layer.moe.routed_down.sharding_config = (
                            _tp_replicate_config()
                        )
                        layer.moe.routed_up.sharding_config = (
                            _tp_replicate_config()
                        )
                        layer.moe.routed_norm.sharding_config = norm_config(
                            enable_sp=False
                        )
                    if enable_ep:
                        # LocalTokenDispatcher only reorders tokens within a
                        # rank, so it hands the experts the GLOBAL per-expert
                        # counts. With the expert weights sharded on E, the
                        # grouped GEMM then sees 32 offsets against 16 local
                        # experts and reports "matrix batch sizes have to
                        # match" -- which reads as a shape bug in the model.
                        # The two configs carry the same fields.
                        dispatcher = layer.moe.routed_experts.token_dispatcher
                        layer.moe.routed_experts.token_dispatcher = (
                            AllToAllTokenDispatcher.Config(
                                num_experts=dispatcher.num_experts,
                                top_k=dispatcher.top_k,
                            )
                        )
                    set_moe_sharding_config(
                        layer.moe,
                        enable_ep=enable_ep,
                        enable_sp=False,
                        expert_param_layout={
                            "w1_EFD": spmd.S(1),
                            "w2_EDF": spmd.S(2),
                            "w3_EFD": spmd.S(1),
                        },
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention_config = self.first_attention
            if not isinstance(attention_config, KimiMLAAttention.Config):
                raise ValueError(
                    "Kimi K3 requires at least one MLA layer for FLOP accounting."
                )
            # KDA and the vision encoder have no dedicated term here, so their
            # parameters only contribute the dense 6*N estimate; reported MFU is
            # approximate.
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention_config.n_heads,
                attention_config.qk_nope_head_dim
                + attention_config.qk_rope_head_dim
                + attention_config.v_head_dim,
                seq_len,
            )

    # Set by apply_cp_kimi_k3; the vision scatter needs it to place this
    # rank's slice of the sequence.
    _cp_group = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        if self._cp_group is not None and dist.get_world_size(self._cp_group) > 1:
            return self._scatter_vision_embeds_cp(
                embeddings_TD,
                tokens=tokens,
                vision_embeds=vision_embeds,
                num_tokens_per_item=num_tokens_per_item,
                placeholder_id=special_tokens["image_id"],
            )
        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        return scatter_vision_embeds(
            embeddings_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

    def _scatter_vision_embeds_cp(
        self,
        embeddings_TD: torch.Tensor,
        *,
        tokens: torch.Tensor,
        vision_embeds: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        placeholder_id: int,
    ) -> torch.Tensor:
        """Scatter vision features when the token sequence is CP-sharded.

        ``get_vision_positions`` reads placeholder runs off the whole sequence
        and requires exactly one run per visual item. A CP rank holds a slice,
        so it may see no runs at all, or half of one -- both of which that
        function rejects, correctly, as a text/vision misalignment.

        The pixel tensors are not sharded (``prepare_context_parallel_input``
        cuts only inputs, labels, positions and masks), so every rank already
        produced every embedding. What is missing is where this rank's slice
        sits in the whole sequence, and one all-gather of the token ids -- an
        int64 vector, once per step, not per layer -- answers that. Runs that
        straddle a rank boundary are handled by intersecting each run with the
        local window rather than assuming a run lies within one rank.

        This assumes contiguous rank-ordered sharding, which is why the config
        rejects a load balancer under CP.
        """
        cp_group = self._cp_group
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        num_local = tokens.shape[0]

        shards = [torch.empty_like(tokens) for _ in range(cp_size)]
        dist.all_gather(shards, tokens.contiguous(), group=cp_group)
        vision_positions = get_vision_positions(
            torch.cat(shards), num_tokens_per_item, placeholder_id
        )

        lo, hi = cp_rank * num_local, (cp_rank + 1) * num_local
        offset = 0
        for _, start, num_tokens in vision_positions:
            begin, end = max(start, lo), min(start + num_tokens, hi)
            if begin < end:
                embeddings_TD[begin - lo : end - lo] = vision_embeds[
                    offset + (begin - start) : offset + (end - start)
                ].to(embeddings_TD.dtype)
            offset += num_tokens
        if offset != vision_embeds.shape[0]:
            raise ValueError(
                f"Vision placeholder runs consume {offset} embeddings but the "
                f"packed vision output contains {vision_embeds.shape[0]}."
            )
        # Keep the tower in every rank's autograd graph. A rank whose slice
        # holds no image tokens consumes no embedding, so the tower would get
        # no gradient there, FSDP would skip that rank's reduce_scatter, and
        # the ranks would deadlock on mismatched collectives -- observed as a
        # 300s watchdog timeout with one rank still in reduce_scatter while
        # the other had moved two collectives ahead. Adding an exact zero
        # leaves the embeddings unchanged.
        return embeddings_TD + vision_embeds.sum() * 0.0

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        block_residual_TND: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        # Under pipeline parallel a middle stage receives its predecessor's
        # two outputs, the hidden states and the accumulated block residual;
        # see the return below for why the residual has to travel.
        block_residual_in = block_residual_TND

        if self.tok_embeddings is not None:
            h_TD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                special_tokens=special_tokens,
            )
        else:
            h_TD = tokens

        num_tokens, D = h_TD.shape
        block_residual_TND = (
            block_residual_in
            if block_residual_in is not None
            else h_TD.new_zeros(num_tokens, 0, D)
        )
        for layer in self.layers.values():
            h_TD, block_residual_TND = layer(
                h_TD,
                block_residual_TND,
                attention_masks,
                positions,
            )

        # The final aggregation belongs to whichever stage owns the head. Under
        # pipeline parallel the other stages have these set to None, the same
        # way norm and lm_head are, and the block residual they accumulated has
        # to travel to the next stage: a block attention residual is defined
        # over the whole stack, so a stage that dropped it would train against
        # a different model. Measured on the debug flavor at pp2, dropping it
        # moved step 3 from 7.44679 to 9.30017.
        if self.output_res_proj is None:
            return h_TD, block_residual_TND
        if self.output_res_proj is not None:
            h_TD = _apply_attention_residual(
                h_TD,
                block_residual_TND,
                self.output_res_proj,
                self.output_res_norm,
            )
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
