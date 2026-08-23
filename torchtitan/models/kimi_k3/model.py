# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch import nn

from torchtitan.hf_datasets.multimodal.mm_datasets import MMSamplePackingConfig

from torchtitan.distributed.fsdp import add_zero_valued_dependency
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
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

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

        # Head divisibility is checked at wiring time, against tp*cp rather
        # than cp; see apply_cp_kimi_k3.
        cp_size = dist.get_world_size(cp_group)
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
        # Smallest image worth partitioning across CP ranks. Below it the
        # replicated encode is cheaper: splitting buys one gather per layer.
        dynamic_cp_min_patches: int = 256
        # KDA runs on fla triton kernels, which do not dispatch through
        # DTensor, so no ShardingConfig can drive its context parallel -- the
        # layer implements both CP modes itself, and the preconditions that
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
                    set_moe_sharding_config(
                        layer.moe,
                        enable_ep=enable_ep,
                        # Not a constant. With EP on, tp becomes a token axis
                        # inside the MoE region -- the sparse mesh folds it into
                        # efsdp -- so keying the desired layouts on enable_sp
                        # alone asks for S(1) -> P(sum), which DTensor rejects.
                        # Declaring SP when both are on makes source and
                        # destination agree. Same expression as the
                        # implementation this was ported from.
                        enable_sp=enable_ep and enable_tp,
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
        self.dynamic_cp_min_patches = config.dynamic_cp_min_patches
        self._dyncp_logged = False


    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group. A replicated DTensor -- what a tp-invariant module
        holds -- issues no all-gather to match, so the test is on the placement
        and not merely on the type.
        """
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_encoder.parameters()
        )

    def _tower_placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The smallest input the tower accepts, for a rank with no images."""
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        grid = torch.tensor(
            [[1, kernel_h, kernel_w]], dtype=torch.long, device=self._device()
        )
        weight = self.vision_encoder.patch_embed.weight
        # A plain tensor, not weight.new_zeros: once FSDP has sharded the
        # tower the weight is a DTensor, and a placeholder inheriting that
        # meets the tower's own plain tensors as "aten.mm got mixed".
        patches = torch.zeros(
            kernel_h * kernel_w,
            weight.shape[-1],
            dtype=weight.dtype,
            device=self._device(),
        )
        return patches, grid

    def _device(self) -> torch.device:
        return next(self.parameters()).device


    def _encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode every image, splitting the large ones across CP ranks.

        Report sec 5.2.3: a single large image is partitioned along the patch
        dimension and attention gathers key-value pairs across ranks, which is
        what reduces the encoder latency of large samples and the cross-device
        imbalance. Small images are left replicated -- splitting one buys a
        gather per layer and saves nothing, so the threshold is on the image's
        own patch count rather than on the batch.

        Without CP, or with every image below the threshold, this is exactly the
        replicated call it replaces.
        """
        from torchtitan.models.kimi_k3.vit_cp_plan import classify
        from torchtitan.models.kimi_k3.vision_encoder import make_cp_patch_plan

        group = self._cp_group
        cp_size = dist.get_world_size(group) if group is not None else 1
        grids = grid_thw.tolist()
        counts = [t * h * w for t, h, w in grids]
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        big = set(classify(counts, cp_size, min_patches=self.dynamic_cp_min_patches))
        # Only images whose height divides the merge kernel: a band boundary
        # inside a (kh, kw) block would ask two ranks to merge halves of one.
        big = {i for i in big if grids[i][1] % kernel_h == 0}
        if not big:
            return self.vision_encoder(pixel_values, grid_thw=grid_thw)
        if not self._dyncp_logged:
            # Said once, because "dynamic CP is on" and "dynamic CP fired" are
            # different claims: with the debug data every image sits under the
            # threshold, so the path is wired and inert, and only a count tells
            # the two apart.
            self._dyncp_logged = True
            logger.info(
                "Dynamic CP partitioning %d of %d image(s) across %d rank(s); "
                "patch counts %s, threshold %d.",
                len(big), len(grids), cp_size, counts, self.dynamic_cp_min_patches,
            )

        rank = dist.get_rank(group)
        outputs = []
        offset = 0
        for i, (grid, count) in enumerate(zip(grids, counts, strict=True)):
            item = pixel_values[offset : offset + count]
            item_grid = torch.tensor([grid], dtype=grid_thw.dtype, device=grid_thw.device)
            offset += count
            if i not in big:
                outputs.append(self.vision_encoder(item, grid_thw=item_grid))
                continue
            plan, ranges = make_cp_patch_plan(
                tuple(grid), group=group, rank=rank, merge_kernel_h=kernel_h
            )
            # Padded to the band, per frame, because _slice_for_shard pads the
            # position tables the same way and the two are added together. A
            # rank short of rows otherwise meets a table longer than its pixels
            # -- which is what "mirroring how the caller lays out the pixels"
            # in that helper means. The padding repeats the last real row
            # rather than being zeroed, so nothing out of range reaches a norm;
            # the padded queries are discarded and the padded keys are masked.
            _, _, grid_w = grid
            per_frame = []
            for lo, hi in ranges:
                rows = item[lo:hi]
                pad = plan.band * grid_w - rows.shape[0]
                if pad > 0:
                    # Zeros, matching the implementation this was ported from.
                    # The padded rows form whole merge blocks -- bands and real
                    # row counts are both multiples of the kernel height -- so
                    # nothing real is averaged with them, and the padded queries
                    # are discarded while the padded keys are masked.
                    rows = torch.cat(
                        [rows, rows.new_zeros(pad, *item.shape[1:])], dim=0
                    )
                per_frame.append(rows)
            shard = torch.cat(per_frame, dim=0)
            mine = self.vision_encoder(shard, grid_thw=item_grid, cp_plan=plan)
            # Every rank needs the whole image's tokens: the text side splices
            # them at positions this rank may or may not hold, and the CP
            # sentinel selection downstream cuts to what it does hold.
            parts = [torch.empty_like(mine) for _ in range(cp_size)]
            dist.all_gather(parts, mine.contiguous())
            # Each rank emitted band-worth of rows, padded; only its real rows
            # carry tokens the text side expects. The counts follow the same
            # ceiling split row_partition performs, so trimming needs no extra
            # collective -- and it cannot be a single trailing trim, because
            # every rank's padding sits at the end of ITS chunk.
            _, full_h, full_w = plan.full_grid
            kernel_w = self.vision_encoder.merge_kernel_size[1]
            per_row = full_w // kernel_w
            trimmed = []
            for r_i, part in enumerate(parts):
                real_rows = min(plan.band, max(0, full_h - r_i * plan.band))
                trimmed.append(part[: (real_rows // kernel_h) * per_row])
            outputs.append(torch.cat(trimmed, dim=0))
        return torch.cat(outputs, dim=0)


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
            # An image-free batch is normal, and skipping the tower on it is
            # what the vision TODO in parallelize.py describes: FSDP2 issues
            # the tower's all-gather from its pre-forward hook, so a rank that
            # does not run it leaves its peers waiting in that collective until
            # the watchdog fires. Run it on a placeholder and keep the graph
            # edge with a zero-valued dependency, so every rank issues the same
            # collectives and the tower's contribution to the data-parallel
            # average is a correct zero.
            if self.vision_encoder is not None and self._tower_needs_collectives():
                placeholder, placeholder_grid = self._tower_placeholder()
                unused = self.vision_encoder(placeholder, grid_thw=placeholder_grid)
                return add_zero_valued_dependency(embeddings_TD, unused)
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self._encode_images(pixel_values, grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        if self._cp_group is not None and dist.get_world_size(self._cp_group) > 1:
            # This rank holds a slice of the sequence but encoded every image,
            # so take the slice of the features its placeholders correspond to
            # and scatter those. get_vision_positions cannot be used on a
            # shard: it requires exactly one whole run per visual item, and a
            # shard legitimately holds none, or half of one.
            local_mask = tokens == special_tokens["image_id"]
            counts = self._exchange_sentinel_counts(int(local_mask.sum().item()))
            mine = self._select_cp_shard(vision_embeds, counts)
            embeddings_TD = embeddings_TD.masked_scatter(
                local_mask.unsqueeze(-1), mine.to(embeddings_TD.dtype)
            )
            return add_zero_valued_dependency(embeddings_TD, vision_embeds)
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

    def _exchange_sentinel_counts(self, local: int) -> torch.Tensor:
        """Per-rank vision-placeholder counts across the CP group.

        Called whenever CP is on, including on ranks with no images: the
        collective's participants are decided by the mesh, never by the batch.
        """
        group = self._cp_group
        counts = torch.zeros(
            dist.get_world_size(group),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        counts[dist.get_rank(group)] = local
        dist.all_reduce(counts, group=group)
        return counts

    def _select_cp_shard(
        self, vision_embeds: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Keep only the visual features belonging to this CP rank's shard.

        ``prepare_context_parallel_input`` shards inputs, labels and positions
        along the sequence but leaves ``pixel_values`` whole, so every rank
        encodes every image while holding only a slice of the placeholders. The
        features are ordered by sequence position and the shards are contiguous
        and equal -- the config rejects a load balancer under CP precisely
        because a permuting one would break that -- so this rank's slice starts
        after however many placeholders the lower ranks hold.

        This is correctness, not the report's sec 5.2.3 optimization: the
        encoder still runs redundantly on every CP rank.
        """
        num_rows = vision_embeds.shape[0]
        if int(counts.sum().item()) != num_rows:
            raise ValueError(
                f"CP ranks hold {int(counts.sum().item())} vision "
                f"placeholder(s) in total but {num_rows} visual token(s) were "
                "encoded; the sequence shard and the image batch disagree"
            )
        rank = dist.get_rank(self._cp_group)
        start = int(counts[:rank].sum().item())
        local = int(counts[rank].item())
        return vision_embeds[start : start + local]

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
