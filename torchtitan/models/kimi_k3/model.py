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
from .sharding import mla_ulysses_attention
from .moe import KimiFeedForward, KimiLatentMoE
from .vision_encoder import KimiK3VisionEncoder

# Shape suffixes: T = packed tokens, D = model dimension, H = heads,
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
        # Head count DERIVED from the projection width, not self.n_heads: the
        # two differ under TP (wq_b/wkv_b are column-parallel, n_heads/tp per
        # rank). Deriving holds under every DTensor backend with no branch.
        q_proj_TE = self.wq_b(self.q_norm(self.wq_a(x_TD)))
        h_local = q_proj_TE.shape[-1] // self.q_head_dim
        q_THK = q_proj_TE.view(num_tokens, h_local, self.q_head_dim)

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = self.wkv_b(self.kv_norm(kv_latent_TC)).view(
            num_tokens,
            h_local,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rope_THK = k_rope_TK.view(num_tokens, 1, self.qk_rope_head_dim).expand(
            -1, h_local, -1
        )
        k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)

        cp_group = self._cp_group
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out_THV = mla_ulysses_attention(
                self, q_THK, kv_THC, k_rope_TK, cp_group
            )
        else:
            out_THV = self.inner_attention(
                q_THK,
                k_THK,
                v_THV,
                attention_masks=attention_masks,
                scale=self.scale,
            )
        out_TD = out_THV.reshape(num_tokens, h_local * self.v_head_dim)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)


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


class _PlainGradBoundary(torch.autograd.Function):
    """Identity forward; forces the incoming gradient to be a plain tensor.

    The vision tower must stay plain in BOTH directions. Its TP and its dynamic
    CP are separate mechanisms from the decoder's, and the CP path runs
    hand-written collectives whose transpose is a reduce_scatter --
    _c10d_functional.reduce_scatter_tensor has no DTensor sharding strategy.

    to_local() alone is not enough and grad_placements is the wrong knob: the
    first re-wraps the gradient with the forward placements, the second states
    which placements to re-wrap WITH. Neither can say "do not re-wrap". That is
    what this states, and only an autograd.Function can.
    """

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad.to_local() if isinstance(grad, DTensor) else grad


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
        # DEP (report sec 5.2.3): the vision tower gets a pipeline stage of its
        # own ahead of the text stages, so its compute leaves the critical path
        # of the stage that owns the embedding. Opt-in: it changes the stage
        # count, which the schedule and the checkpoint layout both see.
        vit_dep: bool = False
        # How many stages the tower is split across. More than one by default:
        # on a single stage the tower's whole forward still serializes against
        # one text stage, which is the imbalance report sec 5.2.3 clause 2
        # exists to remove. Vision stages come OUT of the text budget, so a
        # pipeline with no stage to spare raises rather than dropping to 1.
        vit_dep_stages: int = 2
        # How many micro-batches ahead the tower's encode is issued. 0 keeps
        # the encode inline, which is the measured-nothing default.
        vit_prefetch: int = 0
        # Run the planned encodes in the schedule's idle intervals on the main
        # stream instead of ahead of time on a side stream. The two are
        # alternatives, not layers: this takes over placement when it is on.
        vit_bubble: bool = False
        # One tower forward in units of one text-stage forward. A parameter and
        # not a measurement, because a plan derived from each rank's own timing
        # would stop being identical across ranks.
        vit_bubble_cost_ratio: float = 0.5
        # How many deferred vision backwards may wait; each holds a
        # micro-batch's tower forward graph alive, so this is the backward
        # half's memory window. 0 is unbounded.
        vit_bubble_max_pending: int = 0
        # Whether the tower's attention shards its heads on the tensor-parallel
        # axis. Off means the tower runs replicated there.
        vit_tp_heads: bool = True
        # Ship only the blocks a receiver does not already hold on each
        # pipeline hop, instead of the whole stack. On by default: under
        # pipeline parallelism this is the transport, and the naive one is
        # the fallback, not the other way round. It changes the order the
        # block gradients are summed, so it is not bitwise against that
        # fallback. Engages only on Interleaved1F1B with an even split;
        # anything else warns and passes through.
        attn_res_cache: bool = True
        # DEP (report sec 5.2.3): a patch stream crossing a stage boundary
        # needs a static shape -- pipelining sizes its buffers once. These
        # bound the padded payload; exceeding them raises, never truncates.
        dep_max_images: int = 8
        dep_max_grid_h: int = 64
        dep_max_grid_w: int = 64


        def _validate_cp_backend(self, parallelism) -> None:
            """This model's CP is not ShardingConfig-driven -- the KDA kernels
            are fla triton and never see a DTensor -- so the spmd_types
            requirement does not apply; apply_cp_kimi_k3 checks its own
            preconditions at wiring time."""

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
                # Both CP algorithms read the sequence as rank-ordered
                # contiguous chunks (Ulysses reassembly, KDA's rank-to-rank
                # state). A load balancer silently breaks both.
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
                    # Every norm and the residual projections stay whole: they
                    # sit on the block stream, which TP does not split. Left
                    # undeclared they meet DTensor activations as plain tensors.
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
                        # set_moe_sharding_config does not know it. It stays
                        # whole: it compresses to a rank, not heads or experts.
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
                        # Not a constant: with EP on, tp becomes a token axis
                        # inside the MoE region (the sparse mesh folds it into
                        # efsdp), so SP must be declared when both are on or
                        # DTensor rejects S(1) -> P(sum).
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


    def encode_images(self, pixel_values, grid_thw):
        """Public entry the DEP prefetcher calls.

        The run-ahead issues the encode for a later micro-batch on the vision
        stream, so it needs a name it can call on whichever module owns the
        tower. Delegates to the private path that the forward uses, so the two
        cannot drift.
        """
        return self._encode_images(pixel_values, grid_thw)

    def _vision_stream(self):
        """A dedicated CUDA stream for the tower, created once per module.

        Groundwork for DEP's concurrent design (report 5.2.3), and it is only
        groundwork: running on a side stream and immediately waiting for it cannot
        overlap anything. The overlap needs the encode for micro-batch m+k issued
        during micro-batch m's text compute, which is a scheduling change. What this
        establishes is the part that has to be right first -- cross-stream tensor
        lifetime and the interaction with FSDP2's tower all-gather, neither of which
        the AttnRes PP adapter has ever had to deal with (it touches no streams at
        all).

        Same THREAD, separate stream. Not a worker thread: the adapter keys its
        per-microbatch cache in a ``threading.local``, and its forward reads a
        missing key as "this call is PP's shape inference" and diverts WITHOUT
        raising. A worker thread would therefore take the shape-inference path and
        return wrong shapes with no error.
        """
        if not torch.cuda.is_available():
            return None
        # Only when no autograd graph is being recorded: with prefetch, several
        # micro-batches' backwards accumulate into the tower parameters from two
        # streams with nothing ordering them. Both callers join immediately, so
        # grad-on loses nothing; the deferred design (report 5.2.3) orders first.
        if torch.is_grad_enabled():
            return None
        s = getattr(self, "_vision_side_stream", None)
        if s is None:
            s = torch.cuda.Stream()
            self._vision_side_stream = s
        return s

    def _issue_on_vision_stream(self, fn, *tensors):
        """Issue ``fn`` on the vision stream and return ``(out, event)`` WITHOUT waiting.

        This is the half that makes overlap possible. :meth:`_run_on_vision_stream` joins
        immediately, which is correct for a synchronous encode but means the side stream
        buys nothing -- the caller blocks on it before running anything else. The
        run-ahead needs the encode for micro-batch m+k in flight WHILE m's text compute
        runs, so it issues here and joins later, in :meth:`_join_vision_stream`.

        The input-side edges are the same as the synchronous path and equally required:
        the side stream waits for the current one because ``fn``'s inputs were produced
        there, and each input is ``record_stream``'d so the caching allocator cannot hand
        its memory to another allocation while the side stream still reads it.
        """
        side = self._vision_stream()
        if side is None:
            return fn(), None
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(side)
        # Bracket the encode ON THE SIDE STREAM so its own GPU time is
        # measurable; the issue-to-join span is dominated by text compute and
        # reads the same whether or not the encode ran concurrently.
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            started.record(side)
            out = fn()
            finished.record(side)
        done = finished
        self._last_encode_span = (started, finished)
        return out, done

    def _join_vision_stream(self, out, done) -> None:
        """Make the current stream wait for an issued encode, and hand the outputs over.

        Both halves are needed: without the wait the consumer reads memory the side
        stream is still writing, and without ``record_stream`` on the outputs the
        allocator may reuse buffers the side stream produced while the current stream
        still holds them.
        """
        if done is None:
            return
        cur = torch.cuda.current_stream()
        cur.wait_event(done)
        outs = out if isinstance(out, (list, tuple)) else [out]
        for t in outs:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(cur)

    def _encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode every image, partitioning the large ones (report sec 5.2.3).

        Ported from the reference tree's ``_encode_images_dynamic_cp``. Report
        5.2.3 has two halves and both are needed: a single large image is split
        along the patch dimension with attention gathering key-value pairs
        across ranks, AND each CP group is divided into sub-CP groups with the
        large images distributed across them, which is what keeps the
        communication fraction from growing with scale.

        Every large image is encoded by one sub-CP group, with its patches split
        across that sub-group's ranks. Images below the threshold, or whose grid
        height does not divide the merge kernel, stay whole and are encoded
        replicated -- splitting one buys a gather per layer and saves nothing.
        """
        import torch.distributed._functional_collectives as funcol

        from torchtitan.models.kimi_k3.vision_encoder import CPPatchPlan
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            balance_images,
            classify,
            merged_tokens,
            row_partition,
            subgroup_layout,
        )

        grids = grid_thw.tolist()
        counts = [t * h * w for t, h, w in grids]
        kh, kw = self.vision_encoder.merge_kernel_size
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)

        def _replicated(which: list[int]) -> dict[int, torch.Tensor]:
            """Encode a subset redundantly on every rank."""
            out = {}
            for i in which:
                item = pixel_values[offsets[i] : offsets[i + 1]]
                item_grid = torch.tensor(
                    [grids[i]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                out[i] = self.vision_encoder(item, grid_thw=item_grid)
            return out

        subgroups = getattr(self, "_cp_subgroups", None)
        group_all = self._cp_group
        cp_size = dist.get_world_size(group_all) if group_all is not None else 1
        if not subgroups or cp_size <= 1:
            return torch.cat([_replicated(list(range(len(counts))))[i]
                              for i in range(len(counts))], dim=0)

        large = classify(counts, cp_size, min_patches=self.dynamic_cp_min_patches)
        # Grid heights must divide the merge kernel for a partition to be legal.
        # An image that fails it is left replicated instead of being cut unsafely.
        large = [i for i in large if grids[i][1] % kh == 0]
        if not large:
            return torch.cat([_replicated(list(range(len(counts))))[i]
                              for i in range(len(counts))], dim=0)

        n_sub, g = subgroup_layout(len(large), cp_size)
        group = subgroups.get(n_sub)
        if group is None or g <= 1:
            # No usable sub-group of size > 1 means there is nothing to partition
            # across.
            return torch.cat([_replicated(list(range(len(counts))))[i]
                              for i in range(len(counts))], dim=0)

        cp_rank = dist.get_rank(group_all)
        my_sub = cp_rank // g
        rank_in_sub = cp_rank % g
        group_of = balance_images([counts[i] for i in large], n_sub)
        my_large = [img for img, sub in zip(large, group_of) if sub == my_sub]

        if not self._dyncp_logged:
            self._dyncp_logged = True
            logger.info(
                "Dynamic CP: %d large image(s) of %d over %d sub-CP group(s) of "
                "%d rank(s); min_patches=%d.",
                len(large), len(counts), n_sub, g, self.dynamic_cp_min_patches,
            )

        out: dict[int, torch.Tensor] = {}
        # Every sub-group must run the same NUMBER of passes or the collectives
        # inside them desynchronise. The count is the max over sub-groups, and a
        # sub-group with fewer images pads with an empty pass.
        per_sub = [sum(1 for s in group_of if s == k) for k in range(n_sub)]
        n_passes = max(per_sub) if per_sub else 0

        for p in range(n_passes):
            img = my_large[p] if p < len(my_large) else None
            if img is None:
                # An empty pass still joins this sub-group's collectives. One
                # merge block keeps every shape valid; the output is discarded.
                local = pixel_values.new_zeros(kh * kw, *pixel_values.shape[1:])
                local_grid = torch.tensor(
                    [[1, kh, kw]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=kh * kw * g,
                    full_grid=(1, kh * g, kw),
                    row_start=0,
                    band=kh,
                    real_rows=kh,
                )
            else:
                t, h, w = grids[img]
                shards = row_partition(t, h, w, kh=kh, group_size=g)
                sh = shards[rank_in_sub]
                bands = [s.row_end - s.row_start for s in shards]
                band = max(bands)
                # The ceiling split keeps any deficit on the TRAILING ranks, so
                # every rank's padding lands at the end of the gathered stream
                # rather than inside it. Taking a prefix below depends on that.
                if bands != sorted(bands, reverse=True):
                    raise AssertionError(
                        f"bands {bands} are not non-increasing; padding would land "
                        "inside the gathered token stream and corrupt the order"
                    )
                flat = pixel_values[offsets[img] : offsets[img + 1]]
                # This rank's rows of EVERY frame: the projector's temporal mean
                # spans all frames, so splitting by frame would give each rank the
                # mean of its own frames instead.
                pad_rows = band - (sh.row_end - sh.row_start)
                pieces = []
                for a, b in sh.ranges:
                    pieces.append(flat[a:b])
                    if pad_rows:
                        pieces.append(
                            flat.new_zeros(pad_rows * w, *flat.shape[1:])
                        )
                local = torch.cat(pieces, dim=0)
                local_grid = torch.tensor(
                    [[t, band, w]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=counts[img],
                    full_grid=(t, h, w),
                    row_start=sh.row_start,
                    band=band,
                    real_rows=sh.row_end - sh.row_start,
                )

            local = local.to(self.vision_encoder.patch_embed.weight.dtype)
            feats = self.vision_encoder(local, grid_thw=local_grid, cp_plan=plan)
            # to_local unwraps the value but its backward re-wraps the gradient,
            # and the all_gather below has a reduce_scatter transpose with no
            # DTensor rule.
            if isinstance(feats, DTensor):
                feats = feats.to_local()
            local_feat = _PlainGradBoundary.apply(feats)
            # The boundary belongs on the OUTPUT too: the gradient arrives from
            # downstream, so sealing only the input leaves the transpose
            # receiving a DTensor.
            gathered = _PlainGradBoundary.apply(
                funcol.all_gather_tensor(
                    local_feat.contiguous(), gather_dim=0, group=group
                )
            )
            if img is not None:
                t, h, w = grids[img]
                # NOT counts // merge: the projector collapses time, so a video's
                # token count carries no t.
                out[img] = gathered[: merged_tokens(h, w, kh, kw)]

        rest = [i for i in range(len(counts)) if i not in out]
        if rest:
            out.update(_replicated(rest))
        return torch.cat([out[i] for i in range(len(counts))], dim=0)

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
            # An image-free batch is normal, but FSDP2 issues the tower's
            # all-gather from its pre-forward hook, so every rank must run it.
            # A zero-valued placeholder keeps collectives and the DP average right.
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
            # This rank holds a sequence slice but encoded every image: take the
            # feature slice its placeholders correspond to and scatter it.
            # get_vision_positions needs whole visual items, which a shard lacks.
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

        # The final aggregation belongs to the head-owning stage; other stages
        # have these None, like norm and lm_head. The accumulated block residual
        # must travel on: a block residual is defined over the whole stack, and
        # a stage that dropped it would train against a different model.
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
