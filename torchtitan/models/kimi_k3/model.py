# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch_remat as remat
from torch import nn
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import prepare_context_parallel_batch
from torchtitan.distributed.fsdp import add_zero_valued_dependency
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.common import FeedForward, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    local_head_split,
    VarlenInnerAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    dense_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.kimi_k3.sharding import set_kimi_k3_sharding_config
from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module, ModuleDict

from .kda import KDA
from .moe import KimiLatentMoE
from .mtp import KimiK3MTPLayer, put_mtp_logits
from .vision_encoder import KimiK3VisionEncoder

if TYPE_CHECKING:
    from attn_gym.linear.context_parallel import ContextParallelRouting

logger = logging.getLogger(__name__)

KimiK3AttentionMaskDict = dict[str, BlockMask | VarlenMetadata | None]

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels, H = heads,
# K = query/key head dimension, V = value head dimension,
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
        inner_attention: Module.Config = field(
            default_factory=FlexInnerAttention.Config
        )

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

        q_THK = remat.region(
            self._project_q,
            self.remat_region_name("q"),
            recompute=self.remat_should_recompute("q"),
        )(x_TD)
        k_THK, v_THV = remat.region(
            self._project_kv,
            self.remat_region_name("kv"),
            recompute=self.remat_should_recompute("kv"),
        )(x_TD)
        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK,
            k_THK,
            v_THV,
            attention_masks=attention_masks,
            scale=self.scale,
        )
        gate_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2) * torch.sigmoid(gate_TD)
        out_TD = remat.region(
            self.wo,
            self.remat_region_name("wo"),
            recompute=self.remat_should_recompute("wo"),
        )(out_TD)
        remat.recompute_needs_tensor(out_TD)
        return out_TD

    def _project_q(self, x_TD: torch.Tensor) -> torch.Tensor:
        return local_head_split(
            self.wq_b(self.q_norm(self.wq_a(x_TD))), self.q_head_dim, cp_sharded=True
        )

    def _project_kv(self, x_TD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys (no-position and shared rope parts) and values per head."""
        kv_latent_TC, k_rope_TK = torch.split(
            self.wkv_a(x_TD),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = local_head_split(
            self.wkv_b(self.kv_norm(kv_latent_TC)),
            self.qk_nope_head_dim + self.v_head_dim,
            cp_sharded=True,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # Headless rope slice broadcast onto the local heads, as in DeepSeek-V3's MLA.
        with spmd.local():
            k_rope_THK = k_rope_TK.unsqueeze(1).expand(-1, k_nope_THK.shape[-2], -1)
            k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    k_THK,
                    spmd.SpmdType(
                        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
                        partition_spec=spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    ),
                )
        return k_THK, v_THV


@spmd.register_local_autograd_function
class _AttentionResidualAggregation(torch.autograd.Function):
    """Depth softmax over the block stack, saving only per-token statistics.

    The saved statistics carry no hidden-size factor, so backward recomputes
    the FP32 upcasts instead of retaining them.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        score_weight_D: torch.Tensor,
        norm_weight_D: torch.Tensor,
        eps: float,
        block_residual_TND: torch.Tensor,
        partial_block_TD: torch.Tensor | None,
    ) -> torch.Tensor:
        query_D = score_weight_D.float() * norm_weight_D.float()
        values_TD = list(block_residual_TND.unbind(dim=1))
        if partial_block_TD is not None:
            values_TD.append(partial_block_TD)

        scores, inverse_rms = [], []
        for value_TD in values_TD:
            value_float = value_TD.float()
            inverse_rms.append(torch.rsqrt(value_float.pow(2).mean(dim=-1) + eps))
            scores.append(torch.matmul(value_float, query_D))
        scores_NT = torch.stack(scores)
        inverse_rms_NT = torch.stack(inverse_rms)
        probs_NT = torch.softmax(scores_NT * inverse_rms_NT, dim=0)

        output_TD = None
        for index, value_TD in enumerate(values_TD):
            term_TD = probs_NT[index].unsqueeze(-1) * value_TD.float()
            output_TD = term_TD if output_TD is None else output_TD + term_TD

        ctx.save_for_backward(
            score_weight_D,
            norm_weight_D,
            probs_NT,
            scores_NT,
            inverse_rms_NT,
            block_residual_TND,
            partial_block_TD,
        )
        ctx.has_partial = partial_block_TD is not None
        return output_TD.to(block_residual_TND.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TD: torch.Tensor):  # pyrefly: ignore[bad-override]
        (
            score_weight_D,
            norm_weight_D,
            probs_NT,
            scores_NT,
            inverse_rms_NT,
            block_residual_TND,
            partial_block_TD,
        ) = ctx.saved_tensors
        query_D = score_weight_D.float() * norm_weight_D.float()
        grad_float_TD = grad_output_TD.float()
        dim = block_residual_TND.shape[-1]
        values_TD = list(block_residual_TND.unbind(dim=1))
        if ctx.has_partial:
            values_TD.append(partial_block_TD)

        pairing_NT = torch.stack(
            [(grad_float_TD * v.float()).sum(dim=-1) for v in values_TD]
        )
        grad_logits_NT = probs_NT * (
            pairing_NT - (probs_NT * pairing_NT).sum(dim=0, keepdim=True)
        )
        grad_scores_NT = grad_logits_NT * inverse_rms_NT
        grad_variance_NT = grad_logits_NT * scores_NT * (-0.5) * inverse_rms_NT.pow(3)

        grad_query_D = torch.zeros_like(query_D)
        grad_values_TD = []
        for index, value_TD in enumerate(values_TD):
            value_float = value_TD.float()
            grad_value_TD = (
                probs_NT[index].unsqueeze(-1) * grad_float_TD
                + grad_scores_NT[index].unsqueeze(-1) * query_D
                + (grad_variance_NT[index] * (2.0 / dim)).unsqueeze(-1) * value_float
            )
            grad_values_TD.append(grad_value_TD.to(value_TD.dtype))
            # Accumulated as a GEMV per source so the reduction order is fixed.
            grad_query_D = grad_query_D + torch.matmul(
                value_float.reshape(-1, dim).transpose(0, 1),
                grad_scores_NT[index].reshape(-1),
            )

        grad_partial_TD = grad_values_TD.pop() if ctx.has_partial else None
        grad_stack_TND = (
            torch.stack(grad_values_TD, dim=1)
            if grad_values_TD
            else block_residual_TND.new_zeros(block_residual_TND.shape)
        )
        return (
            (grad_query_D * norm_weight_D.float()).to(score_weight_D.dtype)
            if ctx.needs_input_grad[0]
            else None,
            (grad_query_D * score_weight_D.float()).to(norm_weight_D.dtype)
            if ctx.needs_input_grad[1]
            else None,
            None,
            grad_stack_TND,
            grad_partial_TD,
        )


def _apply_attention_residual(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32."""
    assert norm.eps is not None

    if not is_in_batch_invariant_mode():
        return _AttentionResidualAggregation.apply(
            projection.weight.squeeze(0),
            norm.weight,
            norm.eps,
            block_residual_TND,
            partial_block_TD,
        )

    # Batch-invariant mode keeps the plain form: it deliberately avoids CUDA
    # sum and bmm so the reduction schedule cannot vary between runs.
    values_TND = (
        block_residual_TND
        if partial_block_TD is None
        else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
    )
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    # Do not use CUDA sum/bmm to avoid varying reduction schedules.
    scores_TN = (keys_TND * score_weight_D).mean(dim=-1) * keys_TND.shape[-1]
    probs_T1N = torch.log_softmax(scores_TN, dim=-1).exp().unsqueeze(1)
    output_TD = (probs_T1N.transpose(1, 2) * values_float).mean(
        dim=1
    ) * values_float.shape[1]
    return output_TD.to(values_TND.dtype)


def _checkpointed_attention_residual(
    name: str,
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    args = (partial_block_TD, block_residual_TND, projection, norm)
    if torch.is_grad_enabled() and (
        block_residual_TND.requires_grad
        or (partial_block_TD is not None and partial_block_TD.requires_grad)
    ):
        return remat.checkpoint(region_name=name)(_apply_attention_residual)(*args)
    return _apply_attention_residual(*args)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KDA.Config | None
        feed_forward: FeedForward.Config | None
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
        # A block's first layer closes the previous block and joins the stack.
        self.first_layer_in_block = self.layer_id % self.attn_res_block_size == 0
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
        self.attn_mask_key = (
            "quadratic_attention" if self.attention is not None else "kda"
        )
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
        # False when an activation-checkpointing policy wraps the whole block.
        self.checkpoint_residual = True
        self._region_ac = False

    def configure_remat_regions(self, save_patterns: list[str]) -> None:
        # Called by RegionAC on every block it checkpoints.
        self._region_ac = True
        super().configure_remat_regions(save_patterns)

    def _attention_residual(
        self,
        name: str,
        partial_block_TD: torch.Tensor | None,
        block_residual_TND: torch.Tensor,
        projection: Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """Attention residual whose fp32 intermediates are recomputed in backward."""
        args = (partial_block_TD, block_residual_TND, projection, norm)
        if self._region_ac:
            # A torch_remat checkpoint cannot nest inside RegionAC's block checkpoint.
            return remat.region(
                _apply_attention_residual,
                self.remat_region_name(name),
                recompute=self.remat_should_recompute(name),
            )(*args)
        if not self.checkpoint_residual:
            return _apply_attention_residual(*args)
        return _checkpointed_attention_residual(name, *args)

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_masks: KimiK3AttentionMaskDict | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        kda_cp_routing: "ContextParallelRouting | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_layer_in_block:
            block_residual_TND = torch.cat(
                (block_residual_TND, x_TD.unsqueeze(1)), dim=1
            )
            partial_block_TD = None
        else:
            partial_block_TD = x_TD

        if self.attention_res_proj is None:
            h_TD = x_TD
        else:
            assert self.attention_res_norm is not None
            h_TD = self._attention_residual(
                "attention_res",
                partial_block_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )
        remat.recompute_needs_tensor(h_TD)
        h_TD = self.attention_norm(h_TD)
        layer_mask = (
            attention_masks[self.attn_mask_key] if attention_masks is not None else None
        )
        if self.attention is not None:
            h_TD = self.attention(h_TD, layer_mask, positions)
        else:
            assert self.delta_attention is not None
            h_TD = self.delta_attention(
                h_TD,
                layer_mask,
                positions,
                cu_seqlens=cu_seqlens,
                routing=kda_cp_routing,
            )
        prefix_sum_TD = h_TD if self.first_layer_in_block else x_TD + h_TD

        h_TD = self._attention_residual(
            "ffn_res",
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        remat.recompute_needs_tensor(h_TD)
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD, padding_mask_T=padding_mask)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TND


@spmd.register_local_autograd_function
class _PlainGradBoundary(torch.autograd.Function):
    """Identity forward; the incoming gradient leaves as a plain tensor.

    The vision tower's dynamic CP runs hand-written collectives whose
    transpose is a reduce-scatter with no DTensor sharding strategy;
    ``to_local()`` re-wraps the gradient with the forward placements and
    ``grad_placements`` only says which placements to re-wrap with. Only an
    autograd.Function can say "do not re-wrap".
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
        mtp_layers: list[KimiK3MTPLayer.Config] = field(default_factory=list)
        # The smallest image worth partitioning across CP ranks (report sec
        # 5.2.3); below it the replicated encode is cheaper, since a split buys
        # one gather per layer.
        dynamic_cp_min_patches: int = 256

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.context_parallel_degree > 1:
                if (
                    parallelism.context_parallel_load_balancer.load_balancer_type
                    not in (
                        None,
                        "headtail",
                    )
                ):
                    raise ValueError(
                        "Kimi K3 KDA context parallelism supports only contiguous "
                        "or headtail token partitions."
                    )
                conv_kernel_sizes = {
                    layer.delta_attention.conv_kernel_size
                    for layer in self.layers
                    if layer.delta_attention is not None
                }
                if len(conv_kernel_sizes) != 1:
                    raise ValueError(
                        "Kimi K3 context parallelism requires every KDA layer "
                        "to use the same convolution kernel size."
                    )
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            # Vision attention is also head-sharded; validate its head count.
            tp = parallelism.tensor_parallel_degree
            vision_heads = (
                self.vision_encoder.block.attn.num_heads
                if self.vision_encoder is not None
                else None
            )
            if tp > 1 and vision_heads is not None and vision_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"vision num_heads ({vision_heads})."
                )

            set_kimi_k3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
            tp_sp = tp > 1 and parallelism.enable_sequence_parallel
            if self.mtp_layers and (tp_sp or parallelism.context_parallel_degree > 1):
                # The depth-shifted slice indexes tokens on the stream, which
                # sequence and context parallel shard by position.
                raise ValueError(
                    "MTP does not yet compose with sequence or context "
                    "parallelism: the depth-shifted slice indexes tokens on the "
                    "sequence-sharded stream. Pass "
                    "--parallelism.no-enable-sequence-parallel with tp>1 and "
                    "keep context_parallel_degree at 1."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            kimi_model = cast("KimiK3Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(kimi_model.vision_encoder,),
            )
            attention_op_flops = 0
            for layer in self.layers:
                if isinstance(layer.attention, KimiMLAAttention.Config):
                    attention = layer.attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=(
                            attention.qk_nope_head_dim + attention.qk_rope_head_dim
                        ),
                        v_head_dim=attention.v_head_dim,
                        seq_len=seq_len,
                    )
                elif isinstance(layer.delta_attention, KDA.Config):
                    delta_attention = layer.delta_attention
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=delta_attention.num_heads,
                        key_head_dim=delta_attention.head_dim,
                        v_head_dim=delta_attention.head_dim,
                    )
            for mtp_layer in self.mtp_layers:
                mirror = mtp_layer.block.delta_attention
                if isinstance(mirror, KDA.Config):
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=mirror.num_heads,
                        key_head_dim=mirror.head_dim,
                        v_head_dim=mirror.head_dim,
                    )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        # Registered last so the backbone and the tower draw the same init
        # stream as a model without MTP: their step-1 loss stays bitwise.
        self.mtp_layers = (
            ModuleDict({str(i): cfg.build() for i, cfg in enumerate(config.mtp_layers)})
            if config.mtp_layers
            else None
        )
        self.dynamic_cp_min_patches = config.dynamic_cp_min_patches
        self._dyncp_logged = False

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks and CP metadata, shard inputs, and annotate layouts."""
        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions")
        padding_mask = batch.get("padding_mask", None)
        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(
                inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
            ):
                batch["attention_masks"] = self.get_attention_masks(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        pixel_values = batch.get("pixel_values")
        grid_thw = batch.get("grid_thw")
        special_tokens = batch.get("special_tokens")
        # Under a pipeline every stage sees the batch; only the stage holding the
        # tower (and the embedding it feeds) builds the vision bank.
        if (
            parallel_dims.cp_enabled
            and pixel_values is not None
            and self.vision_encoder is not None
        ):
            if grid_thw is None:
                raise ValueError(
                    "pixel_values were provided but grid_thw was not provided."
                )
            if special_tokens is None or "image_id" not in special_tokens:
                raise ValueError(
                    "pixel_values require special_tokens with an 'image_id' entry."
                )
            if self.tok_embeddings is not None:
                batch["vision_bank_indices_T"] = build_vision_bank_indices(
                    batch["input"],
                    placeholder_id=special_tokens["image_id"],
                )
            batch.pop("special_tokens")

        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
        }
        input_sharding["vision_bank_indices_T"] = token_id_placement()
        if parallel_dims.cp_enabled:
            batch, load_balancer = prepare_context_parallel_batch(
                batch,
                input_shardings=input_sharding,
                cp_mesh=parallel_dims.get_mesh("cp"),
                load_balancer_config=parallelism.context_parallel_load_balancer,
                ptrr_mask_key=parallelism.context_parallel_ptrr_mask_key,
            )
            batch = self._prepare_context_parallel_metadata(batch, load_balancer)

        batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> KimiK3AttentionMaskDict:
        attn_config = self.config.first_attention

        kda_metadata = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )

        if attn_config is None:
            quadratic_attention = None
        elif isinstance(attn_config.inner_attention, VarlenInnerAttention.Config):
            # Under varlen both consumers read the same document offsets.
            quadratic_attention = kda_metadata
        else:
            quadratic_attention = super().get_attention_masks(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
        return {
            "quadratic_attention": quadratic_attention,  # pyrefly: ignore [bad-assignment]
            "kda": kda_metadata,
        }

    def encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """The tower's forward on one micro-batch's images.

        The forward calls this; the pipeline's vision run-ahead calls it too,
        for a later micro-batch, so the two cannot drift. Under context
        parallelism the large images are partitioned across the ranks of a
        sub-CP group (report sec 5.2.3), the rest are encoded replicated.
        """
        assert self.vision_encoder is not None
        group_all = getattr(self, "_cp_group_all", None)
        subgroups = getattr(self, "_cp_subgroups", None)
        if group_all is None or not subgroups:
            pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
            return self.vision_encoder(pixel_values, grid_thw=grid_thw)
        return self._encode_images_partitioned(
            pixel_values, grid_thw, group_all, subgroups
        )

    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group; a replicated DTensor issues no all-gather to match,
        so the test is on the placement, not the type.
        """
        assert self.vision_encoder is not None
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_encoder.parameters()
        )

    def _tower_placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The smallest input the tower accepts, for a rank with no images."""
        assert self.vision_encoder is not None
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        device = next(self.parameters()).device
        grid = torch.tensor([[1, kernel_h, kernel_w]], dtype=torch.long, device=device)
        weight = self.vision_encoder.patch_embed.weight
        # A plain tensor: once FSDP has sharded the tower the weight is a
        # DTensor, and a placeholder inheriting that meets the tower's own
        # plain tensors as a mixed matmul.
        patches = torch.zeros(
            kernel_h * kernel_w, weight.shape[-1], dtype=weight.dtype, device=device
        )
        return patches, grid

    def _encode_images_partitioned(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        group_all: dist.ProcessGroup,
        subgroups: dict[int, dist.ProcessGroup],
    ) -> torch.Tensor:
        """Encode every image, partitioning the large ones (report sec 5.2.3).

        Every large image is encoded by one sub-CP group, its patches split
        across that sub-group's ranks with k/v gathered inside the group;
        images below the threshold, or whose grid height does not divide the
        merge kernel, stay whole and are encoded replicated.
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

        assert self.vision_encoder is not None
        encoder = self.vision_encoder
        weight_dtype = encoder.patch_embed.weight.dtype
        grids = grid_thw.tolist()
        counts = [t * h * w for t, h, w in grids]
        kh, kw = encoder.merge_kernel_size
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)

        def _replicated(which: list[int]) -> dict[int, torch.Tensor]:
            out = {}
            for i in which:
                item = pixel_values[offsets[i] : offsets[i + 1]].to(weight_dtype)
                item_grid = grid_thw[i : i + 1]
                out[i] = encoder(item, grid_thw=item_grid)
            return out

        def _all_replicated() -> torch.Tensor:
            out = _replicated(list(range(len(counts))))
            return torch.cat([out[i] for i in range(len(counts))], dim=0)

        cp_size = dist.get_world_size(group_all)
        if cp_size <= 1:
            return _all_replicated()
        large = classify(counts, cp_size, min_patches=self.dynamic_cp_min_patches)
        # A grid height that does not divide the merge kernel cannot be cut
        # safely; such an image stays replicated.
        large = [i for i in large if grids[i][1] % kh == 0]
        if not large:
            return _all_replicated()
        n_sub, g = subgroup_layout(len(large), cp_size)
        group = subgroups.get(n_sub)
        if group is None or g <= 1:
            return _all_replicated()

        cp_rank = dist.get_rank(group_all)
        my_sub = cp_rank // g
        rank_in_sub = cp_rank % g
        group_of = balance_images([counts[i] for i in large], n_sub)
        my_large = [
            img for img, sub in zip(large, group_of, strict=True) if sub == my_sub
        ]
        if not self._dyncp_logged:
            self._dyncp_logged = True
            logger.info(
                "Dynamic CP: %d large image(s) of %d over %d sub-CP group(s) of "
                "%d rank(s); min_patches=%d.",
                len(large),
                len(counts),
                n_sub,
                g,
                self.dynamic_cp_min_patches,
            )

        out: dict[int, torch.Tensor] = {}
        # Every sub-group runs the same number of passes, or the collectives
        # inside them desynchronise; a sub-group with fewer images pads with
        # an empty pass whose output is discarded.
        per_sub = [sum(1 for s in group_of if s == k) for k in range(n_sub)]
        n_passes = max(per_sub) if per_sub else 0
        for p in range(n_passes):
            img = my_large[p] if p < len(my_large) else None
            if img is None:
                # Derived from the batch's own tensors so they carry its SPMD
                # types; a fresh tensor reads as replicated on every axis.
                local = pixel_values[: kh * kw] * 0
                local_grid = grid_thw[:1].clone()
                local_grid[0, 0], local_grid[0, 1], local_grid[0, 2] = 1, kh, kw
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
                if bands != sorted(bands, reverse=True):
                    raise AssertionError(
                        f"bands {bands} are not non-increasing; padding would land "
                        "inside the gathered token stream"
                    )
                flat = pixel_values[offsets[img] : offsets[img + 1]]
                # This rank's rows of every frame: the projector's temporal
                # mean spans all frames.
                pad_rows = band - (sh.row_end - sh.row_start)
                pieces = []
                for a, b in sh.ranges:
                    pieces.append(flat[a:b])
                    if pad_rows:
                        pieces.append(flat[: pad_rows * w] * 0)
                local = torch.cat(pieces, dim=0)
                local_grid = grid_thw[img : img + 1].clone()
                local_grid[0, 1] = band
                plan = CPPatchPlan(
                    group=group,
                    valid_total=counts[img],
                    full_grid=(t, h, w),
                    row_start=sh.row_start,
                    band=band,
                    real_rows=sh.row_end - sh.row_start,
                )
            feats = encoder(local.to(weight_dtype), grid_thw=local_grid, cp_plan=plan)
            if isinstance(feats, DTensor):
                feats = feats.to_local()
            local_feat = _PlainGradBoundary.apply(feats)
            # The boundary on the output too: the gradient arrives from
            # downstream, and the gather's transpose must not see a DTensor.
            gathered = _PlainGradBoundary.apply(
                funcol.all_gather_tensor(
                    local_feat.contiguous(), gather_dim=0, group=group
                )
            )
            if img is not None:
                t, h, w = grids[img]
                # The projector collapses time: a video's token count carries no t.
                out[img] = gathered[: merged_tokens(h, w, kh, kw)]
        rest = [i for i in range(len(counts)) if i not in out]
        if rest:
            out.update(_replicated(rest))
        return torch.cat([out[i] for i in range(len(counts))], dim=0)

    def _vision_stream(self) -> torch.cuda.Stream | None:
        """A side stream for the tower, created once, only outside autograd.

        With gradients recorded, several micro-batches' tower backwards would
        accumulate into the tower's parameters from two streams with nothing
        ordering them, so the run-ahead stays on the current stream then.
        """
        if not torch.cuda.is_available() or torch.is_grad_enabled():
            return None
        stream = getattr(self, "_vision_side_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self._vision_side_stream = stream
        return stream

    def _issue_on_vision_stream(self, fn, *tensors):
        """Run ``fn`` on the vision stream; return ``(out, event)`` without waiting.

        The side stream waits for the current one, whose products ``fn`` reads,
        and each input is ``record_stream``'d so the allocator does not hand its
        memory on while the side stream still reads it.
        """
        side = self._vision_stream()
        if side is None:
            return fn(), None
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(side)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            started.record(side)
            out = fn()
            finished.record(side)
        self._last_encode_span = (started, finished)
        return out, finished

    def _join_vision_stream(self, out, done) -> None:
        """Make the current stream wait for an issued encode and take its outputs."""
        if done is None:
            return
        current = torch.cuda.current_stream()
        current.wait_event(done)
        outs = out if isinstance(out, (list, tuple)) else [out]
        for t in outs:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(current)

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        vision_bank_indices_T: torch.Tensor | None,
        vision_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            # An image-free batch is normal, but FSDP2 issues the tower's
            # all-gather from its pre-forward hook, so every rank must run it:
            # a zero-valued placeholder keeps the collectives and the DP average.
            if self.vision_encoder is not None and self._tower_needs_collectives():
                placeholder, placeholder_grid = self._tower_placeholder()
                unused = self.vision_encoder(placeholder, grid_thw=placeholder_grid)
                if isinstance(unused, DTensor):
                    unused = unused.to_local()
                return add_zero_valued_dependency(embeddings_TD, unused)
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")

        if vision_embeds is None:
            vision_embeds = self.encode_images(pixel_values, grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        if vision_bank_indices_T is not None:
            return gather_vision_embeds(
                embeddings_TD,
                vision_bank_VD=vision_embeds,
                vision_bank_indices_T=vision_bank_indices_T,
            )
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")
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

    def forward(  # pyrefly: ignore[bad-param-name-override, bad-override]
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
        attention_masks: KimiK3AttentionMaskDict | None = None,
        padding_mask: torch.Tensor | None = None,
        kda_cp_routing: "ContextParallelRouting | None" = None,
        vision_bank_indices_T: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        block_residual_in = block_residual_TND

        if self.tok_embeddings is not None:
            with spmd_local_context("dp"):
                # ``vision_embeds`` are the tower's features when the pipeline
                # encoded this micro-batch ahead of time (DEP); otherwise the
                # tower runs here.
                h_TD = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    special_tokens=special_tokens,
                    vision_bank_indices_T=vision_bank_indices_T,
                    vision_embeds=vision_embeds,
                )
        else:
            h_TD = tokens

        if spmd.is_type_checking():
            # Vision fusion runs on a DP-local mesh. Restore the token layout
            # before constructing and propagating the attention residual state.
            # Tokens shard on DP and CP; the TP layout here is the
            # declarations' (replicated for the vision scatter under TP).
            spmd.assert_type(
                h_TD,
                spmd.SpmdType(
                    {"dp": spmd.V, "cp": spmd.V},
                    partition_spec=spmd.PartitionSpec(("dp", "cp"), None),
                ),
            )

        block_residual_TND = (
            block_residual_in
            if block_residual_in is not None
            else h_TD.unsqueeze(1)[:, :0]
        )
        for layer in self.layers.values():
            h_TD, block_residual_TND = layer(
                h_TD,
                block_residual_TND,
                attention_masks,
                positions,
                padding_mask=padding_mask,
                cu_seqlens=cu_seqlens,
                kda_cp_routing=kda_cp_routing,
            )

        # The aggregation runs on the head-owning stage; other stages hand the stack on.
        if self.output_res_proj is None:
            return h_TD, block_residual_TND
        h_TD = _checkpointed_attention_residual(
            "output_res",
            h_TD,
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        )
        h_pre_norm_TD = h_TD
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self.mtp_layers is not None and self.lm_head is not None:
            if self._skip_lm_head:
                # Raised rather than skipped: skipping would leave
                # take_mtp_logits() empty and the run would LOOK like it
                # trains MTP while it does not. MTP needs full-vocab logits
                # per depth, which is exactly the allocation chunked loss
                # exists to avoid; combining them means per-chunk MTP logits,
                # a change to the loss, not a guard here.
                raise ValueError(
                    "MTP and chunked loss cannot be combined yet: use a "
                    "non-chunked loss for MTP flavors."
                )
            put_mtp_logits(self._compute_mtp_logits(tokens, h_pre_norm_TD))
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD

    def _compute_mtp_logits(
        self, tokens: torch.Tensor, h_pre_norm_TD: torch.Tensor
    ) -> list[torch.Tensor]:
        """Logits for each MTP depth; depth k predicts the token k+1 ahead.

        The depth-k input fuses the backbone's final PRE-norm hidden state
        (the reference feeds hnorm the unnormalised state; normalising twice
        is not an identity and breaks parity against official MTP weights)
        with the embedding of the token k+1 ahead. The last k+1 positions
        have no target and are dropped rather than padded -- padding would
        invent supervision. The folded stream has one token axis, hence the
        [T]-shaped slicing; the multimodal splice is length-preserving here,
        so shift-by-k stays aligned and the visual positions already carry
        IGNORE_INDEX in the labels.
        """
        if self.tok_embeddings is None:
            raise RuntimeError(
                "MTP needs tok_embeddings and lm_head together, and "
                "tok_embeddings is None -- PP has split them across stages. "
                "Keep the embedding and the head on one stage for MTP."
            )
        assert self.mtp_layers is not None and self.lm_head is not None
        out = []
        for k in range(len(self.mtp_layers)):
            shift = k + 1
            ahead_T = tokens[shift:]
            emb_TD = self.tok_embeddings(ahead_T)
            h_TD = h_pre_norm_TD[: ahead_T.shape[0]]
            hidden_TD = self.mtp_layers[str(k)](h_TD, emb_TD)
            if self.norm is not None:
                hidden_TD = self.norm(hidden_TD)
            out.append(self.lm_head(hidden_TD))
        return out
