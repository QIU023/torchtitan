# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields, replace
from typing import cast

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    default_adamw,
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k2_7.config_registry import _per_expert_compute_layout
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, KimiK3Model, model_registry
from .kda import KDA
from .model import KimiMLAAttention


def _kimi_k3_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    processor = dataset.processor
    if not isinstance(processor, MultiModalProcessor.Config):
        raise ValueError("Kimi K3 multimodal data requires MultiModalProcessor.Config")

    processor = MultiModalProcessor.Config(
        sample_processor=processor.sample_processor,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        resize_fn=resize_to_navit_patch_grid,
        min_pixels=56 * 56,
        max_pixels=224 * 224,
        max_patches=256,
        max_patches_per_side=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_batch=8,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=256,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def kimi_k3_debugmodel_lora() -> Trainer.Config:
    """The multimodal debug model with LoRA adapters on the attention output.

    Uses core's LoRAConverter rather than a model-local implementation. The
    Targets are matched on the last segment of the FQN. The set mirrors the
    reference tree's DEFAULT_LORA_TARGETS: the MLA projections, and -- the part
    that matters structurally -- the dense FFN and latent-MoE projections.
    Every decoder layer carries an FFN or MoE, while only one layer in four is
    MLA (K3 is 3 KDA : 1 MLA), so an MLA-only target set leaves an all-KDA
    pipeline stage with zero trainable parameters and the optimizer then raises
    "param_groups pattern matched no parameters". That is what pp8 hit.

    Not covered: the reference also adapts the MLA output gate. Here that module
    is named ``gate``, which is also the router's gate in every MoE layer, and
    last-segment matching cannot separate them -- adding it would silently adapt
    the routers too. Left out rather than guessed.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel", converters=[_kimi_k3_lora_converter()]
    )
    return config


def _kimi_k3_lora_converter(
    *, quantize_base: str | None = None, quantize_experts: str | None = None
):
    from torchtitan.components.lora import LoRAConverter

    return LoRAConverter.Config(
        rank=8,
        alpha=16.0,
        target_modules=[
            # MLA
            "wq_a",
            "wq_b",
            "wkv_a",
            "wkv_b",
            "wo",
            # dense FFN and shared experts
            "w1",
            "w2",
            "w3",
            # latent MoE down/up projections
            "routed_down",
            "routed_up",
        ],
        quantize_base=quantize_base,
        quantize_experts=quantize_experts,
    )


def kimi_k3_debugmodel_qlora_mxfp4_linear() -> Trainer.Config:
    """QLoRA with only the base LINEARS packed (experts stay bf16).

    The packed-TP forward covers colwise/rowwise linears; packed experts
    under expert-TP need a shape-preserving layout and refuse -- this
    flavor is the TP-composable subset.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[_kimi_k3_lora_converter(quantize_base="mxfp4")],
    )
    return config


def kimi_k3_debugmodel_qlora_mxfp4() -> Trainer.Config:
    """The LoRA debug model with MXFP4-packed bases (QLoRA, K3's native
    weight format).

    The packing swaps the base for split storage AT BUILD, before
    parallelize, so FSDP2 shards the packed bytes -- this is the
    pack-then-shard order the nf4 path cannot reach, and the flavor trains
    under the normal sharded flow.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            _kimi_k3_lora_converter(quantize_base="mxfp4", quantize_experts="mxfp4")
        ],
    )
    return config


def kimi_k3_debugmodel_mx_qat() -> Trainer.Config:
    """The debug model under MXFP4-weight / MXFP8-activation fake-quant QAT.

    K3's official quantization scope: the routed experts only, bf16 masters
    training underneath. Fake-quant is bf16 compute, so this runs on any GPU.
    """
    from torchtitan.components.quantization.mx_qat import MXFP4QATConverter

    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel", converters=[MXFP4QATConverter.Config()]
    )
    return config


def kimi_k3_debugmodel_mtp() -> Trainer.Config:
    """The debug model with one multi-token-prediction layer (report sec 3.3).

    Plain (non-chunked) cross entropy: MTP needs full-vocab logits per depth,
    which is exactly the allocation chunked loss exists to avoid -- the model
    raises on the combination rather than silently skipping depths.
    """
    from torchtitan.models.kimi_k3.mtp import KimiMTPLoss

    config = kimi_k3_debugmodel()
    config.model_spec = model_registry("debugmodel", num_mtp_layers=1)
    config.loss = KimiMTPLoss.Config(
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(config.model_spec),
        ),
    )
    return config


def _dist_muon_optimizer(
    model_spec: ModelSpec,
    *,
    lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """DistMuon on the matrix parameters of Kimi K3, AdamW on the rest.

    Muon orthogonalises per head where the weight stacks heads along its
    output dimension: the MLA query and key-value up-projections, and the
    KDA query, key and value projections. Routed experts are batch-first
    stacks of matrices and orthogonalise per expert. Everything else that is
    a plain matrix (down-projections, gates, the latent projections, shared
    experts, the router gate, dense feed-forward) is owned whole. Norms,
    biases, the KDA convolutions and gate parameters, the one-row residual
    projections, embeddings, the LM head and the vision tower stay on AdamW.
    """
    model_config = cast(KimiK3Model.Config, model_spec.model)
    owned = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
    )
    per_expert = _per_expert_compute_layout(parallelism)
    expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")

    def per_head(block_size: int) -> ComputeLayout:
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(dim=0, block_size=block_size)
            },
        )

    def compute_shardings_for_layer(layer_id: int) -> dict[str, ComputeLayout]:
        layer = model_config.layers[layer_id]
        prefix = f"layers.{layer_id}"
        shardings: dict[str, ComputeLayout] = {}
        if layer.attention is not None:
            attention = cast(KimiMLAAttention.Config, layer.attention)
            shardings.update(
                {
                    f"{prefix}.attention.wq_a.weight": owned,
                    f"{prefix}.attention.wq_b.weight": per_head(
                        attention.qk_nope_head_dim + attention.qk_rope_head_dim
                    ),
                    f"{prefix}.attention.wkv_a.weight": owned,
                    f"{prefix}.attention.wkv_b.weight": per_head(
                        attention.qk_nope_head_dim + attention.v_head_dim
                    ),
                    f"{prefix}.attention.wo.weight": owned,
                    f"{prefix}.attention.gate.weight": owned,
                }
            )
        if layer.delta_attention is not None:
            kda = cast(KDA.Config, layer.delta_attention)
            shardings.update(
                {
                    f"{prefix}.delta_attention.{projection}.weight": per_head(
                        kda.head_dim
                    )
                    for projection in ("q_proj", "k_proj", "v_proj")
                }
            )
            shardings.update(
                {
                    f"{prefix}.delta_attention.{projection}.weight": owned
                    for projection in (
                        "output_gate",
                        "output_proj",
                        "beta",
                        "forget_a",
                        "forget_b",
                    )
                }
            )
        if layer.feed_forward is not None:
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": owned
                    for projection in ("w1", "w2", "w3")
                }
            )
        if layer.moe is not None:
            shardings.update(
                {
                    f"{prefix}.moe.routed_experts.inner_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.moe.router.gate.weight"] = owned
            shardings[f"{prefix}.moe.routed_down.weight"] = owned
            shardings[f"{prefix}.moe.routed_up.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.moe.shared_experts.{projection}.weight": owned
                    for projection in ("w1", "w2", "w3")
                }
            )
        return shardings

    num_layers = len(model_config.layers)
    compute_sharding_by_fqn_per_layer = tuple(
        compute_shardings_for_layer(layer_id) for layer_id in range(num_layers)
    )
    compute_sharding_by_fqn = {
        fqn: compute_sharding
        for layer_compute_sharding_by_fqn in compute_sharding_by_fqn_per_layer
        for fqn, compute_sharding in layer_compute_sharding_by_fqn.items()
    }
    # One bucket per pair of layers, the routed experts of the pair in a
    # bucket of their own, as the Kimi K2.5 recipe does.
    bucket_configs_list: list[BucketConfig] = []
    for first_layer_id in range(0, num_layers, 2):
        layer_ids = tuple(range(first_layer_id, min(first_layer_id + 2, num_layers)))
        non_routed = tuple(
            fqn
            for layer_id in layer_ids
            for fqn in compute_sharding_by_fqn_per_layer[layer_id]
            if ".moe.routed_experts.inner_experts." not in fqn
        )
        routed = tuple(
            fqn
            for layer_id in layer_ids
            for fqn in compute_sharding_by_fqn_per_layer[layer_id]
            if ".moe.routed_experts.inner_experts." in fqn
        )
        name = f"layers.{layer_ids[0]}-{layer_ids[-1]}"
        if non_routed:
            bucket_configs_list.append(BucketConfig(name=name, patterns=non_routed))
        if routed:
            bucket_configs_list.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed)
            )
    muon_pattern = (
        r"(?:"
        r"attention\.(?:wq_a|wq_b|wkv_a|wkv_b|wo|gate)\.weight|"
        r"delta_attention\.(?:q_proj|k_proj|v_proj|output_gate|output_proj|beta|forget_a|forget_b)\.weight|"
        r"routed_experts\.inner_experts\.(?:w1_EFD|w2_EDF|w3_EFD)|"
        r"feed_forward\.w[123]\.weight|"
        r"moe\.router\.gate\.weight|"
        r"moe\.routed_(?:down|up)\.weight|"
        r"moe\.shared_experts\.w[123]\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        implementation="foreach",
        param_groups=[
            ParamGroupConfig(
                pattern=muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs={
                    "lr": lr,
                    "weight_decay": 0.1,
                    "foreach": False,
                    "adjust_lr_fn": "match_rms_adamw",
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "bucket_configs": tuple(bucket_configs_list),
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
            }
        },
    )


def _align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Rebuild the routed-expert layouts from the final parallelism config.

    The CLI can override ``expert_parallel_degree`` after the registry built
    the layouts, and that degree decides whether routed experts use the 1-D
    ``dp_shard`` layout or the 2-D EP/EFSDP one.
    """
    factory_kwargs_by_name = {
        name: dict(factory_kwargs)
        for name, factory_kwargs in optimizer_config.optimizer_factory_kwargs_by_name.items()
    }
    dist_muon_kwargs = factory_kwargs_by_name.get("DistMuon")
    if dist_muon_kwargs is None:
        return optimizer_config
    compute_sharding_by_fqn = cast(
        dict[str, ComputeLayout], dist_muon_kwargs["compute_sharding_by_fqn"]
    )
    per_expert = _per_expert_compute_layout(parallelism)
    aligned = {
        fqn: (
            per_expert
            if ".moe.routed_experts.inner_experts." in fqn
            else compute_layout
        )
        for fqn, compute_layout in compute_sharding_by_fqn.items()
    }
    if aligned == compute_sharding_by_fqn:
        return optimizer_config
    dist_muon_kwargs["compute_sharding_by_fqn"] = aligned
    return replace(
        optimizer_config, optimizer_factory_kwargs_by_name=factory_kwargs_by_name
    )


@dataclass(kw_only=True, slots=True)
class _KimiK3MuonTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optimizer, parallelism=self.parallelism
        )
        if self.parallelism.tensor_parallel_degree > 1:
            raise ValueError(
                "Kimi K3 DistMuon currently requires tensor_parallel_degree=1: "
                "tensor parallelism can produce unsupported _StridedShard "
                "parameter layouts."
            )


def kimi_k3_debugmodel_muon() -> Trainer.Config:
    """The debug model trained with DistMuon (per-head MLA and KDA layouts)."""
    base = kimi_k3_debugmodel()
    values = {f.name: getattr(base, f.name) for f in fields(base)}
    values["optimizer"] = _dist_muon_optimizer(
        base.model_spec, lr=8e-4, parallelism=base.parallelism
    )
    return _KimiK3MuonTrainerConfig(**values)
