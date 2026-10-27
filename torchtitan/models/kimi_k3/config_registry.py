# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry


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


def kimi_k3_debugmodel_qb() -> Trainer.Config:
    """The debug model with quantile balancing on the MoE router bias.

    K3 runs sparsity (report sec 2.3) beyond where the auxiliary-loss-free
    bias NUDGE still balances; quantile balancing SOLVES for the bias from
    the accumulated per-expert load histograms instead. The hook replaces
    the default load-balancing hook on the optimizer.
    """
    from torchtitan.components.quantile_balance import register_quantile_balancing

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.post_optimizer_build_fn = register_quantile_balancing
    return config


# PROBE ONLY (not committed): the two hooks with per-step expert-load logging.
def kimi_k3_debugmodel_probe() -> Trainer.Config:
    from torchtitan.components.load_probe import register_load_probe

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    core_hook = config.model_spec.post_optimizer_build_fn  # core's sign-rule registration: the trainer calls ONE fn

    def both(optimizers, model_parts, parallel_dims):
        if core_hook is not None:
            core_hook(optimizers, model_parts, parallel_dims)
        return register_load_probe(optimizers, model_parts, parallel_dims)

    config.model_spec.post_optimizer_build_fn = both
    return config


def kimi_k3_debugmodel_qb_probe() -> Trainer.Config:
    from torchtitan.components.load_probe import register_load_probe
    from torchtitan.components.quantile_balance import register_quantile_balancing

    def both(optimizers, model_parts, parallel_dims):
        # QB's step hook first (registration order), then the probe reads the loads and the solved bias
        balancer = register_quantile_balancing(optimizers, model_parts, parallel_dims)
        return register_load_probe(optimizers, model_parts, parallel_dims, balancer)

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.post_optimizer_build_fn = both
    return config


def kimi_k3_debugmodel_probe_frozen() -> Trainer.Config:  # LOCAL PROBE FLAVOR (not committed): lr 0, the hook alone moves the loads
    config = kimi_k3_debugmodel_probe()
    config.optimizer = default_adamw(lr=0.0)
    return config


def kimi_k3_debugmodel_qb_probe_frozen() -> Trainer.Config:  # LOCAL PROBE FLAVOR (not committed)
    config = kimi_k3_debugmodel_qb_probe()
    config.optimizer = default_adamw(lr=0.0)
    return config
