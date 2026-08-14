# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configurations for Kimi K3."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer, MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry


def kimi_k3_up_debugmodel() -> Trainer.Config:
    """Return the topology-complete Kimi K3 eager/FSDP2 debug config."""
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
        dataloader=MMDataLoader.Config(
            dataset="cc12m-test",
            max_images_per_batch=8,
            patch_size=14,
            temporal_patch_size=1,
            spatial_merge_size=2,
            patch_order="raster",
            resize_fn=resize_to_patch_budget,
            min_pixels=56 * 56,
            max_pixels=224 * 224,
            max_patches=256,
            max_patches_per_side=16,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=256,
            steps=10,
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
    )


def kimi_k3_up_mini_block_attn_res() -> Trainer.Config:
    """The text screening flavor on the upstream model -- migration step 2's gate.

    Training settings are our text arm's, not their debugmodel's: seq_len 4096 in
    float32, found by sweep because 8192 does not fit 16 GB and 2048 trips fla's
    SM120 shared-memory ceiling. Keeping them means a cell here is comparable to
    the same cell on our model rather than only to itself.
    """
    model_spec = model_registry("mini_block_attn_res")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=10,
            dtype="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=None,
    )
