# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
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
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.observability.metrics import MetricsProcessor
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
        max_patches=256,
        max_patches_per_side=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


# ---------------------------------------------------------------------------
# PROBE ONLY (not committed): the c4 text-row flavor and its pipeline variants.
# ---------------------------------------------------------------------------
_C4_ROW_TOKENS = 256  # one row per 256-token micro-batch

_EIGHT_STAGES = [  # the PR's pp2 x vp4 split (tests/unit_tests/cpu/test_kimi_k3_pp_layout.py)
    ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
    ["layers.2", "layers.3", "layers.4"],
    ["layers.5", "layers.6", "layers.7"],
    ["layers.8", "layers.9"],
    ["layers.10", "layers.11"],
    ["layers.12", "layers.13"],
    ["layers.14", "layers.15"],
    ["layers.16", "norm", "lm_head", "output_res_proj", "output_res_norm"],
]


def _process_c4_text_sample(sample, **kwargs):  # PROBE ONLY (not committed)
    """A c4 doc as one text-only row: its first 256 tokens."""
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    out = _process_mm_sample(texts=[sample["text"][:4000]], images=[None], **kwargs)
    if out is None:
        return None
    n = _C4_ROW_TOKENS - 1
    for key in ("input_ids", "labels", "positions"):
        out[key] = out[key][:n]
    return out


def kimi_k3_debugmodel_c4() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource

    config = kimi_k3_debugmodel()
    # ID CHECK: the optimizer the H100 tables were measured with (the recipe's before #4596).
    from torchtitan.components.optimizer import default_adamw

    config.optimizer = default_adamw(lr=8e-4)
    # Every cell loads the seed checkpoint from <dump_folder>/checkpoint and never saves.
    config.checkpointer = CheckpointManager.Config(interval=100000)
    config.dataloader = _kimi_k3_multimodal_dataloader(
        SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"},
            ),
            processor=MultiModalProcessor.Config(sample_processor=_process_c4_text_sample),
            post_filters=(lambda sample: sample is not None,),
        )
    )
    return config


def _naive(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    import os

    os.environ["ATTN_RES_NAIVE"] = "1"  # read by the probe hook in KimiK3Model.pipeline
    return config


def _eight_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [list(s) for s in _EIGHT_STAGES]
    return config


def _sixteen_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    # ID CHECK: the removed pp4 x vp4 recipe's split, spelled out on both heads.
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [
        ["vision_encoder", "tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3", "layers.4"],
        *[[f"layers.{i}"] for i in range(5, 17)],
        ["norm", "lm_head", "output_res_proj", "output_res_norm"],
    ]
    return config


def kimi_k3_debugmodel_c4_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _eight_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_eight_stages(kimi_k3_debugmodel_c4()))


def kimi_k3_debugmodel_c4_16stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _sixteen_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_16stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_sixteen_stages(kimi_k3_debugmodel_c4()))


def kimi_k3_debugmodel_c4_seed() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.checkpointer import CheckpointManager

    config = kimi_k3_debugmodel_c4()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config
