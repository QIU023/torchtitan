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
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
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
    model_spec = model_registry("debugmodel", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=1 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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
def kimi_k3_debugmodel_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3  # pp_review3 round 2: the entry lives in parallelize.py

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache=False
    )
    return config
def kimi_k3_debugmodel_cc12m() -> Trainer.Config:  # PROBE ONLY (not committed)
    """The debug model on the streamed cc12m (no sample repeats in 100 steps; the 32-sample
    test set is memorized by step 90)."""
    config = kimi_k3_debugmodel()
    config.dataloader = _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m"])
    return config


def kimi_k3_debugmodel_cc12m_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3

    config = kimi_k3_debugmodel_cc12m()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache=False
    )
    return config


# One row per micro-batch: set C4_ROW_TOKENS to the micro-batch size (64 for #4500's
# 256 tokens per step as four micro-batches). PROBE ONLY (not committed)
_C4_ROW_TOKENS = int(__import__("os").environ.get("C4_ROW_TOKENS", "256"))


def _process_c4_text_sample(sample, **kwargs):  # PROBE ONLY (not committed)
    """A c4 doc as one text-only row: its first _C4_ROW_TOKENS tokens (the multimodal
    batcher packs whole rows into a micro-batch, so a row never spans two)."""
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    # Cut the text first so the processor's context-length skip keeps long docs.
    out = _process_mm_sample(texts=[sample["text"][:4000]], images=[None], **kwargs)
    if out is None:
        return None
    n = _C4_ROW_TOKENS - 1
    for key in ("input_ids", "labels", "positions"):
        out[key] = out[key][:n]
    return out


def kimi_k3_debugmodel_c4() -> Trainer.Config:  # PROBE ONLY (not committed)
    """The debug model on c4_test as text-only rows, each doc's first 256 tokens:
    2000 rows, about 0.5M tokens, so a 100-step run at 1024-2048 tokens per step
    reads 21-43% of it once; the 32-sample cc12m test set is memorised by step 20."""
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource

    config = kimi_k3_debugmodel()
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


def kimi_k3_debugmodel_c4_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3

    config = kimi_k3_debugmodel_c4()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache=False
    )
    return config
