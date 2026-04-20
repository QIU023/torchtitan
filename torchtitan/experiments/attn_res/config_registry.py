# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the Block AttnRes experiment.

Exposes two sibling configs that share every hyperparameter EXCEPT the model
flavor so the only measurable delta is Block AttnRes itself:

- ``llama3_150m_baseline``: plain ~150M Llama3 dense, standard residuals.
- ``llama3_150m_attn_res``: same shape, Block AttnRes enabled (N=6 blocks).
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.attn_res import model_registry as attn_res_model_registry
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_plain_llama3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
) -> list[TransformerBlock.Config]:
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=ScaledDotProductAttention.Config(),
                    mask_type="causal",
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _llama3_150m_plain_config() -> Llama3Model.Config:
    """Plain ~150M Llama3 dense config (the baseline for AttnRes comparison)."""
    dim = 768
    n_heads = 12
    n_kv_heads = 4
    n_layers = 12
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_plain_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.0
            ),
        ),
    )


def _baseline_model_registry() -> ModelSpec:
    return ModelSpec(
        name="llama3",
        flavor="150M",
        model=_llama3_150m_plain_config(),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def llama3_150m_baseline() -> Trainer.Config:
    """Phase 2 reference run: ~150M dense, standard residuals.

    Paired with ``llama3_150m_attn_res`` for loss-curve alignment. The two
    configs must share every hyperparameter EXCEPT the model flavor so that
    the only difference in the measured loss delta is Block AttnRes itself.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiling=ProfilingConfig(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
            log_freq=10,
        ),
        model_spec=_baseline_model_registry(),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=500,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            seq_len=2048,
            steps=20000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        checkpoint=CheckpointManager.Config(
            # Enable so a mid-run crash (e.g. HF datasets httpx
            # disconnect during C4 streaming) doesn't force a full restart.
            # keep_latest_k=3 bounds disk use at ~3x the model size.
            enable=True,
            interval=1000,
            keep_latest_k=3,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        validator=Validator.Config(freq=500, steps=50),
    )


def llama3_150m_attn_res() -> Trainer.Config:
    """Phase 2 reference run: ~150M dense, Block AttnRes enabled.

    Identical to ``llama3_150m_baseline`` except for the model flavor, so
    the only source of loss-delta is Block AttnRes.
    """
    config = llama3_150m_baseline()
    config.model_spec = attn_res_model_registry("150M_attn_res")
    return config


def _llama3_150m_attn_res_variant(flavor: str) -> Trainer.Config:
    """Helper: baseline Trainer config + a specific attn_res model flavor.

    Used to build num_blocks ablation runs that share every hyperparameter
    with the primary ``llama3_150m_attn_res`` except ``num_blocks``.
    """
    config = llama3_150m_baseline()
    config.model_spec = attn_res_model_registry(flavor)
    return config


def llama3_150m_attn_res_n2() -> Trainer.Config:
    """Ablation: Block AttnRes with N=2 (6 layers per block).

    Tests the low-N end of the paper's "N=2,4,8 roughly equal" claim.
    """
    return _llama3_150m_attn_res_variant("150M_attn_res_n2")


def llama3_150m_attn_res_n3() -> Trainer.Config:
    """Ablation: Block AttnRes with N=3 (4 layers per block)."""
    return _llama3_150m_attn_res_variant("150M_attn_res_n3")


def llama3_150m_attn_res_n4() -> Trainer.Config:
    """Ablation: Block AttnRes with N=4 (3 layers per block)."""
    return _llama3_150m_attn_res_variant("150M_attn_res_n4")


def llama3_150m_attn_res_n12() -> Trainer.Config:
    """Ablation: Block AttnRes with N=12 (1 layer per block).

    Maximum attention granularity at this model size. Tests the
    paper's high-N degradation claim (paper observed N>=16 degrades;
    N=12 is the largest divisor of n_layers=12 available here).
    """
    return _llama3_150m_attn_res_variant("150M_attn_res_n12")


def llama3_150m_attn_res_L16_n8() -> Trainer.Config:
    """16-layer / N=8 variant sized for the Phase-3 8-GPU PP layout.

    Used for the Phase 3 naive-vs-adapter PP smoke on 8 GPUs. The
    launchers (phase3/launch_8gpu_{naive,adapter}.sh) pass PP=8,
    schedule=Interleaved1F1B, layers_per_stage=1, and
    first/last_stage_less_layers=0, which gives:
        (n_layers=16 + first_less=0 + last_less=0) / layers_per_stage=1
        = 16 virtual stages / PP=8 = 2 chunks per rank.
    Two chunks per rank is the minimum Interleaved1F1B requires and is
    what preserves the steady-state overlap the Phase-3 measurement
    relies on (LPS=2 collapses to 1 chunk/rank and loses that). With
    num_blocks=8, every other virtual-stage boundary coincides with a
    block boundary, so the cross-stage caching adapter's "send only
    new blocks" invariant is exercised at half the stage transitions.
    Shares every other hyperparameter with the 12-layer configs so the
    sweep stays apples-to-apples when compared to Phase 2.
    """
    return _llama3_150m_attn_res_variant("150M_attn_res_L16_n8")
