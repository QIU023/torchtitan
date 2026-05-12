# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the Block AttnRes experiment.

Two flavor families are registered here:

1. Dense + GQA (Llama3-shape) — the single-GPU A/B reference:
   - ``llama3_175m_baseline``: plain ~175M Llama3 dense, standard residuals.
   - ``llama3_175m_attn_res``: same shape, Block AttnRes enabled
     (N=6 blocks).
   - Plus N-ablation variants.

2. MoE + MLA (DeepSeek-V3 shape) — the production-adjacent target:
   - ``dsv3_attn_res_debugmodel``: small MoE debug (6 layers, 8 experts,
     N=3). CPU / single-GPU smoke.
   - ``dsv3_attn_res_16b``: ~16B MoE + MLA + AttnRes (N=9, 3 layers per
     block). The A/B baseline for this is upstream
     ``--module deepseek_v3 --config deepseek_v3_16b``; every hyperparameter
     matches that config so the only measured delta is AttnRes.
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
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
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


def _llama3_175m_plain_config() -> Llama3Model.Config:
    """Plain ~175M Llama3 dense config (the baseline for AttnRes comparison)."""
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


def _llama3_175m_plain_L16_config() -> Llama3Model.Config:
    """16-layer plain Llama3 dense config; shape-matched to llama3_175m_attn_res_L16_n8
    minus the AttnRes pseudo-queries and norms.

    Exists so the 4-GPU PP=4 V=2 + layers_per_stage=2 configuration can be
    run as a no-AttnRes baseline under the same ``--module attn_res``
    machinery (no per-model-family launcher duplication) and against the
    same PP slicing as the AttnRes variant. Every other architectural
    field (dim, heads, kv_heads, FFN hidden dim, RoPE, vocab, tying) is
    kept bit-identical to ``_175m_attn_res(n_layers=16,
    enable_weight_tying=False)`` so the only difference in a matched
    A/B is Block AttnRes itself.

    Weight tying is False (required under PP; torchtitan's
    ``parallelize_llama`` raises on tying + PP). ``_EMBEDDING_SKIP_INIT``
    is preserved — nn.Embedding's own ``reset_parameters`` still runs in
    the constructor and leaves the weight at ``N(0, 1)``; the
    experiment-level init just opts out of re-initializing it.
    """
    dim = 768
    n_heads = 12
    n_kv_heads = 4
    n_layers = 16
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=False,
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
        flavor="175M",
        model=_llama3_175m_plain_config(),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def _baseline_L16_model_registry() -> ModelSpec:
    """ModelSpec for the L16 plain baseline. Uses core pipeline_llm (no
    cross-stage caching adapter -- baseline has no AttnRes blocks to
    cache), parallelize_llama, and the stock Llama3 state-dict adapter.
    """
    return ModelSpec(
        name="llama3",
        flavor="175M_L16",
        model=_llama3_175m_plain_L16_config(),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def llama3_175m_baseline() -> Trainer.Config:
    """Phase 2 reference run: ~175M dense, standard residuals.

    Paired with ``llama3_175m_attn_res`` for loss-curve alignment. The two
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
            # keep_latest_k=2 bounds disk use at ~2x the model size.
            enable=True,
            interval=1000,
            keep_latest_k=2,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        validator=Validator.Config(freq=500, steps=50),
    )


def llama3_175m_attn_res() -> Trainer.Config:
    """Phase 2 reference run: ~175M dense, Block AttnRes enabled.

    Identical to ``llama3_175m_baseline`` except for the model flavor, so
    the only source of loss-delta is Block AttnRes.
    """
    config = llama3_175m_baseline()
    config.model_spec = attn_res_model_registry("175M_attn_res")
    return config


def _llama3_175m_attn_res_variant(flavor: str) -> Trainer.Config:
    """Helper: baseline Trainer config + a specific attn_res model flavor.

    Used to build num_blocks ablation runs that share every hyperparameter
    with the primary ``llama3_175m_attn_res`` except ``num_blocks``.
    """
    config = llama3_175m_baseline()
    config.model_spec = attn_res_model_registry(flavor)
    return config


def llama3_175m_attn_res_n2() -> Trainer.Config:
    """Ablation: Block AttnRes with N=2 (6 layers per block).

    Tests the low-N end of the paper's "N=2,4,8 roughly equal" claim.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_n2")


def llama3_175m_attn_res_n3() -> Trainer.Config:
    """Ablation: Block AttnRes with N=3 (4 layers per block)."""
    return _llama3_175m_attn_res_variant("175M_attn_res_n3")


def llama3_175m_attn_res_n4() -> Trainer.Config:
    """Ablation: Block AttnRes with N=4 (3 layers per block)."""
    return _llama3_175m_attn_res_variant("175M_attn_res_n4")


def llama3_175m_attn_res_n12() -> Trainer.Config:
    """Ablation: Block AttnRes with N=12 (1 layer per block).

    Maximum attention granularity at this model size. Tests the
    paper's high-N degradation claim (paper observed N>=16 degrades;
    N=12 is the largest divisor of n_layers=12 available here).
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_n12")


def llama3_175m_attn_res_L16_n8() -> Trainer.Config:
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
    return _llama3_175m_attn_res_variant("175M_attn_res_L16_n8")


def llama3_175m_attn_res_L32_n8() -> Trainer.Config:
    """32-layer / N=8 (4 layers/block) carrier for aggressive PP×VP sweeps.

    Used by phase3/run_pp_pressure_test.sh for the PP=8 × VP=4 stress
    test (needs num_layers >= PP * VP = 32 to satisfy Interleaved1F1B's
    one-chunk-per-stage minimum). Also supports PP=4 × VP=8 (same 32
    chunks total but more aggressive VP).

    Same hyperparameters (dim=768, n_heads=12, n_kv_heads=4, FFN hidden
    via Llama3 SwiGLU formula) as the L16 variant — depth is the only
    delta — so adapter-vs-naive numerics comparison stays apples-to-
    apples within the deeper-carrier family.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L32_n8")


def llama3_175m_attn_res_L16_n16() -> Trainer.Config:
    """L=16 Full AttnRes (N = n_layers). Every transformer-block is
    its own AttnRes-block. Apples-to-apples vs L16_n8 (Block AttnRes,
    2 layers/AttnRes-block) — only the AttnRes geometry differs.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L16_n16")


def llama3_175m_attn_res_L24_n2() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n2")


def llama3_175m_attn_res_L24_n3() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n3")


def llama3_175m_attn_res_L24_n4() -> Trainer.Config:
    """L=24 Block AttnRes with N=4 (6 transformer-blocks per AttnRes-block).

    First L=24 variant tested. dim=768 smoke 50 steps showed inf-grad
    from step 1 (intra-block residual chain S=6 too deep). Sweep at L=24
    N ∈ {2,3,4,6,8,12,24} explores the stability threshold against
    block-size S = L/N — see phase3/PRESSURE_TEST_REPORT_2026-05-12.md.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n4")


def llama3_175m_attn_res_L24_n6() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n6")


def llama3_175m_attn_res_L24_n8() -> Trainer.Config:
    """L=24 N=8: 3 transformer-blocks per AttnRes-block — paper sweet
    spot (matches Kimi 48B's 27/9 = 3 t-blocks/AttnRes-block ratio).
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n8")


def llama3_175m_attn_res_L24_n12() -> Trainer.Config:
    """L=24 N=12: 2 transformer-blocks per AttnRes-block — same ratio
    as proven-stable L16_n8.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n12")


def llama3_175m_attn_res_L24_n24() -> Trainer.Config:
    """L=24 Full AttnRes (N = n_layers, 1 t-block per AttnRes-block).
    Stability upper bound: every layer's residual is a bounded softmax
    mean over preceding sources.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L24_n24")


# Widen-dim carriers for L=32 N=8 Block AttnRes — finding the dim
# threshold where random-init forward stays bf16-finite. All four share
# n_layers=32 num_blocks=8 (4 t-blocks/AttnRes-block, paper sweet spot
# × 1.33), only dim differs.

def llama3_attn_res_L32_n8_d1024() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d1024")


def llama3_attn_res_L32_n8_d1280() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d1280")


def llama3_attn_res_L32_n8_d1536() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d1536")


def llama3_attn_res_L32_n8_d2048() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d2048")


def llama3_175m_attn_res_L32_n16() -> Trainer.Config:
    """L=32 N=16 = 2 transformer-blocks per AttnRes-block.

    Same intra-block residual-chain length as proven-stable L16_n8.
    Tests hypothesis that t-blocks/AttnRes-block is the stability
    driver. Allows PP=8 × VP=4 = 32 chunks at dim=768.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L32_n16")


def llama3_attn_res_L32_n8_d2048_uniform() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d2048_uniform")


def llama3_attn_res_L32_n8_d1280_uniform() -> Trainer.Config:
    return _llama3_175m_attn_res_variant("attn_res_L32_n8_d1280_uniform")


def llama3_175m_attn_res_L32_n32() -> Trainer.Config:
    """L=32 Full AttnRes (N = n_layers). Canonical pair for PP=8 × VP=4
    pressure: 32 chunks × 1 transformer-block per chunk, one AttnRes
    emit per chunk. Worst-case wire bytes for naive (stack grows to
    33 sources at deepest stage), best-case adapter savings.

    Stability: at L=32 standard residual is unstable in bf16 (see L32_n8
    inf-grad notes). Full AttnRes replaces every accumulation with a
    softmax mean over preceding sources; at zero-init pseudo-queries,
    softmax is uniform, output is bounded by max-source-magnitude.
    Expected to remove the L≥32 inf-grad failure mode.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L32_n32")


def llama3_175m_attn_res_L48_n8() -> Trainer.Config:
    """48-layer / N=8 (6 layers/block) carrier — deepest pressure-test
    carrier supported, for PP=8 × VP=6 or PP=4 × VP=12.

    Approaches Llama 3.1 8B's 32-layer depth × 2.4 (or matches 70B's
    80-layer depth × 0.6). Closer to prod-realistic depth than the L16
    toy.
    """
    return _llama3_175m_attn_res_variant("175M_attn_res_L48_n8")


# ------------------------------------------------------------------------- #
# DSv3-shaped MoE + MLA + AttnRes Trainer configs.
#
# Hyperparameters mirror upstream ``torchtitan.models.deepseek_v3.config_registry``
# so the only training-level delta between ``deepseek_v3_16b`` (baseline,
# run via --module deepseek_v3) and ``dsv3_attn_res_16b`` (our variant,
# run via --module attn_res) is Block AttnRes itself.
# ------------------------------------------------------------------------- #


def dsv3_attn_res_debugmodel() -> Trainer.Config:
    """Tiny DSv3-shape MoE + AttnRes debug config.

    6 layers (1 dense + 5 MoE), 8 experts, N=3 AttnRes blocks. Uses the
    bundled test tokenizer and c4_test dataset; finishes in a few seconds
    on CPU. Meant for unit / smoke tests, not a training target.
    """
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=attn_res_model_registry("dsv3_debugmodel_attn_res"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def dsv3_attn_res_16b() -> Trainer.Config:
    """~16B MoE + MLA + AttnRes (N=9). Production-adjacent training target.

    Matches ``deepseek_v3.deepseek_v3_16b`` training hyperparameters
    verbatim; the only delta is Block AttnRes on every layer. For A/B
    comparison, run the baseline as
    ``--module deepseek_v3 --config deepseek_v3_16b`` and this as
    ``--module attn_res --config dsv3_attn_res_16b`` with matching seed
    and data order.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=attn_res_model_registry("dsv3_16b_attn_res"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def _dsv3_attn_res_16b_nvariant(flavor: str) -> Trainer.Config:
    """Helper: baseline 16B trainer config + a specific AttnRes num_blocks flavor.

    Used to build N-ablation runs that share every hyperparameter with the
    primary ``dsv3_attn_res_16b`` except ``num_blocks``.
    """
    config = dsv3_attn_res_16b()
    config.model_spec = attn_res_model_registry(flavor)
    return config


def dsv3_attn_res_16b_n3() -> Trainer.Config:
    """Ablation: N=3 (9 layers per block). Coarse grouping, bandwidth-light."""
    return _dsv3_attn_res_16b_nvariant("dsv3_16b_attn_res_n3")


def dsv3_attn_res_16b_n27() -> Trainer.Config:
    """Ablation: N=27 (1 layer per block = Full-AttnRes on this L=27 shape)."""
    return _dsv3_attn_res_16b_nvariant("dsv3_16b_attn_res_n27")


def llama3_175m_baseline_L16() -> Trainer.Config:
    """16-layer plain Llama3 dense baseline sized to match
    ``llama3_175m_attn_res_L16_n8`` minus AttnRes.

    Purpose: serves as the no-AttnRes reference for all PP-scale
    AttnRes-vs-baseline A/B comparisons. Shares every hyperparameter
    with ``llama3_175m_baseline`` EXCEPT the ``model_spec`` (which
    swaps the 12-layer plain Llama3 for the 16-layer plain Llama3 so
    the PP slicing matches the L16_n8 AttnRes variant, and the 4-GPU
    launchers in ``phase3/`` can point at it directly).

    The 4-GPU PP=4 V=2 reference config is:

        bash phase3/launch_4gpu_baseline_L16.sh

    Run any A/B against this baseline by setting ``STEPS`` identically
    on both sides.
    """
    config = llama3_175m_baseline()
    config.model_spec = _baseline_L16_model_registry()
    return config
