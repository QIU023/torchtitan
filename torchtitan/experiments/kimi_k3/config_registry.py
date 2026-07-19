# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the Attention Residual experiment.

This is the single ``config_registry`` torchtitan's ``ConfigManager``
imports for ``--module kimi_k3``. It exposes three flavor
families:

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

3. Kimi Linear backbone (KDA + MLA + MoE) + AttnRes — the
   scaling-law sweep and the production 447M lineage (defined in the
   "Kimi Linear / K3 trainer configs" section below; architecture-side
   builders live in ``model_configs.py``). The ``kimi_linear_``
   config-name prefix is kept verbatim for backward compatibility with
   the production launch scripts.
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.kimi_k3 import model_registry as attn_res_model_registry

# Re-export every Kimi Linear + AttnRes trainer-config flavor so they are
# discoverable via ``--module kimi_k3 --config kimi_linear_<...>``.
# torchtitan's ConfigManager does ``getattr(config_registry, <config_name>)``,
# so the kimi flavor functions must be module-level attributes here. The
# ``kimi_linear_`` config-name prefix is preserved for backward compatibility
# with production launch scripts (only the ``--module`` value changed).
from torchtitan.experiments.kimi_k3.model_configs import (  # noqa: F401
    _BY_NAME,
    build,
    build_kimi_linear_config,
    flavor_names,
    resolve_num_blocks,
    _alternating_kda_mla_layers,
    SCALING_LAW_TABLE,
    Variant,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    ComplexRoPE,
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
    rope: RoPE.Config,
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
                    rope=rope,
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_plain_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.0
            ),
            rope=ComplexRoPE.Config(
                    dim=dim // n_heads,
                    max_seq_len=8192,
                    theta=500000,
                    scaling="llama",
            ),
        ),
    )


def _llama3_175m_plain_L16_config() -> Llama3Model.Config:
    """16-layer plain Llama3 dense config; shape-matched to llama3_175m_attn_res_L16_n8
    minus the AttnRes pseudo-queries and norms.

    Exists so the 4-GPU PP=4 V=2 + layers_per_stage=2 configuration can be
    run as a no-AttnRes baseline under the same ``--module kimi_k3``
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_plain_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.0
            ),
            rope=ComplexRoPE.Config(
                    dim=dim // n_heads,
                    max_seq_len=8192,
                    theta=500000,
                    scaling="llama",
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
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
            log_freq=10,
        ),
        model_spec=_baseline_model_registry(),
        optimizer=default_adamw(lr=3e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
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
# run via --module kimi_k3) is Block AttnRes itself.
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
        optimizer=default_adamw(lr=8e-4),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def dsv3_attn_res_16b() -> Trainer.Config:
    """~16B MoE + MLA + AttnRes (N=9). Production-adjacent training target.

    Matches ``deepseek_v3.deepseek_v3_16b`` training hyperparameters
    verbatim; the only delta is Block AttnRes on every layer. For A/B
    comparison, run the baseline as
    ``--module deepseek_v3 --config deepseek_v3_16b`` and this as
    ``--module kimi_k3 --config dsv3_attn_res_16b`` with matching seed
    and data order.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=attn_res_model_registry("dsv3_16b_attn_res"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=2.2e-4),
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
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
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


# ----- Kimi Linear / K3 trainer configs (merged from kimi_linear/) ----- #

def _base_trainer_config(size_name: str) -> Trainer.Config:
    """Shared Trainer.Config template for a given paper Table-2 size.

    The peak LR + batch-size come from the paper; other knobs match
    torchtitan common defaults (warmup=500, cosine decay_ratio=0.8,
    min_lr_factor=0.1, FSDP full shard). ``model_spec`` is set by the
    per-flavor wrappers below.
    """
    if size_name not in _BY_NAME:
        raise ValueError(f"Unknown size '{size_name}'")
    spec = _BY_NAME[size_name]
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True, log_freq=10,
        ),
        model_spec=None,  # filled in by the per-flavor wrapper
        optimizer=default_adamw(lr=spec.lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=500,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=max(1, spec.batch_size // 8),  # default 8 DP ranks
            seq_len=8192,  # paper uses 8192 context
            steps=20000,   # placeholder; caller overrides via --training.steps
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1000,
            keep_latest_k=2,  # disk-discipline: at most 2x model size
            last_save_model_only=False,
        ),
        # AC off — kimi_linear/parallelize.py Phase 4c doesn't implement it.
        activation_checkpoint=None,
        validator=Validator.Config(freq=500, steps=50),
    )


def _flavor_trainer_config(size: str, variant: Variant) -> Trainer.Config:
    """Return a Trainer.Config for the requested size+variant with
    ``model_spec`` wired to :func:`model_registry` (imported late to
    avoid a circular import).
    """
    # Late import: model_registry lives in __init__.py which imports
    # from this module. Circular if eager-imported at module top.
    from torchtitan.experiments.kimi_k3 import model_registry

    cfg = _base_trainer_config(size)
    flavor = f"kimi_linear_{size}_{variant}"
    cfg.model_spec = model_registry(flavor)
    return cfg


# ----- Explicit per-flavor entry points (tyro discovers these) ----------- #

def kimi_linear_194m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("194m", "baseline")


def kimi_linear_194m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("194m", "block_attn_res")


def kimi_linear_194m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("194m", "full_attn_res")


def kimi_linear_241m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("241m", "baseline")


def kimi_linear_241m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("241m", "block_attn_res")


def kimi_linear_241m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("241m", "full_attn_res")


def kimi_linear_296m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("296m", "baseline")


def kimi_linear_296m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("296m", "block_attn_res")


def kimi_linear_296m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("296m", "full_attn_res")


def kimi_linear_436m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("436m", "baseline")


def kimi_linear_436m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("436m", "block_attn_res")


def kimi_linear_436m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("436m", "full_attn_res")


def kimi_linear_436m_block_attn_res_n4() -> Trainer.Config:
    """436M Block AttnRes with N=4 (instead of paper-default N=8).

    Paper Fig 6 (S ablation on the 16-layer model from Table 2)
    shows S=2/4/8 — i.e., N=8/4/2 for L=16 — all converging to
    ~1.746 vs baseline 1.766 on validation loss. The choice of
    N is essentially indistinguishable across that range.

    We use N=4 here (S=4 hf_layers/block) instead of paper-canonical
    N=8 (S=2 hf_layers/block) for one purely operational reason:
    halving the per-rank block-cache memory (~3 GiB savings on the
    436M shape) so the AttnRes A/B can run at LOCAL_BS=3 SEQ=2048
    on 4× RTX 5090 32GB without sustained 97% memory utilization +
    CUDA allocation retries that ate ~30% of throughput in the N=8
    variant. On bigger memory boxes (H100/H200/B200) we'd revert to
    paper's canonical N=8.
    """
    from torchtitan.experiments.kimi_k3 import (
        KimiLinearSpec,
        parallelize_kimi_linear,
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    cfg = _base_trainer_config("436m")
    kimi_config = build_kimi_linear_config("436m")
    spec_config = KimiLinearSpec(kimi_config=kimi_config, num_blocks=4)
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_436m_block_attn_res_n4",
        model=spec_config,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4() -> Trainer.Config:
    """447M Block AttnRes with SGLang-friendly head dims (phase 11).

    Same scale as ``kimi_linear_436m_block_attn_res_n4`` — 16 layers,
    16 attention heads, 32 routed experts top-8, 1 shared expert,
    AttnRes N=4 (S=4 layers/block) — but with d_model=1024 (vs 1168)
    so head_dim=64 is divisible by 16. This unblocks SGLang inference
    on SM 12.0 (RTX 5090): the original 436M's head_dim=73 fails
    flashinfer's batch-prefill kernel + cuBLAS strided-batched bmm
    + Triton extend kernel autotune (cudaErrorMisalignedAddress /
    CUBLAS_STATUS_INTERNAL_ERROR / shared-memory OOM respectively).

    All other dims aligned to 8/16 multiples:
    * qk_nope=64, qk_rope=32, v_head=64
    * kv_lora_rank=512 (multiple of 64)
    * head_dim_qk = 96, head_dim_vo = 64 (both flashinfer-accepted)

    intermediate_size / moe_intermediate_size bumped 528 → 768 to keep
    the activated-param budget at ~447M, on par with the original
    436M scaling-law row's compute cost. Same lr (2.20e-3), batch size
    (384 sequences global), and total tokens budget (87.9B) inherited
    from the 436M row in SCALING_LAW_TABLE.

    Trains with the same launcher
    (``phase4/launch_paperhparams_break3.sh``) by setting
    ``CONFIG=kimi_linear_447m_aligned_block_attn_res_n4``. Runs through
    the same parallelize_fn / pipelining_fn / loss_fn as 436M.
    """
    from torchtitan.experiments.kimi_k3 import (
        KimiLinearSpec,
        parallelize_kimi_linear,
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    cfg = _base_trainer_config("447m_aligned")
    kimi_config = build_kimi_linear_config("447m_aligned")
    spec_config = KimiLinearSpec(kimi_config=kimi_config, num_blocks=4)
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_447m_aligned_block_attn_res_n4",
        model=spec_config,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4_fp8() -> Trainer.Config:
    """447M Block AttnRes with FP8 rowwise training.

    Wraps :func:`kimi_linear_447m_aligned_block_attn_res_n4` and adds a
    Float8LinearConverter with the ``rowwise_with_gw_hp`` recipe (weights
    + activations in FP8, grad-output stays high-precision). KDA-specific
    low-dim projections (``kda.*``), MLA LoRA projections, and the
    vocab/router heads are excluded via ``filter_fqns`` — those layers
    have either non-16-aligned shapes or numerical sensitivity that
    regresses under rowwise FP8.

    MoE experts (grouped_mm) stay bf16 — Float8GroupedMMConverter is a
    perf-prototype upstream and not in the dispatch path here.

    Expected speedup on RTX 5090 (SM 12.0): 1.3-1.5× over bf16 for the
    dense MLA / projector / output paths; smaller win at the model level
    because KDA Triton + MoE grouped_mm dominate the per-step compute.
    """
    from torchtitan.components.quantization import Float8LinearConverter

    cfg = kimi_linear_447m_aligned_block_attn_res_n4()
    converter = Float8LinearConverter.Config(
        recipe_name="rowwise",
        filter_fqns=[
            "output",
            "lm_head",
            "router.gate",
            "kda",
            "mla.q_lora_proj",
            "mla.k_lora_proj",
            "attn_res_proj",
            "mlp_res_proj",
            "final_attn_res_proj",
        ],
    )
    cfg.model_spec.model = converter.build().convert(cfg.model_spec.model)
    return cfg


def kimi_linear_528m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("528m", "baseline")


def kimi_linear_528m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "block_attn_res")


def kimi_linear_528m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "full_attn_res")


# ----- Full Kimi Linear 48B-A3B carriers ---------------------------------- #
# Paper §"Training recipe": 27 transformer-blocks = 54 paper-layers,
# Block AttnRes N=9 (= 6 paper-layers per AttnRes-block = 3
# transformer-blocks per AttnRes-block). 48B total / 3B activated.
# Construction-only: requires multi-node + EP to actually train.
# Single-node use case is meta-device build / param-count sanity / PP
# layout planning, NOT actual gradient steps.


def kimi_linear_48b_baseline() -> Trainer.Config:
    return _flavor_trainer_config("48b", "baseline")


def kimi_linear_48b_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("48b", "block_attn_res")


def kimi_linear_48b_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("48b", "full_attn_res")


# ----- 48B downscale variants (single-node feasibility sweep) ------------ #
# Paper 48B (256 experts × dim=2304) doesn't fit 8×32 GiB. These variants
# reduce num_experts (and optionally dim) while keeping n_layers=27 and
# N=9 (paper sweet spot, 3 t-blocks per AttnRes-block). Used to find the
# largest single-node-feasible carrier with paper-aligned architecture.


def _kimi_linear_48b_attnres_downscale(
    *,
    num_experts: int,
    dim: int | None = None,
    n_layers: int | None = None,
    num_blocks: int | None = None,
) -> Trainer.Config:
    """48B Block AttnRes with overridden num_experts (and optionally dim,
    n_layers, num_blocks).

    Defaults: n_layers=27, num_blocks=9 (paper sweet spot 3 t-blocks per
    AttnRes-block), seq_len=4096 (paper). Pass n_layers / num_blocks to
    deviate (e.g. n_layers=24, num_blocks=8 keeps the paper 3:1 ratio
    while making the depth divisible by PP=8 × VP=3 = 24 chunks).
    """
    from torchtitan.experiments.kimi_k3 import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_k3.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kwargs = {"num_experts": num_experts}
    kcfg = build_kimi_linear_config("48b", **kwargs)
    if dim is not None:
        kcfg.hidden_size = dim
        H = kcfg.num_attention_heads
        head_dim_aligned = max(32, (dim // H) & ~15)
        kcfg.qk_nope_head_dim = head_dim_aligned
        kcfg.qk_rope_head_dim = max(16, head_dim_aligned // 2)
        kcfg.v_head_dim = head_dim_aligned
        kcfg.kda_head_dim = head_dim_aligned
        kcfg.kv_lora_rank = (dim // 2) & ~63
        # Paper 48B dense FFN intermediate (layer 0 only) = 4 × dim.
        kcfg.intermediate_size = 4 * dim
    if n_layers is not None:
        kcfg.num_hidden_layers = n_layers
        # Re-derive KDA/MLA pattern with 3:1 ratio.
        kda_layers, full_attn_layers = _alternating_kda_mla_layers(
            n_layers, kda_mla_ratio=3,
        )
        kcfg.kda_layers = kda_layers
        kcfg.full_attn_layers = full_attn_layers

    final_num_blocks = num_blocks if num_blocks is not None else 9
    if n_layers is not None and n_layers % final_num_blocks != 0:
        raise ValueError(
            f"num_blocks={final_num_blocks} must divide n_layers={n_layers}"
        )
    spec_config = KimiLinearSpec(kimi_config=kcfg, num_blocks=final_num_blocks)
    cfg = _base_trainer_config("48b")
    cfg.training.seq_len = 4096
    cfg.training.local_batch_size = 1  # single-node aggressive
    flavor_name = f"kimi_linear_48b_attnres_e{num_experts}"
    if dim is not None:
        flavor_name += f"_d{dim}"
    if n_layers is not None:
        flavor_name += f"_L{n_layers}"
    if num_blocks is not None:
        flavor_name += f"_N{num_blocks}"
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor=flavor_name,
        model=spec_config,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_48b_block_attn_res_e32() -> Trainer.Config:
    """48B carrier, paper dim=2304, num_experts=32 (vs paper 256).
    First feasibility step.
    """
    return _kimi_linear_48b_attnres_downscale(num_experts=32)


def kimi_linear_48b_block_attn_res_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16)


def kimi_linear_48b_block_attn_res_e8() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=8)


def kimi_linear_48b_block_attn_res_d1280_e32() -> Trainer.Config:
    """48B layout (L=27, N=9) at narrower dim=1280, num_experts=32.
    Fallback if paper-dim variants don't fit.
    """
    return _kimi_linear_48b_attnres_downscale(num_experts=32, dim=1280)


def kimi_linear_48b_block_attn_res_d1280_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16, dim=1280)


def kimi_linear_48b_block_attn_res_d1024_e32() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=32, dim=1024)


def kimi_linear_48b_block_attn_res_d1024_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16, dim=1024)


def kimi_linear_48b_block_attn_res_d1280_e32_L24_N8() -> Trainer.Config:
    """48B-layout carrier shrunk to L=24 (vs paper 27) so PP=8 × VP=3 = 24
    chunks divides cleanly. N=8 keeps paper sweet spot 3 transformer-blocks
    per AttnRes-block (24/8 = 3). dim=1280, num_experts=32. seq=2048.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=32, dim=1280, n_layers=24, num_blocks=8,
    )


def kimi_linear_48b_block_attn_res_d1280_e32_L32_N8() -> Trainer.Config:
    """48B-layout at L=32 N=8 (4 transformer-blocks per AttnRes-block,
    1.33× paper sweet spot). Allows PP=8 × VP=4 = 32 chunks × 1 layer.
    dim=1280, num_experts=32.

    NOTE: OOM at step 2 on 8×32 GiB (rank 7 hit 31.34 GiB after cache
    accumulation). Use the e16 variant below instead.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=32, dim=1280, n_layers=32, num_blocks=8,
    )


def kimi_linear_48b_block_attn_res_d1280_e16_L32_N8() -> Trainer.Config:
    """L=32 N=8 carrier with num_experts=16 (vs e32 OOM). Fits PP=8 ×
    VP=4 = 32 chunks paper-aligned, paper-sweet-spot t-blocks/AttnRes-block
    ratio off by 1.33×.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=16, dim=1280, n_layers=32, num_blocks=8,
    )


# ----- PP=4 V=2 lps=2 compatibility variant -------------------------------- #
# Paper's 528M has n_layers=17 (prime), which doesn't divide the 8 virtual
# stages needed by Interleaved1F1B PP=4 V=2 with lps=2. Drop to n_layers=16
# (one fewer layer) so the PP cache adapter layout tables build cleanly.
# All other 528M paper hyperparameters retained (d=1264, d_ff=560,
# lr=2.02e-3, batch=432). The KDA/MLA 3:1 alternation is re-derived for
# L=16 so 4 MLA layers land at the same relative positions.

def _build_528m_l16_config():
    """528M-like Kimi Linear config with n_layers=16 for PP=4 V=2 lps=2
    divisibility. d_model / d_ff / num_heads / LR all match paper's 528M.
    """
    cfg = build_kimi_linear_config("528m")
    cfg.num_hidden_layers = 16
    # Re-derive KDA:MLA = 3:1 pattern for 16 layers
    # (1-indexed). Period 4 → MLA at {4, 8, 12, 16}, KDA at the rest.
    period = 4
    cfg.kda_layers = [i for i in range(1, 17) if i % period != 0]
    cfg.full_attn_layers = [i for i in range(1, 17) if i % period == 0]
    return cfg


def kimi_linear_528m_l16_block_attn_res() -> Trainer.Config:
    """528M-scale Kimi Linear AttnRes with n_layers=16, Block AttnRes N=8.

    PP=4 V=2 lps=2 compatible (8 virtual stages on 4 ranks, 2 layers per
    stage). Every stage is a block boundary → cross-stage cache adapter
    exercised at every stage transition. Paper 528M d/d_ff/heads/LR
    retained; only depth reduced by 1 to satisfy the Interleaved1F1B
    divisibility requirement.
    """
    from torchtitan.experiments.kimi_k3 import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_k3.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=8)
    cfg = _base_trainer_config("528m")  # paper 528M lr / batch template
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_block_attn_res",
        model=spec,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_528m_l16_full_attn_res() -> Trainer.Config:
    """528M-scale Kimi Linear Full AttnRes (num_blocks = n_layers = 16)."""
    from torchtitan.experiments.kimi_k3 import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_k3.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=16)
    cfg = _base_trainer_config("528m")
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_full_attn_res",
        model=spec,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_528m_l16_baseline() -> Trainer.Config:
    """528M-scale Kimi Linear baseline (no AttnRes) with n_layers=16.
    Paired control for the two AttnRes variants above.
    """
    from torchtitan.experiments.kimi_k3 import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_k3.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiLinearSpec(kimi_config=kcfg, num_blocks=None)
    cfg = _base_trainer_config("528m")
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_baseline",
        model=spec,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg
