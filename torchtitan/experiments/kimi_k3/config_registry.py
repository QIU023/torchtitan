# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the Kimi K3 experiment.

This is the ``config_registry`` torchtitan's ConfigManager imports for
``--module kimi_k3``. Flavors: ``kimi_linear_<size>_<variant>`` -- the
AttnRes tech-report Table 2 scaling-law sweep (194m..528m), the
SGLang-aligned 447m carrier (+ fp8 variant), and the 48B-A3B layout
carriers. Architecture-side builders live in ``model_configs.py``.

The dense Llama3-shape / DSv3-shape AttnRes test carrier that previously
shared this registry lives outside this folder; it remains runnable
against earlier history (<= 666cf7ad6).
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.kimi_k3 import model_registry as attn_res_model_registry
from torchtitan.experiments.kimi_k3.model_configs import (  # noqa: F401
    _alternating_kda_mla_layers,
    _BY_NAME,
    build,
    build_kimi_linear_config,
    flavor_names,
    resolve_num_blocks,
    SCALING_LAW_TABLE,
    Variant,
)

# Re-export every Kimi Linear + AttnRes trainer-config flavor so they are
# discoverable via ``--module kimi_k3 --config kimi_linear_<...>``.
# torchtitan's ConfigManager does ``getattr(config_registry, <config_name>)``,
# so the kimi flavor functions must be module-level attributes here. The
# ``kimi_linear_`` config-name prefix is preserved for backward compatibility
# with production launch scripts (only the ``--module`` value changed).
from torchtitan.experiments.kimi_k3.state_dict_adapter import KimiLinearStateDictAdapter
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
        # Plain (non-chunked) CE: matches the numerics of all historical
        # kimi runs, and the KimiLinear* models don't implement the
        # _skip_lm_head forward that ChunkedLossWrapper requires.
        # 163840 = Kimi tokenizer vocab (build_kimi_linear_config
        # default; no flavor overrides it).
        loss=CrossEntropyLoss.Config(global_vocab_size=163840),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
            log_freq=10,
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
            steps=20000,  # placeholder; caller overrides via --training.steps
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1000,
            keep_latest_k=2,  # disk-discipline: at most 2x model size
            last_save_model_only=False,
        ),
        # AC off by default: the debug/scaling flavors fit without it.
        # (AC itself is supported -- see parallelize_kimi_linear.)
        activation_checkpoint=None,
        validator=Validator.Config(freq=500, steps=50),
        # Kimi CP reassembles contiguous rank-ordered seq shards inside
        # KDA/MLA (see model.py); the headtail load balancer permutes the
        # sequence and silently breaks causal order, so it must stay off.
        # parallelize_kimi_linear raises if this is set back to a balancer.
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
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


def kimi_linear_debugmodel_k3faithful() -> Trainer.Config:
    """Debug flavor with the K3-faithful architecture deltas ON:
    Gated MLA + alpha-graft Block AttnRes. CI-scale proof that the K3
    architecture (beyond the plain kimi_linear backbone) trains through
    the real trainer. MXFP4 QAT + Per-Head Muon are applied via their
    module/optimizer hooks (not config flags), see mxfp4_qat.py / muon.py.
    """
    import dataclasses as _dc

    cfg = kimi_linear_debugmodel()
    cfg.model_spec.flavor = "kimi_linear_debugmodel_k3faithful"
    m = cfg.model_spec.model
    m.kimi_config = _dc.replace(m.kimi_config, mla_gated=True)  # Gated MLA
    m.attn_res_gated = True  # alpha graft
    return cfg


def kimi_linear_debugmodel_gated_lora() -> Trainer.Config:
    """Debug flavor with the full post-train graft stack: alpha-gated
    Block AttnRes + LoRA rank-8 (frozen base, alpha-fullparam
    exception). CI-scale rehearsal of the 48B LoRA leg.
    """
    cfg = kimi_linear_debugmodel()
    cfg.model_spec.flavor = "kimi_linear_debugmodel_gated_lora"
    cfg.model_spec.model.attn_res_gated = True
    cfg.model_spec.model.lora_rank = 8
    return cfg


def kimi_linear_debugmodel8h() -> Trainer.Config:
    """8-head debug flavor (d=512, H=8) for deep tp x cp meshes.

    The 4-head debugmodel binds at tp*cp=4 (MLA heads must divide
    tp*cp); this flavor enables tp2cp4 / tp4cp2 cells on 8 ranks.
    """
    import dataclasses as _dc

    cfg = kimi_linear_debugmodel()
    cfg.model_spec.flavor = "kimi_linear_debugmodel8h"
    kimi_config = build_kimi_linear_config(
        "debugmodel8h",
        num_experts=8,
        vocab_size=2016,
    )
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=kimi_config,
        num_blocks=resolve_num_blocks("debugmodel8h", "block_attn_res"),
    )
    return cfg


def kimi_linear_debugmodel_gated_qlora_mxfp4() -> Trainer.Config:
    """Debug QLoRA: gated_lora with the frozen base packed to MXFP4.

    Meta-first trainer flow: the model builds with the PACKED layout
    (base_qdata/base_scale, no base.weight), FSDP shards the packed
    bytes, and the quantized values load from a DCP checkpoint produced
    by an offline streaming quantizer from a bf16 run. CI-scale
    rehearsal of 48B QLoRA on small-VRAM fleets (no rank ever holds the
    full bf16 model).
    """
    cfg = kimi_linear_debugmodel_gated_lora()
    cfg.model_spec.flavor = "kimi_linear_debugmodel_gated_qlora_mxfp4"
    cfg.model_spec.model.lora_quantize_base = "mxfp4"
    return cfg


def kimi_linear_48b_block_attn_res_gated_lora() -> Trainer.Config:
    """48B graft + LoRA rank-16: the 5090-feasible post-training target
    (frozen 48B base sharded at ~12GB/card; only adapters + AttnRes
    params train)."""
    cfg = kimi_linear_48b_block_attn_res_gated()
    cfg.model_spec.flavor = "kimi_linear_48b_block_attn_res_gated_lora"
    cfg.model_spec.model.lora_rank = 16
    return cfg


def kimi_linear_48b_block_attn_res_gated() -> Trainer.Config:
    """48B Block AttnRes with the alpha graft gate enabled.

    The post-training graft flavor: load the official
    Kimi-Linear-48B-A3B weights into the backbone, keep the AttnRes
    params (pseudo-queries + alphas) zero-init -- at step 0 the model
    function EXACTLY equals the original checkpoint (alpha=0 identity);
    alpha then trains away from identity. Use the ungated
    kimi_linear_48b_block_attn_res for from-scratch pretraining.
    """
    cfg = _flavor_trainer_config("48b", "block_attn_res")
    cfg.model_spec.flavor = "kimi_linear_48b_block_attn_res_gated"
    cfg.model_spec.model.attn_res_gated = True
    return cfg


def kimi_linear_debugmodel() -> Trainer.Config:
    """Tiny CI flavor: 4 layers (3 KDA + 1 MLA), d=256, 8 experts,
    Block AttnRes, 2016-token bundled test tokenizer, c4_test dataset.

    Runs a few-step train smoke in seconds on 1 GPU (or a CPU forward
    via the fla fallback); meant for CI and quick regression checks,
    not a training target.
    """
    from torchtitan.experiments.kimi_k3 import (
        KimiLinearSpec,
        parallelize_kimi_linear,
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.experiments.kimi_k3.state_dict_adapter import (
        KimiLinearStateDictAdapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kimi_config = build_kimi_linear_config(
        "debugmodel",
        num_experts=8,
        vocab_size=2016,
    )
    spec_config = KimiLinearSpec(
        kimi_config=kimi_config,
        num_blocks=resolve_num_blocks("debugmodel", "block_attn_res"),
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=2016),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=ModelSpec(
            name="kimi_linear",
            flavor="kimi_linear_debugmodel",
            model=spec_config,
            parallelize_fn=parallelize_kimi_linear,
            pipelining_fn=pipeline_kimi_linear_with_cache_adapter,
            post_optimizer_build_fn=None,
            state_dict_adapter=KimiLinearStateDictAdapter,
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        checkpoint=CheckpointManager.Config(interval=100),
        activation_checkpoint=None,
        # See _base_trainer_config: kimi CP requires contiguous seq shards.
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
    )


def kimi_linear_2p8t_block_attn_res() -> Trainer.Config:
    """PROVISIONAL K3 2.8T-A50B flavor (896 experts / 16 active, Block
    AttnRes). Config-level construction target only -- multi-node + EP
    to materialize; dims are placeholders pending the 7.27 config. Used
    for the 'scale-out is config-level' claim and EP@896 mesh checks.
    """
    return _flavor_trainer_config("2p8t", "block_attn_res")


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
    from torchtitan.experiments.kimi_k3 import KimiLinearSpec, parallelize_kimi_linear
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
            n_layers,
            kda_mla_ratio=3,
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
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


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
