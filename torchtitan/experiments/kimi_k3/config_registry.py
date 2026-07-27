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
from torchtitan.experiments.kimi_k3.state_dict_adapter import (
    KimiLinearStateDictAdapter,
)
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
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4() -> Trainer.Config:
    """447M Block AttnRes with SGLang-friendly head dims.

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

    Selected with
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
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4_fp8() -> Trainer.Config:
    """447M Block AttnRes with FP8 rowwise training.

    Wraps :func:`kimi_linear_447m_aligned_block_attn_res_n4` and adds a
    Float8LinearConverter with the ``rowwise`` recipe. Excluded from the
    swap: every Linear inside a KDA layer (structurally, via
    KimiLinearFloat8Spec -- KDA and MLA share the ``self_attn`` name so
    no FQN substring can single out KDA), the MLA low-rank down-proj
    (``kv_a_proj_with_mqa``), the AttnRes projections, and the
    vocab/router heads -- those layers have either non-16-aligned
    shapes or numerical sensitivity that regresses under rowwise FP8.

    MoE experts (grouped_mm) stay bf16 — Float8GroupedMMConverter is a
    perf-prototype upstream and not in the dispatch path here.

    The Kimi Linear model is built as plain modules (KimiLinearSpec),
    not from a ``Linear.Config`` tree, so ``Float8LinearConverter``'s
    config-traversal ``convert`` cannot apply. The converter is still
    built here for its torchao/SM89 validation and recipe resolution;
    the actual swap is module-level inside
    :class:`KimiLinearFloat8Spec.build` with the same filter semantics.

    Expected speedup on RTX 5090 (SM 12.0): 1.3-1.5× over bf16 for the
    dense MLA / projector / output paths; smaller win at the model level
    because KDA Triton + MoE grouped_mm dominate the per-step compute.
    """
    from torchtitan.components.quantization import Float8LinearConverter
    from torchtitan.experiments.kimi_k3.model import KimiLinearFloat8Spec

    cfg = kimi_linear_447m_aligned_block_attn_res_n4()
    converter = Float8LinearConverter.Config(
        recipe_name="rowwise",
        filter_fqns=[
            "lm_head",
            "router.gate",
            "kv_a_proj_with_mqa",
            "attn_res_proj",
            "mlp_res_proj",
            "final_attn_res_proj",
        ],
    ).build()
    if not converter.enabled:
        # torchao too old for recipe lookup; converter already warned.
        return cfg
    inner = cfg.model_spec.model
    cfg.model_spec.model = KimiLinearFloat8Spec(
        kimi_config=inner.kimi_config,
        num_blocks=inner.num_blocks,
        param_init=inner.param_init,
        torchao_float8_config=converter.torchao_config,
        filter_fqns=list(converter.config.filter_fqns),
    )
    return cfg


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
    # Gated MLA in K3's own parameterization (tech report Eq. 7: full-rank
    # channel-wise sigmoid gate, no bias). The graft flavors below keep
    # per_head_graft instead, where a step-0 no-op is the point.
    m.kimi_config = _dc.replace(
        m.kimi_config, mla_gated=True, attn_gate_param="full_rank"
    )
    m.attn_res_gated = True                                     # alpha graft
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
        "debugmodel8h", num_experts=8, vocab_size=2016,
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
        "debugmodel", num_experts=8, vocab_size=2016,
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
        state_dict_adapter=KimiLinearStateDictAdapter,
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
        state_dict_adapter=KimiLinearStateDictAdapter,
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
        state_dict_adapter=KimiLinearStateDictAdapter,
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
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg
