# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaling-law config registry for Kimi Linear + AttnRes.

Parametric :class:`KimiLinearConfig` constructors for the 5 sizes in
the AttnRes tech-report Table 2 (194M → 528M activated params) plus
the 48B-A3B upscale target (kept for reference only — 48B needs
multi-node). Each ``_build_config`` call returns a tuple of
``(kimi_config, num_blocks)`` so a caller can wire it into either
:class:`KimiLinearModel` (baseline, ``num_blocks=None``) or
:class:`KimiLinearAttnResModel` (AttnRes variant, ``num_blocks=N``).

The paper's Table 2 fields ``d_model``, ``d_ff``, ``L_b`` (= number of
decoder layers), ``lr`` and ``batch_size`` are preserved verbatim. The
vocab / MoE / KDA / MLA knobs default to the 48B-A3B reference config
shape, just scaled by ``d_model``. Specifically:

* Vocab = 163840 (Kimi's native tokenizer, tied in scaling-law to
  keep the non-embedding activated param count matching the paper).
* MoE: ``num_experts_per_token=8``, ``num_shared_experts=1``,
  ``moe_intermediate_size = d_ff``, ``first_k_dense_replace=1``.
* MLA (on ``full_attn_layers``): ``qk_nope_head_dim=128``,
  ``qk_rope_head_dim=64``, ``v_head_dim=128`` scaled to fit
  ``d_model/num_heads``.
* KDA (on ``kda_layers``): head_dim scaled so
  ``num_heads × head_dim ≈ d_model``.
* KDA:MLA = 3:1 ratio matching 48B-A3B pattern (every 4th layer is MLA).

The :attr:`scaling_law_sizes` dict maps size-name → Python constructor;
callers pass a ``num_blocks`` kwarg to pick the AttnRes variant.

This module does NOT yet return torchtitan ``Trainer.Config`` nor
``ModelSpec``. ModelSpec integration is Phase 4c: it requires
refactoring :class:`KimiLinearModel` to inherit from
``torchtitan.protocols.model.BaseModel`` and wrapping
:class:`KimiLinearConfig` inside a ``BaseModel.Config`` shim that
implements ``build()``, ``update_from_config()``, and
``get_nparams_and_flops()``. Until then, use this module to
instantiate models directly for CPU tests / ad-hoc experiments, and
use the Llama3-backed ``attn_res/config_registry.py`` flavors for
actual training (see ``phase4/launch_fsdp_llama3_528m.sh``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torchtitan.experiments.kimi_linear.model import KimiLinearConfig


# ----- Paper Table 2 canonical sizes -------------------------------------- #
# Columns copied verbatim from Kimi Linear AttnRes tech report Table 2.
# d_ff is the MoE per-expert intermediate size (moe_intermediate_size in our
# config). L_b is the number of Kimi decoder layers (= num_hidden_layers).

@dataclass(frozen=True)
class _SweepSize:
    """One row of the tech report's scaling-law sweep (Table 2)."""

    name: str
    activated_params: int   # M parameters (reported, non-embedding)
    tokens: float           # B tokens
    n_layers: int           # L_b in paper (= num_hidden_layers in our config)
    num_heads: int          # H in paper (= num_attention_heads + kda_num_heads)
    d_model: int            # d_model in paper
    d_ff: int               # d_ff in paper (= moe_intermediate_size in our config)
    lr: float               # peak learning rate
    batch_size: int         # global batch size (sequences)


SCALING_LAW_TABLE: tuple[_SweepSize, ...] = (
    _SweepSize("194m", 194, 38.7, 12, 12, 896, 400, 2.99e-3, 192),
    _SweepSize("241m", 241, 45.4, 13, 13, 960, 432, 2.80e-3, 256),
    _SweepSize("296m", 296, 62.1, 14, 14, 1024, 464, 2.50e-3, 320),
    _SweepSize("436m", 436, 87.9, 16, 16, 1168, 528, 2.20e-3, 384),
    _SweepSize("528m", 528, 119.0, 17, 17, 1264, 560, 2.02e-3, 432),
    # Phase 11: SGLang-friendly aligned-dim variant of the 436M row.
    # d=1024 (vs 1168) → head_dim=64 is multiple of 16; qk_rope=32, v=64,
    # kv_lora=512 all 8/16/32-aligned; flashinfer / cublas / triton
    # extend kernels accept this layout on SM 12.0 (RTX 5090). d_ff
    # bumped 528 → 768 to keep activated-param count ~447M, roughly
    # matching the original 436M row's compute budget.
    # Re-uses 436M's lr / batch_size / token_count from the same row.
    _SweepSize("447m_aligned", 447, 87.9, 16, 16, 1024, 768, 2.20e-3, 384),
    # Full Kimi Linear 48B-A3B target. From paper §"Training recipe":
    # "27 Transformer blocks (54 layers)" with Block AttnRes N=9
    # (6 paper-layers per AttnRes-block = 3 transformer-blocks per
    # AttnRes-block). d_ff here is the MoE-per-expert intermediate
    # size (1024 in HF config); the dense FFN at layer 0 uses
    # intermediate_size=9216 (set in build_kimi_linear_48b_a3b_config
    # via the override path, not from this row).
    # NOTE: 48B requires multi-node; this row exists for carrier
    # construction + config-correctness checks, not single-node training.
    # tokens/lr/batch from paper §Training recipe (1T pretrain +
    # 400B mid-train; "global batch size of 8M tokens" → 8M/4096
    # context = 1953 seqs ≈ 2048).
    _SweepSize("48b", 3000, 1400.0, 27, 32, 2304, 1024, 1.0e-3, 2048),
)

_BY_NAME: dict[str, _SweepSize] = {s.name: s for s in SCALING_LAW_TABLE}


# ----- 48B-A3B reference (upscale target, kept for docs) ------------------ #
# Faithful to the HF config.json at moonshotai/Kimi-Linear-48B-A3B-Base.
# Listed here so the full scale sweep is visible in one file; the 48B
# config needs multi-node to train and is OUT OF SCOPE for Phase 4a-d.

_KIMI_48B_A3B_KDA_LAYERS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26)
_KIMI_48B_A3B_FULL_ATTN_LAYERS = (4, 8, 12, 16, 20, 24, 27)


# ----- Sweep config builders ---------------------------------------------- #

def _alternating_kda_mla_layers(
    n_layers: int, kda_mla_ratio: int = 3,
) -> tuple[list[int], list[int]]:
    """Build 1-indexed kda_layers / full_attn_layers lists with given ratio.

    Default ratio 3:1 matches paper + 48B-A3B (3 KDA, 1 MLA, repeat).
    MLA lands every ``kda_mla_ratio+1``-th layer (1-indexed).
    """
    period = kda_mla_ratio + 1
    kda, mla = [], []
    for i in range(1, n_layers + 1):
        if i % period == 0:
            mla.append(i)
        else:
            kda.append(i)
    return kda, mla


def build_kimi_linear_config(
    size: str,
    *,
    num_experts: int | None = None,
    vocab_size: int = 163840,
    tie_word_embeddings: bool | None = None,
    kda_mla_ratio: int = 3,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-5,
    dense_intermediate_size: int | None = None,
    use_grouped_topk: bool | None = None,
) -> KimiLinearConfig:
    """Construct a :class:`KimiLinearConfig` for one scaling-law size.

    Args:
        size: One of ``{"194m","241m","296m","436m","528m","48b"}``.
        num_experts: Total MoE experts (token-choice top-k). Default 32
            for scaling-law sizes; 256 for the full 48B-A3B target.
        vocab_size: Token vocabulary. Default 163840 (Kimi tokenizer).
        tie_word_embeddings: Tie input/output embedding. Default True
            for scaling-law (smaller model, more param-efficient); False
            for 48B-A3B (matches HF config.json).
        kda_mla_ratio: KDA:MLA layer ratio. Default 3 matches paper + 48B.
        rope_theta: RoPE base (unused when ``mla_use_nope=True``, which is
            the Kimi default).
        rms_norm_eps: RMSNorm epsilon.
        dense_intermediate_size: Dense FFN intermediate size used by
            layer 0 only (when ``first_k_dense_replace=1``). Defaults to
            ``spec.d_ff`` (= MoE per-expert intermediate). 48B-A3B
            overrides: dense=9216 while moe-per-expert=1024.
        use_grouped_topk: MoE router grouped-topk gate. Default False
            (simplified); 48B-A3B uses True (matches HF config.json).
    """
    if size not in _BY_NAME:
        raise ValueError(
            f"Unknown size '{size}'. Valid: {sorted(_BY_NAME.keys())}"
        )
    spec = _BY_NAME[size]
    d = spec.d_model
    H = spec.num_heads

    # Size-specific defaults that differ between scaling-law sweep and
    # full 48B-A3B. Each is overridable from the kwargs above.
    if size == "48b":
        num_experts_default = 256
        tie_default = False
        dense_d_ff_default = 9216  # HF config.json:intermediate_size
        use_grouped_topk_default = True  # HF config.json
    else:
        num_experts_default = 32
        tie_default = True
        dense_d_ff_default = spec.d_ff
        use_grouped_topk_default = False
    if num_experts is None:
        num_experts = num_experts_default
    if tie_word_embeddings is None:
        tie_word_embeddings = tie_default
    if dense_intermediate_size is None:
        dense_intermediate_size = dense_d_ff_default
    if use_grouped_topk is None:
        use_grouped_topk = use_grouped_topk_default

    # Head dims — scaled to fit d_model/H, following 48B-A3B where
    # num_heads * head_dim = hidden_size (Kimi has no d_head < hidden/head_count).
    # For KDA: head_dim = d_model / num_heads (round to pow-2 via max(32, ...))
    # For MLA (NoPE): qk_nope + qk_rope + v_head split. Paper's 48B uses
    # qk_nope=128, qk_rope=64, v_head=128 at d=2304, num_heads=32, so each
    # head takes 128 (nope) + 64 (rope, broadcast) + 128 (v) units. We keep
    # qk_rope proportional to d/num_heads * 0.5 (half of nope).
    if size == "48b":
        # Match HF config.json/moonshotai/Kimi-Linear-48B-A3B-Base verbatim.
        head_dim_mla_nope = 128
        head_dim_mla_rope = 64
        head_dim_mla_v = 128
        kda_head_dim = 128
        kv_lora_rank = 512
    elif size.endswith("_aligned"):
        # Phase 11: SGLang flashinfer / cuBLAS / triton extend kernels on
        # SM 12.0 (RTX 5090) require head_dim multiple of 8 (16 preferred
        # so qk_rope = head_dim/2 is also 8-aligned). Round head_dim down
        # to multiple of 16, kv_lora_rank to multiple of 64.
        head_dim_mla_nope = max(32, (d // H) & ~15)
        head_dim_mla_rope = max(16, head_dim_mla_nope // 2)
        head_dim_mla_v = head_dim_mla_nope
        kda_head_dim = head_dim_mla_nope
        kv_lora_rank = (d // 2) & ~63
    else:
        head_dim_mla_nope = max(32, d // H)
        head_dim_mla_rope = max(16, head_dim_mla_nope // 2)
        head_dim_mla_v = head_dim_mla_nope
        kda_head_dim = head_dim_mla_nope
        kv_lora_rank = d // 2  # scale with model; 48B uses 512 at d=2304 ≈ d/4.5

    if size == "48b":
        # HF config.json has 7 MLA layers (full_attn) at indices 4,8,12,16,
        # 20,24,27 (1-indexed) and 20 KDA layers everywhere else. The pattern
        # is "every 4th layer is MLA, plus the last layer 27". Hand-emit this
        # exact split instead of going through _alternating_kda_mla_layers
        # (which would miss layer 27 because 27 % 4 != 0).
        full_attn_layers = [4, 8, 12, 16, 20, 24, 27]
        kda_layers = [i for i in range(1, spec.n_layers + 1) if i not in full_attn_layers]
    else:
        kda_layers, full_attn_layers = _alternating_kda_mla_layers(
            spec.n_layers, kda_mla_ratio=kda_mla_ratio
        )

    return KimiLinearConfig(
        # Vocabulary / embedding
        vocab_size=vocab_size,
        hidden_size=d,
        tie_word_embeddings=tie_word_embeddings,
        # Depth / width
        num_hidden_layers=spec.n_layers,
        intermediate_size=dense_intermediate_size,  # dense FFN (layer 0)
        # MLA
        num_attention_heads=H,
        num_key_value_heads=H,  # no GQA
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=head_dim_mla_nope,
        qk_rope_head_dim=head_dim_mla_rope,
        v_head_dim=head_dim_mla_v,
        mla_use_nope=True,
        rope_theta=rope_theta,
        # KDA
        kda_num_heads=H,
        kda_head_dim=kda_head_dim,
        kda_short_conv_kernel_size=4,
        kda_layers=list(kda_layers),
        full_attn_layers=list(full_attn_layers),
        # MoE
        num_experts=num_experts,
        num_experts_per_token=8,
        moe_intermediate_size=spec.d_ff,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=1,
        topk_group=1,
        # Norm / init
        rms_norm_eps=rms_norm_eps,
        hidden_act="silu",
        initializer_range=0.02,
    )


Variant = Literal["baseline", "block_attn_res", "full_attn_res"]


def resolve_num_blocks(size: str, variant: Variant) -> int | None:
    """Pick ``num_blocks`` for the given (size, variant) combo.

    Returns ``None`` for the baseline (no AttnRes). ``n_layers`` for
    Full AttnRes. A divisor of ``n_layers`` near 8 for Block AttnRes
    (paper's "N≈8" shorthand; when ``n_layers`` is prime we fall back
    to Full AttnRes since no non-trivial divisor exists).
    """
    if size not in _BY_NAME:
        raise ValueError(f"Unknown size '{size}'")
    n_layers = _BY_NAME[size].n_layers
    if variant == "baseline":
        return None
    if variant == "full_attn_res":
        return n_layers
    if variant == "block_attn_res":
        # Nearest divisor of n_layers to 8
        divisors = [d for d in range(2, n_layers + 1) if n_layers % d == 0]
        if not divisors:
            # n_layers == 1, degenerate
            return n_layers
        # Pick divisor minimizing |d - 8|
        return min(divisors, key=lambda d: (abs(d - 8), d))
    raise ValueError(f"Unknown variant '{variant}'")


def build(
    size: str, variant: Variant,
) -> tuple[KimiLinearConfig, int | None]:
    """Top-level entrypoint: return ``(kimi_config, num_blocks)``.

    Pass to :class:`KimiLinearModel` (baseline) or
    :class:`KimiLinearAttnResModel` (AttnRes) depending on
    ``num_blocks is None``.
    """
    return (
        build_kimi_linear_config(size),
        resolve_num_blocks(size, variant),
    )


# ----- Convenience: which (size, variant) pairs exist -------------------- #

def flavor_names() -> list[str]:
    """All registered flavor names: ``kimi_linear_{size}_{variant}``."""
    out: list[str] = []
    for s in SCALING_LAW_TABLE:
        for v in ("baseline", "block_attn_res", "full_attn_res"):
            out.append(f"kimi_linear_{s.name}_{v}")
    return out


# ----- Trainer.Config factories ------------------------------------------ #
# One function per flavor, hand-rolled so the torchtitan ConfigManager
# can import them by name. Pattern matches attn_res/config_registry.py.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer


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
        profiling=ProfilingConfig(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True, log_freq=10,
        ),
        model_spec=None,  # filled in by the per-flavor wrapper
        optimizer=OptimizersContainer.Config(lr=spec.lr),
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
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        validator=Validator.Config(freq=500, steps=50),
    )


def _flavor_trainer_config(size: str, variant: Variant) -> Trainer.Config:
    """Return a Trainer.Config for the requested size+variant with
    ``model_spec`` wired to :func:`model_registry` (imported late to
    avoid a circular import).
    """
    # Late import: model_registry lives in __init__.py which imports
    # from this module. Circular if eager-imported at module top.
    from torchtitan.experiments.kimi_linear import model_registry

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
    from torchtitan.experiments.kimi_linear import (
        KimiLinearSpec,
        parallelize_kimi_linear,
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
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
    from torchtitan.experiments.kimi_linear import (
        KimiLinearSpec,
        parallelize_kimi_linear,
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
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
    from torchtitan.experiments.kimi_linear import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_linear.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
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
    from torchtitan.experiments.kimi_linear import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_linear.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_528m_l16_full_attn_res() -> Trainer.Config:
    """528M-scale Kimi Linear Full AttnRes (num_blocks = n_layers = 16)."""
    from torchtitan.experiments.kimi_linear import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_linear.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg


def kimi_linear_528m_l16_baseline() -> Trainer.Config:
    """528M-scale Kimi Linear baseline (no AttnRes) with n_layers=16.
    Paired control for the two AttnRes variants above.
    """
    from torchtitan.experiments.kimi_linear import (
        parallelize_kimi_linear, KimiLinearSpec,
    )
    from torchtitan.experiments.kimi_linear.pipeline_adapter import (
        pipeline_kimi_linear_with_cache_adapter,
    )
    from torchtitan.components.loss import build_cross_entropy_loss
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
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
    return cfg
