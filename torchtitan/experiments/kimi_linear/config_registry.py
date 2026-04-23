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
    num_experts: int = 32,
    vocab_size: int = 163840,
    tie_word_embeddings: bool = True,
    kda_mla_ratio: int = 3,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-5,
) -> KimiLinearConfig:
    """Construct a :class:`KimiLinearConfig` for one scaling-law size.

    Args:
        size: One of ``{"194m","241m","296m","436m","528m"}``.
        num_experts: Total MoE experts (token-choice top-k). Default 32 is
            a tractable choice for scaling-law experiments; 48B-A3B uses 256.
        vocab_size: Token vocabulary. Default 163840 (Kimi tokenizer).
        tie_word_embeddings: Tie input/output embedding. Default True
            for scaling-law (smaller model, more param-efficient); 48B-A3B
            uses False.
        kda_mla_ratio: KDA:MLA layer ratio. Default 3 matches paper + 48B.
        rope_theta: RoPE base (unused when ``mla_use_nope=True``, which is
            the Kimi default).
        rms_norm_eps: RMSNorm epsilon.
    """
    if size not in _BY_NAME:
        raise ValueError(
            f"Unknown size '{size}'. Valid: {sorted(_BY_NAME.keys())}"
        )
    spec = _BY_NAME[size]
    d = spec.d_model
    H = spec.num_heads

    # Head dims — scaled to fit d_model/H, following 48B-A3B where
    # num_heads * head_dim = hidden_size (Kimi has no d_head < hidden/head_count).
    # For KDA: head_dim = d_model / num_heads (round to pow-2 via max(32, ...))
    # For MLA (NoPE): qk_nope + qk_rope + v_head split. Paper's 48B uses
    # qk_nope=128, qk_rope=64, v_head=128 at d=2304, num_heads=32, so each
    # head takes 128 (nope) + 64 (rope, broadcast) + 128 (v) units. We keep
    # qk_rope proportional to d/num_heads * 0.5 (half of nope).
    head_dim_mla_nope = max(32, d // H)
    head_dim_mla_rope = max(16, head_dim_mla_nope // 2)
    head_dim_mla_v = head_dim_mla_nope
    kda_head_dim = head_dim_mla_nope

    kv_lora_rank = d // 2  # scale with model; 48B uses 512 at d=2304 ≈ d/4.5

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
        intermediate_size=spec.d_ff,  # dense MLP intermediate (layer 0 only)
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
        use_grouped_topk=False,  # simplified; 48B uses True
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
            keep_latest_k=3,
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


def kimi_linear_528m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("528m", "baseline")


def kimi_linear_528m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "block_attn_res")


def kimi_linear_528m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "full_attn_res")
