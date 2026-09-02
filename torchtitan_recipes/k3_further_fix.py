"""Two-feature validation recipes on the true k3_on_4025 tip."""

from torchtitan.models.kimi_k3.config_registry import (
    kimi_k3_debugmodel,
    kimi_k3_debugmodel_32l,
)
from torchtitan.trainer import Trainer


def k3_ac_off() -> Trainer.Config:
    return kimi_k3_debugmodel()


def k3_ac_reuse() -> Trainer.Config:
    config = kimi_k3_debugmodel()
    config.model_spec.model.ac_reuse_attention = True
    return config


def _k3_cache_pp4() -> Trainer.Config:
    config = kimi_k3_debugmodel_32l()
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    config.parallelism.pipeline_parallel_first_stage_less_layers = 0
    config.parallelism.pipeline_parallel_last_stage_less_layers = 0
    return config


def k3_cache_off() -> Trainer.Config:
    return _k3_cache_pp4()


def k3_cache_offload() -> Trainer.Config:
    config = _k3_cache_pp4()
    config.model_spec.model.attn_res_cache_offload = True
    return config


# Per-rank peak-memory probe for the PP balance baseline: metrics print only
# on rank 0, so each worker reports its own peak at exit.
import atexit as _atexit
import os as _os

import torch as _torch


def _report_peak() -> None:
    if _torch.cuda.is_available():
        rank = _os.environ.get("RANK", "?")
        peak = _torch.cuda.max_memory_reserved() / 2**30
        print(f"[PEAK] rank {rank} max_reserved={peak:.2f}GiB", flush=True)


_atexit.register(_report_peak)


def k3_noac() -> Trainer.Config:
    """The debug model with activation checkpointing fully off: the arm where
    the attention-residual checkpoint wrap is visible in memory."""
    config = kimi_k3_debugmodel()
    config.activation_checkpoint = None
    return config


def _k3_cache_pp4_mb8() -> Trainer.Config:
    """pp4 with 8 in-flight microbatches, so warmup imbalance actually exists."""
    config = _k3_cache_pp4()
    config.parallelism.num_pp_microbatches = 8
    return config


def k3_cache_off_mb8() -> Trainer.Config:
    return _k3_cache_pp4_mb8()


def k3_ppbal() -> Trainer.Config:
    """The pp4 arm with activation balancing: ranks 0 and 3 (the measured-heavy
    ones) park their saved tensors on rank 1 (the measured-light one)."""
    config = _k3_cache_pp4_mb8()
    config.model_spec.model.pp_balance_source_ranks = [0, 3]
    config.model_spec.model.pp_balance_dest_rank = 1
    return config


def k3_cache_off_mb8_perturb() -> Trainer.Config:
    """The mb8 baseline plus one dummy 2 MiB CUDA allocation before training:
    a control for allocator-layout sensitivity of the KDA backward."""
    import torch as _t

    if _t.cuda.is_available():
        globals()["_dummy_hold"] = _t.empty(2 << 20, dtype=_t.uint8, device="cuda")
    return _k3_cache_pp4_mb8()
