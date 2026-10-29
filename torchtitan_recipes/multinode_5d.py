# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-node 5D parallelism experiments on the Kimi K3 and MiniMax M3 downscales.

Launch with ``-m torchtitan.train --module torchtitan_recipes.multinode_5d
--config cell``; environment variables select the cell:

- ``MN_MODEL``: ``k3`` (default) or ``m3``.
- ``MN_FLAVOR``: a flavor of that model, default ``k3_16x80g`` / ``m3_16x80g``.
- ``MN_CELL``: degrees such as ``pp=2,dp=2,cp=2,tp=2,ep=8``; missing keys are 1,
  ``dpr`` is the HSDP replicate degree.
- ``MN_SEQ``: tokens per sequence, also the micro-batch per data-parallel rank.
- ``MN_MICROBATCHES``: sequences per data-parallel rank and step (pipeline
  micro-batches under PP, gradient accumulation without); by default the count
  nearest 524288 tokens per step, rounded to a multiple of pp.
- ``MN_SCHEDULE``: ``1F1B`` (default) or ``Interleaved1F1B``.
- ``MN_CP``: ``allgather`` (default, head-tail load balancing) or ``ulysses``.
- ``MN_FSDP_KEEP=1``: no reshard after forward, deferred gradient reduction.
- ``MN_GATE``: KDA bounded gate, ``fused`` (default) or ``reference``.
- ``MN_ASSETS``: tokenizer directory for the downscale flavors; ``MN_SPECIAL``:
  ``model`` (default) or ``qwen`` special tokens.
- ``MN_DATASET`` (``c4``), ``MN_STEPS`` (20), ``MN_WARMUP`` (5), ``MN_LR``
  (3e-4), ``MN_DTYPE`` (``float32``), ``MN_ATTN`` (M3 full-attention backend).
- ``MN_SEED_CKPT``: model-only checkpoint to load; ``seed`` writes one to
  ``MN_SEED_OUT``.
- ``MN_PROFILE``: profile every that many steps; ``MN_DETERMINISTIC=1``.
"""

import dataclasses
import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity

_TOKENS_PER_STEP = 524288
_QWEN_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _degrees(spec: str) -> dict[str, int]:
    degrees = dict.fromkeys(("dpr", "dp", "cp", "tp", "ep", "pp"), 1)
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        if key not in degrees:
            raise ValueError(f"Unknown degree {key!r} in MN_CELL={spec!r}")
        degrees[key] = int(value)
    return degrees


def _microbatches(seq_len: int, degrees: dict[str, int]) -> int:
    explicit = os.environ.get("MN_MICROBATCHES")
    if explicit:
        return int(explicit)
    data_parallel = degrees["dp"] * degrees["dpr"]
    count = max(1, round(_TOKENS_PER_STEP / (seq_len * data_parallel)))
    pp = degrees["pp"]
    return max(pp, round(count / pp) * pp)


def _parallelism(degrees: dict[str, int], microbatches: int) -> ParallelismConfig:
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=degrees["dpr"],
        data_parallel_shard_degree=degrees["dp"],
        context_parallel_degree=degrees["cp"],
        tensor_parallel_degree=degrees["tp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=degrees["pp"],
        enable_sequence_parallel=degrees["tp"] > 1,
    )
    if degrees["pp"] > 1:
        parallelism.pipeline_parallel_schedule = _env("MN_SCHEDULE", "1F1B")
        parallelism.num_pp_microbatches = microbatches
    if _env("MN_FSDP_KEEP", "0") == "1":
        parallelism.fsdp_reshard_after_forward = "never"
        parallelism.fsdp_defer_gradient_reduction = True
    if degrees["cp"] > 1 and _env("MN_CP", "allgather") == "allgather":
        from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer

        parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    return parallelism


def _scaffold(model: str) -> Trainer.Config:
    # The test recipes carry the model's data, loss and remat wiring; the
    # parallelism, model, data and optimizer are replaced below.
    if model == "k3":
        from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

        return kimi_k3_debugmodel(attn_backend="flex")
    if model == "m3":
        from torchtitan_recipes.tests.models.minimax_m3 import minimax_m3_debugmodel

        return minimax_m3_debugmodel()
    raise ValueError(f"MN_MODEL must be k3 or m3, got {model!r}")


def _model_config(model: str, flavor: str, seq_len: int, parallelism):
    if model == "k3":
        from torchtitan.models.kimi_k3 import build_model_config

        return build_model_config(
            flavor, attn_backend="flex", parallelism=parallelism, seq_len=seq_len
        )
    from torchtitan.models.minimax_m3 import build_model_config

    return build_model_config(
        flavor,
        parallelism=parallelism,
        seq_len=seq_len,
        attn_backend=_env("MN_ATTN", "varlen"),
    )


def _special_tokens(model: str) -> dict[str, str]:
    if _env("MN_SPECIAL", "model") == "qwen":
        return _QWEN_SPECIAL_TOKENS
    if model == "k3":
        from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS

        return KIMI_K3_SPECIAL_TOKENS
    from torchtitan.models.minimax_m3 import MINIMAX_M3_SPECIAL_TOKENS

    return MINIMAX_M3_SPECIAL_TOKENS


def _apply_context_parallel(config: Trainer.Config, model: str) -> Trainer.Config:
    if model != "k3":
        raise NotImplementedError("MiniMax M3 has no context parallelism yet.")
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import MLAAttention
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.cp_mla import (
        AllGatherKVMLAFlexAttention,
        UlyssesMLAAttention,
    )
    from torchtitan.models.common.attention.kda import InnerKDA

    mla = (
        UlyssesMLAAttention
        if _env("MN_CP", "allgather") == "ulysses"
        else AllGatherKVMLAFlexAttention
    )
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    MLAAttention: mla,
                    InnerKDA: ContextParallelInnerKDA,
                }
            )
        ],
    )


def _set_kda_gate(config: Trainer.Config, impl: str) -> int:
    from torchtitan.models.common.attention.kda import KDAKernel

    seen: set[int] = set()
    count = 0

    def visit(node) -> None:
        nonlocal count
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, KDAKernel.Config):
            object.__setattr__(node, "gate_impl", impl)
            count += 1
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                visit(getattr(node, field.name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for item in node.values():
                visit(item)

    visit(config.model)
    return count


def _build(degrees: dict[str, int]) -> Trainer.Config:
    model = _env("MN_MODEL", "k3")
    flavor = _env("MN_FLAVOR", f"{model}_16x80g")
    seq_len = int(_env("MN_SEQ", "8192"))
    microbatches = _microbatches(seq_len, degrees)
    tokens = seq_len * degrees["dp"] * degrees["dpr"] * microbatches
    parallelism = _parallelism(degrees, microbatches)

    config = _scaffold(model)
    model_config = _model_config(model, flavor, seq_len, parallelism)
    config.model = model_config
    config.parallelism = parallelism
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
    )
    if not flavor.startswith("debugmodel"):
        config.hf_assets_path = os.environ["MN_ASSETS"]
        config.tokenizer = MultiModalTokenizer.Config(**_special_tokens(model))
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=DATASETS[_env("MN_DATASET", "c4")]
        ),
        max_num_documents=c4_document_capacity(seq_len),
        shuffle=False,
    )
    config.optim = Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=float(_env("MN_LR", "3e-4")))]
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=int(_env("MN_WARMUP", "5")),
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
    )
    training = config.training
    training.max_context_length = seq_len
    training.num_tokens_per_microbatch_per_dp_rank = seq_len
    training.num_tokens_per_train_step = tokens
    training.steps = int(_env("MN_STEPS", "20"))
    training.dtype = _env("MN_DTYPE", "float32")
    training.disable_cuda_graphs = True
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = _env("MN_DETERMINISTIC", "0") == "1"
    if os.environ.get("MN_PROFILE"):
        config.profiler.enable_profiling = True
        config.profiler.profile_freq = int(os.environ["MN_PROFILE"])
    config.checkpointer = None
    if os.environ.get("MN_SEED_CKPT"):
        config.checkpointer = CheckpointManager.Config(
            interval=10**9,
            initial_load_path=os.environ["MN_SEED_CKPT"],
            initial_load_model_only=True,
        )

    if degrees["cp"] > 1:
        config = _apply_context_parallel(config, model)
    gate = _env("MN_GATE", "fused")
    kernels = _set_kda_gate(config, gate) if model == "k3" else 0
    if model == "k3" and kernels == 0:
        raise RuntimeError("No KDAKernel config found to set the gate on.")

    if os.environ.get("RANK", "0") == "0":
        print(
            f"[multinode_5d] model={model} flavor={flavor} cell={degrees} "
            f"seq={seq_len} microbatches={microbatches} tokens/step={tokens} "
            f"schedule={_env('MN_SCHEDULE', '1F1B')} cp={_env('MN_CP', 'allgather')} "
            f"fsdp_keep={_env('MN_FSDP_KEEP', '0')} gate={gate} kda_kernels={kernels} "
            f"dataset={_env('MN_DATASET', 'c4')} dtype={training.dtype}",
            flush=True,
        )
    return config


def cell() -> Trainer.Config:
    return _build(_degrees(os.environ["MN_CELL"]))


def seed() -> Trainer.Config:
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["MN_SEED_OUT"])
    config.create_seed_checkpoint = True
    return config
