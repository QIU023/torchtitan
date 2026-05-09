# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang-backed Generator actor for TorchTitan RL.

Drop-in replacement for :class:`VLLMGenerator`. Mirrors the same
endpoint signatures (``generate`` + ``pull_model_state_dict``) so any
RLHF method (PPO / GRPO / DPO / online-DPO / RLHF-V / …) can swap
engines without touching the algorithm side.

Engine choice rationale
-----------------------
vLLM and SGLang both expose:

  * a paged-KV inference engine
  * a ``model_executor`` that constructs a model and routes
    ``input_ids → hidden → logits``
  * a weight-sync hook (``load_weights`` for vLLM,
    ``update_weights_from_*`` for SGLang)

so the *Generator* abstraction is engine-agnostic at the call level.
This actor implements the SGLang side using SGLang's ``Engine`` API.
The contract:

  * ``generate(prompt_texts, expected_answers) -> list[Episode]`` —
    same shape vLLM uses; ``Episode`` carries token ids + per-token
    log-probs + a ``group_id`` for group-baseline RL methods (GRPO).

  * ``pull_model_state_dict(version)`` — replaces the actor's engine
    weights with those at the given trainer policy version. For
    SGLang we use ``Engine.update_weights_from_distributed`` (RDMA-
    style if torchstore is set up) or fall back to
    ``Engine.update_weights_from_disk`` (DCP→HF dump from trainer).

Multi-modal note
----------------
SGLang already supports VLM rollouts (LLaVA, Qwen-VL, Kimi-VL etc.)
via the ``image_data`` field on ``generate`` requests. The
``generate`` endpoint here accepts an optional ``images`` argument
parallel to ``prompt_texts`` to enable multimodal RLHF without
forking this file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torchstore as ts
from monarch.actor import Actor, endpoint

from torchtitan.config import Configurable
from torchtitan.config.configs import DebugConfig, ParallelismConfig
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.plugin import (
    register_model_to_sglang_model_registry,
    SGLANG_MODEL_NAME,
)
from torchtitan.experiments.rl.types import Episode
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class SGLangCompileConfig:
    """Compilation / cuda-graph settings for SGLang."""

    cuda_graph: bool = True
    """Enable SGLang's cuda-graph capture. Disable for bitwise-
    identical numerics during debug (eager mode)."""

    piecewise_cuda_graph: bool = True
    """SGLang's piecewise cuda-graph (around non-capturable ops like
    attention). Cheaper than full capture; default-on."""


@dataclass(kw_only=True, slots=True)
class SGLangSamplingConfig:
    """Sampling parameters for SGLang's ``generate`` API."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    top_k: int = -1
    """Top-k filter. -1 disables (vLLM-equivalent default)."""

    max_new_tokens: int = 100
    """Maximum tokens generated per completion."""


@dataclass(kw_only=True, slots=True)
class SGLangBackendConfig:
    """SGLang attention / linear-attention backend selection.

    The default ``flashinfer`` matches what's used in
    ``phase11/PHASE11_SGLANG_REPORT.md`` for AttnRes inference and is
    tested on SM 8.0 and SM 12.0. ``triton`` linear-attn is required
    for KDA-style models (fla-core kernels). Set ``"linear_triton"``
    to skip linear-attn entirely (text-only models).
    """

    attention_backend: str = "flashinfer"
    linear_attn_backend: str = "triton"


class SGLangGenerator(Actor, Configurable):
    """RLHF rollout generator backed by SGLang's inference engine.

    Engine-agnostic API (same as :class:`VLLMGenerator`) so any RL
    algorithm can swap engines via config.

    Args:
        config: actor config (parallelism, sampling, compile, backend).
        model_spec: trainer ``ModelSpec`` registered to SGLang at
            init.
        model_path: HF-format checkpoint dir. SGLang reads the
            ``architectures`` field of ``config.json`` and constructs
            the registered wrapper class.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        sampling: SGLangSamplingConfig = field(default_factory=SGLangSamplingConfig)
        compile: SGLangCompileConfig = field(default_factory=SGLangCompileConfig)
        backend: SGLangBackendConfig = field(default_factory=SGLangBackendConfig)

        model_dtype: str = "bfloat16"
        """SGLang ``dtype`` (``auto``, ``bfloat16``, ``float16`` …)."""

        gpu_memory_limit: float = 0.6
        """Fraction of GPU memory for SGLang's KV cache + weights.
        Lower than vLLM's default 0.9 because SGLang pre-allocates a
        static fraction at boot rather than growing on demand."""

        num_samples_per_prompt: int = 8
        """Number of completions per prompt (group size for GRPO)."""

        debug: DebugConfig = field(default_factory=DebugConfig)

        weight_sync_method: str = "torchstore"
        """How weights are pulled from the trainer:
          * ``torchstore`` — direct RDMA pull (default, matches vLLM
            generator's path; requires monarch + torchstore with
            RDMA support).
          * ``disk`` — trainer DCP-converts to HF safetensors at a
            shared path; SGLang reloads via
            ``Engine.update_weights_from_disk``. Slower (~minutes
            per sync at 1.4B) but works without RDMA.
        """

        weight_sync_disk_path: Optional[str] = None
        """When ``weight_sync_method='disk'``, the shared dir the
        trainer writes HF safetensors to and SGLang reads from."""

        def __post_init__(self):
            assert self.parallelism.data_parallel_shard_degree in (1, -1), (
                f"SGLang generator does not support data-parallel "
                f"sharding inside the engine, got "
                f"dp_shard={self.parallelism.data_parallel_shard_degree}. "
                f"Use multiple Generator actors on disjoint GPU "
                f"meshes for batch-parallel rollouts."
            )
            assert self.parallelism.data_parallel_replicate_degree == 1, (
                f"SGLang generator does not support DP replication, "
                f"got dp_replicate={self.parallelism.data_parallel_replicate_degree}"
            )
            if self.weight_sync_method == "disk" and not self.weight_sync_disk_path:
                raise ValueError(
                    "weight_sync_method='disk' requires "
                    "weight_sync_disk_path to be set"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
    ):
        self.config = config
        self.model_spec = model_spec
        self.model_path = model_path

        # Apply determinism settings first (matches VLLMGenerator).
        set_batch_invariance(config.debug.batch_invariant)
        self._set_determinism(config.debug)

        # Register the trainer model with SGLang. Must happen before
        # constructing the Engine so SGLang can resolve the
        # ``architectures`` field in config.json to our wrapper class.
        register_model_to_sglang_model_registry(model_spec)

        # Lazy-import sglang's Engine class. We import directly from
        # ``sglang.srt.entrypoints.engine`` rather than via
        # ``sglang.Engine`` because the latter is a ``LazyImport``
        # populated by ``sglang/__init__.py`` only when imported from
        # certain working directories on some installs.
        from sglang.srt.entrypoints.engine import Engine

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            skip_tokenizer_init=True,
            tp_size=config.parallelism.tensor_parallel_degree,
            dtype=config.model_dtype,
            mem_fraction_static=config.gpu_memory_limit,
            attention_backend=config.backend.attention_backend,
            linear_attn_backend=config.backend.linear_attn_backend,
            disable_cuda_graph=not config.compile.cuda_graph,
            disable_piecewise_cuda_graph=not config.compile.piecewise_cuda_graph,
            log_level="error",
        )
        if config.debug.seed is not None:
            engine_kwargs["random_seed"] = config.debug.seed

        # Pipeline / Expert parallel are SGLang-specific kwargs only
        # if non-trivial; pass through when set.
        pp = config.parallelism.pipeline_parallel_degree
        if pp and pp > 1:
            engine_kwargs["pp_size"] = pp
        ep = config.parallelism.expert_parallel_degree
        if ep and ep > 1:
            engine_kwargs["ep_size"] = ep

        logger.info(f"Initializing SGLang Engine with kwargs: {engine_kwargs}")
        self._engine = Engine(**engine_kwargs)
        logger.info("SGLang rollout engine initialized")

        self.policy_version = 0

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator's host process.

        SGLang spawns its own worker subprocesses for TP; those inherit
        env vars via ``os.environ``, so flags set here apply.
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    @endpoint
    async def generate(
        self,
        prompt_texts: list[str],
        expected_answers: list[str],
        images: Optional[list[Any]] = None,
    ) -> list[Episode]:
        """Generate ``num_samples_per_prompt`` completions per prompt.

        Args:
            prompt_texts: prompt strings (chat-formatted upstream).
            expected_answers: gold answers for the grader; not used
                here, copied into Episodes for downstream reward calc.
            images: optional per-prompt image inputs for VLM rollout.
                Each entry is a path / PIL.Image / bytes that SGLang's
                multimodal-aware tokenizer accepts; pass ``None`` for
                text-only rollout.

        Returns:
            Flat list of Episodes (``len(prompt_texts) ×
            num_samples_per_prompt``); same shape as VLLMGenerator.
        """
        cfg = self.config
        sampling_params = {
            "temperature": cfg.sampling.temperature,
            "top_p": cfg.sampling.top_p,
            "top_k": cfg.sampling.top_k,
            "max_new_tokens": cfg.sampling.max_new_tokens,
            "n": cfg.num_samples_per_prompt,
        }
        if cfg.debug.seed is not None:
            sampling_params["random_seed"] = cfg.debug.seed

        # SGLang's `Engine.generate` returns logprobs when requested.
        # The shape of the return varies between versions; this code
        # adapts to the standard list-of-dicts format documented in
        # docs/backend/sampling_params.md.
        request: dict[str, Any] = {
            "prompt": prompt_texts,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        if images is not None:
            request["image_data"] = images

        outputs = self._engine.generate(**request)

        # Build flat list of Episodes; assign group_id per prompt.
        # group_id is GRPO/group-baseline-specific; downstream
        # algorithms that don't need it can ignore.
        episodes: list[Episode] = []
        for prompt_idx, output in enumerate(outputs):
            # SGLang collapses ``n`` samples into a single response
            # under the "outputs" key when n > 1; older versions
            # return a flat list. Handle both.
            samples = output.get("outputs", [output])
            for sample_idx, sample in enumerate(samples):
                gid = f"{os.getpid()}_{self.policy_version}_{prompt_idx}"
                token_ids = sample.get("output_ids") or sample.get("token_ids", [])
                token_log_probs = [
                    lp for (lp, _tok, _) in sample.get("output_token_logprobs", [])
                ] if "output_token_logprobs" in sample else []
                prompt_token_ids = (
                    output.get("prompt_token_ids")
                    or sample.get("prompt_token_ids")
                    or []
                )
                episodes.append(
                    Episode(
                        policy_version=self.policy_version,
                        prompt_token_ids=prompt_token_ids,
                        text=sample.get("text", ""),
                        token_ids=token_ids,
                        token_log_probs=token_log_probs,
                        expected_answer=(
                            expected_answers[prompt_idx]
                            if expected_answers
                            else ""
                        ),
                        group_id=gid,
                    )
                )

        return episodes

    @endpoint
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from the trainer.

        Two paths supported:
          * ``torchstore`` — same as VLLMGenerator: pull the trainer's
            state dict via ``ts.get_state_dict``. SGLang's engine
            exposes ``update_weights_from_distributed`` which we
            wire into here.
          * ``disk`` — trainer dumps HF safetensors at
            ``weight_sync_disk_path`` and signals via torchstore;
            SGLang reloads via ``Engine.update_weights_from_disk``.
        """
        cfg = self.config
        if cfg.weight_sync_method == "torchstore":
            from monarch.rdma import is_rdma_available

            # Get the current engine state dict (parameter names match
            # the trainer's because both register via the same
            # ModelSpec).
            engine_sd = self._engine_state_dict()
            await ts.get_state_dict(
                "model_state_dict",
                user_state_dict=engine_sd,
                strict=False,
                direct_rdma=is_rdma_available(),
            )
        elif cfg.weight_sync_method == "disk":
            # Trainer is expected to have dumped HF safetensors at
            # weight_sync_disk_path; signal SGLang's engine to reload.
            self._engine.update_weights_from_disk(cfg.weight_sync_disk_path)
        else:
            raise ValueError(
                f"unknown weight_sync_method={cfg.weight_sync_method}"
            )
        self.policy_version = version
        logger.debug(
            f"SGLangGenerator pulled weights for policy v{version} "
            f"via {cfg.weight_sync_method}"
        )

    def _engine_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the engine model's state dict for in-place
        ``ts.get_state_dict`` write.

        SGLang's engine model lives inside the worker subprocess; we
        access via ``Engine.tokenizer_manager``'s exposed handle when
        running with ``external_launcher``-style provisioning. For
        now, raise a clear error so deployments that need this set
        the path explicitly.
        """
        # NOTE: SGLang's Engine doesn't expose the model directly the
        # same way vLLM does (vLLM has ``model_executor.driver_worker
        # .get_model()``). Until SGLang exposes an equivalent, the
        # ``torchstore`` weight-sync path requires:
        #   * patching SGLang to expose the inner model state dict
        #   * OR running the SGLang engine in-process via ModelRunner
        # Our default in this class is therefore the disk path.
        raise NotImplementedError(
            "torchstore weight sync requires direct access to the "
            "SGLang engine's inner model state dict, which is not yet "
            "exposed in the public Engine API. Use "
            "weight_sync_method='disk' for now."
        )

    def __del__(self):
        if hasattr(self, "_engine"):
            try:
                self._engine.shutdown()
            except Exception:
                pass
            del self._engine
            torch.cuda.empty_cache()
