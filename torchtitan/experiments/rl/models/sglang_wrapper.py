# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang model wrapper for TorchTitan models.

Mirrors ``vllm_wrapper.TorchTitanVLLMModelWrapper`` for the SGLang
inference engine. Where vLLM's wrapper has to thread its own attention
backend (``vLLM Attention`` class registered via ``@register_backend``),
SGLang's wrapper is much thinner: SGLang already supplies the
attention path (flashinfer/triton backends + KV cache management), so
the wrapper only needs to:

  1. Build the trainer's ``ModelSpec`` model in inference mode.
  2. Expose the standard SGLang model class API (``forward``,
     ``load_weights``, ``compute_logits``, etc.).
  3. Hand SGLang's KV-cache manager the right per-layer attention
     module reference.

This keeps the wrapper engine-specific boilerplate small and lets
SGLang's existing per-arch model adapters (KimiLinear, Qwen3, …)
serve as the implementation when the trainer uses one of those
architectures.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class TorchTitanSGLangModelWrapper(nn.Module):
    """SGLang model wrapper for a TorchTitan ``ModelSpec``.

    The wrapper builds the trainer's exact model graph (so weights
    bit-load via ``load_weights``) and adapts its forward signature to
    SGLang's expected (input_ids, positions, forward_batch) tuple.

    Args:
        model_spec: trainer-side TorchTitan ``ModelSpec`` (the same
            instance that the trainer constructs from).
        config: SGLang's per-model ``Config`` object, passed through
            from the engine (mostly carries dtype + device).
        quant_config: optional quantization config; passed through to
            sub-modules that support it.
        prefix: parameter-name prefix used for sharded weight loading
            (matches SGLang's convention).
    """

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        config,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.model_spec = model_spec
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix

        # Build the trainer model in inference mode. The model_spec's
        # builder is responsible for constructing the parameter shapes
        # the trainer uses; SGLang's engine then attaches its KV cache
        # to whichever Linear/Attention modules the model exposes.
        # NOTE: trainer-side builders typically expect `ParallelDims`,
        # which on SGLang's side comes from
        # ``sglang.srt.distributed`` (TP world from
        # ``get_tensor_model_parallel_world_size``). Builders are model-
        # specific and must be reviewed when adding a new arch.
        self.model = model_spec.cls(model_spec.config)

        logger.info(
            "TorchTitanSGLangModelWrapper built "
            f"(model_spec={model_spec.name}, flavor={model_spec.flavor}, "
            f"prefix={prefix!r})"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        **kwargs,
    ) -> torch.Tensor:
        """SGLang-shape forward.

        SGLang's model-runner calls ``forward(input_ids, positions,
        forward_batch)``. Trainer models typically take ``input_ids``
        only. We forward the extra args via kwargs so model-specific
        wrappers can inspect ``forward_batch`` for KV-cache state.
        """
        return self.model(input_ids, positions=positions, forward_batch=forward_batch)

    def load_weights(self, weights):
        """Load weights via the trainer's state-dict naming.

        SGLang ships a default ``load_weights`` on its model classes;
        we delegate to the trainer model's ``load_state_dict`` so the
        loader sees the *trainer's* parameter names. Cross-engine
        weight sync (TorchStore RDMA, DCP-disk, …) writes to those
        names so loading is name-equivalent.
        """
        # weights is an iterable of (name, tensor); convert to dict.
        sd = {n: t for n, t in weights}
        self.model.load_state_dict(sd, strict=False)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Forward the lm head. Delegate to model if it has its own
        ``compute_logits``, otherwise call ``lm_head`` directly."""
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return self.model.lm_head(hidden_states)
