# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan RL training with engine-agnostic rollouts.

Inference engines are imported lazily. Importing this package no longer
pulls in vLLM (or SGLang); each engine wrapper is loaded only when its
:func:`register_model_to_*` helper is called.

Usage:
    # vLLM rollout (default)
    from torchtitan.experiments.rl.plugin import (
        register_model_to_vllm_model_registry,
    )
    register_model_to_vllm_model_registry(model_spec)

    # SGLang rollout (engine-agnostic Generator path)
    from torchtitan.experiments.rl.plugin import (
        register_model_to_sglang_model_registry,
    )
    register_model_to_sglang_model_registry(model_spec)

The lazy-import pattern was introduced so the framework can be used
in environments where only one of {vLLM, SGLang} is installed (e.g.
SGLang's sgl_kernel ABI is locked to torch 2.9 stable, while vLLM's
pre-built wheels are PyTorch nightly only).
"""

from torchtitan.experiments.rl.plugin import (
    register_model_to_sglang_model_registry,
    register_model_to_vllm_model_registry,
)


__all__ = [
    "register_model_to_sglang_model_registry",
    "register_model_to_vllm_model_registry",
]
