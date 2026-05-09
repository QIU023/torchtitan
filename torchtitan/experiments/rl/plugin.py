# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference-engine plugins for TorchTitan models.

Both vLLM and SGLang share the model-spec convention: the trainer's
``ModelSpec`` is registered with the engine under a stable name so the
engine can construct a wrapped model that matches the trainer's
parameter layout for weight sync.

Usage:
    # vLLM
    from torchtitan.experiments.rl.plugin import (
        register_model_to_vllm_model_registry,
    )
    register_model_to_vllm_model_registry(model_spec)

    # SGLang
    from torchtitan.experiments.rl.plugin import (
        register_model_to_sglang_model_registry,
    )
    register_model_to_sglang_model_registry(model_spec)
"""

from torchtitan.protocols.model_spec import ModelSpec

# Model-agnostic name used for vLLM model registration.
# Must match the hf_overrides["architectures"] value passed to EngineArgs.
VLLM_MODEL_NAME = "TorchTitanCausalLM"

# Model-agnostic name used for SGLang model registration. SGLang reads
# from the HF config's ``architectures`` field; the wrapper class is
# registered into ``sglang.srt.models.registry`` under this name.
SGLANG_MODEL_NAME = "TorchTitanCausalLM"


def register_model_to_vllm_model_registry(
    model_spec: ModelSpec,
) -> None:
    """
    Register a TorchTitan model with vLLM's ModelRegistry.

    Must be called before creating a vLLM engine that uses this model.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and components
    """
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

    from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper

    logger = init_logger(__name__)

    # Create dynamic model class capturing ModelSpec in the closure
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_spec=model_spec,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    # Set the class name so vLLM can identify it
    TorchTitanVLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    TorchTitanVLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME

    # Register with vLLM
    ModelRegistry.register_model(VLLM_MODEL_NAME, TorchTitanVLLMModelFromSpec)

    logger.info(
        f"Registered {VLLM_MODEL_NAME} with vLLM "
        f"(model={model_spec.name}, flavor={model_spec.flavor})"
    )


def register_model_to_sglang_model_registry(
    model_spec: ModelSpec,
) -> None:
    """Register a TorchTitan model with SGLang's model registry.

    Mirrors :func:`register_model_to_vllm_model_registry` for the
    SGLang engine. SGLang's model-class loader reads from
    ``HF config['architectures']``; setting that to ``SGLANG_MODEL_NAME``
    and registering our wrapper here lets SGLang construct the model
    with the same parameter layout the trainer uses.

    Must be called before constructing an :class:`SGLangGenerator`.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and
            components.
    """
    import logging

    from torchtitan.experiments.rl.models.sglang_wrapper import (
        TorchTitanSGLangModelWrapper,
    )

    logger = logging.getLogger(__name__)

    # Create dynamic model class capturing ModelSpec in the closure.
    class TorchTitanSGLangModelFromSpec(TorchTitanSGLangModelWrapper):
        def __init__(self, *, config, quant_config=None, prefix=""):
            super().__init__(
                model_spec=model_spec,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            )

    TorchTitanSGLangModelFromSpec.__name__ = SGLANG_MODEL_NAME
    TorchTitanSGLangModelFromSpec.__qualname__ = SGLANG_MODEL_NAME

    # SGLang model registry: ``ModelRegistry.models`` is a dict in
    # newer versions, the recommended path is ``register`` on the
    # registry singleton.
    try:
        from sglang.srt.models.registry import ModelRegistry as SGLangModelRegistry
        SGLangModelRegistry.register({SGLANG_MODEL_NAME: TorchTitanSGLangModelFromSpec})
    except (ImportError, AttributeError):
        # Older sglang exposes a dict directly.
        from sglang.srt.models.registry import ModelRegistry as SGLangModelRegistry
        SGLangModelRegistry.models[SGLANG_MODEL_NAME] = TorchTitanSGLangModelFromSpec

    logger.info(
        f"Registered {SGLANG_MODEL_NAME} with SGLang "
        f"(model={model_spec.name}, flavor={model_spec.flavor})"
    )
