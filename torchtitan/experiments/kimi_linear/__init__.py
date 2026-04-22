# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Linear experiment — torchtitan-idiom port of
MoonshotAI/Kimi-Linear (MoE Transformer with KDA + MLA + AttnRes).

See ``README.md`` in this directory for scope, and
``../../phase4/README.md`` for the overall plan.

Phase 4a (current session): skeleton + reference files fetched from HF.
ModelSpec registration lands in Phase 4c once the full model class,
AttnRes subclass, and config flavors exist.
"""

from torchtitan.experiments.kimi_linear.model import (
    KimiDeltaAttention,
    KimiDecoderLayer,
    KimiLinearConfig,
    KimiLinearModel,
    KimiMLAAttention,
    KimiMLP,
    KimiMoE,
    KimiRMSNorm,
)

__all__ = [
    "KimiDeltaAttention",
    "KimiDecoderLayer",
    "KimiLinearConfig",
    "KimiLinearModel",
    "KimiMLAAttention",
    "KimiMLP",
    "KimiMoE",
    "KimiRMSNorm",
]
