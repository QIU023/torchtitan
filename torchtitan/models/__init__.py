# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_models = frozenset(
    [
        "deepseek_v3",
        "flux",
        "gpt_oss",
        "kimi_k2_7",
        "kimi_k3",
        # Vendored copy of the upstream K3 reference model, registered alongside ours so
        # both run in the same tree during the alignment migration.
        "kimi_k3_up",
        "llama3",
        "muse_glimmer",
        "qwen3",
        "qwen3_5",
    ]
)
