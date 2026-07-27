# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing for the Kimi K3 MoE router (K3 delta, provisional).

K3 replaces DeepSeek's aux-loss-free sign-based expert-bias correction
(``coeff * sign(mean - load)``, torchtitan
``register_moe_load_balancing_hook``) with a Quantile Balancing method.
The exact K3 algorithm is blog-only; this is a faithful PROVISIONAL
form of the CONCEPT: bias each expert by its position in the load
distribution's empirical CDF rather than a binary above/below-mean
sign, giving a smooth, magnitude-aware balancing signal.

    rank_e = (empirical CDF position of expert e's load) in [0, 1]
    delta_e = coeff * (target_quantile - rank_e)     # target 0.5 = median
    delta   = delta - mean(delta)                     # zero-sum (like DSv3)

Underused experts (low load -> low CDF rank) get boosted proportionally
to how deep in the low-load tail they sit; overused experts (high rank)
are suppressed. Graftable (training-time bias, no weight-structure
change, so it is graftable onto pretrained weights). Exact form
reconciles at the official config release.
"""

import torch


def quantile_balance_delta(
    tokens_per_expert_E: torch.Tensor,
    coeff: float,
    target_quantile: float = 0.5,
) -> torch.Tensor:
    """Zero-sum expert-bias delta from load quantile positions.

    Args:
        tokens_per_expert_E: (E,) token count routed to each expert.
        coeff: update magnitude (analogue of load_balance_coeff).
        target_quantile: the load-CDF position pulled toward (0.5=median).
    """
    load = tokens_per_expert_E.float()
    E = load.numel()
    # Empirical CDF rank of each expert's load in [0, 1). Ties get an
    # averaged rank so equal loads receive equal bias.
    order = load.argsort()
    ranks = torch.empty_like(load)
    ranks[order] = torch.arange(E, device=load.device, dtype=load.dtype)
    # average ties
    uniq, inv, counts = torch.unique(
        load, return_inverse=True, return_counts=True
    )
    csum = torch.cumsum(counts, 0)
    start = csum - counts
    avg_rank = (start + (counts - 1) / 2.0)[inv]
    cdf = avg_rank / max(E - 1, 1)
    delta = coeff * (target_quantile - cdf)
    return delta - delta.mean()


def register_quantile_balancing_hook(
    optimizers,
    model_parts,
    parallel_dims,
    target_quantile: float = 0.5,
):
    """Optimizer pre-step hook that applies Quantile Balancing to every
    MoE layer's expert_bias_E (drop-in alternative to
    ``register_moe_load_balancing_hook``). Reduces token counts across
    the loss mesh first (same as the core hook), then applies the
    quantile delta and resets counters.
    """
    import torch.nn as nn

    def _update(*_a, **_k):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        with torch.no_grad():
            for model_part in model_parts:
                layers = model_part.get_submodule("layers")
                assert isinstance(layers, nn.ModuleDict)
                for block in layers.values():
                    moe = getattr(block, "moe", None) or getattr(
                        getattr(block, "ffn", None), "_moe", None
                    )
                    if moe is None or not getattr(block, "moe_enabled", True):
                        continue
                    coeff = getattr(moe, "load_balance_coeff", None)
                    if coeff is None:
                        continue
                    tpe = moe.tokens_per_expert_E
                    if isinstance(tpe, torch.distributed.tensor.DTensor):
                        tpe = tpe.full_tensor()
                    if loss_mesh is not None:
                        torch.distributed.all_reduce(
                            tpe, group=loss_mesh.get_group()
                        )
                    moe.expert_bias_E.add_(
                        quantile_balance_delta(tpe, coeff, target_quantile)
                    )
                    moe.tokens_per_expert_E.zero_()

    optimizers.register_step_pre_hook(_update)
    return optimizers
