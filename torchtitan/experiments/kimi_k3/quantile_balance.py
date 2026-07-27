# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing for the Kimi K3 MoE router (tech report sec 2.3.3).

Auxiliary-loss-free routing adds a per-expert bias ``b`` to the router score
used for Top-k SELECTION only::

    T_i = argtopk(s_i + b),    p_ij = s_ij / sum_{r in T_i} s_ir      (Eq. 13)

so ``b`` regulates dispatch without altering the mixture weights or the
router's gradients. The original rule nudges the bias by a fixed step,
``b_j += gamma * sign(mean_load - load_j)``, where gamma trades slow adaptation
against load oscillation -- and that trade-off gets worse as the routed pool
grows to 896 experts per layer.

Quantile Balancing removes the step size by SOLVING for the bias rather than
nudging it. Route with Top-(k+1) on the biased score: the first k entries are
the routes actually taken, and the (k+1)-th is the cutoff ``alpha_i`` that an
expert must exceed to enter token i's Top-k. Under a candidate bias the count
routed to expert j is ``sum_i 1[s_ij + b_j > alpha_i]``, monotonically
decreasing in ``-b_j``, so setting that count to the target load
``q = m*k/n`` puts ``-b_j`` at the (q+1)-th largest margin ``s_ij - alpha_i``.
Since ``q/m = k/n``::

    b_hat_j = -quantile_{1-k/n}( s_{:,j} - alpha )                    (Eq. 14)
    b       = b_hat - mean(b_hat)

The second line removes a common offset, which leaves Top-k selection
unchanged. The update takes effect on the NEXT step -- a batch is never routed
with a bias derived from itself -- and the bias is frozen at inference.

At scale the quantile spans the whole global batch (millions of margins across
ranks and accumulation steps), so gathering them exactly is not viable.
:func:`quantile_balance_bias_histogram` instead reads each expert's quantile
from pooled histograms of its margins: counts are additive, so one all-reduce
of the per-rank bin counts represents the whole global batch regardless of how
tokens are sharded, and the estimate is exact up to the bin width.

Superseded: this module previously shipped a PROVISIONAL rule that took the
quantile of the expert LOAD distribution (a smooth stand-in for the sign rule).
That is a different algorithm -- it still nudges by a coefficient, whereas QB
solves for the bias that hits the target load exactly.
"""

from __future__ import annotations

import torch


def topk_with_cutoff(
    scores_TE: torch.Tensor,
    bias_E: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-(k+1) routing: the k taken routes, plus the cutoff.

    Args:
        scores_TE: ``(T, E)`` raw router scores ``s = Sigmoid(W_r x)``.
        bias_E: ``(E,)`` current expert bias (selection only).
        top_k: ``k``.

    Returns:
        ``(expert_ids_TK, cutoff_T)``. The cutoff is the ``(k+1)``-th biased
        score, i.e. the threshold an expert must exceed to enter that token's
        Top-k; taking it from Top-(k+1) routing avoids a separate token-side
        quantile.
    """
    E = scores_TE.size(-1)
    if top_k + 1 > E:
        raise ValueError(
            f"Quantile Balancing routes with Top-(k+1), so top_k+1="
            f"{top_k + 1} must not exceed num_experts={E}"
        )
    vals, ids = torch.topk(scores_TE + bias_E, top_k + 1, dim=-1)
    return ids[..., :top_k], vals[..., top_k]


def quantile_balance_bias(
    scores_TE: torch.Tensor,
    cutoff_T: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Exact QB bias (Eq. 14). Reference form for small batches and tests.

    Args:
        scores_TE: ``(T, E)`` raw router scores.
        cutoff_T: ``(T,)`` cutoffs from :func:`topk_with_cutoff`.
        top_k: ``k``.

    Returns:
        ``(E,)`` zero-mean bias, to be used on the NEXT step.
    """
    n = scores_TE.size(-1)
    margins_TE = (scores_TE - cutoff_T.unsqueeze(-1)).float()
    # Per expert, over tokens. "lower" interpolation keeps the result on an
    # actual margin value, which is what makes the count land on the target
    # exactly rather than between two order statistics.
    b_hat = -torch.quantile(
        margins_TE, 1.0 - top_k / n, dim=0, interpolation="lower"
    )
    return b_hat - b_hat.mean()


def margin_histogram(
    scores_TE: torch.Tensor,
    cutoff_T: torch.Tensor,
    *,
    num_bins: int = 512,
    lo: float = -1.0,
    hi: float = 1.0,
) -> torch.Tensor:
    """Per-expert histogram of the margins ``s_{:,j} - alpha``.

    Counts are ADDITIVE across ranks and accumulation steps, which is what
    lets one all-reduce reconstruct the whole-batch distribution.

    Returns:
        ``(E, num_bins)`` int64 counts. Margins outside ``[lo, hi]`` are
        clamped into the end bins, so no token is dropped.
    """
    E = scores_TE.size(-1)
    margins_TE = (scores_TE - cutoff_T.unsqueeze(-1)).float()
    edges = torch.linspace(lo, hi, num_bins + 1, device=margins_TE.device)
    idx = torch.bucketize(margins_TE.clamp(lo, hi), edges[1:-1])
    counts = torch.zeros(E, num_bins, dtype=torch.long, device=margins_TE.device)
    idx_ET = idx.t().contiguous()
    counts.scatter_add_(1, idx_ET, torch.ones_like(idx_ET))
    return counts


def quantile_balance_bias_histogram(
    counts_EB: torch.Tensor,
    top_k: int,
    *,
    lo: float = -1.0,
    hi: float = 1.0,
) -> torch.Tensor:
    """QB bias read from pooled margin histograms -- the method used at scale.

    Args:
        counts_EB: ``(E, num_bins)`` pooled counts. Sum the per-rank
            histograms with a single all-reduce before calling this.
        top_k: ``k``.

    Returns:
        ``(E,)`` zero-mean bias for the next step, exact up to the bin width.
    """
    E, num_bins = counts_EB.shape
    total = counts_EB.sum(dim=1, keepdim=True).clamp(min=1)
    target = 1.0 - top_k / E
    cdf = counts_EB.cumsum(dim=1).float() / total.float()
    bin_idx = (cdf < target).sum(dim=1).clamp(max=num_bins - 1)
    edges = torch.linspace(lo, hi, num_bins + 1, device=counts_EB.device)
    b_hat = -edges[bin_idx]
    return b_hat - b_hat.mean()


def expert_loads(
    scores_TE: torch.Tensor,
    bias_E: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """``(E,)`` token count each expert would receive under ``bias_E``.

    The property QB is defined by: applying the bias it returns should make
    these counts hit the target load ``q = m*k/n``.
    """
    ids, _ = topk_with_cutoff(scores_TE, bias_E, top_k)
    return torch.bincount(ids.reshape(-1), minlength=scores_TE.size(-1))
