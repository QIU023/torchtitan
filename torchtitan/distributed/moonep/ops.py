# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP dispatch, expert compute and combine as ``torch.library`` ops."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

_lib = torch.library.Library("moonep", "DEF")
_lib.define(
    "dispatch(Tensor x, Tensor weights, Tensor ids, Tensor counts) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "experts(Tensor x, Tensor w13, Tensor w2, Tensor cu_seqlens, Tensor plan_id) "
    "-> Tensor"
)
_lib.define("combine(Tensor x, Tensor plan_id, bool will_backward) -> Tensor")


@dataclass
class _Plan:
    plan: object
    compute: Callable[..., torch.Tensor] | None = None


# Ops take only tensors, so a plan crosses them as a CPU id into this table; combine removes it.
_plans: dict[int, _Plan] = {}
_next_plan_id = 0


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchtitan.distributed.moonep.moonep import current_buffer

    global _next_plan_id
    buffer, _ = current_buffer()
    hidden, route_weights, cu_seqlens, plan = buffer.dispatch(x, weights, ids, counts)
    _next_plan_id += 1
    _plans[_next_plan_id] = _Plan(plan)
    return hidden, route_weights, cu_seqlens, torch.tensor([_next_plan_id])


def _dispatch_setup_context(ctx, inputs, output) -> None:
    ctx.entry = _plans.get(int(output[3]))


def _dispatch_backward(ctx, grad_hidden, grad_route_weights, *_):
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    grad_x, grad_weights, _ = buffer.combine(
        plan=ctx.entry.plan,
        hidden_nvsh=grad_hidden.to(torch.bfloat16).contiguous(),
        route_weights_nvs=grad_route_weights.float().contiguous(),
    )
    return grad_x, grad_weights, None, None


@torch.library.impl(_lib, "experts", "CUDA")
def _experts_impl(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan_id: torch.Tensor,
) -> torch.Tensor:
    from torchtitan.distributed.moonep.moonep import current_buffer, prefetch_rows

    buffer, group = current_buffer()
    entry = _plans[int(plan_id)]
    compute = entry.compute
    assert compute is not None
    rows = prefetch_rows(buffer, entry.plan, group, w13, w2)
    with torch.no_grad():
        return compute(x, rows, cu_seqlens)


def _experts_setup_context(ctx, inputs, output) -> None:
    x, w13, w2, cu_seqlens, plan_id = inputs
    ctx.entry = _plans.get(int(plan_id))
    ctx.save_for_backward(x, w13, w2, cu_seqlens)


def _experts_backward(ctx, grad_out):
    from torchtitan.distributed.moonep.moonep import (
        current_buffer,
        prefetch_rows,
        reduce_rows,
    )

    buffer, group = current_buffer()
    x, w13, w2, cu_seqlens = ctx.saved_tensors
    plan, compute = ctx.entry.plan, ctx.entry.compute
    # The pools are shared by every layer: refill them for this plan and recompute.
    rows = {
        name: row.detach().requires_grad_()
        for name, row in prefetch_rows(buffer, plan, group, w13, w2).items()
    }
    x_leaf = x.detach().requires_grad_()
    with torch.enable_grad():
        out = compute(x_leaf, rows, cu_seqlens)
    inputs: list[torch.Tensor] = [x_leaf, *rows.values()]
    grad_x, *row_grads = torch.autograd.grad(out, inputs, grad_out)
    grad_w13, grad_w2 = reduce_rows(buffer, plan, group, dict(zip(rows, row_grads)))
    return grad_x, grad_w13.to(w13.dtype), grad_w2.to(w2.dtype), None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_impl(
    x: torch.Tensor, plan_id: torch.Tensor, will_backward: bool
) -> torch.Tensor:
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    key = int(plan_id)
    entry = _plans[key] if will_backward else _plans.pop(key)
    out, _, _ = buffer.combine(plan=entry.plan, hidden_nvsh=x)
    return out


def _combine_setup_context(ctx, inputs, output) -> None:
    ctx.entry = _plans.pop(int(inputs[1]), None)


def _combine_backward(ctx, grad_out):
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    grad_x, _, _, _ = buffer.dispatch(
        grad_out.to(torch.bfloat16).contiguous(), plan=ctx.entry.plan
    )
    return grad_x, None, None


torch.library.register_autograd(
    "moonep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "moonep::experts", _experts_backward, setup_context=_experts_setup_context
)
torch.library.register_autograd(
    "moonep::combine", _combine_backward, setup_context=_combine_setup_context
)


@torch.compiler.disable
def dispatch_tokens(
    x_SH: torch.Tensor,
    weights_SK: torch.Tensor,
    ids_SK: torch.Tensor,
    counts_E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens to expert rows; returns the rows, their weights, ``cu_seqlens`` and the plan id."""
    return torch.ops.moonep.dispatch(
        x_SH.to(torch.bfloat16),
        weights_SK.float(),
        ids_SK.to(torch.int32),
        counts_E.to(torch.int32),
    )


@torch.compiler.disable
def routed_experts(
    compute: Callable[..., torch.Tensor],
    x_RD: torch.Tensor,
    w13_e2FD: torch.Tensor,
    w2_eDF: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan_id: torch.Tensor,
) -> torch.Tensor:
    """Run ``compute`` over this rank's expert rows and the copies prefetched for the plan."""
    entry = _plans.get(int(plan_id))
    # A selective-AC recompute replays the saved output after combine has removed the plan.
    if entry is not None:
        entry.compute = compute
    return torch.ops.moonep.experts(x_RD, w13_e2FD, w2_eDF, cu_seqlens, plan_id)


@torch.compiler.disable
def combine_tokens(hidden_NH: torch.Tensor, plan_id: torch.Tensor) -> torch.Tensor:
    """Sum each token's expert rows; the backward is a dispatch on the same plan."""
    return torch.ops.moonep.combine(
        hidden_NH.to(torch.bfloat16).contiguous(),
        plan_id,
        torch.is_grad_enabled() and hidden_NH.requires_grad,
    )
