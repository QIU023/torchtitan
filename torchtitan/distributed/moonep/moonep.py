# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP primitives for MoE expert parallelism: one ``Buffer`` and one set of
NVLink expert pools per process, shared by every MoE layer."""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    import moonep
    from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
except ImportError as e:
    raise ImportError(
        "MoonEP is required for this module. Install from: "
        "https://github.com/MoonshotAI/MoonEP"
    ) from e

_PROJECTIONS = ("gate", "up", "down")

_buffer: "moonep.Buffer | None" = None
_buffer_key: tuple | None = None
_pools: dict[tuple, torch.Tensor] = {}


def get_buffer(
    group: ProcessGroup,
    *,
    num_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> "moonep.Buffer":
    """Get or create the process-global MoonEP ``Buffer``."""
    global _buffer, _buffer_key
    key = (group, num_tokens_per_rank, hidden, num_topk, num_experts)
    if _buffer is not None:
        if _buffer_key != key:
            raise ValueError(
                "MoonEP sizes its buffer statically; every MoE layer must share "
                "one token count, hidden size, top-k, expert count and EP group."
            )
        return _buffer
    _buffer = moonep.Buffer(
        S=num_tokens_per_rank,
        H=hidden,
        K=num_topk,
        E=num_experts,
        num_ep_ranks=group.size(),
        group=group,
    )
    _buffer_key = key
    return _buffer


def _pool(
    name: str,
    rows: int,
    row_shape: tuple[int, ...],
    dtype: torch.dtype,
    group: ProcessGroup,
) -> torch.Tensor:
    key = (name, rows, row_shape, dtype, group)
    pool = _pools.get(key)
    if pool is None:
        padded = pad_dim0_for_alignment([rows, *row_shape], dtype)
        size = group.size()
        pool = create_nvl_dist_tensor(
            [padded, *row_shape], dtype, dist.get_rank(group), size, group=group
        ).view(size, padded, *row_shape)[:, :rows]
        _pools[key] = pool
    return pool


def _pools_for(
    group: ProcessGroup,
    num_local_experts: int,
    shapes: dict[str, tuple[int, ...]],
    dtype: torch.dtype,
    kind: str,
) -> dict[str, torch.Tensor]:
    return {
        name: _pool(f"{kind}.{name}", 2 * num_local_experts, shapes[name], dtype, group)
        for name in _PROJECTIONS
    }


def _projections(
    w13_e2FD: torch.Tensor, w2_eDF: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {"gate": w13_e2FD[:, 0], "up": w13_e2FD[:, 1], "down": w2_eDF}


def prefetch_rows(
    buffer: "moonep.Buffer",
    plan: object,
    group: ProcessGroup,
    w13_e2FD: torch.Tensor,
    w2_eDF: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fill this rank's rows and slots for ``plan``; return the ``[2e]`` GEMM rows."""
    num_local_experts = w2_eDF.shape[0]
    local = _projections(w13_e2FD, w2_eDF)
    shapes = {name: tuple(w.shape[1:]) for name, w in local.items()}
    pools = _pools_for(group, num_local_experts, shapes, torch.bfloat16, "weight")
    rank = dist.get_rank(group)
    with torch.no_grad():
        for name in _PROJECTIONS:
            pools[name][rank, :num_local_experts].copy_(local[name])
    buffer.prefetch_weight(
        plan=plan,
        **{
            f"local_{name}_weight": pools[name][rank, :num_local_experts]
            for name in _PROJECTIONS
        },
        **{
            f"{name}_prefetch_buffer": pools[name][:, num_local_experts:]
            for name in _PROJECTIONS
        },
    )
    return {name: pools[name][rank] for name in _PROJECTIONS}


def reduce_rows(
    buffer: "moonep.Buffer",
    plan: object,
    group: ProcessGroup,
    row_grads: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Send slot gradients home; return the ``w13`` and ``w2`` gradients of this rank."""
    num_local_experts = row_grads["down"].shape[0] // 2
    shapes = {name: tuple(g.shape[1:]) for name, g in row_grads.items()}
    pools = _pools_for(group, num_local_experts, shapes, torch.float32, "grad")
    rank = dist.get_rank(group)
    for name in _PROJECTIONS:
        pools[name][rank].copy_(row_grads[name])
    buffer.reduce_grad(
        plan=plan,
        **{
            f"local_{name}_grad": pools[name][rank, :num_local_experts]
            for name in _PROJECTIONS
        },
        **{
            f"{name}_reduce_buffer": pools[name][:, num_local_experts:]
            for name in _PROJECTIONS
        },
    )
    local = {name: pools[name][rank, :num_local_experts] for name in _PROJECTIONS}
    return torch.stack([local["gate"], local["up"]], dim=1), local["down"].clone()


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, buffer, x_SH, weights_SK, ids_SK, counts_E
    ):
        hidden_NH, weights_N, cu_seqlens, plan = buffer.dispatch(
            x_SH, weights_SK, ids_SK, counts_E
        )
        ctx.buffer, ctx.plan = buffer, plan
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_NH, weights_N, cu_seqlens, plan

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_hidden_NH, grad_weights_N, *_
    ):
        grad_x_SH, grad_weights_SK, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_NH.to(torch.bfloat16).contiguous(),
            route_weights_nvs=grad_weights_N.float().contiguous(),
        )
        return None, grad_x_SH, grad_weights_SK, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, plan, hidden_NH):  # pyrefly: ignore[bad-override]
        out_SH, _, _ = buffer.combine(plan=plan, hidden_nvsh=hidden_NH)
        ctx.buffer, ctx.plan = buffer, plan
        return out_SH

    @staticmethod
    def backward(ctx, grad_out_SH):  # pyrefly: ignore[bad-override]
        grad_hidden_NH, _, _, _ = ctx.buffer.dispatch(
            grad_out_SH.to(torch.bfloat16).contiguous(), plan=ctx.plan
        )
        return None, None, grad_hidden_NH


def dispatch_tokens(
    buffer: "moonep.Buffer",
    x_SH: torch.Tensor,
    weights_SK: torch.Tensor,
    ids_SK: torch.Tensor,
    counts_E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """Route tokens to expert rows; the backward is a combine on the same plan."""
    return _Dispatch.apply(
        buffer,
        x_SH.to(torch.bfloat16),
        weights_SK.float(),
        ids_SK.to(torch.int32),
        counts_E.to(torch.int32),
    )


def combine_tokens(
    buffer: "moonep.Buffer", plan: object, hidden_NH: torch.Tensor
) -> torch.Tensor:
    """Sum each token's expert rows; the backward is a dispatch on the same plan."""
    return _Combine.apply(buffer, plan, hidden_NH.to(torch.bfloat16).contiguous())
