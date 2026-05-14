# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only distributed regression test for AttnRes TP gradient placement.

Reproduces (in miniature, on the gloo backend) the 4D-mesh SFT bug where
``block_attn_res`` produced exploding ``grad_norm`` because the pseudo-query
projection's ``DTensor.to_local()`` defaulted its backward gradient placement
to ``Replicate`` on the tp mesh dim. Replicate tells DTensor the gradient is
already consistent across tp ranks and SKIPS the all-reduce — but the einsum's
grad w.r.t. the (replicated) query is a per-tp-rank Partial contribution that
MUST be all-reduced. The result is a mis-reduced ``proj.weight.grad`` whose
norm blows up once ``clip_grad_norm_`` stacks it across the (fsdp, tp) mesh.

The fix in ``attn_res.block_attn_res`` passes ``grad_placements=[Partial()]``
on the tp dim. This test runs ``block_attn_res`` on a ``(fsdp=1, tp=2)`` mesh
with tp-rank-divergent K/V inputs and asserts the resulting weight gradient is
identical across tp ranks (the all-reduce fired) and equals the reference
sum-of-rank-contributions. Run via:

    python -m torchtitan.experiments.attn_res.tests.test_attn_res_tp_grad
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.experiments.attn_res.attn_res import block_attn_res
from torchtitan.models.common.rmsnorm import RMSNorm


def _unit_norm(dim: int) -> RMSNorm:
    config = RMSNorm.Config(normalized_shape=dim, param_init={"weight": nn.init.ones_})
    norm = config.build()
    norm.init_states()
    return norm


WORLD_SIZE = 2
DIM = 8
B, T = 2, 3
N_BLOCKS = 2


def _worker(rank: int, world_size: int, result_queue: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # 4D-shaped mesh in miniature: (fsdp=1, tp=2). The real failing run is
        # (pp, dp_shard, tp, ep); only the tp dim governs block_attn_res's
        # query-gradient reduction, so a (fsdp, tp) mesh exercises the exact
        # code path under test.
        mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("fsdp", "tp"))

        torch.manual_seed(0)
        # proj.weight: Linear(DIM, 1) — the pseudo-query. Replicated on tp,
        # exactly as NoParallel + FSDP would place it (Shard(0) on the size-1
        # fsdp dim collapses to Replicate; Replicate on tp).
        full_w = torch.randn(1, DIM, dtype=torch.float32)

        class _Proj(torch.nn.Module):
            def __init__(self, w: torch.Tensor) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(w.clone())

        # Reference (single device, no TP): grad of the SUMMED-over-tp-ranks
        # forward. Each tp rank consumes its own divergent K/V; the true
        # gradient of the replicated query is the sum of per-rank grads.
        def _make_kv(seed: int) -> tuple[list[torch.Tensor], torch.Tensor]:
            g = torch.Generator().manual_seed(seed)
            blocks = [torch.randn(B, T, DIM, generator=g) for _ in range(N_BLOCKS - 1)]
            partial = torch.randn(B, T, DIM, generator=g)
            return blocks, partial

        norm = _unit_norm(DIM)
        # Per-rank divergent inputs — emulates K/V carrying tp-divergent
        # gradients back from rowwise-sharded o_proj / down_proj.
        all_kv = [_make_kv(100 + r) for r in range(world_size)]

        # ---- reference: plain (no DTensor) summed gradient ----
        ref_proj = _Proj(full_w)
        ref_grad = torch.zeros_like(ref_proj.weight)
        for blocks, partial in all_kv:
            ref_proj.weight.grad = None
            h = block_attn_res(blocks, partial, ref_proj, norm)
            h.sum().backward()
            ref_grad = ref_grad + ref_proj.weight.grad
        ref_grad = ref_grad.detach()

        # ---- system under test: DTensor proj on (fsdp, tp) mesh ----
        dt_proj = _Proj(full_w)
        dt_proj.weight = torch.nn.Parameter(
            distribute_tensor(dt_proj.weight.detach(), mesh, [Replicate(), Replicate()])
        )
        blocks, partial = all_kv[rank]
        h = block_attn_res(blocks, partial, dt_proj, norm)
        h.sum().backward()
        grad = dt_proj.weight.grad
        assert isinstance(grad, DTensor), f"rank{rank}: grad not DTensor"
        # After the Partial->Replicate all-reduce, every tp rank holds the
        # full (summed) gradient.
        local_grad = grad.full_tensor()

        result_queue.put((rank, local_grad.clone(), ref_grad.clone()))
    finally:
        dist.destroy_process_group()


def main() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, WORLD_SIZE, q)) for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(WORLD_SIZE)]
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    results.sort()
    ref = results[0][2]
    for rank, grad, _ in results:
        # 1. all-reduce fired: every tp rank sees the SAME gradient.
        torch.testing.assert_close(
            grad, results[0][1], msg=f"rank{rank} grad differs across tp ranks"
        )
        # 2. that gradient equals the reference sum-of-rank-contributions
        #    (i.e. the Partial placement was correctly all-reduced, not
        #    silently dropped as it would be with a bare to_local()).
        torch.testing.assert_close(
            grad, ref, msg=f"rank{rank} grad != reference summed grad"
        )
    print(
        "PASS: block_attn_res query gradient is Partial-all-reduced across "
        f"tp ranks (grad_norm={ref.norm().item():.6f}, finite and O(1))."
    )


if __name__ == "__main__":
    main()
