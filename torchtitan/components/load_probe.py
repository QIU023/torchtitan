# PROBE ONLY (run worktree, not committed): per-step expert-load statistics for the MoE layers, for the
# quantile-balancing evidence. Registers a router forward hook per MoE layer that bincounts the routed
# expert ids, and an optimizer step pre-hook that all-reduces the counts over the loss mesh, logs the load
# coefficient of variation, max/mean, min/mean and the bias extremes per layer, and zeroes the counts.
from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


class LoadProbe:
    def __init__(self, model_parts, parallel_dims, balancer=None):
        from torchtitan.models.common.moe import MoE

        self.moes = []
        for part in model_parts:
            for m in part.modules():
                if isinstance(m, MoE):
                    self.moes.append(m)
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        self.group = None if loss_mesh is None else loss_mesh.get_group()
        self.counts = []
        self.step_idx = 0
        self.balancer = balancer
        self.routes = {}
        self.scores = {}
        for i, moe in enumerate(self.moes):
            bias = moe.expert_bias_E
            bias = bias.to_local() if isinstance(bias, DTensor) else bias
            self.counts.append(torch.zeros(bias.numel(), dtype=torch.long, device=bias.device))
            moe.router.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(router, args, output):
            ids = output[1]
            ids = ids.to_local() if isinstance(ids, DTensor) else ids
            with torch.no_grad():
                self.counts[i].add_(torch.bincount(ids.reshape(-1), minlength=self.counts[i].numel()))
                # routing census: keep every micro-batch's routed ids of the steps named in
                # QB_PROBE_ROUTE_STEPS (default 2,10), so two runs on the same data can be diffed offline
                import os
                steps = {int(x) for x in os.environ.get("QB_PROBE_ROUTE_STEPS", "2,10").split(",")}
                if (self.step_idx + 1) in steps:
                    self.routes.setdefault((self.step_idx + 1, i), []).append(ids.detach().to("cpu").clone())
                # score dump (QB_PROBE_SCORE_DIR): the router's raw scores of the same steps, for offline solves
                if os.environ.get("QB_PROBE_SCORE_DIR") and (self.step_idx + 1) in steps:
                    sc = output[2]
                    sc = sc.to_local() if isinstance(sc, DTensor) else sc
                    self.scores.setdefault((self.step_idx + 1, i), []).append(sc.detach().float().reshape(-1, sc.shape[-1]).to("cpu").clone())
        return hook

    @torch.no_grad()
    def step(self):
        self.step_idx += 1
        stacked = torch.stack(self.counts)
        if self.group is not None and dist.is_initialized():
            dist.all_reduce(stacked, group=self.group, op=dist.ReduceOp.SUM)
        rank = dist.get_rank() if dist.is_initialized() else 0
        lines = []
        for i, moe in enumerate(self.moes):
            c = stacked[i].float()
            mean = c.mean().clamp(min=1e-9)
            bias = moe.expert_bias_E
            bias = bias.to_local() if isinstance(bias, DTensor) else bias
            lines.append(f"LOADPROBE step={self.step_idx} layer={i} tokens={int(c.sum())} cv={(c.std(unbiased=False) / mean):.4f} "
                         f"max/mean={(c.max() / mean):.3f} min/mean={(c.min() / mean):.3f} bmin={bias.min():.4f} bmax={bias.max():.4f}")
        if dist.is_initialized() and dist.get_world_size() > 1:
            # every rank must hold the same bias (the solve reads the same pooled histogram everywhere)
            for i, moe in enumerate(self.moes):
                bias = moe.expert_bias_E
                bias = (bias.to_local() if isinstance(bias, DTensor) else bias).detach().float().contiguous()
                gathered = [torch.empty_like(bias) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, bias)
                spread = max((g - gathered[0]).abs().max().item() for g in gathered)
                if spread > 0:
                    lines.append(f"LOADPROBE step={self.step_idx} layer={i} BIAS-DISAGREES-ACROSS-RANKS max|d|={spread:.3e}")
        if rank == 0:
            print("\n".join(lines), file=sys.stderr, flush=True)
        import os
        sdump = os.environ.get("QB_PROBE_SCORE_DIR")
        if sdump and any(k[0] == self.step_idx for k in self.scores):
            os.makedirs(sdump, exist_ok=True)
            torch.save({k: torch.cat(v) for k, v in self.scores.items() if k[0] == self.step_idx}, os.path.join(sdump, f"scores_step{self.step_idx}_rank{rank}.pt"))
            self.scores = {k: v for k, v in self.scores.items() if k[0] != self.step_idx}
        dump = os.environ.get("QB_PROBE_ROUTE_DIR")
        if dump and any(k[0] == self.step_idx for k in self.routes):
            os.makedirs(dump, exist_ok=True)
            torch.save({k: v for k, v in self.routes.items() if k[0] == self.step_idx}, os.path.join(dump, f"routes_step{self.step_idx}_rank{rank}.pt"))
            self.routes = {k: v for k, v in self.routes.items() if k[0] != self.step_idx}
        for c in self.counts:
            c.zero_()


def register_load_probe(optimizers, model_parts, parallel_dims, balancer=None):
    probe = LoadProbe(model_parts, parallel_dims, balancer)
    optimizers.register_step_pre_hook(lambda *a, **kw: probe.step())
    return probe
