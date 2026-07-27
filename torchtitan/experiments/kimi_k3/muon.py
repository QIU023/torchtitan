# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""(Per-Head) Muon optimizer for the Kimi K3 experiment.

Muon (Jordan et al., 2024): momentum SGD whose update direction is
orthogonalized via a Newton-Schulz iteration -- for 2-D weight
matrices, replace the raw momentum G with ~ (G G^T)^-1/2 G, an
approximate orthogonal factor. K3 uses a Per-Head Muon variant; the
"per-head" part orthogonalizes each attention head's projection block
independently (heads share no orthogonality).

Scope (honest): the BASE Muon algorithm is published and implemented
faithfully here. The exact K3 Per-Head variant (which projections,
head grouping, Nesterov details) reconciles at 7.27; this provides a
correct, testable base + a per-head reshape hook. Non-2-D params
(embeddings, norms, biases, KDA vectors) fall back to AdamW, as in the
reference Muon recipe.
"""

import torch
from torch.optim.optimizer import Optimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of a 2-D matrix via Newton-Schulz.

    Quintic iteration (Jordan's coefficients). Operates in bf16 for
    speed; returns same shape as G.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for 2-D matrices; AdamW fallback for everything else.

    Args:
        lr: learning rate for the Muon (matrix) group.
        momentum: heavy-ball momentum.
        nesterov: use Nesterov momentum.
        ns_steps: Newton-Schulz iterations.
        per_head: if set, a param whose ``_muon_heads`` attribute is an
            int H reshapes to (H, out/H, in) and orthogonalizes each
            head block independently (Per-Head Muon).
        adamw_lr / adamw_betas / adamw_eps / weight_decay: fallback
            AdamW hyperparameters for non-2-D params.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        per_head: bool = True,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            per_head=per_head,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Muon applies to 2-D weight matrices; else AdamW.
                if g.ndim == 2 and min(g.shape) > 1:
                    self._muon_update(p, g, group)
                else:
                    self._adamw_update(p, g, group)
        return loss

    def _muon_update(self, p, g, group):
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g)
        buf = st["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(g)
        d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

        heads = getattr(p, "_muon_heads", None)
        if group["per_head"] and heads and d.size(0) % heads == 0:
            # Orthogonalize each head's row-block independently.
            hd = d.view(heads, d.size(0) // heads, d.size(1))
            o = torch.stack(
                [_newton_schulz(hd[i], group["ns_steps"]) for i in range(heads)]
            ).view_as(d)
        else:
            o = _newton_schulz(d, group["ns_steps"])
        # scale by sqrt(max(1, rows/cols)) per the Muon recipe
        scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
        p.add_(o, alpha=-group["lr"] * scale)

    def _adamw_update(self, p, g, group):
        st = self.state[p]
        if "step" not in st:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(g)
            st["exp_avg_sq"] = torch.zeros_like(g)
        st["step"] += 1
        b1, b2 = group["adamw_betas"]
        st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
        st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1 ** st["step"]
        bc2 = 1 - b2 ** st["step"]
        denom = (st["exp_avg_sq"].sqrt() / (bc2 ** 0.5)).add_(group["adamw_eps"])
        if group["weight_decay"]:
            p.mul_(1 - group["adamw_lr"] * group["weight_decay"])
        p.addcdiv_(st["exp_avg"], denom, value=-group["adamw_lr"] / bc1)
