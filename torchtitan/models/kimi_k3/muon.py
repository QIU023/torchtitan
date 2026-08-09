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

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.logging import logger


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

    def _warn_if_per_head_is_inert(self) -> None:
        """Per-head Muon needs ``_muon_heads`` tags; without any it is just
        Muon. That degeneration is invisible in the loss, so say so once."""
        if getattr(self, "_per_head_checked", False):
            return
        self._per_head_checked = True
        for group in self.param_groups:
            if not group.get("per_head") or not group.get("use_muon", True):
                continue
            if any(getattr(p, "_muon_heads", None) for p in group["params"]):
                continue
            logger.warning(
                "Muon(per_head=True) but no parameter in this group carries "
                "_muon_heads, so every update falls back to full-matrix "
                "orthogonalization. Call tag_per_head_muon(model) before "
                "building the optimizer."
            )

    @torch.no_grad()
    def step(self, closure=None):
        self._warn_if_per_head_is_inert()
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
        denom = (st["exp_avg_sq"].sqrt() / (bc2**0.5)).add_(group["adamw_eps"])
        if group["weight_decay"]:
            p.mul_(1 - group["adamw_lr"] * group["weight_decay"])
        p.addcdiv_(st["exp_avg"], denom, value=-group["adamw_lr"] / bc1)


# Report sec 2.5 scopes the per-head refinement to the Q, K and V projections:
# "instead of applying Newton-Schulz orthogonalization to the full Q, K, and V
# projection matrices, we partition their momentum matrices along the head
# dimension and orthogonalize each head's block separately." o_proj is excluded
# deliberately -- it is the head axis on its INPUT side, so a row partition
# would not correspond to heads at all.
_PER_HEAD_MLA = ("q_proj", "q_b_proj", "kv_b_proj")
_PER_HEAD_KDA = ("q_proj", "k_proj", "v_proj")


def tag_per_head_muon(model: nn.Module) -> int:
    """Mark every Q/K/V projection with its head count. Returns the count.

    Per-Head Muon is driven by a ``_muon_heads`` attribute on the parameter,
    which nothing set outside the tests -- so a real run silently degenerated to
    plain full-matrix Muon. Call this before building the optimizer.

    The head count is read from the owning attention module rather than guessed
    from shapes, and a projection whose output width is not a multiple of its
    head count is left untagged instead of partitioned wrongly.
    """
    from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiMLAAttention

    tagged = 0
    for module in model.modules():
        if isinstance(module, KimiMLAAttention):
            names, heads = _PER_HEAD_MLA, module.num_heads
        elif isinstance(module, KimiDeltaAttention):
            names, heads = _PER_HEAD_KDA, module.num_heads
        else:
            continue
        for name in names:
            proj = getattr(module, name, None)
            if proj is None:
                continue
            weight = getattr(proj, "weight", None)
            if weight is None or weight.dim() != 2:
                continue
            if weight.size(0) % heads != 0:
                # e.g. a fused projection whose rows do not tile by head. Better
                # to run full-matrix Muon on it than to partition into blocks
                # that are not heads.
                continue
            # kv_b_proj's per-head block holds that head's K_nope rows AND its V
            # rows; "partition along the head dimension" keeps them together,
            # which is what a fused KV matrix makes them.
            weight._muon_heads = heads
            tagged += 1
    return tagged


# ----- Wiring Muon into torchtitan's optimizer container ------------------ #


class KimiOptimizersContainer(OptimizersContainer):
    """``OptimizersContainer`` that also knows about Muon.

    Core's ``_resolve_optimizer_cls`` hardcodes ``{Adam, AdamW}`` and raises
    ``NotImplementedError`` for anything else, and CLAUDE.md rules out editing
    core to accommodate an experiment. Subclassing keeps the addition local: the
    Config's ``_owner`` machinery builds this class, so a flavor pointing at
    ``KimiOptimizersContainer.Config`` gets Muon resolution and nothing else
    changes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        """Needed even though it adds no fields.

        Configurable sets ``_owner`` per Config CLASS. Inheriting the parent's
        Config verbatim means ``_owner`` still points at OptimizersContainer, so
        ``build()`` returns core's container and Muon resolution never happens --
        the smoke failed with "Optimizer Muon not added" for exactly that reason.
        """

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Muon":
            return Muon
        return OptimizersContainer._resolve_optimizer_cls(name)


# Report sec 2.5: Muon for the matrix parameters, with the per-head refinement on
# the attention projections. Everything that is not a 2-D weight matrix -- norms,
# biases, the 1-D KDA parameters, embeddings and the LM head -- stays on AdamW,
# which is the standard Muon recipe rather than something specific to K3.
_MUON_EXCLUDE_PATTERNS = (
    r".*norm.*",
    r".*\.bias$",
    r".*embed_tokens.*",
    r".*lm_head.*",
    r".*A_log$",
    r".*dt_bias$",
    r".*_res_proj\.weight$",  # AttnRes pseudo-queries are [1, D], not matrices
    r".*conv1d.*",
)


def default_muon(
    lr: float = 2e-2,
    *,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> "OptimizersContainer.Config":
    """Muon on the matrix parameters, AdamW on everything else.

    The two learning rates are deliberately different: Muon's update is
    orthogonalized, so its scale is decoupled from the gradient magnitude and it
    wants a much larger lr than AdamW on the same model. Passing one lr for both
    is the usual way to make Muon look bad.
    """
    from torchtitan.components.optimizer import ParamGroupConfig

    exclude = "|".join(_MUON_EXCLUDE_PATTERNS)
    return KimiOptimizersContainer.Config(
        param_groups=[
            # AdamW first: the container assigns each parameter to the FIRST
            # matching pattern, so the narrower exclusion set has to precede the
            # catch-all Muon group.
            ParamGroupConfig(
                pattern=exclude,
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": adamw_lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Muon",
                optimizer_kwargs={
                    "lr": lr,
                    "momentum": momentum,
                    "ns_steps": ns_steps,
                    "per_head": True,
                },
            ),
        ],
        implementation="for-loop",
    )
