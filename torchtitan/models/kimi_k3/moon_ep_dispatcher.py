# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP token dispatch for the K3 MoE (report sec 5.2.1).

MoonshotAI released the transport as MoonEP
(https://github.com/MoonshotAI/MoonEP). It slots into torchtitan's existing
seam: ``BaseEPTokenDispatcher`` is an ABC with exactly ``dispatch`` and
``combine``, ``wire_meshes`` installs the EP mesh, and ``init_buffer`` is the
hook a persistent-buffer backend needs. The MoE module is unchanged.

Kept in the model folder rather than ``models/common``: like fla, MoonEP is a
non-PyTorch dependency, and core's dispatcher factory stays free of it.

Mapping onto MoonEP's released API (per its README):

* ``Buffer(S, H, K, E, num_ep_ranks, num_sms, token_padding)`` preallocates
  once per EP group; ``S`` is the lifetime per-rank token bound, matching the
  base class's ``num_max_tokens_per_rank`` storage-bound convention.
* ``buffer.dispatch(hidden_sh, route_weights_sk, topk_experts_sk,
  tokens_per_expert)`` returns ``(hidden_nvsh, route_weights_nvs, cu_seqlens,
  plan)``: this rank's received tokens, their routing weights, the per-local-
  expert boundaries, and an opaque plan.
* ``buffer.combine(plan, hidden_nvsh, route_weights_nvs)`` returns
  ``(output_sh, gathered_route_weights_sk, _)`` -- the weighted sum back in
  input order, which is the same combine-side score application the standard
  ``AllToAllTokenDispatcher`` performs.
* dispatch/combine carry their own backward per the README ("dispatch bwd",
  "combine bwd"), so no autograd.Function wrapper is layered here; the
  gradient-reaches-dispatch-input property is still verification item #2.
* ``prefetch_weight`` / ``reduce_grad`` (expert-weight prefetch) are NOT wired:
  they change which rank holds expert weights mid-step and belong to a later
  unit, after plain dispatch/combine parity is established.

Two properties to verify first on NVLink hardware, in this order:

1. Token conservation: dispatch-then-combine returns one row per input row in
   input order, for every routing pattern including a zero-token expert.
   ``AllToAllTokenDispatcher`` on the same inputs is the reference.
2. The backward: ``combine``'s gradient must reach ``dispatch``'s input. A
   forward-only path looks correct while silently dropping expert gradients.

Points that could not be pinned from the README are marked ``ON-BOX`` below;
each is one line to adjust with the real package installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    LocalTokenDispatcher,
)
from torchtitan.tools.logging import logger


def _import_moon_ep():
    """Import MoonEP, or explain what is missing.

    Optional in the same sense as fla and DeepEP: absent on a machine that
    cannot run it, and the error names the package rather than surfacing as an
    AttributeError deep in dispatch.
    """
    try:
        import moon_ep  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            "MoonEP is not installed. It is an optional dependency, like "
            "DeepEP: install from https://github.com/MoonshotAI/MoonEP, and "
            "note that it requires NVLink-connected GPUs. Use another "
            "comm_backend on hardware without that topology."
        ) from err
    return moon_ep


@dataclass
class MoonEPDispatchMetadata:
    """What ``combine`` needs to invert the routing."""

    plan: object
    route_weights_nvs: torch.Tensor


class MoonEPTokenDispatcher(BaseEPTokenDispatcher):
    """Balanced EP dispatch (report sec 5.2.1), through MoonEP's kernels.

    Slots into the same place as ``AllToAllTokenDispatcher``; the MoE module
    is unchanged. EP=1 falls back to local dispatch exactly as the standard
    dispatcher does, so a flavor carrying this config still runs unsharded.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        hidden_dim: int | None = None
        """Feature width of the tokens entering dispatch (sizes the buffer)."""

        num_max_tokens_per_rank: int = 8192
        """Lifetime storage bound per rank, NOT the per-call token count --
        MoE pads the sequence before routing, so the per-call count is
        ``x_TD.shape[0]`` and is the same on every rank."""

        num_sms: int = 32
        """SMs MoonEP's kernels may occupy (its default)."""

        token_padding: int = 128
        """MoonEP's internal alignment (its default)."""

    def __init__(self, config: "MoonEPTokenDispatcher.Config") -> None:
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.num_sms = config.num_sms
        self.token_padding = config.token_padding
        self._buffer = None

    def wire_meshes(self, *, ep_mesh) -> None:
        # Unlike MinimalAsyncEP this does not REQUIRE an EP mesh: with EP off
        # the local fallback below runs and MoonEP is never imported.
        super().wire_meshes(ep_mesh=ep_mesh)

    def init_buffer(self) -> None:
        """Allocate MoonEP's persistent buffer once the EP mesh is known.

        The buffer's creation is collective, so every rank has to reach it the
        same number of times in the same order -- hence once, from
        ``wire_meshes``, never per step.
        """
        if self.ep_mesh is None:
            return
        if self.hidden_dim is None:
            raise ValueError(
                "MoonEPTokenDispatcher.Config.hidden_dim is unset; the model "
                "registry must size the buffer with the dispatched feature "
                "width."
            )
        moon_ep = _import_moon_ep()
        ep_size = self.ep_mesh.size()
        # ON-BOX: the README's Buffer signature carries no process group --
        # only num_ep_ranks -- so either it binds the default group or there
        # is a group kwarg the README omits. Resolve against the installed
        # package; the EP group is ``self.ep_mesh.get_group()``.
        self._buffer = moon_ep.Buffer(
            S=self.num_max_tokens_per_rank,
            H=self.hidden_dim,
            K=self.top_k,
            E=self.num_experts,
            num_ep_ranks=ep_size,
            num_sms=self.num_sms,
            token_padding=self.token_padding,
        )
        logger.info(
            "MoonEP dispatcher: buffer S=%d H=%d K=%d E=%d on an ep mesh of %d",
            self.num_max_tokens_per_rank,
            self.hidden_dim,
            self.top_k,
            self.num_experts,
            ep_size,
        )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Route padded local tokens to their experts' ranks.

        Matches the base contract: returns ``(routed_input_RD,
        num_tokens_per_local_expert_e, metadata)`` with the routed tokens in
        expert-major order for this rank's local experts.
        """
        if self.ep_mesh is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )
        if self._buffer is None:
            raise RuntimeError(
                "MoonEP dispatcher used before wire_meshes(); the EP mesh has "
                "to be installed first, and init_buffer allocates on that mesh "
                "collectively."
            )
        if x_TD.shape[0] > self.num_max_tokens_per_rank:
            raise ValueError(
                f"{x_TD.shape[0]} tokens exceed the MoonEP buffer bound "
                f"num_max_tokens_per_rank={self.num_max_tokens_per_rank}; "
                "raise it in the dispatcher config."
            )
        # README dtypes: hidden bf16, weights fp32, ids int32, counts int32.
        # ON-BOX: whether hidden dtypes other than bf16 are accepted.
        (routed_input_RD, route_weights_nvs, cu_seqlens, plan,) = self._buffer.dispatch(
            x_TD,
            topk_scores_TK.to(torch.float32),
            topk_expert_ids_TK.to(torch.int32),
            num_local_tokens_per_expert_E.to(torch.int32),
        )
        # ON-BOX: cu_seqlens is read as the per-local-expert boundaries of the
        # received (expert-major) tokens; the assert below is the tripwire if
        # the actual layout differs.
        num_local_experts = self.num_experts // self.ep_mesh.size()
        if cu_seqlens.numel() != num_local_experts + 1:
            raise RuntimeError(
                f"MoonEP cu_seqlens has {cu_seqlens.numel()} entries; expected "
                f"num_local_experts+1={num_local_experts + 1}. The boundary "
                "layout differs from the assumed one -- fix the count "
                "derivation here."
            )
        num_tokens_per_local_expert_e = cu_seqlens[1:] - cu_seqlens[:-1]
        metadata = MoonEPDispatchMetadata(
            plan=plan, route_weights_nvs=route_weights_nvs
        )
        return routed_input_RD, num_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: MoonEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ``dispatch``: one weighted row per original token, in order.

        MoonEP applies the routing weights inside its combine, which is the
        same combine-side score application the standard dispatcher performs
        (fp32 multiply before the scatter-add).
        """
        if self.ep_mesh is None:
            return LocalTokenDispatcher.combine(self, routed_output_RD, metadata, x_TD)
        if self._buffer is None:
            raise RuntimeError("MoonEP dispatcher used before wire_meshes().")
        out_TD, _, _ = self._buffer.combine(
            plan=metadata.plan,
            hidden_nvsh=routed_output_RD,
            route_weights_nvs=metadata.route_weights_nvs,
        )
        if out_TD.shape != x_TD.shape:
            raise RuntimeError(
                f"MoonEP combine returned {tuple(out_TD.shape)} for input "
                f"{tuple(x_TD.shape)}; token conservation is broken (or the "
                "buffer returned its padded S rows -- slice here if so)."
            )
        return out_TD.to(x_TD.dtype)
