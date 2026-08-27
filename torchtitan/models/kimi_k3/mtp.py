# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-token prediction for Kimi K3 (report sec 3.3), folded layout.

Ported from the reference tree's ``mtp_loss.py`` with its GB200 fixes in from
the start (keyword-only ``positions``, cross-document masking). Indexing is the
folded stream's: one token axis, no batch dimension.

The multimodal restriction the reference carried does not apply here: its
wrapper splice EXPANDED the sequence (one sentinel to many visual tokens), so a
shift of ``input_ids`` no longer aligned with the hidden states and it refused
to run. This layout's collator pre-expands sentinels and the splice is a
masked_scatter -- length-preserving -- so shift-by-k alignment holds, and the
visual positions already carry IGNORE_INDEX in the labels (mm_datasets masks
every special token), so no depth trains on predicting a sentinel.
"""

from dataclasses import dataclass, field

import torch

from torchtitan.components.loss import BaseLoss, CrossEntropyLoss, IGNORE_INDEX

# Rank-local hand-off from the model's forward to the loss. Written by
# KimiK3Model.forward when MTP layers are configured, taken (and cleared) here.
_PENDING: list[torch.Tensor] | None = None


def put_mtp_logits(logits: list[torch.Tensor]) -> None:
    global _PENDING
    _PENDING = logits


def take_mtp_logits() -> list[torch.Tensor] | None:
    global _PENDING
    logits, _PENDING = _PENDING, None
    return logits


class KimiMTPLoss(BaseLoss):
    """Main next-token cross-entropy plus the MTP depths' cross-entropy.

    Reduces to exactly the inner loss when no depth logits arrive, so the same
    loss config can serve flavors with and without MTP layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        mtp_weight: float = 0.3
        """Weight on the mean per-depth MTP loss."""

        loss_fn: BaseLoss.Config = field(default_factory=CrossEntropyLoss.Config)
        """Loss applied to the main head and to each MTP depth."""

    def __init__(self, config: Config, *, compile_config=None):
        self.mtp_weight = config.mtp_weight
        self.inner = config.loss_fn.build(compile_config=compile_config)
        # BaseLoss.__call__ would use self.fn; this class overrides __call__ and
        # delegates to the inner loss instead, so self.fn stays the inner's.
        self.fn = self.inner.fn

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens=None,
        *,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # ``positions`` is more than plumbing: on a packed stream the position
        # ids restart at every document boundary, which is the ONLY way this
        # loss can see the boundaries -- without it a depth-k target can be the
        # first tokens of the next document.
        main_loss, metrics = self.inner(pred, labels, global_valid_tokens, **kwargs)

        mtp_logits = take_mtp_logits()
        if not mtp_logits:
            return main_loss, metrics

        depth_losses = []
        for k, logits in enumerate(mtp_logits):
            shift = k + 1
            # Depth k's prediction at position t targets the token at t+shift;
            # the model already dropped the tail positions with no target, so
            # the labels line up by taking the same shift off the front. The
            # folded stream has one token axis, hence the [-1]-anchored slices.
            target = labels[..., shift:]
            n = min(logits.shape[-2], target.shape[-1])
            if n <= 0:
                continue
            target = target[..., :n]
            if positions is not None:
                # A target belongs to the SAME document exactly when the ids
                # are still climbing: after a packing restart
                # pos[t+shift] < pos[t] + shift. Masked with IGNORE_INDEX, the
                # convention the inner loss already applies to padding and to
                # the visual sentinels.
                pos_t = positions[..., :n]
                pos_target = positions[..., shift : shift + n]
                same_doc = pos_target == pos_t + shift
                target = torch.where(
                    same_doc, target, torch.full_like(target, IGNORE_INDEX)
                )
            depth_loss, _ = self.inner(
                logits[..., :n, :], target, global_valid_tokens
            )
            depth_losses.append(depth_loss)

        if not depth_losses:
            return main_loss, metrics

        mtp_mean = torch.stack(depth_losses).mean()
        metrics = {**metrics, "loss/mtp": mtp_mean.detach()}
        return main_loss + self.mtp_weight * mtp_mean, metrics
