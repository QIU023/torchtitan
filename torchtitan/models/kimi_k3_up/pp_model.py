# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Make the upstream K3 model's AttnRes carrier survive a pipeline stage boundary.

`KimiK3Model.forward` builds `block_residual_TND` as a local and returns only the
hidden state. torchtitan's PP runs that same forward on each stage over a subset
of layers, so stage 1 restarts the carrier at N=0 and everything stage 0
committed is gone -- with nothing raised and the loss still falling.

Measured (`matrix_scripts/attnres_pp_carrier_probe.py`, 21 layers cut after 10):

    carrier N:  whole=2  split=1  carried=2
    whole vs split (carrier reset at the cut): 1.321e+00
    whole vs carried (carrier handed across):  0.000e+00

So the cut itself is innocent and the whole defect is that the carrier is not in
the model's signature. This subclass puts it there. Their model file is left
byte-identical, which is what keeps rebases onto the upstream PR cheap.

This is the CORRECTNESS half of pipeline support. The memory half is separate and
still needed: N grows monotonically with depth, so shipping the carrier through
every stage boundary costs [T, N, D] per hop -- about 0.88 GiB per microbatch on
the last hop at 2.8T dims, times the microbatches in flight under pp x vp. The
rank-local cache in `kimi_k3.pipeline_adapter` plus a recv_delta is what makes
the per-hop cost constant, and it goes on top of this rather than instead of it.
"""

from __future__ import annotations

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.kimi_k3_up.model import (
    _apply_attention_residual,
    KimiK3Model,
)


class KimiK3PipelineModel(KimiK3Model):
    """`KimiK3Model` with the AttnRes carrier threaded through the signature.

    On a stage that owns `tok_embeddings` the carrier starts empty, exactly as
    upstream. On any later stage it arrives as an argument. The tail -- the
    output attention residual, the final norm and `lm_head` -- runs only where
    `lm_head` survives the split, because applying the output residual on every
    stage would aggregate a partial block stack into the hidden state repeatedly.
    """

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        block_residual_TND: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        if self.tok_embeddings is not None:
            h_BLD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                special_tokens=special_tokens,
            )
        else:
            h_BLD = tokens

        B, L, D = h_BLD.shape
        if block_residual_TND is None:
            block_residual_TND = h_BLD.new_zeros(B * L, 0, D)

        for layer in self.layers.values():
            h_BLD, block_residual_TND = layer(
                h_BLD,
                block_residual_TND,
                attention_masks,
                positions,
            )

        is_last_stage = self.lm_head is not None or self._skip_lm_head
        if not is_last_stage:
            # Hand both onward. The carrier is returned even when this stage
            # committed no new column: routing it back out of the module is what
            # keeps FSDP's backward hooks on it, the same reason their block
            # returns it unchanged rather than None.
            return h_BLD, block_residual_TND

        h_BLD = _apply_attention_residual(
            h_BLD.reshape(-1, D),
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        ).view(B, L, D)
        h_BLD = self.norm(h_BLD) if self.norm is not None else h_BLD
        if self._skip_lm_head:
            return h_BLD
        return self.lm_head(h_BLD) if self.lm_head is not None else h_BLD
