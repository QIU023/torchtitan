# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.attention import create_attention_mask
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import VisionAttention, local_head_split
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _tpool_patch_merger,
    MoonViTEncoder,
)
from torchtitan.protocols.module import Module


@dataclass
class CPPatchPlan:
    """Dynamic CP: one large image split along the PATCH dimension (report 5.2.3).

    Report 5.2.3, verbatim in substance: "A single large image is partitioned
    along the patch dimension across multiple devices, and attention is computed
    by gathering key-value pairs (gather-KV) across CP ranks."

    This is the half that the earlier image-level round-robin did NOT provide, and
    it is the load-bearing one -- the report's stated purpose for it is to reduce
    "the encoder latency of large visual samples and the cross-device load
    imbalance, allowing the remaining encoder computation to be hidden in pipeline
    bubbles", so DEP depends on it rather than the other way round.

    Each rank holds ``shard_len`` consecutive patches of the image and computes q
    for those alone; k and v are all-gathered across ``group`` so every rank
    attends over the whole image. The gather is differentiable, so its transpose
    is the reduce-scatter that returns each rank the gradient for the patches it
    owns.

    ``valid_total`` is the image's true patch count. A partition needs an equal
    shard on every rank for a fixed-shape collective, so the tail is padded and
    the padded KEY positions are masked out of attention. Without the mask the
    padding would contribute to every softmax -- silently, since the shapes are
    all correct.

    ``full_grid`` and ``patch_start`` exist because KimiK3VisionEncoder carries position
    information TWICE -- the divided_fixed absolute embedding added at the patch
    embed, and 2-D RoPE applied to q/k in every block -- and both are built from
    the grid starting at row 0. Describing a shard as a standalone image therefore
    gives every rank the same positions, so rank 1's patches would be encoded as
    if they were rank 0's. Measured before this was carried: the partitioned path
    differed from the replicated one by 2.3e-03 in step-1 loss, which is far too
    large for a reduction-order effect. The tables are built for the whole image
    and sliced.
    """

    group: dist.ProcessGroup
    valid_total: int
    """The image's true patch count, for the padded-key mask."""

    full_grid: tuple[int, int, int] = (0, 0, 0)
    """(t, h, w) of the WHOLE image, not of this shard."""

    row_start: int = 0
    """First patch-grid ROW this rank owns, in the whole image's coordinates."""

    band: int = 0
    """Rows in this rank's tensor, including padding: the shard is (t, band, w)."""

    real_rows: int = 0
    """How many of ``band`` are real; the rest are padding."""


def _slice_for_shard(table: torch.Tensor, plan: "CPPatchPlan"):
    """Take this rank's ROW BAND out of a table built for the WHOLE image.

    The band is strided once the image is a video: the rank owns rows
    ``[row_start, row_start + real_rows)`` of EVERY frame, because the projector's
    temporal mean spans all frames and splitting by frame would break it. So the
    table is gathered frame by frame and padded to ``band`` rows per frame, exactly
    mirroring how the caller lays out the pixels.

    Padding rows repeat the last real row rather than being zeroed: a zeroed RoPE
    factor is not a rotation. Neither choice changes the result -- padded queries
    are discarded and padded keys are masked -- but staying in range keeps a NaN
    out of the softmax, where it would reach real rows.
    """
    t, h, w = plan.full_grid
    per_frame = []
    for f in range(t):
        base = f * h * w
        lo = base + plan.row_start * w
        hi = lo + plan.real_rows * w
        rows = table[lo:hi]
        pad_rows = plan.band - plan.real_rows
        if pad_rows > 0:
            src = rows[-1:] if rows.size(0) else table[base : base + 1]
            rows = torch.cat([rows, src.expand(pad_rows * w, *table.shape[1:])], dim=0)
        per_frame.append(rows)
    return torch.cat(per_frame, dim=0)


def _padded_key_keep(plan: "CPPatchPlan", total: int, device) -> torch.Tensor:
    """Which of the gathered key positions are real patches.

    NOT a prefix. ``_slice_for_shard`` pads PER FRAME -- it takes each frame's
    band rows, tops that frame up to ``band``, and only then concatenates the
    frames -- so a deficit rank's stream is [frame0 real, frame0 pad, frame1
    real, frame1 pad, ...] and the padding is INTERLEAVED. A prefix mask admits
    frame 0's padding into the softmax and masks frame 1's real keys instead:
    silently wrong encoder output whenever t > 1 and some rank is short. (t == 1
    has one frame, so a prefix happens to be right, which is why a test with a
    single frame would pass either way.)

    Each rank's real row count follows from the same ceiling split
    ``row_partition`` performs, so this needs no extra field and no collective:
    rank r holds ``min(band, max(0, h - r * band))`` real rows.
    """
    keep = torch.zeros(total, dtype=torch.bool, device=device)
    t, h, _w = plan.full_grid
    group_size = dist.get_world_size(plan.group)
    if t > 0 and plan.band > 0 and total % (group_size * t * plan.band) == 0:
        row_len = total // (group_size * t * plan.band)
        pos = 0
        for r in range(group_size):
            real_rows = min(plan.band, max(0, h - r * plan.band))
            for _frame in range(t):
                keep[pos : pos + real_rows * row_len] = True
                pos += plan.band * row_len
    else:
        # A plan with no grid describes a flat patch split where the padding IS
        # a trailing run. Falling back matters: the branch above would compute
        # from zeros, mark nothing valid, and mask every key.
        keep[: plan.valid_total] = True
    return keep


class KimiK3VisionCPAttention(VisionAttention):
    """Vision attention over an image whose patches are split across ranks.

    q is this rank's patch shard; k and v are gathered so the shard attends over
    the whole image (report sec 5.2.3's gather-KV). The gather is
    differentiable, so its transpose is the reduce-scatter that returns each
    rank the gradient for the patches it owns -- ``dist.all_gather`` would
    detach and the tower would train on gradients missing every other rank's
    contribution.

    Without a plan this is exactly ``VisionAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VisionAttention.Config):
        """Same fields; a distinct Config so build() returns this class."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply,
        attention_mask,
        cp_plan: "CPPatchPlan | None" = None,
    ) -> torch.Tensor:
        # An argument rather than module state: activation checkpointing
        # recomputes this forward from its saved arguments, and state set
        # around the call has been cleared by recompute time.
        plan = cp_plan
        if plan is None:
            return super().forward(
                x,
                rope_cache=rope_cache,
                rope_apply=rope_apply,
                attention_mask=attention_mask,
            )

        import torch.distributed.nn.functional as dist_nn

        num_tokens = x.shape[0]
        q_THDh = local_head_split(self.wq(x), self.head_dim)
        k_THDh = local_head_split(self.wk(x), self.head_dim)
        v_THDh = local_head_split(self.wv(x), self.head_dim)
        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)

        k_full = torch.cat(
            dist_nn.all_gather(k_THDh.contiguous(), group=plan.group), dim=0
        )
        v_full = torch.cat(
            dist_nn.all_gather(v_THDh.contiguous(), group=plan.group), dim=0
        )
        total = k_full.size(0)

        # Built unconditionally: flex requires a BlockMask, and with no padding
        # the keep vector is all true, which is the dense mask the replicated
        # path would use for a single image.
        keep = _padded_key_keep(plan, total, x.device)

        def _mask_mod(b, h, q_idx, kv_idx):
            del b, h, q_idx
            return keep[kv_idx]

        mask = create_attention_mask(
            _mask_mod, None, None, q_THDh.size(0), total, device=x.device
        )
        out_THDh = self.flex_attention(
            q_THDh, k_full, v_full, attention_masks=mask
        )
        return self.proj(out_THDh.reshape(num_tokens, -1))


class KimiK3VisionProjector(Module):
    """PatchMergerMLPV2 projector from merged vision features to text width."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear_1: Linear.Config
        linear_2: Linear.Config
        post_norm: RMSNorm.Config
        activation: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.post_norm = config.post_norm.build()
        self.activation = config.activation.build()

    def forward(self, merged_MF: torch.Tensor) -> torch.Tensor:
        projected_MO = self.linear_2(self.activation(self.linear_1(merged_MF)))
        return self.post_norm(projected_MO)


class KimiK3VisionEncoder(MoonViTEncoder):
    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]  # pyrefly: ignore [bad-override]
        max_num_frames: int

        final_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        projector: KimiK3VisionProjector.Config  # pyrefly: ignore [bad-override]

    def forward(  # pyrefly: ignore [bad-override]
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        part: str | None = None,
        upto_block: int | None = None,
        lo: int | None = None,
        hi: int | None = None,
        from_block: int | None = None,
    ) -> torch.Tensor:
        """The shared tower's forward, plus report sec 5.2.3's patch partition.

        Without ``cp_plan`` this defers to the tower unchanged. With one,
        ``pixel_values`` is this rank's row band of ONE image and ``grid_thw``
        still describes the whole image: the position tables are built for the
        whole image and sliced, because both the learned absolute embedding and
        the 2-D RoPE index from row 0, and describing a shard as a standalone
        image would give every rank rank 0's positions.

        Kept here rather than in the shared tower because the partition is a
        Kimi K3 feature -- it is what its report describes -- and k2.5 has no
        use for it.
        """
        # ``part`` selects one share of a tower spanning PP stages (report sec
        # 5.2.3 clause 2), reached THROUGH this forward: FSDP2 hooks __call__,
        # and calling forward_head directly meets still-sharded DTensor weights.
        if part is not None:
            if part == "head":
                return self.forward_head(
                    pixel_values, grid_thw, upto_block=upto_block
                )
            if part == "body":
                return self.forward_body(pixel_values, grid_thw, lo=lo, hi=hi)
            if part == "tail":
                return self.forward_tail(
                    pixel_values, grid_thw, from_block=from_block or 0
                )
            raise ValueError(f"unknown tower part {part!r}")

        if cp_plan is None:
            return super().forward(pixel_values, grid_thw=grid_thw)

        grids = grid_thw.tolist()
        if len(grids) != 1:
            raise ValueError(
                "a CP patch plan describes one image, but grid_thw carries "
                f"{len(grids)}; a mixed stream needs a per-segment plan."
            )
        # Position tables are built for the WHOLE image, then sliced to this
        # rank's band: ``grid_thw`` describes only the local shard, and building
        # from it gives every rank rank 0's positions. The full grid rides the plan.
        x, rope_cache = self._embed_patches(pixel_values, cp_plan)
        x = self._run_blocks(x, rope_cache=rope_cache, cp_plan=cp_plan)
        return self._merge_and_project(self.final_norm(x), cp_plan)

    # ---- DEP: the tower split across pipeline stages (report sec 5.2.3) ----
    #
    # The report asks for vision forward and backward balanced across PP
    # stages, so the tower runs as contiguous block ranges: head / body / tail.

    def _embed_patches(self, pixel_values, cp_plan):
        """Patch embed plus position tables. Returns (x, rope_cache)."""
        full_grid = [list(cp_plan.full_grid)]
        learned_pos, rope_cache = self.compute_position_embeddings(full_grid)
        learned_pos = _slice_for_shard(learned_pos, cp_plan)
        rope_cache = _slice_for_shard(rope_cache, cp_plan)
        return self.patch_embed(pixel_values) + learned_pos, rope_cache

    def _run_blocks(
        self, x, *, rope_cache, cp_plan, block_slice=None, attention_mask=None
    ):
        blocks = list(self.layers.values())
        if block_slice is not None:
            blocks = blocks[block_slice]
        for block in blocks:
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=attention_mask,
                cp_plan=cp_plan,
            )
        return x

    def _merge_and_project(self, x, cp_plan):
        # The merge sees the SHARD's grid: this rank holds a band of rows for
        # every frame, so the (kh, kw) blocking and the temporal mean are over
        # its own (t, band, w). The positions above needed the whole image.
        t, _, w = cp_plan.full_grid
        merged = _tpool_patch_merger(x, [[t, cp_plan.band, w]], self.merge_kernel_size)
        return self.projector(merged)

    def block_bounds(self, num_shares: int) -> list[tuple[int, int]]:
        """Split the tower's blocks into ``num_shares`` contiguous ranges.

        Report 5.2.3 balances vision passes across PP stages, so shares are as
        even as possible. A remainder goes to the LAST shares, because share 0
        also carries ``patch_embed`` and the final share's projector is cheaper
        than that -- giving share 0 an extra block as well would make the least
        balanced stage worse.
        """
        n = len(self.layers)
        if num_shares < 1 or num_shares > n:
            raise ValueError(
                f"cannot split {n} encoder block(s) into {num_shares} share(s)"
            )
        base, extra = divmod(n, num_shares)
        bounds, lo = [], 0
        for i in range(num_shares):
            hi = lo + base + (1 if i >= num_shares - extra else 0)
            bounds.append((lo, hi))
            lo = hi
        return bounds

    # Every share recomputes position tables and the block-diagonal mask from
    # grid_thw: RoPE indices and segment bounds do not survive PP's dummy
    # metadata values, so only float activations cross a stage boundary. DEP
    # and CP are mutually exclusive, so no share here carries a patch plan.

    def _share_block_inputs(self, grid_thw, num_tokens, device):
        """Position tables and attention mask for a share, from the grid alone."""
        from torchtitan.models.common.vision_encoder import create_block_diagonal_mask
        import spmd_types as spmd

        grids = grid_thw.tolist()
        learned_pos, rope_cache = self.compute_position_embeddings(grids)
        with spmd.no_typecheck():
            mask = create_block_diagonal_mask(
                grid_thw.prod(dim=-1), num_tokens, device
            )
        return grids, learned_pos, rope_cache, mask

    def forward_head(self, pixel_values, grid_thw, *, upto_block: int):
        """Patch embed plus blocks ``[0, upto_block)``, without the final norm.

        The first share when the tower spans PP stages. Returns patch hidden
        states, not features -- the projector belongs to the last share.
        """
        _, learned_pos, rope_cache, mask = self._share_block_inputs(
            grid_thw, pixel_values.shape[0], pixel_values.device
        )
        x = self.patch_embed(pixel_values) + learned_pos
        return self._run_blocks(
            x,
            rope_cache=rope_cache,
            cp_plan=None,
            block_slice=slice(0, upto_block),
            attention_mask=mask,
        )

    def forward_body(self, x, grid_thw, *, lo: int, hi: int):
        """Blocks ``[lo, hi)`` only -- a middle share, no norm, no projector."""
        _, _, rope_cache, mask = self._share_block_inputs(
            grid_thw, x.shape[0], x.device
        )
        return self._run_blocks(
            x,
            rope_cache=rope_cache,
            cp_plan=None,
            block_slice=slice(lo, hi),
            attention_mask=mask,
        )

    def forward_tail(self, x, grid_thw, *, from_block: int):
        """Blocks ``[from_block, end)``, the final norm, the merge, the projector.

        The last share, and the only one that produces features.
        """
        grids, _, rope_cache, mask = self._share_block_inputs(
            grid_thw, x.shape[0], x.device
        )
        x = self._run_blocks(
            x,
            rope_cache=rope_cache,
            cp_plan=None,
            block_slice=slice(from_block, len(self.layers)),
            attention_mask=mask,
        )
        x = self.final_norm(x)
        return self.projector(_tpool_patch_merger(x, grids, self.merge_kernel_size))
