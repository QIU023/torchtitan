# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- N = number of visual items
- T = total packed patches
- D = vision hidden dimension
- H = number of attention heads
- K = attention head dimension
- C = number of complex-valued head-dimension pairs
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import create_attention_mask
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    create_block_diagonal_mask,
    local_head_split,
)
from torchtitan.protocols.module import Module, ModuleDict


def _get_temporal_pos_embed(
    num_frames: int,
    embed_dim: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return fixed 1D sinusoidal embeddings for video frame positions."""
    grid = torch.arange(num_frames, dtype=torch.float32, device=device)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device) / (
        embed_dim / 2.0
    )
    omega = 1.0 / 10000.0**omega
    angles = torch.outer(grid, omega)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _compute_learned_pos_embeds(
    pos_embed: torch.Tensor,
    grids: list[list[int]],
    interpolation_mode: str,
    max_num_frames: int,
) -> torch.Tensor:
    """Interpolate the learned 2D table and add fixed temporal embeddings."""
    height, width, dim = pos_embed.shape
    pos_grid = pos_embed.permute(2, 0, 1).unsqueeze(0).float()

    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    positions = []
    for num_frames, grid_h, grid_w in grids:
        if num_frames > max_num_frames:
            raise ValueError(
                f"Vision grid has {num_frames} frames, exceeding "
                f"max_num_frames={max_num_frames}."
            )
        spatial = cached_spatial.get((grid_h, grid_w))
        if spatial is None:
            if (grid_h, grid_w) == (height, width):
                spatial = pos_embed.flatten(end_dim=1)
            else:
                spatial = (
                    F.interpolate(
                        pos_grid,
                        size=(grid_h, grid_w),
                        mode=interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(grid_h * grid_w, dim)
                    .to(pos_embed.dtype)
                )
            cached_spatial[(grid_h, grid_w)] = spatial

        if num_frames == 1:
            item_pos = spatial
        else:
            temporal = _get_temporal_pos_embed(num_frames, dim, device=pos_embed.device)
            item_pos = spatial.unsqueeze(0) + temporal.unsqueeze(1).to(spatial.dtype)
            item_pos = item_pos.reshape(num_frames * grid_h * grid_w, dim)
        positions.append(item_pos)

    return torch.cat(positions)


def _compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grids: list[list[int]],
    head_dim: int,
) -> torch.Tensor:
    """Build the real-valued 2D RoPE cache in raster patch order."""
    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    item_angles = []
    for num_frames, grid_h, grid_w in grids:
        spatial = cached_spatial.get((grid_h, grid_w))
        if spatial is None:
            flat = torch.arange(grid_h * grid_w, device=freq_table.device)
            x_angles = freq_table[flat % grid_w]
            y_angles = freq_table[flat // grid_w]
            spatial = torch.stack((x_angles, y_angles), dim=-1).reshape(
                grid_h * grid_w, head_dim // 2
            )
            cached_spatial[(grid_h, grid_w)] = spatial
        item_angles.append(spatial.repeat(num_frames, 1))

    angles = torch.cat(item_angles)
    # ComplexRoPE.apply_rotary_emb multiplies in complex64; float() only widens
    # the container, so cos/sin keep whatever precision angles were computed in.
    cos_sin = torch.stack((angles.cos(), angles.sin()), dim=-1).float()
    return torch.view_as_complex(cos_sin).unsqueeze(1)


def _temporal_pool_and_merge(
    hidden_TD: torch.Tensor,
    grids: list[list[int]],
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Temporally pool and concatenate neighboring spatial patch features."""
    dim = hidden_TD.shape[-1]
    kernel_h, kernel_w = merge_kernel_size
    merged_dim = kernel_h * kernel_w * dim

    merged_items = []
    offset = 0
    for num_frames, grid_h, grid_w in grids:
        num_patches = num_frames * grid_h * grid_w
        item = hidden_TD[offset : offset + num_patches]
        offset += num_patches
        merged_h = grid_h // kernel_h
        merged_w = grid_w // kernel_w
        item = item.view(
            num_frames,
            merged_h,
            kernel_h,
            merged_w,
            kernel_w,
            dim,
        )
        item = item.permute(0, 1, 3, 2, 4, 5).mean(dim=0)
        merged_items.append(item.reshape(merged_h * merged_w, merged_dim))

    return torch.cat(merged_items)


class VisionRotaryEmbedding2D(Module):
    """Per-axis frequency table for MoonViT's interleaved 2D RoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        if config.head_dim % 4 != 0:
            raise ValueError(
                "Vision 2D RoPE head_dim must be divisible by 4, "
                f"got {config.head_dim}."
            )
        self.head_dim = config.head_dim
        self.theta = config.theta
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, *, device: torch.device | None = None) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    4,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_dim
            )
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.inv_freq.device
        self.inv_freq = self._compute_inv_freq(device=device)

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


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
        # A plan carrying no grid describes a flat patch split with no frame
        # structure, where the padding IS a trailing run. Falling back rather
        # than computing from zeros matters: the branch above would mark
        # nothing valid and mask every key.
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

    # Set by the encoder for the duration of one forward; None means replicated.
    _cp_plan: "CPPatchPlan | None" = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply,
        attention_mask,
    ) -> torch.Tensor:
        plan = self._cp_plan
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



def make_cp_patch_plan(
    grid: tuple[int, int, int],
    *,
    group,
    rank: int,
    merge_kernel_h: int,
) -> tuple["CPPatchPlan", tuple[int, int]]:
    """Build this rank's plan for one image, and the flat range it owns.

    Bands are multiples of the merge kernel height: the projector merges each
    (kh, kw) block, so a cut inside kh rows would ask two ranks to merge halves
    of one block. Every rank keeps every frame -- the temporal pool is a mean
    over all of them, so splitting by frame would give each rank the mean of its
    own frames instead.

    The ceiling split leaves any deficit on the trailing ranks, which is what
    makes the padding land at the end of each frame's band rather than inside
    it; ``_padded_key_keep`` relies on exactly that arithmetic.
    """
    from torchtitan.models.kimi_k3.vit_cp_plan import row_partition

    t, h, w = grid
    group_size = dist.get_world_size(group)
    shards = row_partition(t, h, w, kh=merge_kernel_h, group_size=group_size)
    band = max(s.row_end - s.row_start for s in shards)
    mine = shards[rank]
    plan = CPPatchPlan(
        group=group,
        valid_total=t * h * w,
        full_grid=(t, h, w),
        row_start=mine.row_start,
        band=band,
        real_rows=mine.row_end - mine.row_start,
    )
    return plan, mine.ranges


class KimiK3VisionBlock(Module):
    """MoonViT pre-norm attention and MLP block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: RMSNorm.Config
        norm2: RMSNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x_TD = x_TD + self.attn(
            self.norm1(x_TD),
            rope_cache=rope_cache,
            rope_apply=ComplexRoPE.apply_rotary_emb,
            attention_mask=attention_mask,
        )
        return x_TD + self.mlp(self.norm2(x_TD))


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


class KimiK3VisionEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_layers: int
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]
        init_pos_emb_height: int
        init_pos_emb_width: int
        max_num_frames: int
        interpolation_mode: str
        patch_embed_proj: Linear.Config
        rotary_pos_emb: VisionRotaryEmbedding2D.Config
        block: KimiK3VisionBlock.Config
        final_norm: RMSNorm.Config
        projector: KimiK3VisionProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size
        self.max_num_frames = config.max_num_frames
        self.interpolation_mode = config.interpolation_mode
        self.patch_embed = config.patch_embed_proj.build()
        self.pos_embed = torch.nn.Parameter(
            torch.empty(
                config.init_pos_emb_height,
                config.init_pos_emb_width,
                config.dim,
            )
        )
        self.rotary_pos_emb = config.rotary_pos_emb.build()
        self.register_buffer("_cached_freq_table", None, persistent=False)
        self.layers = ModuleDict(
            {
                str(layer_idx): config.block.build()
                for layer_idx in range(config.num_layers)
            }
        )
        self.final_norm = config.final_norm.build()
        self.projector = config.projector.build()

    def _compute_position_embeddings(
        self, grids: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_grid_side = max(max(grid_h, grid_w) for _, grid_h, grid_w in grids)
        if (
            self._cached_freq_table is None
            or self._cached_freq_table.shape[0] < max_grid_side
        ):
            self._cached_freq_table = self.rotary_pos_emb(max_grid_side)
        learned_pos = _compute_learned_pos_embeds(
            self.pos_embed,
            grids,
            self.interpolation_mode,
            self.max_num_frames,
        )
        rope_cache = _compute_2d_rope_cache(
            self._cached_freq_table,
            grids,
            self.rotary_pos_emb.head_dim,
        )
        return learned_pos, rope_cache

    def set_cp_patch_plan(self, plan: "CPPatchPlan | None") -> None:
        """Hand every block's attention the plan, or clear it.

        The plan lives on the attention rather than being threaded through the
        block signature because the block is shared with the replicated path
        and has nothing to do with it.
        """
        for block in self.layers.values():
            block.attn._cp_plan = plan

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
    ) -> torch.Tensor:
        """Encode packed raster-order patches and return packed text features.

        With ``cp_plan``, ``pixel_values`` is this rank's patch shard of ONE
        image and ``grid_thw`` still describes the whole image: the position
        tables are built for the whole image and sliced, because both the
        learned absolute embedding and the 2-D RoPE are indexed from row 0 and
        describing a shard as a standalone image would give every rank rank 0's
        positions.
        """
        grids = grid_thw.tolist()

        kernel_h, kernel_w = self.merge_kernel_size
        for _, grid_h, grid_w in grids:
            if grid_h % kernel_h != 0 or grid_w % kernel_w != 0:
                raise ValueError(
                    f"Vision grid {grid_h}x{grid_w} is not divisible by "
                    f"merge kernel {self.merge_kernel_size}."
                )

        segment_lengths = grid_thw.prod(dim=-1)
        total_tokens = pixel_values.shape[0]
        expected_tokens = sum(t * h * w for t, h, w in grids)
        if cp_plan is None and total_tokens != expected_tokens:
            raise ValueError(
                f"pixel_values contains {total_tokens} patches but grid_thw "
                f"describes {expected_tokens}."
            )
        if cp_plan is not None and len(grids) != 1:
            raise ValueError(
                "a CP patch plan describes one image, but grid_thw carries "
                f"{len(grids)}; a mixed stream needs a per-segment plan."
            )

        learned_pos, rope_cache = self._compute_position_embeddings(grids)
        if cp_plan is not None:
            learned_pos = _slice_for_shard(learned_pos, cp_plan)
            rope_cache = _slice_for_shard(rope_cache, cp_plan)
        hidden_TD = self.patch_embed(pixel_values) + learned_pos

        attention_mask = None
        if cp_plan is None:
            with spmd.no_typecheck():
                attention_mask = create_block_diagonal_mask(
                    segment_lengths,
                    total_tokens,
                    hidden_TD.device,
                )
        self.set_cp_patch_plan(cp_plan)
        try:
            for block in self.layers.values():
                hidden_TD = block(
                    hidden_TD,
                    rope_cache=rope_cache,
                    attention_mask=attention_mask,
                )
        finally:
            # Cleared unconditionally: a plan left behind would silently make
            # the next replicated forward gather across ranks.
            self.set_cp_patch_plan(None)
        hidden_TD = self.final_norm(hidden_TD)
        # The merge sees the SHARD's grid, not the image's: this rank holds a
        # band of rows for every frame, so its own (t, band, w) is what the
        # (kh, kw) blocking and the temporal mean are over. The positions above
        # needed the whole image; this needs the piece.
        merge_grids = (
            [[cp_plan.full_grid[0], cp_plan.band, cp_plan.full_grid[2]]]
            if cp_plan is not None
            else grids
        )
        merged_MF = _temporal_pool_and_merge(
            hidden_TD, merge_grids, self.merge_kernel_size
        )
        return self.projector(merged_MF)
