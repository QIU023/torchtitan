# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Planning for dynamic CP in the vision encoder (report 5.2.3).

Pure functions, no collectives and no torch tensors in the signatures, so the
scheduling decisions can be tested without spawning ranks. The distributed half
lives in ``multimodal_model`` and ``moonvit``.

Report 5.2.3 asks for two things:

1. "A single large image is partitioned along the patch dimension across multiple
   devices, and attention is computed by gathering key-value pairs (gather-KV)
   across CP ranks."
2. "we divide each CP group into several sub-CP groups and distribute multiple
   large images across them in a load-balanced manner, preventing the
   communication fraction from growing with scale."

The reason (2) exists is in its own clause: gather-KV over the WHOLE CP group
makes every rank exchange every large image's keys, so the communication fraction
grows with the group. Partitioning one image over a sub-group of 2 while another
image occupies a different sub-group keeps the exchange local and the ranks busy.

**The merge kernel constrains where a partition may cut.** The projector merges
each ``(kh, kw)`` block of patches into one output token, so a cut inside a block
would ask two ranks to merge halves of the same block. The safe unit is a
MERGE-ROW BLOCK -- ``kh`` consecutive grid rows, ``kh * w`` patches. Since patches
are laid out row-major over ``(t, h, w)``, such a block is contiguous in the
packed stream and consecutive blocks abut, including across a video's frame
boundary. Cutting on arbitrary patch counts is merge-unsafe; cutting "rows r0..r1
of every frame" is merge-safe but NOT contiguous once ``t > 1``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageShard:
    """One rank's slice of one partitioned image, as flat patch offsets."""

    patch_start: int
    patch_end: int
    """Half-open offsets into the image's flat patch stream. May be empty when the
    image has fewer merge-row blocks than the sub-group has ranks."""

    num_blocks: int
    """How many merge-row blocks this shard covers; 0 for an empty shard."""


def row_partition(
    t: int, h: int, w: int, *, kh: int, group_size: int
) -> list[ImageShard]:
    """Split one image's patches across ``group_size`` ranks at safe boundaries.

    The unit is a MERGE-ROW BLOCK: ``kh`` consecutive grid rows, ``kh * w``
    patches. Two properties make it the right unit, and both are needed:

    * **Merge-safe.** The projector merges each ``(kh, kw)`` block into one token,
      so a cut inside ``kh`` rows would ask two ranks to merge halves of the same
      block. Cutting only at multiples of ``kh`` rows cannot.
    * **Contiguous in the packed stream.** Patches are laid out row-major over
      ``(t, h, w)``, so a merge-row block is contiguous and consecutive blocks are
      adjacent -- including across the frame boundary of a video, where the last
      block of frame f abuts the first of frame f+1. An earlier version cut "rows
      [r0, r1) of every frame", which is contiguous only when ``t == 1`` and gave
      silently wrong offsets for video.

    An image with fewer blocks than ranks leaves the tail ranks empty rather than
    raising: a batch does not get to choose its image sizes, and the caller pads
    for the fixed-shape collective anyway. A non-multiple chunk, by contrast, is
    always a bug, so that is refused.
    """
    if h % kh:
        raise ValueError(
            f"patch grid height {h} must divide the merge kernel height {kh}; "
            "the projector merges (kh, kw) blocks and a partition cannot cut "
            "inside one"
        )
    block_patches = kh * w
    blocks = t * (h // kh)
    # Ceiling split so earlier ranks take the extra block; the tail may be empty.
    per = -(-blocks // group_size)
    shards: list[ImageShard] = []
    for r in range(group_size):
        b0 = min(r * per, blocks)
        b1 = min((r + 1) * per, blocks)
        shards.append(
            ImageShard(
                patch_start=b0 * block_patches,
                patch_end=b1 * block_patches,
                num_blocks=b1 - b0,
            )
        )
    return shards


def subgroup_layout(num_large: int, cp_size: int) -> tuple[int, int]:
    """Choose (number of sub-CP groups, ranks per sub-group).

    One large image and a CP group of 8 gives (1, 8) -- the report's "a single
    large image is partitioned across multiple devices". Four large images and 8
    ranks gives (4, 2), so each image is exchanged inside a pair instead of across
    all eight, which is the communication-fraction argument.

    Sub-groups are equal in size because a process group is formed from a rank
    list and an uneven split would leave a sub-group whose gather is a different
    shape on different ranks. So the count is the largest divisor of ``cp_size``
    that does not exceed ``num_large``.
    """
    if num_large <= 0 or cp_size <= 1:
        return (1, cp_size)
    best = 1
    for n in range(1, cp_size + 1):
        if cp_size % n == 0 and n <= num_large:
            best = n
    return (best, cp_size // best)


def balance_images(sizes: list[int], num_groups: int) -> list[int]:
    """Assign each image to a sub-group, longest-processing-time-first.

    Returns ``group_of[i]`` for every entry of ``sizes``. LPT rather than
    round-robin: round-robin on sizes [100, 10, 10, 10] with two groups gives 110
    against 20, while LPT gives 100 against 30. The report asks for "a
    load-balanced manner" and the imbalance it is trying to remove is exactly this
    one.
    """
    if num_groups <= 1:
        return [0] * len(sizes)
    load = [0] * num_groups
    group_of = [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda j: -sizes[j]):
        g = min(range(num_groups), key=lambda x: load[x])
        group_of[i] = g
        load[g] += sizes[i]
    return group_of


def classify(counts: list[int], cp_size: int, *, min_patches: int) -> list[int]:
    """Indices of the images worth partitioning within a sub-group.

    An image is only worth splitting if the split leaves each rank real work: the
    threshold is on the image's own patch count, not on the batch. Below it the
    image-level round-robin already balances better, because splitting a small
    image buys one gather per layer for nothing.
    """
    if cp_size <= 1:
        return []
    return [i for i, c in enumerate(counts) if c >= min_patches]
