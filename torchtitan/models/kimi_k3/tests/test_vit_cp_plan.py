# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic CP scheduling: sub-group layout, load balance, merge-aligned cuts."""

import unittest

from torchtitan.models.kimi_k3.vit_cp_plan import (
    balance_images,
    classify,
    row_partition,
    subgroup_layout,
)


class TestRowPartition(unittest.TestCase):
    def test_cuts_land_on_merge_block_boundaries(self):
        # h=8, kh=2 -> 4 merge blocks over 2 ranks -> 2 blocks (4 rows) each.
        shards = row_partition(1, 8, 3, kh=2, group_size=2)
        self.assertEqual(
            [(s.patch_start, s.patch_end) for s in shards], [(0, 12), (12, 24)]
        )
        for s in shards:
            self.assertEqual(s.patch_start % (2 * 3), 0)
            self.assertEqual((s.patch_end - s.patch_start) % (2 * 3), 0)

    def test_patch_offsets_are_contiguous_and_cover_the_image(self):
        shards = row_partition(1, 8, 3, kh=2, group_size=2)
        self.assertEqual(shards[0].patch_start, 0)
        for a, b in zip(shards[:-1], shards[1:]):
            self.assertEqual(a.patch_end, b.patch_start)
        self.assertEqual(shards[-1].patch_end, 8 * 3)

    def test_video_frames_stay_contiguous(self):
        """t>1: blocks run frame by frame through the flat stream. Cutting 'rows
        r0..r1 of every frame' would not be contiguous and was the earlier bug."""
        t, h, w, kh = 3, 4, 5, 2
        shards = row_partition(t, h, w, kh=kh, group_size=3)
        self.assertEqual(shards[0].patch_start, 0)
        for a, b in zip(shards[:-1], shards[1:]):
            self.assertEqual(a.patch_end, b.patch_start)
        self.assertEqual(shards[-1].patch_end, t * h * w)
        self.assertEqual(sum(s.num_blocks for s in shards), t * (h // kh))

    def test_uneven_split_gives_an_empty_tail_not_a_broken_block(self):
        shards = row_partition(1, 6, 2, kh=2, group_size=2)
        self.assertEqual(
            [(s.patch_start, s.patch_end) for s in shards], [(0, 8), (8, 12)]
        )
        # 1 block over 4 ranks: rank 0 takes it, the rest are empty rather than
        # being handed a fraction of a merge block.
        shards = row_partition(1, 2, 2, kh=2, group_size=4)
        self.assertEqual([s.num_blocks for s in shards], [1, 0, 0, 0])
        self.assertEqual(
            [(s.patch_start, s.patch_end) for s in shards],
            [(0, 4), (4, 4), (4, 4), (4, 4)],
        )

    def test_height_not_divisible_by_the_kernel_is_refused(self):
        with self.assertRaises(ValueError):
            row_partition(1, 7, 2, kh=2, group_size=2)


class TestSubgroupLayout(unittest.TestCase):
    def test_one_large_image_uses_the_whole_group(self):
        self.assertEqual(subgroup_layout(1, 8), (1, 8))

    def test_four_large_images_on_eight_ranks_pair_up(self):
        self.assertEqual(subgroup_layout(4, 8), (4, 2))

    def test_sub_group_count_divides_the_cp_size(self):
        # 3 large images on 8 ranks: 3 does not divide 8, so 2 groups of 4.
        self.assertEqual(subgroup_layout(3, 8), (2, 4))

    def test_more_images_than_ranks_caps_at_one_rank_each(self):
        self.assertEqual(subgroup_layout(20, 8), (8, 1))


class TestBalance(unittest.TestCase):
    def test_lpt_beats_round_robin_on_a_skewed_batch(self):
        sizes = [100, 10, 10, 10]
        g = balance_images(sizes, 2)
        loads = [sum(s for s, gg in zip(sizes, g) if gg == i) for i in range(2)]
        self.assertEqual(sorted(loads), [30, 100])
        rr = [i % 2 for i in range(4)]
        rr_loads = [sum(s for s, gg in zip(sizes, rr) if gg == i) for i in range(2)]
        self.assertEqual(sorted(rr_loads), [20, 110])
        self.assertLess(max(loads), max(rr_loads))

    def test_single_group_is_a_no_op(self):
        self.assertEqual(balance_images([5, 1, 3], 1), [0, 0, 0])


class TestClassify(unittest.TestCase):
    def test_threshold_is_on_the_image_not_the_batch(self):
        self.assertEqual(classify([1000, 10, 2000], 4, min_patches=512), [0, 2])

    def test_no_cp_means_nothing_to_partition(self):
        self.assertEqual(classify([1000, 2000], 1, min_patches=512), [])


if __name__ == "__main__":
    unittest.main()
