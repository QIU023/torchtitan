# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rank store's offload path: what it parks and what it hands back."""

import unittest

import torch

from torchtitan.models.kimi_k3.pipeline_parallel.stage import PPRankLocalCache


class TestRankStoreOffload(unittest.TestCase):
    def test_without_a_device_the_store_hands_back_what_it_was_given(self):
        store = PPRankLocalCache()
        block = torch.randn(4, 8)
        store.put(0, 0, block)
        self.assertIs(store.blocks(0)[0], block)

    @unittest.skipUnless(torch.cuda.is_available(), "the park is a device to host copy")
    def test_with_a_device_the_block_parks_on_the_host_and_comes_back(self):
        device = torch.device("cuda", torch.cuda.current_device())
        store = PPRankLocalCache(device=device)
        block = torch.randn(4, 8, device=device)
        store.put(0, 0, block)
        parked = store._blocks[0][0]
        self.assertEqual(parked.device.type, "cpu")
        self.assertTrue(parked.is_pinned())
        back = store.blocks(0)[0]
        self.assertEqual(back.device, device)
        torch.testing.assert_close(back, block)


if __name__ == "__main__":
    unittest.main()
