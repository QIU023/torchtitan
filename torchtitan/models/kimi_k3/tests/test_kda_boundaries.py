# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document boundaries for the KDA and its short convolution come from the positions."""

import torch

from torchtitan.models.kimi_k3.kda import cu_seqlens_from_positions


def test_restarts_become_packed_offsets():
    positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    assert cu_seqlens_from_positions(positions, 9).tolist() == [0, 3, 5, 9]


def test_a_single_document_keeps_the_unpacked_kernels():
    assert cu_seqlens_from_positions(torch.arange(16), 16) is None
    assert cu_seqlens_from_positions(None, 16) is None


def test_a_shard_that_starts_mid_document_gets_a_leading_offset():
    # A context-parallel shard can begin inside a document: the first offset is
    # still 0, the restart inside the shard is a real boundary.
    positions = torch.tensor([5, 6, 7, 0, 1])
    assert cu_seqlens_from_positions(positions, 5).tolist() == [0, 3, 5]


def test_the_batched_stream_is_unfolded():
    positions = torch.tensor([[0, 1, 0, 1]])
    assert cu_seqlens_from_positions(positions, 4).tolist() == [0, 2, 4]
