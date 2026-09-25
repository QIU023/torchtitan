# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

from torchtitan.distributed.activation_storage import ActivationStorage, HostBackend


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.w(x)) + x


class _Stage(nn.Module):
    def __init__(self, dim: int, n: int, wrap: str) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): _Block(dim) for i in range(n)})
        for name, block in list(self.layers.items()):
            if wrap == "torch":
                self.layers[name] = checkpoint_wrapper(
                    block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
            else:
                block.forward = remat.checkpoint(region_name=f"layers.{name}")(block.forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers.values():
            x = block(x)
        return x


def _grads(stage: nn.Module, x: torch.Tensor, storage: ActivationStorage | None, prefetch: bool):
    stage.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    if storage is None:
        y = stage(x)
    else:
        with storage.forward(0, 0, keep=(x,)):
            y = stage(x)
        if prefetch:
            storage.prefetch_first(0, 0)
    y.square().sum().backward()
    if storage is not None:
        storage.finish(0, 0)
    return [p.grad.clone() for p in stage.parameters()] + [x.grad.clone()]


class TestActivationStorage(unittest.TestCase):
    def _check(self, wrap: str) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, wrap)
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        for prefetch in (False, True):
            storage = ActivationStorage(
                torch.device("cpu"),
                {"host": HostBackend()},
                lambda tensor, chunk: "host",
                min_tensor_bytes=0,
            )
            storage.register_stage(0, stage.layers)
            got = _grads(stage, x, storage, prefetch)
            for a, b in zip(ref, got):
                self.assertTrue(torch.equal(a, b))
            self.assertGreater(storage.stats["host_bytes"], 0)
            # the first layer's input is the stage input the caller keeps; later layers' inputs move
            self.assertEqual(storage.stats["late_fetches"], 0 if prefetch else 1)
            self.assertFalse(storage._chunks)

    def test_torch_checkpoint_saves_round_trip_bitwise(self) -> None:
        self._check("torch")

    def test_remat_checkpoint_saves_round_trip_bitwise(self) -> None:
        self._check("remat")


if __name__ == "__main__":
    unittest.main()
