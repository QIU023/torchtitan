# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parking a pipeline rank's saved activations leaves every gradient bitwise unchanged, on gloo."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_storage import (
    ActivationStorage,
    HostBackend,
    RemoteBackend,
)
from torchtitan.models.kimi_k3.pipeline_parallel.activations import ActivationPlan
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    _grad_send_wait_points,
    _GradSendWaits,
    AttnResPipelineStage,
)

TOKENS, DIM, LAYERS, BLOCK, MICROBATCHES = 8, 16, 8, 2, 4


class _Layer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.w = nn.Linear(DIM, DIM, dtype=torch.float64)

    def forward(self, hidden: torch.Tensor, stack: torch.Tensor):
        if self.layer % BLOCK == 0:
            stack = torch.cat((stack, hidden.unsqueeze(1)), dim=1)
        weights = torch.softmax(stack @ self.w.weight[0], dim=1)
        mixed = (weights.unsqueeze(-1) * stack).sum(1)
        return torch.tanh(self.w(mixed)) + hidden, stack


class _Stage(nn.Module):
    def __init__(self, layers: list[int], *, first: bool, last: bool) -> None:
        super().__init__()
        self.first, self.last = first, last
        self.layers = nn.ModuleDict(
            {
                str(layer): checkpoint_wrapper(_Layer(layer), checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                for layer in layers
            }
        )

    def forward(self, hidden: torch.Tensor, stack: torch.Tensor | None = None):
        if self.first:
            stack = hidden.new_zeros(hidden.shape[0], 0, DIM)
        for layer in self.layers.values():
            hidden, stack = layer(hidden, stack)
        if self.last:
            return hidden.sum(-1)
        return hidden, stack


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target)


class TestPipelineActivationOffload(DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 2

    def _train_pipeline(self, park: int, remote: bool = False):
        torch.manual_seed(0)
        split = [[0, 1], [2, 3], [4, 5], [6, 7]]
        modules_all = [_Stage(l, first=s == 0, last=s == 3) for s, l in enumerate(split)]
        mine = list(range(self.rank, 4, self.world_size))
        stages = [
            AttnResPipelineStage(modules_all[s], s, 4, torch.device("cpu")) for s in mine
        ]
        schedule = ScheduleInterleaved1F1B(
            list(stages), n_microbatches=MICROBATCHES, loss_fn=_loss, scale_grads=False
        )
        stage_to_rank = dict(stages[0].stage_index_to_group_rank)
        layout = infer_block_layout_tables(
            stage_to_rank=stage_to_rank,
            n_layers=LAYERS,
            layers_per_block=BLOCK,
            layer_to_stage={l: s for s, ls in enumerate(split) for l in ls},
        )
        store = PPRankLocalCache()
        waits = _GradSendWaits(_grad_send_wait_points(schedule.pipeline_order, stage_to_rank, self.rank))
        storage = None
        if park:
            backends: dict = {"host": HostBackend()}
            moves = [("host", park, 1)]
            if remote:
                # rank 0 parks on rank 1 through mooncake's TCP transport
                backends["remote"] = RemoteBackend(
                    stages[0].group,
                    dests={0: 1},
                    pool_bytes=64 << 20,
                    staging_bytes=16 << 20,
                    device=torch.device("cpu"),
                )
                moves = [("remote", park, 1)] if self.rank == 0 else []
            plan = ActivationPlan(
                schedule.pipeline_order[self.rank],
                {stage.stage_index: len(stage.submod.layers) for stage in stages},
                moves,
            )
            storage = ActivationStorage(
                torch.device("cpu"),
                backends,
                lambda tensor, chunk: plan.backend(chunk[0], chunk[1]),
                min_tensor_bytes=0,
            )
        for stage in stages:
            stage.set_routing(layout, store, wait_sends_at_backward=True, grad_send_waits=waits)
            if storage is not None:
                storage.register_stage(stage.stage_index, stage.submod.layers)
                stage.set_activation_storage(storage, plan)
        inputs = torch.randn(MICROBATCHES * TOKENS, DIM, dtype=torch.float64)
        targets = torch.randn(MICROBATCHES * TOKENS, dtype=torch.float64)
        losses: list[torch.Tensor] = []
        if 3 in mine:
            schedule.step(target=targets, losses=losses)
        else:
            schedule.step(inputs)
        grads = {
            (stage.stage_index, name): p.grad.clone()
            for stage in stages
            for name, p in stage.submod.named_parameters()
        }
        return grads, [loss.detach() for loss in losses], storage

    @with_comms
    def test_parked_saves_leave_every_gradient_bitwise(self):
        reference, ref_losses, _ = self._train_pipeline(park=0)
        for park in (2, 8):
            grads, losses, storage = self._train_pipeline(park=park)
            self.assertEqual(reference.keys(), grads.keys())
            for key, grad in grads.items():
                torch.testing.assert_close(grad, reference[key], rtol=0, atol=0)
            for a, b in zip(losses, ref_losses):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(storage.stats["host_bytes"], 0)
            # each backward's first layer came back ahead of it, the rest one layer ahead
            self.assertEqual(storage.stats["late_fetches"], 0)
            self.assertFalse(storage._chunks)

    @with_comms
    def test_saves_parked_on_a_peer_leave_every_gradient_bitwise(self):
        try:
            from torchtitan.distributed.activation_storage import _load_transfer_engine

            _load_transfer_engine()
        except ImportError as err:
            self.skipTest(str(err))
        reference, ref_losses, _ = self._train_pipeline(park=0)
        grads, losses, storage = self._train_pipeline(park=8, remote=True)
        for key, grad in grads.items():
            torch.testing.assert_close(grad, reference[key], rtol=0, atol=0)
        for a, b in zip(losses, ref_losses):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if self.rank == 0:
            self.assertGreater(storage.stats["remote_bytes"], 0)
        self.assertFalse(storage._chunks)
