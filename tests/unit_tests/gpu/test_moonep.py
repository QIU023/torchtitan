# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.transform import convert_config_type
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointing,
    FullAC,
    RegionAC,
    SelectiveAC,
)
from torchtitan.distributed.moonep import ops
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import MoonEPRoutedExperts
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict

pytestmark = pytest.mark.multi_gpu

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 4.0, 25.0


def _moonep_runs_here() -> bool:
    try:
        import moonep  # noqa: F401
        from torch._C._distributed_c10d import _SymmetricMemory
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return all(
        _SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i)
        for i in range(torch.cuda.device_count())
    )


class _Block(Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        self.experts = experts

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.experts(*args)


class _Model(Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        self.layers = ModuleDict({"0": _Block(experts)})


def _reference(x_TD, weights_TK, ids_TK, w13_E2FD, w2_EDF):
    activation = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA).build()
    out_TD = torch.zeros_like(x_TD)
    for k in range(K):
        e = ids_TK[:, k]
        gate_TF = torch.einsum("td,tfd->tf", x_TD, w13_E2FD[e, 0])
        up_TF = torch.einsum("td,tfd->tf", x_TD, w13_E2FD[e, 1])
        out_TD = out_TD + weights_TK[:, k : k + 1] * torch.einsum(
            "tf,tdf->td", activation(gate_TF, up_TF), w2_EDF[e]
        )
    return out_TD


@pytest.mark.skipif(
    not _moonep_runs_here(), reason="needs the moonep package and NVLink multicast"
)
class TestMoonEPRoutedExperts(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _check(
        self, routing: str, ac: ActivationCheckpointing.Config | None = None
    ) -> None:
        rank, size = self.rank, self.world_size
        device = torch.device("cuda", torch.cuda.current_device())
        mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)

        torch.manual_seed(1)
        w13_E2FD = torch.randn(E, 2, F, D) * 0.05
        w2_EDF = torch.randn(E, D, F) * 0.05
        config = convert_config_type(
            make_routed_experts_config(
                dim=D, hidden_dim=F, num_experts=E, top_k=K, param_init={}
            ),
            MoonEPRoutedExperts,
        )
        config.token_dispatcher = MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S
        )
        config.activation_fn = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA)
        experts = config.build().to(device)
        lo, hi = rank * (E // size), (rank + 1) * (E // size)
        experts.w13.weight = nn.Parameter(w13_E2FD[lo:hi].to(device, torch.bfloat16))
        experts.w2.weight = nn.Parameter(w2_EDF[lo:hi].to(device, torch.bfloat16))
        model = _Model(experts)
        if ac is not None:
            ac.build().apply(model)

        torch.manual_seed(100 + rank)
        x_TD = (torch.randn(S, D, device=device) * 0.5).to(torch.bfloat16)
        if routing == "hot":
            # Every token goes to experts homed on rank 0, so balancing needs the slots.
            ids_TK = torch.arange(K, device=device).repeat(S, 1)
            weights_TK = torch.softmax(torch.randn(S, K, device=device), dim=-1)
        else:
            weights_TK, ids_TK = torch.rand(S, E, device=device).topk(K, dim=-1)
            weights_TK = weights_TK / weights_TK.sum(-1, keepdim=True)
        counts_E = torch.bincount(ids_TK.flatten(), minlength=E)

        dispatched = []
        dispatch = experts.token_dispatcher.dispatch

        def recording_dispatch(*args):
            result = dispatch(*args)
            dispatched.append(result)
            return result

        experts.token_dispatcher.dispatch = recording_dispatch
        x_in_TD = x_TD.clone().requires_grad_(True)
        first_plan = ops._next_plan_id
        with set_current_spmd_mesh(mesh):
            experts.token_dispatcher.init_buffer()
            out_TD = model.layers["0"](x_in_TD, weights_TK, ids_TK, counts_E)
            out_TD.float().sum().backward()
        torch.cuda.synchronize()
        self.assertEqual(ops._next_plan_id - first_plan, 1, "the dispatch was replayed")
        self.assertEqual(ops._plans, {}, "a plan outlived its combine")

        slot_rows = dispatched[0][1][E // size :].sum().reshape(1)
        dist.all_reduce(slot_rows)

        gathered = []
        for t in (x_TD, weights_TK, ids_TK):
            parts = [torch.empty_like(t) for _ in range(size)]
            dist.all_gather(parts, t)
            gathered.append(torch.cat(parts))
        x_all_TD = gathered[0].float().requires_grad_(True)
        w13_ref = w13_E2FD.to(device, torch.bfloat16).float().requires_grad_(True)
        w2_ref = w2_EDF.to(device, torch.bfloat16).float().requires_grad_(True)
        ref_TD = _reference(x_all_TD, gathered[1].float(), gathered[2], w13_ref, w2_ref)
        ref_TD.sum().backward()
        mine = slice(rank * S, (rank + 1) * S)
        experts.token_dispatcher.buffer.destroy()

        if routing == "hot":
            self.assertGreater(int(slot_rows.item()), 0, "no token reached a slot")
        # bf16 weights and activations against an fp32 reference.
        torch.testing.assert_close(out_TD.float(), ref_TD[mine], atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            x_in_TD.grad.float(), x_all_TD.grad[mine], atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            experts.w13.weight.grad.float(), w13_ref.grad[lo:hi], atol=1e-1, rtol=5e-2
        )
        torch.testing.assert_close(
            experts.w2.weight.grad.float(), w2_ref.grad[lo:hi], atol=1e-1, rtol=5e-2
        )

    @with_comms
    def test_hot_routing_fills_slots_and_matches_dense_reference(self):
        self._check("hot")

    @with_comms
    def test_uniform_routing_matches_dense_reference(self):
        self._check("uniform")

    @with_comms
    def test_selective_ac_matches_dense_reference(self):
        self._check("hot", SelectiveAC.Config())

    @with_comms
    def test_full_ac_matches_dense_reference(self):
        self._check("hot", FullAC.Config())

    @with_comms
    def test_region_ac_matches_dense_reference(self):
        self._check("hot", RegionAC.Config(save_regions=[]))


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
