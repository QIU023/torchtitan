# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts around the CP wiring that only a GPU run has checked so far.

Every fix in this file was established by running the 58-cell gate or a probe, and a
gate run is not a repeatable check: it needs eight GPUs and 75 minutes, and it cannot be
run against a patch before that patch lands. These pin the same contracts on CPU.

What is covered, and why each one needed a check rather than an argument:

* ``conv_with_halo`` must hand fla a PLAIN weight. KDA is NoParallel under TP, so its
  short-conv weight is a DTensor(Replicate), and passing that to the triton kernel does
  not raise anything legible -- it surfaced as CUBLAS_STATUS_INTERNAL_ERROR and illegal
  memory accesses across thirteen gate cells. The Ulysses path unwraps the same weight
  in its own conv_subset; the KCP path did not, and nothing noticed because KCP had only
  ever run in a flavor without TP.
* ``build_kcp_context`` must pass through real document boundaries. It used to hardcode
  a single document, which is right for the caller it has and wrong to bake in.
* ``verify_params_distributed`` must name the parameter. Its absence let a plain
  parameter reach ``clip_grad_norm_``, which reports
  ``aten._foreach_mul_.Tensor got mixed`` and names neither the parameter nor the
  mechanism that skipped it.
* ``verify_ep_applied`` must accept an empty plan. Under PP a rank can hold only the
  vision-tower stage and therefore no MoE at all; treating that as a failure is what
  took down every ep+pp cell on the multimodal arms.
"""

from __future__ import annotations

import unittest

import torch


class TestConvWithHaloUnwrapsWeights(unittest.TestCase):
    """The KCP conv must not hand a DTensor to fla's kernel."""

    def _run(self, make_weight):
        import sys
        import types

        import torch.distributed as dist

        from torch.distributed.device_mesh import init_device_mesh

        seen = {}

        # Stand in for fla's kernel: the contract under test is what it RECEIVES, and
        # calling the real one needs CUDA, triton and a live CP context.
        def fake_causal_conv1d_cp(*, x, weight, bias, activation, cp_context):
            seen["weight"] = weight
            seen["bias"] = bias
            return x

        module = types.ModuleType("fla.modules.conv.cp.ops")
        module.causal_conv1d_cp = fake_causal_conv1d_cp
        saved = sys.modules.get("fla.modules.conv.cp.ops")
        sys.modules["fla.modules.conv.cp.ops"] = module
        try:
            from torchtitan.models.kimi_k3.kcp import conv_with_halo

            conv = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=True)
            conv.weight = torch.nn.Parameter(make_weight(conv.weight.data))
            conv.bias = torch.nn.Parameter(make_weight(conv.bias.data))
            conv_with_halo(
                conv, torch.zeros(1, 8, 4), cp_context=object(), activation=None
            )
        finally:
            if saved is None:
                sys.modules.pop("fla.modules.conv.cp.ops", None)
            else:
                sys.modules["fla.modules.conv.cp.ops"] = saved
        del init_device_mesh, dist
        return seen

    def test_a_plain_weight_passes_through(self):
        seen = self._run(lambda t: t)
        self.assertIsInstance(seen["weight"], torch.Tensor)
        self.assertNotIn("DTensor", type(seen["weight"]).__name__)

    def test_a_dtensor_weight_is_unwrapped(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor, DTensor, Replicate

        if not dist.is_initialized():
            import os

            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29511")
            dist.init_process_group("gloo", rank=0, world_size=1)
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))

        def as_dtensor(t):
            return distribute_tensor(t, mesh, [Replicate()])

        seen = self._run(as_dtensor)
        self.assertNotIsInstance(
            seen["weight"], DTensor, "fla's kernel received a DTensor weight"
        )
        self.assertNotIsInstance(seen["bias"], DTensor)


class TestKcpContextBoundaries(unittest.TestCase):
    """Document boundaries are the caller's to state, not the helper's to assume."""

    def _capture(self, **kwargs):
        import sys
        import types

        import torch.distributed as dist

        seen = {}

        def fake_build_cp_context(cu_seqlens, *, group, conv1d_kernel_size=None):
            seen["cu_seqlens"] = cu_seqlens
            seen["conv1d_kernel_size"] = conv1d_kernel_size
            return object()

        module = types.ModuleType("fla.ops.cp.context")
        module.build_cp_context = fake_build_cp_context
        saved = sys.modules.get("fla.ops.cp.context")
        sys.modules["fla.ops.cp.context"] = module
        try:
            from torchtitan.models.kimi_k3.kcp import build_kcp_context

            if not dist.is_initialized():
                import os

                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29512")
                dist.init_process_group("gloo", rank=0, world_size=1)
            build_kcp_context(
                16,
                dist.group.WORLD,
                torch.device("cpu"),
                conv1d_kernel_size=4,
                **kwargs,
            )
        finally:
            if saved is None:
                sys.modules.pop("fla.ops.cp.context", None)
            else:
                sys.modules["fla.ops.cp.context"] = saved
        return seen

    def test_the_default_is_one_document_spanning_the_global_sequence(self):
        seen = self._capture()
        # world size 1, local 16 -> global 16.
        self.assertEqual(seen["cu_seqlens"].tolist(), [0, 16])
        self.assertEqual(seen["conv1d_kernel_size"], 4)

    def test_real_boundaries_are_passed_through_unchanged(self):
        packed = torch.tensor([0, 5, 11, 16], dtype=torch.int32)
        seen = self._capture(cu_seqlens=packed)
        self.assertEqual(seen["cu_seqlens"].tolist(), [0, 5, 11, 16])


class TestParamDistributionVerifier(unittest.TestCase):
    """A plain parameter must be named here, not inside clip_grad_norm_."""

    def _mesh(self):
        import os

        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29513")
            dist.init_process_group("gloo", rank=0, world_size=1)
        return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))

    def test_a_plain_parameter_raises_and_is_named(self):
        from torchtitan.models.kimi_k3.parallelize import verify_params_distributed

        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with self.assertRaises(ValueError) as ctx:
            verify_params_distributed(model)
        # The point of the check is that the message points at the parameter.
        self.assertIn("0.weight", str(ctx.exception))
        self.assertIn("plain Tensor", str(ctx.exception))

    def test_all_dtensor_parameters_pass(self):
        from torch.distributed.tensor import distribute_tensor, Replicate

        from torchtitan.models.kimi_k3.parallelize import verify_params_distributed

        mesh = self._mesh()
        model = torch.nn.Linear(4, 4, bias=False)
        model.weight = torch.nn.Parameter(
            distribute_tensor(model.weight.data, mesh, [Replicate()])
        )
        verify_params_distributed(model)  # must not raise

    def test_a_model_with_no_parameters_is_not_a_failure(self):
        from torchtitan.models.kimi_k3.parallelize import verify_params_distributed

        verify_params_distributed(torch.nn.Identity())


class TestEpVerifierOnAnEmptyPlan(unittest.TestCase):
    """A rank holding no MoE is a normal state under PP, not a missing plan.

    ``ep_expected`` is assigned only when this rank has MoE layers, while the verify call
    is guarded on ``ep_enabled`` -- a property of the JOB. Under PP a rank can hold only
    the vision-tower stage, and reading the unset local there took down every ep+pp cell
    on the multimodal arms.
    """

    def test_an_empty_plan_verifies_vacuously(self):
        from torchtitan.models.kimi_k3.parallelize import verify_ep_applied

        verify_ep_applied([])  # must not raise

    def test_a_layer_whose_experts_are_missing_is_reported(self):
        from torchtitan.models.kimi_k3.parallelize import verify_ep_applied

        moe = torch.nn.Module()  # no routed_experts at all
        with self.assertRaises(ValueError) as ctx:
            verify_ep_applied([(3, moe)])
        self.assertIn("layer 3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestKcpBatchLoop(unittest.TestCase):
    """The batch axis is handled by looping, and the loop's shape is the contract.

    fla's ``causal_conv1d_cp`` asserts ``[1, T, D]``, so the CP path cannot take a batch
    at all -- it raised for B > 1, which is most of the gate's cells, until the default
    moved to KCP and the loop was added. A GPU parity probe measures that the numbers come
    out right; what it cannot show is the STRUCTURE: that each row is handed over on its
    own, in order, and reassembled in the same order. A loop that passed the whole batch
    to one call, or that reused row 0's slice, could still produce plausible numbers.

    Flattening into one packed sequence instead would be wrong rather than merely
    awkward: ``build_cp_context`` cuts the GLOBAL packed sequence into contiguous
    rank-ordered pieces, while a rank holds piece r of EVERY sequence, so the layouts
    coincide only at B = 1.
    """

    def _kda(self):
        from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiK3Config

        flat = KimiK3Config(
            hidden_size=32,
            kda_num_heads=2,
            kda_head_dim=16,
            kda_short_conv_kernel_size=4,
            kda_use_full_rank_gate=True,
            kda_cp_mode="kcp",
        )
        return KimiDeltaAttention.make_config(flat, layer_idx=0).build()

    def test_each_row_is_handed_over_alone_and_in_order(self):
        kda = self._kda()
        seen = []

        def fake_one(x, cp_group):
            seen.append(x)
            # Return something row-identifiable so the concatenation order is checkable.
            return x[..., :1] * 0 + len(seen)

        kda._forward_kcp_one = fake_one
        x = torch.arange(3 * 5 * 32, dtype=torch.float32).reshape(3, 5, 32)
        out = kda._forward_kcp(x, cp_group=object())

        self.assertEqual(len(seen), 3, "one call per batch row")
        for b, got in enumerate(seen):
            self.assertEqual(tuple(got.shape), (1, 5, 32), "each call gets [1, L, D]")
            torch.testing.assert_close(got, x[b : b + 1], rtol=0, atol=0)
        # Reassembled in call order, so row b of the output came from call b.
        self.assertEqual(tuple(out.shape), (3, 5, 1))
        torch.testing.assert_close(out[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))

    def test_a_single_row_does_not_take_the_loop(self):
        """B = 1 must reach the same call the loop would make, without a cat."""
        kda = self._kda()
        seen = []

        def fake_one(x, cp_group):
            seen.append(x)
            return x

        kda._forward_kcp_one = fake_one
        x = torch.zeros(1, 4, 32)
        out = kda._forward_kcp(x, cp_group=object())
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], x, "the single-row path should not slice or copy")
        self.assertIs(out, x)
