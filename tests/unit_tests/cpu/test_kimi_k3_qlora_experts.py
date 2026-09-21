# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for MXFP4-packed grouped experts under the LoRA converter."""

import pytest
import torch

pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")

from torchtitan.config.transform.lora import (
    LoRAConverter,
    merge_lora_state_dict,
    MXFP4ExpertsBase,
)
from torchtitan.models.kimi_k3 import model_registry


def _build():
    torch.manual_seed(0)
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=4,
                alpha=8.0,
                target_modules=["wo"],
                quantize_experts="mxfp4",
            ),
        ],
    )
    model = spec.model.build()
    model.init_states()
    return model


def test_experts_pack_at_build_and_dequant_property():
    model = _build()
    packed = [m for _, m in model.named_modules() if isinstance(m, MXFP4ExpertsBase)]
    assert packed
    m = packed[0]
    for name, shape in m._mxfp4_shapes.items():
        assert name not in m._parameters
        assert m._parameters[name + "_qdata"].dtype == torch.uint8
        # The property restores the logical [E, A, B] view in bf16.
        w = getattr(m, name)
        assert tuple(w.shape) == shape
        assert w.abs().sum() > 0, "packed experts left zero-initialized"


def test_experts_merge_restores_original_keys():
    model = _build()
    merged = merge_lora_state_dict(model)
    packed = [
        (n, m) for n, m in model.named_modules() if isinstance(m, MXFP4ExpertsBase)
    ]
    name, m = packed[0]
    for wname, shape in m._mxfp4_shapes.items():
        assert f"{name}.{wname}" in merged
        assert tuple(merged[f"{name}.{wname}"].shape) == shape
        assert f"{name}.{wname}_qdata" not in merged
        # The packed layout is back in place after the export.
        assert wname not in m._parameters


def test_experts_pack_under_the_declared_sharding():
    """The trainer's path: the declarations go on before the build, and the
    packed pair inherits the experts' entry (this is where main's SpmdType
    broke the packing at the seed build)."""
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_qlora_mxfp4

    config = kimi_k3_debugmodel_qlora_mxfp4()
    assert config.model_spec is not None
    # The experts declare a state sharding only under expert parallelism.
    config.parallelism.expert_parallel_degree = 2
    model_config = config.model_spec.model
    model_config.update_from_config(config=config)
    with torch.device("meta"):
        model = model_config.build()
    packed = [m for _, m in model.named_modules() if isinstance(m, MXFP4ExpertsBase)]
    assert packed
    # The packing rewrites the declaration the module was built with: the
    # packed pair inherits the experts' entry, the original key goes.
    seen = 0
    for m in packed:
        sharding = m._sharding_config
        if sharding is None or not sharding.state_shardings:
            continue
        seen += 1
        for wname in m._mxfp4_shapes:
            assert wname not in sharding.state_shardings
            assert wname + "_qdata" in sharding.state_shardings
            assert wname + "_scale" in sharding.state_shardings
    assert seen, "no packed experts carried a sharding declaration"


def test_packed_fused_projection_keeps_the_stacked_layout():
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_qlora_mxfp4

    config = kimi_k3_debugmodel_qlora_mxfp4()
    assert config.model_spec is not None
    model = config.model_spec.model.build()
    model.init_states()
    w13 = model.layers["0"].feed_forward.w13
    n, f, d = 2, w13.out_features, w13.in_features
    sd = model.state_dict()
    assert tuple(sd["layers.0.feed_forward.w13.base_qdata"].shape) == (n, f, d // 2)
    assert tuple(sd["layers.0.feed_forward.w13.base_scale"].shape) == (n, f, d // 32)
    assert tuple(w13.lora_b.weight.shape) == (n, f, w13.lora_a.weight.shape[0])
    x = torch.randn(3, d, dtype=w13.lora_a.weight.dtype)
    with torch.no_grad():
        w13.lora_b.weight.normal_()
    dense = w13._dequant_base_mxfp4().to(x.dtype)
    expected = torch.nn.functional.linear(x, dense.flatten(0, 1)) + w13._lora_scaling * (
        torch.nn.functional.linear(
            torch.nn.functional.linear(x, w13.lora_a.weight), w13.lora_b.weight.flatten(0, 1)
        )
    )
    torch.testing.assert_close(w13(x), expected.unflatten(-1, (n, f)))
    model.load_state_dict(sd, strict=True)
    merged = merge_lora_state_dict(model)
    assert tuple(merged["layers.0.feed_forward.w13.weight"].shape) == (n, f, d)
    assert not [k for k in merged if "base_qdata" in k or "lora_" in k]
