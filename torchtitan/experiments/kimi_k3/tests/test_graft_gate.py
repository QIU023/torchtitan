# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graft-gate identity tests (HANDOFF sec 5 anchor).

alpha-gated zero-init AttnRes must EXACTLY reproduce the plain
backbone's function at step 0; the ungated zero-init read is a uniform
source-average and must NOT (that distinction is the reason the gate
exists -- lock both directions in).
"""

import unittest

import torch

from torchtitan.experiments.kimi_k3 import config_registry
from torchtitan.experiments.kimi_k3.model import KimiLinearSpec


def _pair(gated: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    kimi_config = config_registry.kimi_linear_debugmodel().model_spec.model.kimi_config
    graft_spec = KimiLinearSpec(
        kimi_config=kimi_config, num_blocks=4, attn_res_gated=gated
    )
    base_spec = KimiLinearSpec(kimi_config=kimi_config, num_blocks=None)
    with torch.device(device):
        graft = graft_spec.build()
        graft.init_weights()
        base = base_spec.build()
        base.init_weights()
    # Share the backbone: copy the key intersection graft -> base
    # (graft-only extras: *_res_proj / *_res_norm / *_res_alpha).
    bsd = base.state_dict()
    shared = {k: v for k, v in graft.state_dict().items() if k in bsd}
    assert set(shared) == set(bsd)
    base.load_state_dict(shared, strict=True)
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 2016, (2, 128), generator=g).to(device)
    graft.eval()
    base.eval()
    with torch.no_grad():
        return graft(tokens).float(), base(tokens).float()


class TestGraftGate(unittest.TestCase):
    def test_gated_zero_init_is_exact_identity(self):
        lg, lb = _pair(gated=True)
        self.assertTrue(
            torch.equal(lg, lb),
            f"gated graft must be bit-identical at step 0; "
            f"max delta {(lg - lb).abs().max().item():.3e}",
        )

    def test_ungated_zero_init_is_not_identity(self):
        lg, lb = _pair(gated=False)
        self.assertGreater(
            (lg - lb).abs().max().item(), 1e-4,
            "ungated zero-init read is a uniform source-average and is "
            "expected to differ from the plain backbone -- if this ever "
            "matches exactly, the read semantics changed",
        )


if __name__ == "__main__":
    unittest.main()
