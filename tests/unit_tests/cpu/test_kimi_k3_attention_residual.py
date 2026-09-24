# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import AttentionResidual


class TestKimiK3AttentionResidual(unittest.TestCase):
    def test_the_aggregation_is_a_config_node(self):
        # The override mechanism swaps Config nodes, so every aggregation sits
        # in the model's config tree while the weights stay on the block.
        config = model_registry("debugmodel")
        for node in (
            config.layers[0].ffn_res_fn,
            config.layers[1].attention_res_fn,
            config.output_res_fn,
        ):
            self.assertIsInstance(node, AttentionResidual.Config)
        model = config.build()
        self.assertEqual(
            [name for name, _ in model.named_parameters() if "_res_fn" in name], []
        )


if __name__ == "__main__":
    unittest.main()
