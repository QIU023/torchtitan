# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constructs every registered flavor end-to-end on CPU.

Catches upstream config-API drift in flavors the unit tests never touch
(the pressure-test carriers, the 48B downscales, the fp8 variant). Pure
config construction -- no weights are materialized.
"""

import inspect
import unittest

import torchtitan.experiments.kimi_k3 as kimi_k3
from torchtitan.experiments.kimi_k3 import config_registry


class TestFlavorRegistrySweep(unittest.TestCase):
    def test_every_dense_model_spec_builds(self):
        for flavor in sorted(kimi_k3.attn_res_configs):
            with self.subTest(flavor=flavor):
                spec = kimi_k3.model_registry(flavor)
                self.assertIsNotNone(spec.parallelize_fn)

    def test_every_kimi_model_spec_builds(self):
        for flavor in config_registry.flavor_names():
            with self.subTest(flavor=flavor):
                spec = kimi_k3.model_registry(flavor)
                self.assertIsNotNone(spec.parallelize_fn)

    def test_every_trainer_config_builds(self):
        for name, fn in sorted(vars(config_registry).items()):
            if not (
                inspect.isfunction(fn)
                and fn.__module__ == config_registry.__name__
                and name.startswith(("llama3_", "dsv3_", "kimi_linear_"))
            ):
                continue
            with self.subTest(flavor=name):
                try:
                    cfg = fn()
                except ValueError as e:
                    # Float8 swap requires SM89+; the fp8 flavor is
                    # hardware-gated, not a config error.
                    if "float8 is only supported" in str(e):
                        self.skipTest("float8 requires SM89+ hardware")
                    raise
                self.assertIsNotNone(cfg.model_spec)

    def test_unknown_flavor_raises_value_error(self):
        with self.assertRaises(ValueError):
            kimi_k3.model_registry("no_such_flavor")


if __name__ == "__main__":
    unittest.main()
