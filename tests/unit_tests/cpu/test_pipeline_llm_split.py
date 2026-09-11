# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part


def _layers_per_stage(fqns):
    return [sum(1 for n in stage if n.startswith("layers.")) for stage in fqns]


class TestGenerateLLMFqnPerModelPart(unittest.TestCase):
    def test_default_splits_are_unchanged(self):
        """The shapes the models rely on, with no pinned modules."""
        self.assertEqual(
            _generate_llm_fqn_per_model_part(1, 2),
            [["tok_embeddings", "layers.0", "layers.1", "norm", "lm_head"]],
        )
        self.assertEqual(
            _generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2),
            [
                ["tok_embeddings", "layers.0", "layers.1"],
                ["layers.2", "norm", "lm_head"],
            ],
        )
        fqns = _generate_llm_fqn_per_model_part(4, 8)
        self.assertEqual(_layers_per_stage(fqns), [2, 3, 2, 1])
        self.assertEqual(fqns[0][0], "tok_embeddings")
        self.assertEqual(fqns[-1][-2:], ["norm", "lm_head"])

    def test_pinned_modules_ride_with_the_embedding_and_the_head(self):
        """Extra modules land on the first / last stage and count as no layer."""
        fqns = _generate_llm_fqn_per_model_part(
            2,
            3,
            first_stage_modules=("vision_encoder",),
            last_stage_modules=("output_res_proj", "output_res_norm"),
        )
        self.assertEqual(
            _layers_per_stage(fqns),
            _layers_per_stage(_generate_llm_fqn_per_model_part(2, 3)),
        )
        self.assertEqual(fqns[0][:2], ["tok_embeddings", "vision_encoder"])
        self.assertEqual(
            fqns[-1][-4:], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )
        single = _generate_llm_fqn_per_model_part(
            1, 2, first_stage_modules=("vision_encoder",), last_stage_modules=("tail",)
        )
        self.assertEqual(
            single,
            [
                [
                    "tok_embeddings",
                    "vision_encoder",
                    "layers.0",
                    "layers.1",
                    "norm",
                    "lm_head",
                    "tail",
                ]
            ],
        )

    def test_a_unit_count_no_shape_divides_still_splits(self):
        """33 layers are 35 units with the embedding and the head: over 32
        stages the split is uneven and the last stage holds the head alone."""
        fqns = _generate_llm_fqn_per_model_part(
            32, 33, last_stage_modules=("output_res_proj", "output_res_norm")
        )
        self.assertEqual(len(fqns), 32)
        self.assertEqual(sum(_layers_per_stage(fqns)), 33)
        self.assertEqual(_layers_per_stage(fqns)[-1], 0)
        self.assertEqual(
            fqns[-1], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )


if __name__ == "__main__":
    unittest.main()
