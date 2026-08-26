# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline stage FQN injection, on CPU.

This exists because the injection used to return early on a Config-tree model
-- it read the layer count from a flat config's ``num_hidden_layers`` -- and
said nothing when it did. The split then fell back to core's, which left the
AttnRes aggregation modules off the last stage, and the only visible symptom
was a loss that trained to a different place.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch.nn as nn

from torchtitan.models.kimi_k3.pipeline_adapter import _inject_kimi_k3_fqns


class _Model(nn.Module):
    """Enough of the model for the injection to recognise it: the two AttnRes
    tail modules and core's spellings for the embedding and the head."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(4, 4)
        self.layers = nn.ModuleDict({str(i): nn.Linear(4, 4) for i in range(8)})
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4)
        self.output_res_norm = nn.LayerNorm(4)
        self.output_res_proj = nn.Linear(4, 4)


class _MultimodalModel(_Model):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4)


class _TextModel(_Model):
    """A text model as the real one is built: the attribute EXISTS and is None.

    ``_Model`` omits it entirely, so a ``hasattr`` test passes there and fails
    here -- which is the whole point of testing this shape separately.
    """

    def __init__(self):
        super().__init__()
        self.vision_encoder = None


def _kwargs(num_layers: int):
    parallelism = SimpleNamespace(
        module_fqns_per_model_part=None,
        pipeline_parallel_first_stage_less_layers=1,
        pipeline_parallel_last_stage_less_layers=1,
        pipeline_parallel_layers_per_stage=None,
        pipeline_parallel_schedule="1F1B",
    )
    # A Config-tree model carries the layers themselves rather than a count.
    model_config = SimpleNamespace(layers=[object()] * num_layers)
    return {
        "parallelism": parallelism,
        "model_config": model_config,
        "parallel_dims": SimpleNamespace(pp=2),
    }


class TestFQNInjection(unittest.TestCase):
    def test_a_config_tree_model_still_gets_a_split(self):
        kwargs = _kwargs(8)
        _inject_kimi_k3_fqns(_Model(), kwargs)
        fqns = kwargs["parallelism"].module_fqns_per_model_part
        self.assertIsNotNone(fqns, "injection returned early and said nothing")
        self.assertEqual(len(fqns), 2)

    def test_the_attn_res_tail_lands_on_the_last_stage(self):
        kwargs = _kwargs(8)
        _inject_kimi_k3_fqns(_Model(), kwargs)
        last = kwargs["parallelism"].module_fqns_per_model_part[-1]
        self.assertIn("output_res_proj", last)
        self.assertIn("output_res_norm", last)

    def test_every_emitted_name_matches_a_child(self):
        """Core sets every non-matching child to None, so an FQN that matches
        nothing is a stage with pieces missing rather than an error."""
        model = _Model()
        kwargs = _kwargs(8)
        _inject_kimi_k3_fqns(model, kwargs)
        children = {name for name, _ in model.named_children()}
        for stage in kwargs["parallelism"].module_fqns_per_model_part:
            for fqn in stage:
                root = fqn.split(".", 1)[0]
                self.assertIn(root, children, f"{fqn} matches no child")


    def test_the_vision_tower_gets_a_stage(self):
        """Vision features are spliced into the embeddings, so the tower rides
        with whichever stage kept the embedding. Left unnamed it is None on
        every stage, and the first multimodal batch reports "pixel_values were
        provided without a vision encoder"."""
        kwargs = _kwargs(8)
        _inject_kimi_k3_fqns(_MultimodalModel(), kwargs)
        fqns = kwargs["parallelism"].module_fqns_per_model_part
        owner = [s for s in fqns if "vision_encoder" in s]
        self.assertEqual(len(owner), 1, "the tower must land on exactly one stage")
        self.assertIn("tok_embeddings", owner[0])


    def test_a_text_model_is_untouched_by_dep(self):
        """DEP is on by default, and a text model has no tower to give a stage.

        The failure this pins is silent: ``vision_encoder`` is None rather than
        absent, so a ``hasattr`` gate charged the text stack a vision stage and
        emitted an FQN matching no child. pp=2 then trained with the embedding
        alone on stage 0 and every layer on stage 1 -- a legal run, a wrong
        pipeline, and nothing in the loss to show for it.
        """
        text, mm = _kwargs(8), _kwargs(8)
        _inject_kimi_k3_fqns(_TextModel(), text)
        _inject_kimi_k3_fqns(_Model(), mm)
        fqns = text["parallelism"].module_fqns_per_model_part
        self.assertEqual(
            fqns,
            mm["parallelism"].module_fqns_per_model_part,
            "a None tower must split exactly like no tower at all",
        )
        self.assertNotIn("vision_encoder", [f for stage in fqns for f in stage])


if __name__ == "__main__":
    unittest.main()
