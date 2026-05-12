# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class Episode:
    """
    A single prompt + completion pair with reward and advantage.

    The generator creates Episodes (with group_id, no reward yet).
    The grader fills in the reward.
    The controller computes advantages across episodes sharing a group_id.
    The trainer consumes the final Episodes with advantages set.

    Attributes:
        policy_version: Version of policy that produced this episode.
        prompt_token_ids: Token IDs for the prompt. For VLM rollouts
            this should include any image-token splicing the
            generator's multimodal processor performed (e.g.
            ``[image_token_id] * num_vision_tokens`` interspersed
            with text tokens), so the trainer concats
            ``prompt_token_ids + token_ids`` and reproduces the same
            sequence the generator's LM saw.
        text: Decoded completion text.
        token_ids: Completion token IDs.
        token_log_probs: Per-token log probabilities from the generator.
        expected_answer: Expected answer for reward computation.
            Passed to Episode by the generator — the generator
            does not read this field.
        reward: Scalar reward assigned by the grader.
        group_id: Identifies which group this episode belongs to.
            Episodes with the same group_id share a prompt and have
            their advantages normalized together.
        advantage: Advantage value computed by the controller (GRPO:
            reward minus group mean reward).
        image_path: Optional path to the input image for VLM rollouts.
            A vision-aware trainer can use this to re-run the vision
            tower + projector during ``compute_token_log_probs`` and
            inject the same vision_embeds at image-token positions
            that the generator's LM saw. Trainers that ignore this
            field still see the right input_ids (image tokens spliced
            into ``prompt_token_ids``) but their recomputed log-probs
            at vision positions diverge from the generator's because
            the LM embedding for the image_token_id is the literal-
            text embedding, not the projected vision feature.
    """

    policy_version: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    token_log_probs: list[float]
    expected_answer: str = ""
    reward: float = 0.0
    group_id: str = ""
    advantage: float = 0.0
    image_path: str | None = None
