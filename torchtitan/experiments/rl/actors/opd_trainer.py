# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""On-Policy Distillation trainer (GKD, Agarwal et al. 2024).

``OPDTrainer`` is a sibling of ``PolicyTrainer``: same FSDP / optimizer /
DCP load / monarch actor surface, but the per-step loss is a generalized-
JSD distillation against a frozen teacher's logits at the student's own
rollout positions, NOT a scalar-advantage policy gradient.

Design notes:
  * Subclasses ``PolicyTrainer`` so the FSDP / parallelize_fn / DCP /
    vision-tower wiring is reused verbatim. Only ``step`` is overridden
    and ``__init__`` is extended with two optional knobs (teacher
    scorer + tokenizer for prompt/response decode).
  * Teacher is a duck-typed scorer object with a single method
    ``score(image_path, prompt_text, response_text) -> (logits, ids)``.
    Reference impl lives outside torchtitan (HF
    ``LlavaNextForConditionalGeneration``); injecting it here keeps
    ``actors/`` ``transformers``-free.
  * Vocab slice and label-masking live in the launcher-side
    ``opd_loss`` adapter (which calls ``trl.experimental.gkd.
    GKDTrainer.generalized_jsd_loss`` for the actual loss math).
    Trainer is loss-function-agnostic; the only contract is
    ``opd_loss(student_logits, teacher_logits, labels) -> scalar``.
"""

import logging
import os
from typing import Any, Callable

import torch
from monarch.actor import endpoint

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.actors.utils import compute_response_logits
from torchtitan.experiments.rl.types import Episode

logger = logging.getLogger(__name__)


# Type alias for the teacher scoring callable. Decoupled from the
# launcher's specific TeacherScorer class so this file stays HF-free.
#   args:    image_path (str|None), prompt_text (str), response_text (str)
#   returns: (teacher_logits[T_resp, V_teacher], teacher_ids[T_resp])
TeacherScoreFn = Callable[
    [str | None, str, str], tuple[torch.Tensor, torch.Tensor]
]

# Type alias for the loss function. Identical signature to the launcher-
# side ``opd_loss(student_logits, teacher_logits, labels)``. Decoupled
# from the trl dep so this file stays trl-free.
#   args:    student_logits[B, T, V_s], teacher_logits[B, T, V_t],
#            labels[B, T] (use -100 to ignore positions)
#   returns: scalar loss
OPDLossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


class OPDTrainer(PolicyTrainer):
    """On-policy distillation (GKD) sibling of ``PolicyTrainer``.

    Args:
        *args, **kwargs: forwarded to ``PolicyTrainer.__init__``.
            ``kl_coef`` should typically be 0 — distillation provides
            its own KL/JSD against the teacher and a separate ref-model
            KL is rarely useful.

    Late-injected (via ``set_teacher_scorer`` / ``set_opd_loss_fn``):
        teacher_score_fn: callable returning per-response-position
            teacher logits + teacher-tokenized ids.
        opd_loss_fn: callable computing the scalar loss from student
            logits, teacher logits, and labels.
        tokenizer: a HF-style tokenizer with ``.decode(ids,
            skip_special_tokens=True) -> str``. Used to convert
            ``Episode.prompt_token_ids`` and ``Episode.token_ids`` back
            into text for the teacher prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_score_fn: TeacherScoreFn | None = None
        self._opd_loss_fn: OPDLossFn | None = None
        self._tokenizer: Any = None
        # OPDTrainer ignores reward / advantage; warn if a ref model
        # was built — it's wasted memory under distillation.
        if self.ref_model is not None:
            logger.warning(
                "OPDTrainer built with kl_coef>0 reference model — "
                "distillation provides its own teacher KL, so the ref "
                "model is unused. Consider kl_coef=0 to save memory."
            )

    @endpoint
    async def set_teacher_scorer(self, scorer: Any) -> None:
        """Inject the teacher-scoring callable after actor construction.

        ``scorer`` may be either a callable matching ``TeacherScoreFn``
        or an object with a ``.score(image_path, prompt_text,
        response_text)`` method (e.g. the launcher's ``TeacherScorer``).
        """
        if callable(scorer) and not hasattr(scorer, "score"):
            self._teacher_score_fn = scorer
        else:
            self._teacher_score_fn = scorer.score

    @endpoint
    async def set_opd_loss_fn(self, fn: OPDLossFn) -> None:
        """Inject the OPD loss callable after actor construction."""
        self._opd_loss_fn = fn

    @endpoint
    async def set_tokenizer(self, tokenizer: Any) -> None:
        """Inject the tokenizer used to decode Episode token ids to text."""
        self._tokenizer = tokenizer

    @endpoint
    async def step(self, episodes: list[Episode]) -> dict:
        """One OPD training step over the given Episodes.

        Skips reward / advantage / ref-model entirely. Per episode:
          (1) decode prompt + response text from Episode token ids,
          (2) call ``teacher_score_fn`` to get teacher logits + ids at
              the student's response positions,
          (3) call ``compute_response_logits`` for student logits at
              the same response positions,
          (4) accumulate ``opd_loss_fn(student, teacher, labels)``.
        ``loss.backward()`` + optimizer step at end.

        Returns:
            Training metrics dict (``loss``, ``grad_norm``, ``lr``,
            ``policy_version``, ``num_response_tokens``,
            ``sample_completion``).
        """
        if self._teacher_score_fn is None or self._opd_loss_fn is None:
            raise RuntimeError(
                "OPDTrainer.step called before set_teacher_scorer + "
                "set_opd_loss_fn. The actor's controller must inject "
                "both before driving any step."
            )
        if self._tokenizer is None:
            raise RuntimeError(
                "OPDTrainer.step called before set_tokenizer. "
                "Need a tokenizer to decode Episode token ids back "
                "to text for the teacher prompt."
            )

        # Match PolicyTrainer's dynamo cache_size_limit pattern (kept
        # consistent for shared-codeobject recompile behaviour).
        torch._dynamo.config.recompile_limit = 16

        logger.debug(
            f"{os.getpid()=} OPDTrainer starting step {self.policy_version}"
        )

        # Shard episodes across DP ranks (same pattern as
        # PolicyTrainer.step — unique slice per rank).
        total_samples = len(episodes)
        my_indices = list(range(self.dp_rank, total_samples, self.dp_size))
        my_episodes = [episodes[i] for i in my_indices]

        if not my_episodes:
            logger.warning(
                f"OPDTrainer dp_rank={self.dp_rank} got empty episode "
                f"shard (total={total_samples}, dp_size={self.dp_size}); "
                "skipping step."
            )
            return {"loss": 0.0, "num_response_tokens": 0,
                    "policy_version": self.policy_version}

        self.optimizers.zero_grad()
        loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)
        n_resp_tokens = 0

        for ep in my_episodes:
            prompt_text = self._tokenizer.decode(
                ep.prompt_token_ids, skip_special_tokens=True,
            )
            response_text = self._tokenizer.decode(
                ep.token_ids, skip_special_tokens=True,
            )
            if not response_text:
                # Empty generation — no response positions to score.
                continue

            # (1) Teacher logits at response positions.
            #     Returns (teacher_logits[T_resp, V_t], teacher_ids[T_resp])
            #     where teacher_ids may have a different T from len(ep.token_ids)
            #     because the teacher's tokenizer re-tokenizes the response text.
            with torch.no_grad():
                teacher_logits, teacher_ids = self._teacher_score_fn(
                    ep.image_path, prompt_text, response_text,
                )
            teacher_logits = teacher_logits.to(self.device, torch.float32)
            teacher_ids = teacher_ids.to(self.device)

            # (2) Student logits at the SAME response positions.
            #     We use the student's own ep.token_ids (the bytes it
            #     actually sampled), so student response length is
            #     len(ep.token_ids).
            student_logits = compute_response_logits(
                self.model,
                ep.prompt_token_ids,
                ep.token_ids,
                self.device,
                vision_tower=self._vision_tower,
                projector=self._projector,
                image_path=ep.image_path,
                image_token_id=self._image_token_id,
            )  # [T_resp_student, V_student]

            # (3) Length reconciliation. Student and teacher tokenize
            #     the same response text but may produce slightly different
            #     id lengths if the response uses tokens at a vocab merge
            #     boundary or the student emits a Kimi-padding id outside
            #     Llama-3 base. Truncate to min length and mask the rest.
            T = min(student_logits.shape[0], teacher_logits.shape[0])
            if T == 0:
                continue
            student_logits = student_logits[:T]
            teacher_logits = teacher_logits[:T]
            labels = teacher_ids[:T].clone()

            # (4) Generalized JSD via injected loss fn. opd_loss handles
            #     vocab alignment (slice to shared 128256) and label mask.
            #     Wrap in batch dim [1, T, V] / [1, T].
            loss = self._opd_loss_fn(
                student_logits.unsqueeze(0),
                teacher_logits.unsqueeze(0),
                labels.unsqueeze(0),
            )
            loss_accum = loss_accum + loss
            n_resp_tokens += int(T)

        # Average loss across episodes on this rank so the magnitude
        # is independent of batch size (mirrors GRPO PG averaging).
        if len(my_episodes) > 0:
            loss_accum = loss_accum / float(len(my_episodes))

        loss_accum.backward()

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )
        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        metrics = {
            "loss": loss_accum.item(),
            "num_response_tokens": n_resp_tokens,
            "policy_version": self.policy_version,
            "sample_completion": episodes[0].text[:80],
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            "lr": self.lr_schedulers.schedulers[0].get_last_lr()[0]
                  if hasattr(self.lr_schedulers, "schedulers")
                  and self.lr_schedulers.schedulers
                  else float("nan"),
        }
        logger.debug(
            f"{os.getpid()=} OPDTrainer finished step {self.policy_version} "
            f"loss={metrics['loss']:.4f} resp_tokens={n_resp_tokens}"
        )
        return metrics
