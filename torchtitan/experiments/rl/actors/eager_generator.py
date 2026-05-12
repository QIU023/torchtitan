# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager torchtitan generator — drop-in replacement for SGLangGenerator
when the SGLang inference path produces NaN logits (e.g. flashinfer_mla
on consumer Blackwell SM 12.0 with under-converged AttnRes models).

Loads the multimodal model in pure-torch eager mode via torchtitan's
``KimiLinearAttnResModel.forward`` — the same forward path used during
SFT training. Numerically stable but slow (no KV cache, no CUDA graph
capture) — suitable for research-grade VLM GRPO/PPO rollouts where
correctness matters more than throughput.

API surface mirrors :class:`SGLangGenerator` so it can be substituted
without changes to the GRPO controller in
``run_grpo_llava_caption.py`` / ``run_grpo_kimi_attn_res.py``:

  * ``generate(prompt_texts, expected_answers, images)`` → list[Episode]
  * ``pull_model_state_dict(version)`` → no-op (we control the model
    instance directly; weight sync handled outside this generator)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from monarch.actor import Actor, endpoint

from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import DebugConfig, ParallelismConfig
from torchtitan.experiments.rl.types import Episode
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class EagerSamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    max_new_tokens: int = 80


@dataclass(kw_only=True, slots=True)
class EagerBackendConfig:
    dtype: str = "bfloat16"
    image_token_id: int = 32000
    num_vision_tokens: int = 196
    vision_tower_path: str = "google/siglip-base-patch16-224"


class EagerTorchtitanGenerator(Actor, Configurable):
    """Numerically-stable eager forward generator — bypasses SGLang.

    Each rollout runs ``model(input_ids, vision_embeds=, image_mask=)``
    in pure torchtitan eager mode for both prefill and decode (no KV
    cache reuse — every decode step re-runs the full prefix forward).
    Slow but produces non-NaN logits where SGLang flashinfer_mla
    fails on this specific AttnRes model.

    Args:
        config: EagerTorchtitanGenerator.Config
        model_spec: torchtitan ModelSpec (e.g. kimi_linear flavor)
        dcp_path: torch DCP checkpoint dir to load weights from
        tokenizer_path: HF tokenizer dir / id (Llama-3.1)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        sampling: EagerSamplingConfig = field(default_factory=EagerSamplingConfig)
        backend: EagerBackendConfig = field(default_factory=EagerBackendConfig)
        num_samples_per_prompt: int = 4
        debug: DebugConfig = field(default_factory=DebugConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        dcp_path: str,
        tokenizer_path: str = "NousResearch/Meta-Llama-3.1-8B",
    ):
        self.config = config
        self.policy_version = 0
        # Single-GPU eager: this actor instance owns one GPU (rank 0),
        # other ranks (if Monarch spawned them) idle.
        self._is_lead = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if not self._is_lead:
            return

        device = torch.device("cuda")
        torch.cuda.set_device(0)
        dtype = TORCH_DTYPE_MAP[config.backend.dtype]
        torch.set_default_dtype(dtype)

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Build the LM (no vision tower yet — we load it lazily if
        # an image-bearing prompt arrives).
        spec = model_spec.model
        with torch.device("meta"):
            with torch.amp.autocast("cuda", enabled=False):
                model = type(spec).build(spec)  # KimiLinearAttnResModel
        model.to_empty(device=device)
        model.init_weights(buffer_device=None)

        # Load DCP weights.
        sd = model.state_dict()
        logger.info(f"loading DCP from {dcp_path} ({len(sd)} keys)")
        dcp.load(sd, checkpoint_id=str(dcp_path))
        model.load_state_dict(sd, strict=False, assign=False)
        model.eval()
        self._model = model
        self._device = device

        # Lazy vision tower + projector (LLaVA-1.5 stage 1 layout).
        self._vision_tower = None
        self._projector = None
        self._dcp_path_for_projector = dcp_path
        logger.info("EagerTorchtitanGenerator ready (lead actor)")

    def _ensure_vision(self) -> None:
        """Lazy-load the SigLIP vision tower + projector. Idempotent."""
        if self._vision_tower is not None:
            return
        from transformers import SiglipVisionModel
        vt_path = self.config.backend.vision_tower_path
        logger.info(f"loading frozen vision tower {vt_path}")
        vt = SiglipVisionModel.from_pretrained(vt_path)
        for p in vt.parameters():
            p.requires_grad = False
        vt.eval().to(self._device)
        self._vision_tower = vt

        # Projector lives in DCP at mm_projector.projector.{fc1,fc2}.{weight,bias}.
        import torch.nn as nn
        vision_hidden = vt.config.hidden_size
        try:
            llm_hidden = self._model.embed_tokens.weight.shape[1]
        except AttributeError:
            llm_hidden = self._model.tok_embeddings.weight.shape[1]
        proj_dtype = next(self._model.parameters()).dtype

        class _Projector(nn.Module):
            def __init__(self, vd, ld):
                super().__init__()
                self.fc1 = nn.Linear(vd, ld, bias=True)
                self.fc2 = nn.Linear(ld, ld, bias=True)
            def forward(self, x):
                import torch.nn.functional as F
                return self.fc2(F.gelu(self.fc1(x)))

        with torch.device(self._device):
            with torch.amp.autocast("cuda", enabled=False):
                torch.set_default_dtype(proj_dtype)
                proj = _Projector(vision_hidden, llm_hidden)

        # Pull projector keys from the same DCP dir.
        proj_sd_dcp = {
            "mm_projector.projector.fc1.weight": proj.fc1.weight,
            "mm_projector.projector.fc1.bias": proj.fc1.bias,
            "mm_projector.projector.fc2.weight": proj.fc2.weight,
            "mm_projector.projector.fc2.bias": proj.fc2.bias,
        }
        try:
            dcp.load(proj_sd_dcp, checkpoint_id=str(self._dcp_path_for_projector))
            proj.fc1.weight.copy_(proj_sd_dcp["mm_projector.projector.fc1.weight"])
            proj.fc1.bias.copy_(proj_sd_dcp["mm_projector.projector.fc1.bias"])
            proj.fc2.weight.copy_(proj_sd_dcp["mm_projector.projector.fc2.weight"])
            proj.fc2.bias.copy_(proj_sd_dcp["mm_projector.projector.fc2.bias"])
            for p in proj.parameters():
                p.requires_grad = False
            proj.eval().to(proj_dtype).to(self._device)
            self._projector = proj
            logger.info(
                f"projector loaded ({vision_hidden}->{llm_hidden})"
            )
        except Exception as e:
            logger.warning(
                f"projector DCP load failed ({e}); generator will use random projector "
                "(VLM rollouts will be ungrounded)"
            )
            self._projector = proj.eval().to(proj_dtype).to(self._device)

    def _encode_image(self, image_path: str | None) -> tuple[torch.Tensor | None, list[int]]:
        """Run vision tower + projector. Returns
        (vision_embeds [1, 1, N_vis, D_llm] or None, list of image_token_ids).
        """
        if image_path is None:
            return None, []
        self._ensure_vision()
        from PIL import Image
        from transformers import SiglipImageProcessor
        if not hasattr(self, "_ip"):
            self._ip = SiglipImageProcessor.from_pretrained(
                self.config.backend.vision_tower_path
            )
        pil = Image.open(image_path).convert("RGB")
        px = self._ip(images=pil, return_tensors="pt").pixel_values.to(self._device)
        px = px.to(next(self._vision_tower.parameters()).dtype)
        with torch.no_grad():
            vt_out = self._vision_tower(pixel_values=px)
        vfeat = vt_out.last_hidden_state.to(next(self._projector.parameters()).dtype)
        # [1, N_vis, D_vis]
        N_vis = vfeat.shape[1]
        flat = vfeat.reshape(-1, vfeat.shape[-1])
        projected = self._projector(flat)  # [N_vis, D_llm]
        projected = projected.unsqueeze(0).unsqueeze(0)  # [1, 1, N_vis, D_llm]
        # Image-token IDs to splice into input_ids
        img_ids = [self.config.backend.image_token_id] * N_vis
        return projected, img_ids

    def _generate_one(
        self, prompt_text: str, image_path: str | None,
    ) -> tuple[str, list[int], list[int], list[float]]:
        """Greedy/sampling decode for one prompt. Returns
        (decoded_text, prompt_token_ids, gen_token_ids, gen_logprobs).
        """
        # Tokenize text, splice image tokens at <image> placeholder
        sampling = self.config.sampling
        img_token_id = self.config.backend.image_token_id

        # Run vision once if image supplied
        vision_embeds, img_ids = self._encode_image(image_path)

        # Splice
        if "<image>" in prompt_text and img_ids:
            chunks = prompt_text.split("<image>")
            prompt_ids: list[int] = []
            for i, chunk in enumerate(chunks):
                ids = self._tokenizer.encode(chunk, add_special_tokens=(i == 0))
                prompt_ids.extend(ids)
                if i < len(chunks) - 1:
                    prompt_ids.extend(img_ids)
        else:
            prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=True)
            if image_path is not None and img_ids:
                # Prepend image tokens if no placeholder
                prompt_ids = img_ids + prompt_ids

        gen_ids: list[int] = []
        gen_logprobs: list[float] = []
        full_ids = list(prompt_ids)
        eos_id = self._tokenizer.eos_token_id
        max_new = sampling.max_new_tokens
        temp = sampling.temperature
        top_p = sampling.top_p

        with torch.no_grad():
            for _ in range(max_new):
                tok_t = torch.tensor([full_ids], dtype=torch.long, device=self._device)
                kwargs: dict[str, Any] = {}
                if vision_embeds is not None:
                    image_mask = (tok_t == img_token_id)
                    if image_mask.any():
                        kwargs["vision_embeds"] = vision_embeds
                        kwargs["image_mask"] = image_mask
                logits = self._model(tok_t, **kwargs)  # [1, T, V]
                last = logits[0, -1, :].float()
                if temp <= 0:
                    next_id = int(last.argmax().item())
                    next_lp = float(torch.log_softmax(last, dim=-1)[next_id].item())
                else:
                    probs = torch.softmax(last / max(temp, 1e-6), dim=-1)
                    if 0 < top_p < 1:
                        sorted_probs, sorted_idx = probs.sort(descending=True)
                        cumulative = sorted_probs.cumsum(0)
                        cutoff = int((cumulative <= top_p).sum().item()) + 1
                        keep_idx = sorted_idx[:cutoff]
                        keep_probs = sorted_probs[:cutoff]
                        keep_probs = keep_probs / keep_probs.sum()
                        choice = torch.multinomial(keep_probs, 1).item()
                        next_id = int(keep_idx[choice].item())
                    else:
                        next_id = int(torch.multinomial(probs, 1).item())
                    next_lp = float(torch.log(probs[next_id] + 1e-12).item())
                gen_ids.append(next_id)
                gen_logprobs.append(next_lp)
                full_ids.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break

        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, prompt_ids, gen_ids, gen_logprobs

    @endpoint
    async def generate(
        self,
        prompt_texts: list[str],
        expected_answers: list[str],
        images: Optional[list[Any]] = None,
    ) -> list[Episode]:
        if not self._is_lead:
            return []
        n_per = max(1, self.config.num_samples_per_prompt)
        episodes: list[Episode] = []
        for prompt_idx, p in enumerate(prompt_texts):
            img = None
            if images is not None and prompt_idx < len(images):
                cand = images[prompt_idx]
                img = cand if isinstance(cand, str) else None
            for _ in range(n_per):
                t0 = time.perf_counter()
                text, prompt_ids, gen_ids, gen_logprobs = self._generate_one(p, img)
                dt = time.perf_counter() - t0
                gid = f"{os.getpid()}_{self.policy_version}_{prompt_idx}"
                episodes.append(Episode(
                    policy_version=self.policy_version,
                    prompt_token_ids=prompt_ids,
                    text=text,
                    token_ids=gen_ids,
                    token_log_probs=gen_logprobs,
                    expected_answer=(
                        expected_answers[prompt_idx]
                        if expected_answers and prompt_idx < len(expected_answers)
                        else ""
                    ),
                    group_id=gid,
                    image_path=img,
                ))
                logger.debug(
                    f"prompt {prompt_idx} sample done in {dt:.1f}s "
                    f"(gen_len={len(gen_ids)})"
                )
        return episodes

    @endpoint
    async def pull_model_state_dict(self, version: int) -> None:
        """No-op for the eager generator. Weight sync between trainer
        and generator should happen out-of-band (we share the same
        DCP path at startup; live updates require an explicit reload
        which can be added later)."""
        self.policy_version = version

    @endpoint
    async def push_model_state_dict(self) -> None:
        return None
