# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenMetadata

# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def build_varlen_metadata(
    input_sequences: list[tuple[torch.Tensor, int, int]], device: torch.device
) -> VarlenMetadata:
    """Build VarlenMetadata for all sequences in a batch."""
    cu_seqs = torch.cumsum(
        torch.tensor(
            [0] + [token_ids.shape[0] for token_ids, _, _ in input_sequences],
            dtype=torch.int32,
            device=device,
        ),
        0,
        dtype=torch.int32,
    )
    max_len = max(token_ids.shape[0] for token_ids, _, _ in input_sequences)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


def compute_token_log_probs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
    *,
    vision_tower: torch.nn.Module | None = None,
    projector: torch.nn.Module | None = None,
    image_path: str | None = None,
    image_token_id: int = 32000,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for generated tokens.
    TODO Only batch size 1 is supported for now.

    Args:
        model: The model to use for computing logits
        prompt_ids: Prompt token IDs (for VLM rollouts these already
            include ``[image_token_id] * num_vision_tokens`` spliced
            at each ``<image>`` placeholder by the generator's
            multimodal processor).
        gen_ids: Generated token IDs
        device: Device to run computation on
        vision_tower: Optional frozen vision encoder (HF SigLIP-style).
            When provided alongside ``projector`` and ``image_path``,
            vision features are injected at image-token positions to
            match what the generator's LM saw. Without these, the LM
            sees the literal-text embedding for ``image_token_id``,
            which diverges from the generator's logits.
        projector: Optional 2-layer MLP that maps vision features to
            ``llm_hidden_size``. Must be paired with ``vision_tower``.
        image_path: Optional path to the image associated with this
            prompt (e.g. from ``Episode.image_path``).
        image_token_id: Sentinel id marking image-feature positions in
            ``token_ids`` (32000 = Llama-3.1 reserved special token,
            phase5/multimodal_dataset.py convention).

    Returns:
        Per-token log probabilities for the generated tokens
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    attention_masks = build_varlen_metadata([(token_ids, prompt_len, gen_len)], device)

    full_tensor = token_ids.unsqueeze(0)

    # NOTE: We should move towards batching to improve efficiency here
    # See https://github.com/pytorch/torchtitan/issues/2674
    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which breaks torch.compile with symbolic shapes.
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # Vision injection: if a vision tower + projector + image are all
    # supplied, run them and call the LM with vision_embeds= and
    # image_mask= kwargs. Models that support these kwargs (e.g.
    # KimiLinearAttnResModel via phase5/multimodal_model.py) will
    # replace embed_tokens(image_token_id) with the projected vision
    # features at image positions. Models that don't recognise the
    # kwargs ignore them and use literal-text embedding (logprobs at
    # image positions then diverge from the generator's).
    if (
        vision_tower is not None
        and projector is not None
        and image_path is not None
    ):
        vision_embeds, image_mask = _encode_image_for_logprob(
            vision_tower=vision_tower,
            projector=projector,
            image_path=image_path,
            input_ids=full_tensor,
            image_token_id=image_token_id,
            device=device,
        )
        logits = model(
            full_tensor,
            attention_masks=attention_masks,
            positions=positions,
            vision_embeds=vision_embeds,
            image_mask=image_mask,
        )
    else:
        logits = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # Convert to float32 for numerical stability
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    target_tokens = full_tensor[:, 1:]

    # Extract log probs for generated tokens only
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len

    gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
    gen_token_ids_tensor = target_tokens[0, gen_start_idx:gen_end_idx]
    token_lps = gen_token_logprobs.gather(
        1, gen_token_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_lps


def compute_response_logits(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
    *,
    vision_tower: torch.nn.Module | None = None,
    projector: torch.nn.Module | None = None,
    image_path: str | None = None,
    image_token_id: int = 32000,
) -> torch.Tensor:
    """Forward pass returning full-vocab logits at response positions.

    Sibling to ``compute_token_log_probs``: shares the prompt+gen tensor
    build, varlen metadata, explicit positions, and vision-tower /
    projector / image-mask injection path. The only difference is the
    return — ``[T_resp, V]`` float32 logits at response positions instead
    of ``[T_resp]`` gathered-by-token-id log probabilities.

    Intended consumer is on-policy distillation
    (``OPDTrainer``, GKD / Agarwal et al. 2024) which feeds these
    logits into a generalized-JSD loss against a frozen teacher's
    logits at the same response positions. Caller is responsible for
    enabling / disabling gradients on ``model`` parameters; this
    function itself does NOT wrap the forward in ``torch.no_grad`` so
    the student can be trained.

    TODO Only batch size 1 is supported for now, matching
    ``compute_token_log_probs``.

    Args:
        model: The student model to score at its own rollout positions.
        prompt_ids: Prompt token IDs (already including image-token
            placeholders for VLM rollouts; see ``compute_token_log_probs``
            docstring).
        gen_ids: Student-generated token IDs.
        device: Device to run computation on.
        vision_tower: Optional frozen vision encoder; see
            ``compute_token_log_probs``.
        projector: Optional 2-layer MLP paired with ``vision_tower``.
        image_path: Optional path to the image associated with this
            prompt.
        image_token_id: Sentinel id marking image-feature positions.

    Returns:
        Per-response-token full-vocab logits, shape ``[T_resp, V]``
        in float32 (matching the precision-stability convention used
        by ``compute_token_log_probs`` before its log_softmax).
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    attention_masks = build_varlen_metadata([(token_ids, prompt_len, gen_len)], device)

    full_tensor = token_ids.unsqueeze(0)

    # Explicit positions avoid the dynamic rope_cache[0:seqlen] slice that
    # would break torch.compile with symbolic shapes (mirrors the
    # rationale in compute_token_log_probs).
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    if (
        vision_tower is not None
        and projector is not None
        and image_path is not None
    ):
        vision_embeds, image_mask = _encode_image_for_logprob(
            vision_tower=vision_tower,
            projector=projector,
            image_path=image_path,
            input_ids=full_tensor,
            image_token_id=image_token_id,
            device=device,
        )
        logits = model(
            full_tensor,
            attention_masks=attention_masks,
            positions=positions,
            vision_embeds=vision_embeds,
            image_mask=image_mask,
        )
    else:
        logits = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # float32 for numerical stability (parity with compute_token_log_probs).
    # The :-1 slice mirrors the next-token-prediction shift: position i's
    # logits predict token i+1.
    logits_f32 = logits[:, :-1, :].to(torch.float32)

    # Same gen-window indexing as compute_token_log_probs:
    #   gen_start_idx = prompt_len - 1  (first position whose target is gen_ids[0])
    #   gen_end_idx   = gen_start_idx + gen_len
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len

    return logits_f32[0, gen_start_idx:gen_end_idx, :]


def _encode_image_for_logprob(
    vision_tower: torch.nn.Module,
    projector: torch.nn.Module,
    image_path: str,
    input_ids: torch.Tensor,
    image_token_id: int,
    device: torch.device,
):
    """Run the vision tower + projector for a single image, return
    ``(vision_embeds, image_mask)`` shaped to match
    ``KimiLinearAttnResModel.forward``'s expectations.

    Output shapes:
      * vision_embeds: ``[B=1, num_images=1, N_vision, llm_hidden_size]``
      * image_mask:    ``[B=1, T]`` boolean

    The caller's ``prompt_token_ids`` already include
    ``[image_token_id] * num_vision_tokens`` at the image position,
    so the LM forward replaces those positions' text-embeddings with
    the projected vision features.
    """
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")

    # Run the SigLIP image processor to get the standard normalised
    # 224x224 tensor. We import lazily — most callers will already
    # have transformers available, but this only matters when
    # vision_tower is supplied.
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(
            getattr(vision_tower.config, "_name_or_path",
                    "google/siglip-base-patch16-224")
        )
    except Exception:
        from transformers import SiglipImageProcessor
        proc = SiglipImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
    px = proc(images=pil, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        vt_out = vision_tower(pixel_values=px)
        vision_features = vt_out.last_hidden_state  # [1, N_vis, D_vis]

    # Cast to projector's dtype (vision tower is fp32 by default,
    # projector + LM are bf16 in our setup).
    proj_dtype = next(projector.parameters()).dtype
    vision_features = vision_features.to(proj_dtype)
    projected = projector(vision_features)  # [1, N_vis, D_llm]

    # Multimodal model expects [B, num_images, N_vis, D_llm].
    vision_embeds = projected.unsqueeze(1)  # [1, 1, N_vis, D_llm]
    image_mask = (input_ids == image_token_id)  # [1, T]
    return vision_embeds, image_mask


def compute_policy_gradient_loss(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    ref_token_log_probs: list[torch.Tensor] | None = None,
    kl_coef: float = 0.0,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    *,
    vision_tower: torch.nn.Module | None = None,
    projector: torch.nn.Module | None = None,
    image_paths: list[str | None] | None = None,
    image_token_id: int = 32000,
) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
    """
    Compute GRPO policy gradient loss with entropy bonus.

    Uses per-token log probs averaged across tokens per sample. Advantages
    are expected to be group-relative (reward - group mean), computed upstream.

    When a reference model is provided (ref_token_log_probs is not None),
    adds a KL divergence penalty and uses PPO-clipped ratios.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        ref_token_log_probs: Per-token log probs from reference model (frozen).
            If None, KL divergence is not included in the loss.
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Total loss (PG + entropy + optional KL)
        metrics: Training metrics dict
        batch_token_log_probs: List of per-token log probs for each sample (for verification)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute per-token log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []

    for i, (prompt_toks, gen_toks) in enumerate(
        zip(prompt_token_ids, vllm_token_ids)
    ):
        ip = image_paths[i] if image_paths is not None and i < len(image_paths) else None
        token_lps = compute_token_log_probs(
            model,
            prompt_toks,
            gen_toks,
            device,
            vision_tower=vision_tower,
            projector=projector,
            image_path=ip,
            image_token_id=image_token_id,
        )
        batch_token_log_probs.append(token_lps)

    if ref_token_log_probs is not None:
        # Per-token log ratios and KL, averaged across tokens per sample
        per_sample_mean_log_ratio = []
        per_sample_mean_kl = []
        all_token_log_probs = []

        for policy_token_lps, ref_token_lps in zip(
            batch_token_log_probs, ref_token_log_probs
        ):
            # Per-token log ratio: log(pi/pi_ref) for each token
            token_log_ratio = policy_token_lps - ref_token_lps.detach()
            # Average across tokens in this sequence
            per_sample_mean_log_ratio.append(token_log_ratio.mean())
            # Per-token KL: E[ratio - 1 - log_ratio] (Schulman approx)
            token_ratio = torch.exp(token_log_ratio)
            token_kl = token_ratio - 1 - token_log_ratio
            per_sample_mean_kl.append(token_kl.mean())
            all_token_log_probs.append(policy_token_lps)

        mean_log_ratio = torch.stack(per_sample_mean_log_ratio)  # [batch]
        mean_kl = torch.stack(per_sample_mean_kl)  # [batch]

        # PPO clipped objective using per-token-averaged ratio
        ratio = torch.exp(mean_log_ratio)
        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
        clipped_loss = clipped_ratio * advantages
        pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        # KL divergence penalty (averaged across samples)
        kl_div = mean_kl.mean()

        # Entropy bonus (averaged across all tokens)
        all_token_lps = torch.cat(all_token_log_probs)
        entropy = -all_token_lps.mean()
        entropy_bonus = -entropy_coef * entropy

        # Total loss
        total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

        metrics = {
            "pg_loss": pg_loss.item(),
            "entropy": entropy.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
            .float()
            .mean()
            .item(),
        }
    else:
        # No reference model: policy gradient loss without KL penalty
        all_token_lps = torch.cat(batch_token_log_probs)
        per_sample_mean_lps = torch.stack([lps.mean() for lps in batch_token_log_probs])
        pg_loss = -(per_sample_mean_lps * advantages).mean()

        entropy = -all_token_lps.mean()
        entropy_bonus = -entropy_coef * entropy

        total_loss = pg_loss + entropy_bonus

        metrics = {
            "pg_loss": pg_loss.item(),
            "entropy": entropy.item(),
        }

    return total_loss, metrics, batch_token_log_probs


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
