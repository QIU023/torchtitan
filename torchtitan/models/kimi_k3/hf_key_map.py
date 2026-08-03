# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 released-checkpoint key names <-> ours.

``state_dict_adapter.py`` targets the Kimi-Linear-48B naming, which K3's release
does not use. Rather than mutate a validated adapter, the K3 translation lives
here and the adapter delegates to it when it sees the K3 layout.

Every pattern below is transcribed from ``model.safetensors.index.json`` of
``moonshotai/Kimi-K3``, and ``test_hf_key_map.py`` asserts coverage in both
directions against that file, so an unhandled key is a test failure rather than
a silently dropped tensor.

Naming differences worth knowing, in rough order of how easy they are to miss:

* Everything text-side sits under ``language_model.model.`` -- the release is
  the multimodal wrapper, so even a text-only load has to strip that.
* Block Attention Residuals ARE in the release, as ``self_attention_res_proj`` /
  ``self_attention_res_norm`` (93 each) plus ``mlp_res_proj`` / ``mlp_res_norm``,
  with the final aggregation at ``model.output_attn_res_proj``. We call the
  per-layer pair ``attn_res_proj`` / ``attn_res_norm`` and the final one
  ``final_attn_res_proj``.
* The MoE layers' module is ``block_sparse_moe``; the single dense layer's is
  ``mlp``. Both map into our ``ffn``.
* Routed experts use ``w1`` / ``w2`` / ``w3`` while the SHARED experts use
  ``gate_proj`` / ``up_proj`` / ``down_proj`` -- the same block uses both
  conventions, so a single global rename gets one of them wrong. w1 is the gate,
  w3 the up, w2 the down (annotated as such in the reference).
* The Gated-MLA output gate is ``g_proj``, the same name KDA uses for its own
  output gate. We call the MLA one ``attn_gate_proj`` because our module also
  supports a per-head graft parameterization the release does not have.
* The router's load-balance bias is ``gate.e_score_correction_bias``, which is a
  BUFFER on our side (``expert_bias_E``), not a parameter.
* Routed-expert weights are MXFP4, stored as ``.weight_packed`` plus
  ``.weight_scale`` rather than ``.weight``. Nothing else in the checkpoint is
  quantized, which is the scope ``quant_scope.py`` encodes.
"""

from __future__ import annotations

import re

TEXT_PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"

# Per-layer names that differ only by spelling.
_LAYER_RENAME = {
    "self_attention_res_proj": "attn_res_proj",
    "self_attention_res_norm": "attn_res_norm",
    "mlp_res_proj": "mlp_res_proj",
    "mlp_res_norm": "mlp_res_norm",
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
}

# Attention leaves that keep their name. Both attention types are covered; the
# official g_proj is ambiguous between them, so it is resolved by layer type.
_ATTN_SAME = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "q_a_proj", "q_a_layernorm", "q_b_proj",
    "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
    "f_a_proj", "f_b_proj", "b_proj",
    "q_conv1d", "k_conv1d", "v_conv1d", "o_norm",
    "A_log", "dt_bias",
)

# Routed experts: w1 gate, w3 up, w2 down (reference annotates these).
EXPERT_W_TO_SUFFIXED = {"w1": "w1_EFD", "w2": "w2_EDF", "w3": "w3_EFD"}

_MOE_BLOCK_RENAME = {
    "routed_expert_down_proj": "latent.down",
    "routed_expert_up_proj": "latent.up",
    "routed_expert_norm": "latent.norm",
}

VISION_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


class UnmappedKey(ValueError):
    """A checkpoint key with no destination. Never ignored silently."""


def _mla_layer(layer_idx: int, kda_layers: set[int]) -> bool:
    """The release uses 1-BASED layer indices in linear_attn_config, while
    checkpoint keys are 0-based, so the caller's kda_layers must already be
    normalized to whichever base it uses. See is_kda_layer in the reference."""
    return layer_idx not in kda_layers


def official_to_titan(key: str, *, kda_layers: set[int]) -> tuple[str, str]:
    """Translate one released key. Returns ``(our_key, kind)``.

    ``kind`` is one of ``"param"``, ``"buffer"``, ``"expert_packed"``,
    ``"expert_scale"``, ``"vision"``. Raises :class:`UnmappedKey` otherwise --
    a checkpoint tensor we cannot place is a bug, not something to skip.
    """
    if key == LM_HEAD:
        return "lm_head.weight", "param"
    if key.startswith(VISION_PREFIX) or key.startswith(PROJECTOR_PREFIX):
        # Our MoonViT holds the projector as a child, so mm_projector.* becomes
        # a child path and vision_tower.* loses its prefix.
        if key.startswith(PROJECTOR_PREFIX):
            return f"vision_tower.mm_projector.{key[len(PROJECTOR_PREFIX):]}", "vision"
        return f"vision_tower.{key[len(VISION_PREFIX):]}", "vision"
    if not key.startswith(TEXT_PREFIX):
        raise UnmappedKey(key)

    rest = key[len(TEXT_PREFIX) :]
    if rest == "embed_tokens.weight":
        return "embed_tokens.weight", "param"
    if rest == "norm.weight":
        return "norm.weight", "param"
    if rest == "output_attn_res_proj.weight":
        return "final_attn_res_proj.weight", "param"
    if rest == "output_attn_res_norm.weight":
        return "final_attn_res_norm.weight", "param"

    m = _LAYER_RE.match(rest)
    if m is None:
        raise UnmappedKey(key)
    idx, tail = int(m.group(1)), m.group(2)
    head = tail.split(".", 1)[0]

    if head in _LAYER_RENAME:
        return f"layers.{idx}.{_LAYER_RENAME[head]}.weight", "param"

    if head == "self_attn":
        leaf = tail.split(".", 1)[1]
        name = leaf.rsplit(".", 1)[0] if leaf.endswith(".weight") else leaf
        if name == "g_proj":
            # KDA keeps g_proj; MLA's gate is attn_gate_proj on our side.
            ours = "g_proj" if not _mla_layer(idx, kda_layers) else "attn_gate_proj"
            return f"layers.{idx}.self_attn.{ours}.weight", "param"
        if name in _ATTN_SAME:
            suffix = ".weight" if leaf.endswith(".weight") else ""
            return f"layers.{idx}.self_attn.{name}{suffix}", "param"
        raise UnmappedKey(key)

    if head == "mlp":
        # the single dense layer (first_k_dense_replace)
        leaf = tail.split(".", 1)[1]
        return f"layers.{idx}.ffn.{leaf}", "param"

    if head == "block_sparse_moe":
        leaf = tail.split(".", 1)[1]
        first = leaf.split(".", 1)[0]
        if first in _MOE_BLOCK_RENAME:
            return f"layers.{idx}.ffn.{_MOE_BLOCK_RENAME[first]}.weight", "param"
        if first == "shared_experts":
            return f"layers.{idx}.ffn.{leaf}", "param"
        if leaf == "gate.weight":
            return f"layers.{idx}.ffn._moe.router.gate.weight", "param"
        if leaf == "gate.e_score_correction_bias":
            return f"layers.{idx}.ffn._moe.expert_bias_E", "buffer"
        em = re.match(r"^experts\.(\d+)\.(w[123])\.(.+)$", leaf)
        if em:
            expert, w, suffix = int(em.group(1)), em.group(2), em.group(3)
            base = (
                f"layers.{idx}.ffn._moe.routed_experts.inner_experts."
                f"{EXPERT_W_TO_SUFFIXED[w]}"
            )
            if suffix == "weight_packed":
                return f"{base}[{expert}]", "expert_packed"
            if suffix == "weight_scale":
                return f"{base}[{expert}]", "expert_scale"
            if suffix == "weight":
                return f"{base}[{expert}]", "param"
            raise UnmappedKey(key)
        raise UnmappedKey(key)

    raise UnmappedKey(key)


def titan_to_official(
    key: str, *, kda_layers: set[int], expert_idx: int | None = None
) -> str:
    """Inverse of :func:`official_to_titan` for a single tensor.

    Expert weights need ``expert_idx`` because one stacked ``w1_EFD`` on our
    side corresponds to ``num_experts`` separate official keys.
    """
    inv_layer = {v: k for k, v in _LAYER_RENAME.items()}
    inv_moe = {v: k for k, v in _MOE_BLOCK_RENAME.items()}
    inv_expert = {v: k for k, v in EXPERT_W_TO_SUFFIXED.items()}

    if key == "lm_head.weight":
        return LM_HEAD
    if key.startswith("vision_tower.mm_projector."):
        return PROJECTOR_PREFIX + key[len("vision_tower.mm_projector.") :]
    if key.startswith("vision_tower."):
        return VISION_PREFIX + key[len("vision_tower.") :]
    if key in ("embed_tokens.weight", "norm.weight"):
        return TEXT_PREFIX + key
    if key == "final_attn_res_proj.weight":
        return TEXT_PREFIX + "output_attn_res_proj.weight"
    if key == "final_attn_res_norm.weight":
        return TEXT_PREFIX + "output_attn_res_norm.weight"

    m = _LAYER_RE.match(key)
    if m is None:
        raise UnmappedKey(key)
    idx, tail = int(m.group(1)), m.group(2)
    prefix = f"{TEXT_PREFIX}layers.{idx}."

    stem = tail.rsplit(".weight", 1)[0]
    if stem in inv_layer:
        return f"{prefix}{inv_layer[stem]}.weight"

    if tail.startswith("self_attn."):
        leaf = tail[len("self_attn.") :]
        name = leaf.rsplit(".", 1)[0] if leaf.endswith(".weight") else leaf
        official = "g_proj" if name in ("g_proj", "attn_gate_proj") else name
        suffix = ".weight" if leaf.endswith(".weight") else ""
        return f"{prefix}self_attn.{official}{suffix}"

    if tail.startswith("ffn."):
        leaf = tail[len("ffn.") :]
        base = leaf.rsplit(".weight", 1)[0]
        if base in inv_moe:
            return f"{prefix}block_sparse_moe.{inv_moe[base]}.weight"
        if leaf.startswith("shared_experts."):
            return f"{prefix}block_sparse_moe.{leaf}"
        if leaf == "_moe.router.gate.weight":
            return f"{prefix}block_sparse_moe.gate.weight"
        if leaf == "_moe.expert_bias_E":
            return f"{prefix}block_sparse_moe.gate.e_score_correction_bias"
        em = re.match(r"^_moe\.routed_experts\.inner_experts\.(w\d_\w+)$", leaf)
        if em:
            if expert_idx is None:
                raise UnmappedKey(
                    f"{key} is a stacked expert tensor; expert_idx is required"
                )
            w = inv_expert[em.group(1)]
            return f"{prefix}block_sparse_moe.experts.{expert_idx}.{w}.weight"
        # dense FFN
        return f"{prefix}mlp.{leaf}"

    raise UnmappedKey(key)
