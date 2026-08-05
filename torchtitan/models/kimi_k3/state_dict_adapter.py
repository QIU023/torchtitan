# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> torchtitan state-dict adapter for the Kimi Linear (+AttnRes) LM.

Promotion of the standalone HF<->DCP converters (424/424 keys
validated at meta-49.12B) into the titan folder, wired as
``ModelSpec.state_dict_adapter`` so both offline conversion and the
Trainer's ``initial_load_in_hf`` path (and veRL's torchtitan engine,
which sets ``initial_load_in_hf=True``) work.

Key-space notes:

* tt keys are the KimiLinear(AttnRes)Model module tree: ``layers.{i}.
  self_attn.*`` (KDA and MLA share the attribute name; per-key names
  already match HF), ``ffn.{gate,up,down}_proj`` on dense layers,
  ``ffn._moe.{router.gate,expert_bias,experts.w*,shared_experts.w*}``
  on MoE layers, plus the AttnRes extras (``attn_res_proj`` etc. and
  the model-level ``final_attn_res_*``).
* HF checkpoints appear with two MoE prefixes in the wild: the official
  Kimi export style (``mlp.*`` with gate_proj/up_proj/down_proj expert
  linears) and the block-sparse style (``block_sparse_moe.*`` with
  w1/w2/w3 routed + gate/up/down_proj shared). Reading accepts both;
  writing emits the official Kimi-Linear-48B export style
  (``block_sparse_moe.*``; dense layer-0 MLP stays ``mlp.*``).
* KDA ``A_log`` is ``[1, 1, H, 1]`` in HF and ``[H]`` in tt; reading
  reshapes, writing passes the tt shape through (the SGLang overlay
  accepts it -- keep in sync with the overlay if this changes).

Quantized (packed) checkpoints: NOT silently accepted. K3 official
weights are expected to ship packed MXFP4 + scales; until the exact
packing is known (2026-07-27 report), any quantization sidecar key or
sub-byte dtype raises with an explicit message instead of being treated
as an ordinary value.
"""

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger


_W_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_HF_TO_W = {v: k for k, v in _W_TO_HF.items()}

# Post-merge GroupedExperts params carry shape suffixes (Noam convention):
# w1/w3 are [E, F, D], w2 is [E, D, F].
_EXPERT_W_SUFFIXED = {"w1": "w1_EFD", "w2": "w2_EDF", "w3": "w3_EFD"}
_EXPERT_SUFFIXED_TO_W = {v: k for k, v in _EXPERT_W_SUFFIXED.items()}

# Sidecar/packed key suffixes that signal a quantized HF checkpoint.
_QUANT_KEY_MARKERS = (
    ".weight_scale",
    ".weight_scale_inv",
    ".scales",
    ".weight_packed",
    ".qweight",
    ".weight_blocks",
    ".qzeros",
)

_DIRECT_MAP_FROM_HF = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    "model.final_attn_res_proj.weight": "final_attn_res_proj.weight",
    "model.final_attn_res_norm.weight": "final_attn_res_norm.weight",
    "model.final_attn_res_alpha": "final_attn_res_alpha",
}

_PASSTHROUGH_LAYER_TAGS = (
    "attn_res_alpha",
    "mlp_res_alpha",
    "attn_res_proj.weight",
    "attn_res_norm.weight",
    "mlp_res_proj.weight",
    "mlp_res_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


class KimiLinearStateDictAdapter(MoEStateDictAdapter):
    """StateDictAdapter for KimiK3Model / KimiK3AttnResModel."""

    def __init__(self, model_config, hf_assets_path: str | None):
        # model_config is a KimiK3Spec (duck-typed shim); the base
        # class only reads the safetensors index from hf_assets_path.
        super().__init__(model_config, hf_assets_path)
        self.kimi_config = model_config.kimi_config
        # LoRA renames every wrapped projection's weight (q_proj.weight ->
        # q_proj.base.weight). to_hf already strips that on export; loading
        # needs the inverse, or a plain base checkpoint cannot be loaded into a
        # LoRA model at all -- which is the 48B graft path: take official
        # weights, attach adapters, train. Without it the load dies on
        # "Missing key: ...base.weight".
        self._lora_rank = getattr(model_config, "lora_rank", None)
        self._lora_targets: tuple[str, ...] = ()
        if self._lora_rank is not None:
            from torchtitan.models.kimi_k3.lora import DEFAULT_LORA_TARGETS

            self._lora_targets = DEFAULT_LORA_TARGETS

    def _add_lora_base(self, tt_key: str) -> str:
        """Insert ``.base`` for LoRA-wrapped projections, if LoRA is enabled.

        Matches the same leaf/qualified-suffix rule apply_lora uses, so the two
        cannot disagree about which modules are wrapped.
        """
        if not self._lora_targets or not tt_key.endswith((".weight", ".bias")):
            return tt_key
        stem, _, suffix = tt_key.rpartition(".")
        leaf = stem.rpartition(".")[2]
        matched = leaf in self._lora_targets or any(
            "." in t and stem.endswith(f".{t}") for t in self._lora_targets
        )
        return f"{stem}.base.{suffix}" if matched else tt_key

    # ----- quantization guard -------------------------------------- #

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            raise NotImplementedError(
                "Quantized (packed) Kimi checkpoints are not supported yet: "
                "the MXFP4 unpack path lands once the K3 report fixes the "
                "exact packing. Refusing to silently treat packed weights "
                "as ordinary values."
            )
        return HuggingFaceStorageReader(path)

    @staticmethod
    def _check_not_packed(hf_state_dict: dict[str, Any]) -> None:
        packed = [
            k
            for k in hf_state_dict
            if k.endswith(_QUANT_KEY_MARKERS)
            or (
                isinstance(hf_state_dict[k], torch.Tensor)
                and hf_state_dict[k].dtype
                in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)
            )
        ]
        if packed:
            raise NotImplementedError(
                "HF checkpoint contains quantized/packed tensors "
                f"(e.g. {packed[:4]}); the MXFP4/packed unpack path is not "
                "implemented yet. Refusing to silently treat packed weights "
                "as ordinary values."
            )

    # ----- tt -> HF -------------------------------------------------- #

    def _is_text_only(self) -> bool:
        """No vision tower -> the release's multimodal wrapper prefix names a
        module this model does not have, so emit the bare ``model.`` spelling."""
        return getattr(self.model_config, "vision_config", None) is None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert tt state dict to HF naming; split stacked experts."""
        hf_state_dict: dict[str, Any] = {}
        num_experts = self.kimi_config.num_experts

        for key, value in state_dict.items():
            # LoRA wrapping renames base weights (q_proj.weight ->
            # q_proj.base.weight); the HF destination is the original
            # name, and the value stays a view of the same storage so
            # the online read path fills the real param in place.
            key = key.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
            if (
                "attn_res" in key
                or "mlp_res" in key
                or "lora_a" in key
                or "lora_b" in key
            ):
                # Graft/LoRA extras have no HF-format destination: the HF
                # key space is the ORIGINAL Kimi architecture (so official
                # checkpoints load into graft flavors without phantom read
                # keys). Trained graft/adapter params ship as the
                # fork-native trainable_state_dict payload instead.
                continue
            if ".ffn._moe.routed_experts.inner_experts." in key:
                # layers.{i}.ffn._moe.routed_experts.inner_experts.w1_EFD
                # -> per-expert HF linears
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                w_suffixed = key.rsplit(".", 1)[-1]
                w_tag = _EXPERT_SUFFIXED_TO_W[w_suffixed]
                # Official Kimi-Linear-48B export style: routed experts are
                # block_sparse_moe.experts.{e}.w{1,2,3}.weight (w-naming),
                # while shared experts use gate/up/down_proj naming.
                hf_abstract_key = (
                    "model.layers.{}.block_sparse_moe.experts.{}." + w_tag + ".weight"
                )
                if isinstance(value, DTensor):
                    # Online (sharded) path: record placement metadata so
                    # from_hf can rebuild the DTensor, emit local experts.
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    hf_state_dict.update(
                        self._get_local_experts_weights(
                            hf_abstract_key, abstract_key, layer_num, value
                        )
                    )
                else:
                    split_values = self._split_experts_weights(value, num_experts)
                    for e in range(num_experts):
                        hf_state_dict[
                            hf_abstract_key.format(layer_num, e)
                        ] = split_values[e].squeeze()
                continue

            if key.endswith("self_attn.A_log"):
                # File-side KDA A_log is [1, 1, H, 1]; the model holds [H].
                # The online HF reader validates placeholder shapes against
                # the saved file, so the view must happen on this side too
                # (from_hf flattens back).
                value = value.reshape(1, 1, -1, 1)
            hf_state_dict[self._tt_key_to_hf(key, self._is_text_only())] = value

        return hf_state_dict

    @staticmethod
    def _tt_key_to_hf(key: str, text_only: bool = False) -> str:
        """Single-tensor tt -> HF key mapping (experts handled separately)."""
        direct = {v: k for k, v in _DIRECT_MAP_FROM_HF.items()}
        if key in direct:
            return direct[key]
        if not key.startswith("layers."):
            raise ValueError(f"Unmapped tt key: {key!r}")
        rest = key[len("layers.") :]
        idx_s, _, sub = rest.partition(".")
        prefix = f"model.layers.{idx_s}"

        if sub in _PASSTHROUGH_LAYER_TAGS or sub.startswith("self_attn."):
            return f"{prefix}.{sub}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if sub == f"ffn.{proj}.weight":
                return f"{prefix}.mlp.{proj}.weight"
        if sub == "ffn._moe.router.gate.weight":
            return f"{prefix}.block_sparse_moe.gate.weight"
        if sub == "ffn._moe.expert_bias_E":
            return f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"
        if sub.startswith("ffn._moe.shared_experts."):
            tail = sub[len("ffn._moe.shared_experts.") :]
            w_tag, _, suff = tail.partition(".")
            return f"{prefix}.block_sparse_moe.shared_experts.{_W_TO_HF[w_tag]}.{suff}"
        # K3's layout (latent MoE projections, the released AttnRes and gate
        # names) is owned by hf_key_map, which is tested for full coverage
        # against the released checkpoint index. This adapter predates it and
        # targets the Kimi-Linear-48B naming, so K3-only keys arrive here;
        # delegate rather than keep the same table in two places that can drift.
        from torchtitan.models.kimi_k3.hf_key_map import titan_to_official, UnmappedKey

        try:
            return titan_to_official(key, kda_layers=set(), text_only=text_only)
        except UnmappedKey:
            pass
        raise ValueError(f"Unmapped tt key: {key!r}")

    # ----- HF -> tt -------------------------------------------------- #

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to tt naming; stack per-expert weights."""
        self._check_not_packed(hf_state_dict)

        state_dict: dict[str, Any] = {}
        num_experts = self.kimi_config.num_experts
        # {layer: {titan_abstract_key: {expert_id: tensor}}}
        expert_weights_by_layer: dict[str, dict[str, dict[int, Any]]] = {}

        # Iterate over a key snapshot and pop each entry as it is
        # consumed: on the online (sharded initial-load) path
        # hf_state_dict holds every loaded per-expert slice, and keeping
        # those references alive while the stacked copies are built
        # doubles the peak -- enough to OOM the 48B load on 32 GiB
        # cards. Consuming the input dict is part of this method's
        # contract (the caller replaces it with the returned dict).
        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)
            expert_m = re.match(
                r"model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\."
                r"(\d+)\.(\w+)\.weight",
                key,
            )
            if expert_m is not None:
                layer_num, expert_num, proj = expert_m.groups()
                w_tag = _HF_TO_W.get(proj, proj)  # w1/w2/w3 or gate_proj-style
                if w_tag not in ("w1", "w2", "w3"):
                    raise ValueError(f"Unknown expert projection in {key!r}")
                titan_abstract_key = (
                    "layers.{}.ffn._moe.routed_experts.inner_experts."
                    + _EXPERT_W_SUFFIXED[w_tag]
                )
                new_key = titan_abstract_key.format(layer_num)

                layer_bucket = expert_weights_by_layer.setdefault(layer_num, {})
                layer_bucket.setdefault(titan_abstract_key, {})[int(expert_num)] = value

                if titan_abstract_key in self.local_experts_indices:
                    # Online path: to_hf() ran first and recorded shards.
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer, titan_abstract_key, layer_num
                    )
                else:
                    stacked = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked
                continue

            tt_key, value = self._hf_key_to_tt(key, value)
            tt_key = self._add_lora_base(tt_key) if tt_key else tt_key
            if tt_key is not None:
                state_dict[tt_key] = value

        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            # Kimi scaling-law configs tie lm_head to the embedding and the
            # HF export omits the alias. For a genuinely untied model with a
            # missing head this is wrong -- warn loudly either way.
            logger.warning(
                "HF checkpoint has no lm_head.weight; aliasing "
                "embed_tokens.weight (Kimi tied-embedding convention)."
            )
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        return state_dict

    @staticmethod
    def _hf_key_to_tt(key: str, value: Any) -> tuple[str | None, Any]:
        """Single-tensor HF -> tt key mapping (experts handled separately).

        Returns (None, value) for HF keys with no tt destination (e.g.
        vision tower tensors in a multimodal export).
        """
        if key in _DIRECT_MAP_FROM_HF:
            return _DIRECT_MAP_FROM_HF[key], value

        m = re.match(r"model\.layers\.(\d+)\.(.+)", key)
        if m is None:
            return None, value
        idx_s, sub = m.groups()
        tt_prefix = f"layers.{idx_s}"

        if sub in _PASSTHROUGH_LAYER_TAGS:
            return f"{tt_prefix}.{sub}", value
        if sub.startswith("self_attn."):
            if (
                sub == "self_attn.A_log"
                and isinstance(value, torch.Tensor)
                and value.dim() == 4
            ):
                value = value.reshape(-1)
            return f"{tt_prefix}.{sub}", value

        # Dense MLP (both HF prefixes)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if sub == f"mlp.{proj}.weight":
                return f"{tt_prefix}.ffn.{proj}.weight", value

        # Router / bias (both HF prefixes)
        router_m = re.match(
            r"(?:mlp|block_sparse_moe)\.gate\.(weight|e_score_correction_bias)",
            sub,
        )
        if router_m is not None:
            tail = router_m.group(1)
            if tail == "weight":
                return f"{tt_prefix}.ffn._moe.router.gate.weight", value
            return f"{tt_prefix}.ffn._moe.expert_bias_E", value

        # Shared experts (both HF prefixes, both naming styles)
        shared_m = re.match(
            r"(?:mlp|block_sparse_moe)\.shared_experts\.(\w+)\.(.+)", sub
        )
        if shared_m is not None:
            proj, suff = shared_m.groups()
            w_tag = _HF_TO_W.get(proj, proj)
            if w_tag not in ("w1", "w2", "w3"):
                raise ValueError(f"Unknown shared-expert projection in {key!r}")
            return f"{tt_prefix}.ffn._moe.shared_experts.{w_tag}.{suff}", value

        # Unknown per-layer key: skip with a debug note rather than failing
        # (multimodal exports carry vision/projector keys the LM ignores).
        logger.debug("KimiLinearStateDictAdapter: skipping HF key %s", key)
        return None, value
