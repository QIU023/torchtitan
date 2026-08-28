# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, fields

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def _lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """Derive LoRA adapter sharding from the base linear's TP sharding.

    For colwise base linears, ``lora_a`` is TP-replicated and ``lora_b``
    mirrors the base output-dim shard.

    For rowwise base linears, ``lora_a`` mirrors the base input-dim shard and
    ``lora_b`` is TP-replicated, producing the same partial-output shape as the
    base linear.
    """
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None, None

    replicated_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    if base_weight_sharding == dense_param_placement(tp=spmd.S(0)):
        lora_b_sharding = ShardingConfig(
            state_shardings={"weight": base_weight_sharding},
        )
        return replicated_weight, lora_b_sharding
    else:
        assert base_weight_sharding == dense_param_placement(tp=spmd.S(1))
        lora_a_sharding = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        )
        return lora_a_sharding, replicated_weight


class LoRALinearBase:
    """Marker base of every dynamically created LoRA linear class.

    ``_get_lora_cls`` builds one subclass per parent linear class, so there is
    no single concrete class to isinstance against; this empty base is baked
    into each of them and is the stable way to find LoRA modules on a built
    model (``merge_lora_state_dict`` walks it).
    """


_lora_class_cache: dict[type, type] = {}
_frozen_config_class_cache: dict[type, type] = {}


def _get_lora_cls(parent_cls: type) -> type:
    """Get or create a LoRA subclass for *parent_cls* (e.g. Linear, Float8Linear).

    The returned class has a proper ``Config`` that extends the parent's Config
    with ``rank`` and ``alpha``.  Adapters are built in ``__init__`` from the
    base config's dimensions and sharding.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]

    class LoRALinear(parent_cls, LoRALinearBase):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            for param in nn.Module.parameters(self):
                param.requires_grad_(False)
            self._lora_scaling = config.alpha / config.rank
            lora_a_sharding, lora_b_sharding = _lora_adapter_sharding(
                config.sharding_config
            )
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=config.rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={
                    "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
                },
            ).build()
            self.lora_b = Linear.Config(
                in_features=config.rank,
                out_features=config.out_features,
                bias=False,
                sharding_config=lora_b_sharding,
                param_init={"weight": nn.init.zeros_},
            ).build()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(input)
            lora_out = self.lora_b(self.lora_a(input))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


def _get_frozen_config_cls(
    config_cls: type[Module.Config],
) -> type[Module.Config]:
    """Get or create a config subclass that freezes direct build parameters."""
    if config_cls in _frozen_config_class_cache:
        return _frozen_config_class_cache[config_cls]

    class FrozenConfig(config_cls):  # type: ignore[valid-type, misc]
        def build(self, **kwargs):
            instance = config_cls.build(self, **kwargs)
            for param in instance.parameters(recurse=False):
                param.requires_grad_(False)
            return instance

    FrozenConfig.__name__ = f"Frozen{config_cls.__name__}"
    FrozenConfig.__qualname__ = f"Frozen{config_cls.__qualname__}"
    _frozen_config_class_cache[config_cls] = FrozenConfig
    return FrozenConfig


def _make_frozen_config(cfg: Module.Config) -> Module.Config:
    """Create a frozen config that still passes checks for the original type."""
    frozen_cls = _get_frozen_config_cls(type(cfg))
    return frozen_cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init})


class LoRAConverter(ModelConfigConverter):
    """Apply LoRA adapters to Linear layers in a model.

    Operates on the model config tree: target Linear configs are replaced
    with ``LoRALinear.Config`` (which builds a LoRA subclass with frozen base
    and trainable adapters). Non-target modules are replaced with dynamic
    frozen config subclasses that freeze direct parameters at build time.

    When ``target_modules`` is None (default), every ``Linear.Config`` is
    converted.  When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        rank: int = 8
        """Rank of the LoRA matrices."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Module names to apply LoRA to (matched against the last segment of the FQN).
        None means all Linear layers. An empty list means no layers."""

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules is not None else None
        )
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def _make_lora_config(self, cfg: Linear.Config):
        """Create a LoRALinear.Config from a base Linear.Config."""
        assert cfg._owner is not None
        lora_cls = _get_lora_cls(cfg._owner)
        return lora_cls.Config(  # pyrefly: ignore [missing-attribute]
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=self.rank,
            alpha=self.alpha,
        )

    def convert(self, model_config: Module.Config) -> Module.Config:
        """Walk the module config tree from leaves to root.

        Target Linear modules get their config replaced with
        ``LoRALinear.Config``. All other module configs become frozen config
        subclasses so LoRA training updates only adapter parameters.
        """
        converted_root = model_config
        matched = set()
        configs = list(model_config.traverse(Module.Config, recurse=True))

        for fqn, cfg, parent, attr in reversed(configs):
            assert isinstance(cfg, Module.Config)
            last_segment = fqn.rsplit(".", 1)[-1]
            is_target = isinstance(cfg, Linear.Config) and (
                self.target_modules is None or last_segment in self.target_modules
            )

            if is_target:
                new_cfg = self._make_lora_config(cfg)
                matched.add(last_segment)
            else:
                new_cfg = _make_frozen_config(cfg)

            if parent is None:
                converted_root = new_cfg
            elif isinstance(parent, list):
                assert isinstance(attr, int)
                parent[attr] = new_cfg
            else:
                assert isinstance(attr, str)
                setattr(parent, attr, new_cfg)

        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"Linear.Config in the model config tree."
            )
        return converted_root


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter-only checkpoint payload: every parameter left trainable.

    Under LoRA that is the adapters (plus anything a model deliberately
    unfroze); shipping only these is what makes adapter checkpoints and
    trainer-to-rollout weight syncs light.
    """
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


# Wrapper segments that appear in ``named_modules()`` paths but NOT in
# ``state_dict()`` keys, because each wrapper installs a hook that strips its
# own prefix. Activation checkpointing, FSDP and torch.compile all do this.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
)


def _state_dict_prefix(mod_name: str, sd: dict) -> str:
    """The state-dict prefix for a module reached at ``mod_name``.

    The two namings differ once anything wraps the module: activation
    checkpointing turns ``layers.0.feed_forward.w1`` into
    ``layers.0._checkpoint_wrapped_module.feed_forward.w1`` in
    ``named_modules()``, while ``state_dict()`` strips it back out. Composing
    keys from the module path then writes a name nothing else recognises AND
    leaves the adapter keys in place, because the pops miss too.

    An unknown wrapper raises rather than guessing: a wrong name here is a
    weight that never reaches the consumer, which is not a failure that
    announces itself.
    """
    stripped = ".".join(p for p in mod_name.split(".") if p not in _WRAPPER_SEGMENTS)
    for candidate in (stripped, mod_name):
        if f"{candidate}.lora_a.weight" in sd:
            return candidate
    raise KeyError(
        f"LoRA module at {mod_name!r} has no matching state_dict entry (tried "
        f"{stripped!r}); an unrecognised module wrapper is in the path, and "
        "merging under a guessed name would ship weights nothing can load"
    )


def merge_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Fold LoRA adapters into base weights and return a state dict keyed by
    ORIGINAL param names (no ``lora_a``/``lora_b``). The model is unchanged
    when this returns.

    For each LoRA linear, ``W_merged = W_base + (alpha / rank) * (B @ A)``.
    This is the deployable/exportable form: a state-dict adapter can convert
    it exactly as it converts the unadapted model's, whereas the raw state
    dict carries adapter keys nothing downstream recognises.

    The merge happens IN the modules (temporarily, restored from clones on
    the way out) rather than in the returned dict, because a base linear's
    serialization is not necessarily ``<fqn>.weight``: the fused attention
    linear exports split ``wq``/``wk``/``wv`` keys through a state-dict hook,
    and writing composed key names would miss every such hook. Taking
    ``state_dict()`` with the merged weights in place lets each module's own
    serialization produce the right keys.

    Under TP nothing needs materializing: the adapter shardings mirror the
    base (``_lora_adapter_sharding``), so ``B @ A`` composes to the base
    weight's placement and the add dispatches as DTensors.
    """
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinearBase)
    ]
    originals = [module.weight.data.clone() for _, module in lora_modules]
    with torch.no_grad():
        for _, module in lora_modules:
            # fp32 delta for deployable precision, cast back to the base dtype.
            delta = module._lora_scaling * (
                module.lora_b.weight.float() @ module.lora_a.weight.float()
            )
            module.weight.data.copy_(
                (module.weight.data.float() + delta).to(module.weight.dtype)
            )
    try:
        # state_dict() values ALIAS the live parameter storage (hook-split
        # views included), so the restore below would silently revert them in
        # the returned dict too: clone while the merge is in place.
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    finally:
        with torch.no_grad():
            for (_, module), original in zip(lora_modules, originals):
                module.weight.data.copy_(original)
    for mod_name, _ in lora_modules:
        # named_modules() and state_dict() disagree once a wrapper is in the
        # path.
        prefix = _state_dict_prefix(mod_name, sd)
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            sd.pop(f"{prefix}{suffix}", None)
    return sd
