# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchstore as ts
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.utils import (
    compute_policy_gradient_loss,
    compute_token_log_probs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.types import Episode
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """
    Updates policy based on collected Episode using TorchTitan components.

    Uses ModelSpec for model construction, parallelization, and weight loading.

    Args:
        config: PolicyTrainer.Config for model/optimizer/parallelism settings.
        model_spec: Model specification (model config, parallelize_fn, state_dict_adapter).
        hf_assets_path: Path to HF assets folder for checkpoint loading.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """PolicyTrainer configuration for optimizer, training, and parallelism."""

        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        """Communication configuration for distributed initialization."""
        compile: CompileConfig = field(default_factory=CompileConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        hf_assets_path: str = "",
        transfer_dtype: str = "",
        kl_coef: float = 0.0,
        dcp_initial_load_path: str = "",
        vision_tower_path: str = "",
        projector_dcp_path: str = "",
        image_token_id: int = 32000,
    ):
        # ``dcp_initial_load_path``: optional path to a torch DCP-format
        # checkpoint to load instead of HF safetensors. Used when the
        # model_spec has no ``state_dict_adapter`` (e.g. our Kimi AttnRes
        # 447M, where the HF↔torchtitan key remap is large enough that
        # we'd rather just round-trip torchtitan-native DCP). Mutually
        # exclusive with ``hf_assets_path`` — if both are non-empty,
        # DCP wins.
        self._dcp_initial_load_path = dcp_initial_load_path
        # Optional VLM hooks: when ``vision_tower_path`` and
        # ``projector_dcp_path`` are set, compute_token_log_probs
        # injects vision features at image-token positions, matching
        # what the generator's LM saw. ``vision_tower_path`` is an HF
        # id (e.g. ``google/siglip-base-patch16-224``);
        # ``projector_dcp_path`` is the torchtitan-native DCP dir whose
        # top-level ``mm_projector.projector.{fc1,fc2}.{weight,bias}``
        # keys are loaded into a freshly-built 2-layer MLP (typically
        # the same dir as ``dcp_initial_load_path``).
        self._vision_tower_path = vision_tower_path
        self._projector_dcp_path = projector_dcp_path
        self._image_token_id = image_token_id
        self._vision_tower = None
        self._projector = None
        self.config = config
        self.model_spec = model_spec
        self.kl_coef = kl_coef
        # Only cast if transfer dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        requested = TORCH_DTYPE_MAP[transfer_dtype] if transfer_dtype else None
        self._transfer_dtype = requested if requested != training_dtype else None

        # The policy and ref models share code objects, so dynamo's
        # per-code-object cache must hold entries for both grad modes
        # (grad for policy, no_grad for ref). The default limit of 8
        # is not enough; 16 accommodates both without recompile storms.
        # TODO: @Lucaskabela fix recompiles in general as these increase startup
        torch._dynamo.config.cache_size_limit = 16

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.debug.batch_invariant)

        world_size = dist_utils.init_distributed(config.comm)

        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)

        # Set determinism flags and seed via core torchtitan utility
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # Initialize state dict adapter for HF checkpoint loading
        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_spec.model, hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Create training policy model
        model = self._build_model(model_spec, config, device_type, hf_assets_path)
        model.train()
        self.model = model
        self.model_parts = [model]

        # Conditionally build frozen reference model for KL penalty
        if kl_coef > 0:
            ref_model = self._build_model(
                model_spec, config, device_type, hf_assets_path
            )
            ref_model.eval()
            ref_model.requires_grad_(False)
            self.ref_model = ref_model
            logger.info(f"Built frozen reference model (kl_coef={kl_coef})")
        else:
            self.ref_model = None

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        # Optional vision tower + projector for VLM logprob recompute.
        # Loaded once at __init__; used by ``step`` to inject vision
        # features into the LM forward at image-token positions.
        if self._vision_tower_path and self._projector_dcp_path:
            self._load_vision_components(device_type)

        self.policy_version = 0
        self.generator: Any | None = None

        # Data parallelism: determine this rank's shard of the batch.
        self.dp_size = self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
        self.dp_rank = dist.get_rank() // self.parallel_dims.non_data_parallel_size
        self.dp_enabled = self.parallel_dims.dp_enabled

        logger.debug(
            f"PolicyTrainer initialized (dp_rank={self.dp_rank}, dp_size={self.dp_size})"
        )

    def _load_initial_hf_weights(self, model, checkpoint_path: str) -> None:
        """Load model weights from HF checkpoint using DCP and state_dict_adapter.

        Args:
            model: The model to load weights into.
            checkpoint_path: Path to HF checkpoint directory.
        """
        if self.sd_adapter is None:
            logger.warning(
                "No state_dict_adapter available, skipping initial weight load"
            )
            return

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' does not exist. "
                "Please provide a valid path to a HuggingFace checkpoint directory."
            )

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=True),
        )
        logger.info(
            f"Loaded initial weights from {checkpoint_path} "
            f"({len(torchtitan_state_dict)} parameters)"
        )

    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
        hf_assets_path: str,
    ):
        """Build, parallelize, and initialize a model from checkpoint.
        Will be used to build trainer's policy model and reference model.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, checkpoint path, etc.).
            device_type: Device type string (e.g. "cuda").
            hf_assets_path: Path to HF assets folder for checkpoint loading.

        Returns:
            Initialized model with weights loaded from checkpoint.
        """

        # TODO Also support flex attention backend later.
        from torchtitan.models.common.attention import VarlenAttention

        # Soft check (was hard assert): warn instead of crash when the
        # model_spec doesn't follow the canonical Qwen3 layout
        # (``model.layers[0].attention.inner_attention``). Non-Qwen3
        # specs (e.g. Kimi Linear's ``ModuleDict`` layers, MoE-rich
        # decoder blocks, or non-VarlenAttention attention impls) can
        # still build + parallelize via their own ``parallelize_fn``;
        # the trainer's assumptions about ``compute_token_log_probs``
        # downstream are what actually constrain the model surface.
        try:
            inner = model_spec.model.layers[0].attention.inner_attention
            if not isinstance(inner, VarlenAttention.Config):
                logger.warning(
                    f"Inner attention is {type(inner).__name__}, not "
                    f"VarlenAttention.Config. Trainer assumes the model's "
                    f"forward signature accepts (input_ids, attention_masks, "
                    f"positions); non-standard models may need a custom "
                    f"compute_token_log_probs. Continuing best-effort."
                )
        except (AttributeError, IndexError, TypeError, KeyError):
            logger.warning(
                "model_spec.model layout is non-standard (no "
                "``.layers[0].attention.inner_attention``). Trainer "
                "assumes ``model.build()`` returns nn.Module with "
                "``forward(input_ids, attention_masks=, positions=)``. "
                "Continuing best-effort."
            )

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            parallelism=config.parallelism,
            compile_config=config.compile,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights. DCP path wins when set (skips
        # HF↔torchtitan adapter remap entirely, useful for models
        # that don't ship a ``state_dict_adapter``).
        if self._dcp_initial_load_path:
            self._load_initial_dcp_weights(model, self._dcp_initial_load_path)
        else:
            self._load_initial_hf_weights(model, hf_assets_path)

        return model

    def _load_initial_dcp_weights(self, model, dcp_path: str) -> None:
        """Load model weights from a torch DCP-format checkpoint.

        Bypasses the HF state_dict_adapter pipeline. The trainer's
        torchtitan-native state_dict layout must already match what
        was saved (true when both sides use the same model_spec, which
        is the case for our 447M AttnRes whose torchtitan training and
        RL trainer use the same ``KimiLinearAttnResModel``).
        """
        if not os.path.isdir(dcp_path):
            raise FileNotFoundError(
                f"dcp_initial_load_path does not exist: {dcp_path}"
            )
        sd = model.state_dict()
        logger.info(
            f"loading torchtitan-native DCP from {dcp_path} "
            f"({len(sd)} keys)"
        )
        dcp.load(sd, checkpoint_id=dcp_path)
        # ``dcp.load`` populates ``sd`` in place; assign back so
        # FSDP-wrapped params see the loaded weights.
        set_model_state_dict(
            model=model,
            model_state_dict=sd,
            options=StateDictOptions(strict=False),
        )
        logger.info(f"DCP weights loaded ({len(sd)} parameters)")

    def _load_vision_components(self, device_type: str) -> None:
        """Build a frozen SigLIP vision tower and a 2-layer MLP projector
        for VLM logprob recompute.

        Vision tower: HF SigLIP via ``transformers.SiglipVisionModel``.
        Projector: 2-layer MLP matching ``phase5/multimodal_model.py``'s
        ``Projector`` (fc1 vision_d->llm_d with bias, GELU, fc2 llm_d->llm_d
        with bias). Weights loaded from
        ``mm_projector.projector.{fc1,fc2}.{weight,bias}`` in the DCP
        ckpt at ``self._projector_dcp_path``.
        """
        try:
            from transformers import SiglipVisionModel
        except ImportError as e:
            logger.warning(
                f"transformers.SiglipVisionModel not available ({e}); "
                "skipping vision-tower load. VLM logprob match will diverge."
            )
            return

        logger.info(f"loading vision tower from {self._vision_tower_path}")
        vt = SiglipVisionModel.from_pretrained(self._vision_tower_path)
        for p in vt.parameters():
            p.requires_grad = False
        vt.eval()
        vt = vt.to(self.device)
        self._vision_tower = vt

        # Build the projector. fc1 vision_hidden -> llm_hidden, fc2
        # llm_hidden -> llm_hidden, both with bias. Match the trained
        # geometry from phase5/multimodal_model.py:Projector.
        vision_hidden = vt.config.hidden_size
        # Probe the LM hidden size from the model's embed_tokens.
        try:
            llm_hidden = self.model.embed_tokens.weight.shape[1]
        except AttributeError:
            llm_hidden = self.model.tok_embeddings.weight.shape[1]
        proj = torch.nn.Sequential()  # placeholder; build proper module below
        # Use a small custom module to match the saved keys
        # (mm_projector.projector.fc1/.fc2).
        class _Projector(torch.nn.Module):
            def __init__(self, vd, ld):
                super().__init__()
                self.fc1 = torch.nn.Linear(vd, ld, bias=True)
                self.fc2 = torch.nn.Linear(ld, ld, bias=True)
            def forward(self, x):
                import torch.nn.functional as F
                return self.fc2(F.gelu(self.fc1(x)))
        proj_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        with torch.device(device_type):
            with utils.set_default_dtype(proj_dtype):
                proj = _Projector(vision_hidden, llm_hidden)

        # Load mm_projector keys from DCP. The DCP saved them under
        # mm_projector.projector.fc1.{weight,bias} etc.
        sd = {
            f"mm_projector.projector.{k}": v
            for k, v in proj.state_dict().items()
        }
        try:
            dcp.load(sd, checkpoint_id=self._projector_dcp_path)
        except Exception as e:
            logger.warning(
                f"projector DCP load failed ({e}); leaving projector at random "
                "init — VLM logprob recompute will be ungrounded."
            )
            self._projector = None
            return
        # Strip the prefix back out and load into the module.
        proj_sd = {
            k[len("mm_projector.projector."):]: v
            for k, v in sd.items()
            if k.startswith("mm_projector.projector.")
        }
        proj.load_state_dict(proj_sd, strict=True)
        for p in proj.parameters():
            p.requires_grad = False
        proj.eval()
        self._projector = proj.to(self.device)
        logger.info(
            f"VLM components loaded: vision_tower {vision_hidden}-d, "
            f"projector {vision_hidden}->{llm_hidden}-d (frozen)"
        )

    @endpoint
    async def save_dcp(self, save_dir: str, step: int) -> str:
        """Save the policy model as a DCP checkpoint under
        ``{save_dir}/step-{step}/`` and return the path written.

        Model weights only (no optimizer state) — RL fine-tunes are short
        and re-init optim on restart; the saved DCP is a resume point for
        --dcp-load-path and the deliverable policy. Mirrors
        ``OPDTrainer.save_dcp`` so the GRPO path (PolicyTrainer, no OPD
        subclass) can also checkpoint.
        """
        out_dir = os.path.join(save_dir, f"step-{step}")
        os.makedirs(out_dir, exist_ok=True)
        dcp.save(self.model.state_dict(), checkpoint_id=out_dir)
        logger.info(f"policy ckpt saved: {out_dir} (step {step})")
        return out_dir

    @endpoint
    async def push_model_state_dict(self) -> None:
        """Publish model weights for generator consumption via TorchStore.

        When ``direct_rdma=True``, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When ``False``, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple ``is_rdma_available()`` with ``direct_rdma`` here,
        but the two concepts are not identical — StorageVolumes can also
        use RDMA as their transport layer. ``direct_rdma`` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".
        """
        # NOTE: pinned to vendored torchstore 0.1.2 API (state_dict, key,
        # store_name). direct_rdma / transfer_dtype kwargs existed only in
        # newer torchstore (not on PyPI as of 2026-05-12).
        await ts.put_state_dict(
            self.model.state_dict(),
            "model_state_dict",
        )

    @endpoint
    async def step(self, episodes: list[Episode]) -> dict:
        """Perform one training step.

        Expects a flat list of Episodes with ``advantage`` already computed
        by the controller. Shards episodes across DP ranks so each rank
        processes a unique slice of the data.

        Args:
            episodes: Flat list of Episodes with advantages set.

        Returns:
            Training metrics
        """
        # The policy and ref models share code objects, so dynamo's
        # per-code-object cache must hold entries for both grad modes
        # (grad for policy, no_grad for ref). The default limit of
        # is not enough; 16 accommodates both without recompile storms.
        # TODO: @Lucaskabela fix recompiles in general as these increase startup
        torch._dynamo.config.recompile_limit = 16

        logger.debug(
            f"{os.getpid()=} PolicyTrainer starting step {self.policy_version} "
        )

        advantages = torch.tensor([ep.advantage for ep in episodes])

        all_token_ids: list[list[int]] = [ep.token_ids for ep in episodes]
        all_prompt_token_ids: list[list[int]] = [ep.prompt_token_ids for ep in episodes]
        all_token_log_probs: list[list[float]] = [ep.token_log_probs for ep in episodes]
        all_image_paths: list[str | None] = [
            getattr(ep, "image_path", None) for ep in episodes
        ]

        all_rewards_tensor = torch.tensor([ep.reward for ep in episodes])

        # Shard flattened completions across DP ranks so each rank processes
        # a unique subset of the data.
        total_samples = len(all_token_ids)
        my_indices = list(range(self.dp_rank, total_samples, self.dp_size))
        my_token_ids = [all_token_ids[i] for i in my_indices]
        my_prompt_token_ids = [all_prompt_token_ids[i] for i in my_indices]
        my_token_log_probs = [all_token_log_probs[i] for i in my_indices]
        my_image_paths = [all_image_paths[i] for i in my_indices]
        my_advantages = advantages[my_indices]

        # Compute reference model log probs if KL penalty is enabled
        ref_token_log_probs = None
        if self.ref_model is not None:
            ref_token_log_probs = []
            with torch.no_grad():
                for prompt_toks, gen_toks, img_path in zip(
                    my_prompt_token_ids, my_token_ids, my_image_paths,
                ):
                    ref_lps = compute_token_log_probs(
                        self.ref_model, prompt_toks, gen_toks, self.device,
                        vision_tower=self._vision_tower,
                        projector=self._projector,
                        image_path=img_path,
                        image_token_id=self._image_token_id,
                    )
                    ref_token_log_probs.append(ref_lps)

        loss, loss_metrics, batch_token_log_probs = compute_policy_gradient_loss(
            self.model,
            my_token_ids,
            my_prompt_token_ids,
            my_advantages,
            ref_token_log_probs=ref_token_log_probs,
            kl_coef=self.kl_coef,
            vision_tower=self._vision_tower,
            projector=self._projector,
            image_paths=my_image_paths,
            image_token_id=self._image_token_id,
        )

        # Verify logprob identity (local shard)
        verification_result = verify_logprob_identity(
            my_token_log_probs,
            batch_token_log_probs,
        )

        logger.debug(
            f"Logprob verification: bitwise_identical={verification_result['logprob_bitwise_identical']}, "
            f"max_delta={verification_result['logprob_max_delta']:.6e}, "
            f"diff_mean={verification_result['logprob_diff_mean']:.6e}, "
            f"diff_max={verification_result['logprob_diff_max']:.6e}, "
            f"tokens_checked={verification_result['total_tokens_checked']}"
        )

        # Update weights
        self.optimizers.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "reward_mean": all_rewards_tensor.mean().item(),
            "reward_std": all_rewards_tensor.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "sample_completion": episodes[0].text[:80],
            "policy_version": self.policy_version,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            # Trainer vs generator log prob divergence
            "logprob_diff_mean": verification_result["logprob_diff_mean"],
            "logprob_diff_max": verification_result["logprob_diff_max"],
            "logprob_max_delta": verification_result["logprob_max_delta"],
            "logprob_bitwise_identical": verification_result[
                "logprob_bitwise_identical"
            ],
            **loss_metrics,
        }
        logger.debug(
            f"{os.getpid()=} PolicyTrainer finished step {self.policy_version}"
        )
        return metrics
