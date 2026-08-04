# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the Kimi K3 experiment.

This is the ``config_registry`` torchtitan's ConfigManager imports for
``--module kimi_k3``. Flavors: ``kimi_linear_<size>_<variant>`` -- the
AttnRes tech-report Table 2 scaling-law sweep (194m..528m), the
SGLang-aligned 447m carrier (+ fp8 variant), and the 48B-A3B layout
carriers. Architecture-side builders live in ``model_configs.py``.

The dense Llama3-shape / DSv3-shape AttnRes test carrier that previously
shared this registry lives outside this folder; it remains runnable
against earlier history (<= 666cf7ad6).
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.kimi_k3 import model_registry as attn_res_model_registry
from torchtitan.models.kimi_k3.model_configs import (  # noqa: F401
    _alternating_kda_mla_layers,
    _BY_NAME,
    build,
    build_kimi_linear_config,
    flavor_names,
    resolve_num_blocks,
    SCALING_LAW_TABLE,
    Variant,
)

# Re-export every Kimi Linear + AttnRes trainer-config flavor so they are
# discoverable via ``--module kimi_k3 --config kimi_linear_<...>``.
# torchtitan's ConfigManager does ``getattr(config_registry, <config_name>)``,
# so the kimi flavor functions must be module-level attributes here. The
# ``kimi_linear_`` config-name prefix is preserved for backward compatibility
# with production launch scripts (only the ``--module`` value changed).
from torchtitan.models.kimi_k3.state_dict_adapter import KimiLinearStateDictAdapter
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


# ----- Kimi Linear / K3 trainer configs (merged from kimi_linear/) ----- #


def _base_trainer_config(size_name: str) -> Trainer.Config:
    """Shared Trainer.Config template for a given paper Table-2 size.

    The peak LR + batch-size come from the paper; other knobs match
    torchtitan common defaults (warmup=500, cosine decay_ratio=0.8,
    min_lr_factor=0.1, FSDP full shard). ``model_spec`` is set by the
    per-flavor wrappers below.
    """
    if size_name not in _BY_NAME:
        raise ValueError(f"Unknown size '{size_name}'")
    spec = _BY_NAME[size_name]
    return Trainer.Config(
        # Plain (non-chunked) CE: matches the numerics of all historical
        # kimi runs, and the KimiLinear* models don't implement the
        # _skip_lm_head forward that ChunkedLossWrapper requires.
        # 163840 = Kimi tokenizer vocab (build_kimi_linear_config
        # default; no flavor overrides it).
        loss=CrossEntropyLoss.Config(global_vocab_size=163840),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
            log_freq=10,
        ),
        model_spec=None,  # filled in by the per-flavor wrapper
        optimizer=default_adamw(lr=spec.lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=500,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=max(1, spec.batch_size // 8),  # default 8 DP ranks
            seq_len=8192,  # paper uses 8192 context
            steps=20000,  # placeholder; caller overrides via --training.steps
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1000,
            keep_latest_k=2,  # disk-discipline: at most 2x model size
            last_save_model_only=False,
        ),
        # AC off by default: the debug/scaling flavors fit without it.
        # (AC itself is supported -- see parallelize_kimi_k3.)
        activation_checkpoint=None,
        validator=Validator.Config(freq=500, steps=50),
        # Kimi CP reassembles contiguous rank-ordered seq shards inside
        # KDA/MLA (see model.py); the headtail load balancer permutes the
        # sequence and silently breaks causal order, so it must stay off.
        # parallelize_kimi_k3 raises if this is set back to a balancer.
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
    )


def _flavor_trainer_config(size: str, variant: Variant) -> Trainer.Config:
    """Return a Trainer.Config for the requested size+variant with
    ``model_spec`` wired to :func:`model_registry` (imported late to
    avoid a circular import).
    """
    # Late import: model_registry lives in __init__.py which imports
    # from this module. Circular if eager-imported at module top.
    from torchtitan.models.kimi_k3 import model_registry

    cfg = _base_trainer_config(size)
    flavor = f"kimi_linear_{size}_{variant}"
    cfg.model_spec = model_registry(flavor)
    return cfg


# ----- Explicit per-flavor entry points (tyro discovers these) ----------- #


def kimi_linear_194m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("194m", "baseline")


def kimi_linear_194m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("194m", "block_attn_res")


def kimi_linear_194m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("194m", "full_attn_res")


def kimi_linear_241m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("241m", "baseline")


def kimi_linear_241m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("241m", "block_attn_res")


def kimi_linear_241m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("241m", "full_attn_res")


def kimi_linear_296m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("296m", "baseline")


def kimi_linear_296m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("296m", "block_attn_res")


def kimi_linear_296m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("296m", "full_attn_res")


def kimi_linear_436m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("436m", "baseline")


def kimi_linear_436m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("436m", "block_attn_res")


def kimi_linear_436m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("436m", "full_attn_res")


def kimi_linear_436m_block_attn_res_n4() -> Trainer.Config:
    """436M Block AttnRes with N=4 (instead of paper-default N=8).

    Paper Fig 6 (S ablation on the 16-layer model from Table 2)
    shows S=2/4/8 — i.e., N=8/4/2 for L=16 — all converging to
    ~1.746 vs baseline 1.766 on validation loss. The choice of
    N is essentially indistinguishable across that range.

    We use N=4 here (S=4 hf_layers/block) instead of paper-canonical
    N=8 (S=2 hf_layers/block) for one purely operational reason:
    halving the per-rank block-cache memory (~3 GiB savings on the
    436M shape) so the AttnRes A/B can run at LOCAL_BS=3 SEQ=2048
    on 4× RTX 5090 32GB without sustained 97% memory utilization +
    CUDA allocation retries that ate ~30% of throughput in the N=8
    variant. On bigger memory boxes (H100/H200/B200) we'd revert to
    paper's canonical N=8.
    """
    from torchtitan.models.kimi_k3 import (
        KimiK3Spec,
        parallelize_kimi_k3,
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    cfg = _base_trainer_config("436m")
    kimi_config = build_kimi_linear_config("436m")
    spec_config = KimiK3Spec(kimi_config=kimi_config, num_blocks=4)
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_436m_block_attn_res_n4",
        model=spec_config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4() -> Trainer.Config:
    """447M Block AttnRes with SGLang-friendly head dims.

    Same scale as ``kimi_linear_436m_block_attn_res_n4`` — 16 layers,
    16 attention heads, 32 routed experts top-8, 1 shared expert,
    AttnRes N=4 (S=4 layers/block) — but with d_model=1024 (vs 1168)
    so head_dim=64 is divisible by 16. This unblocks SGLang inference
    on SM 12.0 (RTX 5090): the original 436M's head_dim=73 fails
    flashinfer's batch-prefill kernel + cuBLAS strided-batched bmm
    + Triton extend kernel autotune (cudaErrorMisalignedAddress /
    CUBLAS_STATUS_INTERNAL_ERROR / shared-memory OOM respectively).

    All other dims aligned to 8/16 multiples:
    * qk_nope=64, qk_rope=32, v_head=64
    * kv_lora_rank=512 (multiple of 64)
    * head_dim_qk = 96, head_dim_vo = 64 (both flashinfer-accepted)

    intermediate_size / moe_intermediate_size bumped 528 → 768 to keep
    the activated-param budget at ~447M, on par with the original
    436M scaling-law row's compute cost. Same lr (2.20e-3), batch size
    (384 sequences global), and total tokens budget (87.9B) inherited
    from the 436M row in SCALING_LAW_TABLE.

    Selected with
    ``CONFIG=kimi_linear_447m_aligned_block_attn_res_n4``. Runs through
    the same parallelize_fn / pipelining_fn / loss_fn as 436M.
    """
    from torchtitan.models.kimi_k3 import (
        KimiK3Spec,
        parallelize_kimi_k3,
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    cfg = _base_trainer_config("447m_aligned")
    kimi_config = build_kimi_linear_config("447m_aligned")
    spec_config = KimiK3Spec(kimi_config=kimi_config, num_blocks=4)
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_447m_aligned_block_attn_res_n4",
        model=spec_config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_447m_aligned_block_attn_res_n4_fp8() -> Trainer.Config:
    """447M Block AttnRes with FP8 rowwise training.

    Wraps :func:`kimi_linear_447m_aligned_block_attn_res_n4` and adds a
    Float8LinearConverter with the ``rowwise`` recipe. Excluded from the
    swap: every Linear inside a KDA layer (structurally, via
    KimiK3Float8Spec -- KDA and MLA share the ``self_attn`` name so
    no FQN substring can single out KDA), the MLA low-rank down-proj
    (``kv_a_proj_with_mqa``), the AttnRes projections, and the
    vocab/router heads -- those layers have either non-16-aligned
    shapes or numerical sensitivity that regresses under rowwise FP8.

    MoE experts (grouped_mm) stay bf16 — Float8GroupedMMConverter is a
    perf-prototype upstream and not in the dispatch path here.

    The Kimi Linear model is built as plain modules (KimiK3Spec),
    not from a ``Linear.Config`` tree, so ``Float8LinearConverter``'s
    config-traversal ``convert`` cannot apply. The converter is still
    built here for its torchao/SM89 validation and recipe resolution;
    the actual swap is module-level inside
    :class:`KimiK3Float8Spec.build` with the same filter semantics.

    Expected speedup on RTX 5090 (SM 12.0): 1.3-1.5× over bf16 for the
    dense MLA / projector / output paths; smaller win at the model level
    because KDA Triton + MoE grouped_mm dominate the per-step compute.
    """
    from torchtitan.components.quantization import Float8LinearConverter
    from torchtitan.models.kimi_k3.model import KimiK3Float8Spec

    cfg = kimi_linear_447m_aligned_block_attn_res_n4()
    converter = Float8LinearConverter.Config(
        recipe_name="rowwise",
        filter_fqns=[
            "lm_head",
            "router.gate",
            "kv_a_proj_with_mqa",
            "attn_res_proj",
            "mlp_res_proj",
            "final_attn_res_proj",
        ],
    ).build()
    if not converter.enabled:
        # torchao too old for recipe lookup; converter already warned.
        return cfg
    inner = cfg.model_spec.model
    cfg.model_spec.model = KimiK3Float8Spec(
        kimi_config=inner.kimi_config,
        num_blocks=inner.num_blocks,
        param_init=inner.param_init,
        torchao_float8_config=converter.torchao_config,
        filter_fqns=list(converter.config.filter_fqns),
    )
    return cfg


def kimi_k3_debugmodel_k3faithful() -> Trainer.Config:
    """Debug flavor with the K3-faithful architecture deltas ON:
    Gated MLA + alpha-graft Block AttnRes. CI-scale proof that the K3
    architecture (beyond the plain kimi_linear backbone) trains through
    the real trainer. MXFP4 QAT + Per-Head Muon are applied via their
    module/optimizer hooks (not config flags), see mxfp4_qat.py / muon.py.
    """
    import dataclasses as _dc

    cfg = kimi_k3_debugmodel()
    cfg.model_spec.flavor = "kimi_k3_debugmodel_k3faithful"
    m = cfg.model_spec.model
    # Gated MLA in K3's own parameterization (tech report Eq. 7: full-rank
    # channel-wise sigmoid gate, no bias). The graft flavors below keep
    # per_head_graft instead, where a step-0 no-op is the point.
    m.kimi_config = _dc.replace(
        m.kimi_config, mla_gated=True, attn_gate_param="full_rank"
    )
    m.attn_res_gated = True  # alpha graft
    return cfg


def kimi_k3_debugmodel_gated_lora() -> Trainer.Config:
    """Debug flavor with the full post-train graft stack: alpha-gated
    Block AttnRes + LoRA rank-8 (frozen base, alpha-fullparam
    exception). CI-scale rehearsal of the 48B LoRA leg.
    """
    cfg = kimi_k3_debugmodel()
    cfg.model_spec.flavor = "kimi_k3_debugmodel_gated_lora"
    cfg.model_spec.model.attn_res_gated = True
    cfg.model_spec.model.lora_rank = 8
    return cfg


def kimi_k3_mini_vl() -> Trainer.Config:
    """K3-faithful multimodal downscale: text k3mini plus a shrunk MoonViT-V2.

    K3 is natively multimodal, so a debug flavor that drops the vision tower
    misrepresents the architecture. But the tower at its released size is 447.4M
    parameters against k3mini's 80.9M text side -- 5.5x -- so a debug run would
    be dominated by the encoder rather than exercising the K3 structure under
    test.

    The tower is therefore SHRUNK, not simplified: 4 layers and hidden 256
    instead of 27 and 1024, while every structural feature of MoonViT-V2 is
    kept -- the single varlen attention pass (not the factorized one the report
    describes), 2D RoPE with the divided_fixed absolute embedding, sd2_tpool,
    and PatchMergerMLPV2. So the multimodal path is genuinely exercised: NaViT
    packing, the projector, the image_mask splice into the LM hidden states.

    Head count drops to 4 to keep head_dim at 64, matching the released tower.
    """
    import dataclasses as _dc

    from torchtitan.components.tokenizer import MultiModalTokenizer
    from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
    from torchtitan.models.kimi_k3.moonvit import MoonViTConfig
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalSpec

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_vl"
    kc = cfg.model_spec.model.kimi_config
    vision = MoonViTConfig(
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        qkv_hidden_size=384,
        intermediate_size=1024,
        text_hidden_size=kc.hidden_size,
    )
    # KimiK3MultimodalConfig, not KimiMultimodalConfig: it is the
    # release-faithful one. The projector belongs to the tower (mm_projector is
    # a MoonViT child in the checkpoint) and the tower is NOT frozen -- report
    # sec 2.4 trains MoonViT-V2 from scratch jointly with the text model, and
    # freezing it reproduces the opposite recipe.
    # The bundled tokenizer appends the media tokens ABOVE the 2016-token text
    # vocab (image 2016, vision_start 2017, vision_end 2018, pad 2019), so the
    # embedding must cover 2020 or every image row indexes out of range and the
    # run dies in a CUDA device-side assert. vision_token_id must be the
    # tokenizer's image id, not the LLaVA -200 default -- at -200 the sentinel
    # scan never matches and forward silently takes its text-only branch.
    import dataclasses as _dc2

    kc = _dc2.replace(kc, vocab_size=2020)
    cfg.loss = _dc2.replace(cfg.loss, global_vocab_size=2020)
    cfg.model_spec.model = KimiK3MultimodalSpec(
        kimi_config=kc,
        vision_config=vision,
        num_blocks=cfg.model_spec.model.num_blocks,
        vision_token_id=2016,
    )
    # Without these the flavor inherits the TEXT dataloader, which emits no
    # patches -- forward then takes its text-only branch and the tower never
    # runs, so a "multimodal" run silently validates nothing vision-side.
    # patch_size and spatial_merge_size must match MoonViTConfig's patch_size
    # and merge_kernel_size. The bundled test tokenizer already carries the
    # media tokens the collator needs.
    cfg.tokenizer = MultiModalTokenizer.Config(
        image_token="<|media_pad|>",
        video_token="<|media_pad|>",
        vision_start_token="<|media_begin|>",
        vision_end_token="<|media_end|>",
        pad_token="[PAD]",
    )
    cfg.dataloader = MMDataLoader.Config(
        dataset="cc12m-test",
        max_images_per_batch=8,
        patch_size=vision.patch_size,
        temporal_patch_size=1,
        spatial_merge_size=vision.merge_kernel_size[0],
        patch_order="raster",
        resize_fn=resize_to_patch_budget,
        max_patches=1024,
        max_patches_per_side=64,
        min_pixels=65536,
        max_pixels=1048576,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return cfg


def kimi_k3_mini_block_attn_res() -> Trainer.Config:
    """K3-FAITHFUL downscale: every structural choice is K3's, extents shrink.

    SiTU-GLU, Gated MLA with q-compression and a full-rank output gate, KDA with
    the lower-bounded decay (g_min = -5) and a full-rank output gate, Stable
    LatentMoE with 2 shared experts, AttnRes with K3's block size 12 over 21
    layers (2 blocks + a 9-layer tail, mirroring 93 = 7*12 + 9), and head_dim
    128 -- the last one deliberately, since FlashKDA requires K = V = 128, so
    this is the only small flavor that can exercise the official inference
    kernel.

    Use this, not debugmodel, whenever the question is "does it behave like K3".

    One deliberate deviation: vocab is the bundled 2016-token test tokenizer's,
    not K3's 163840, so the flavor runs without downloading assets. Vocab is an
    embedding EXTENT, not a mechanism -- nothing in the architecture branches on
    it -- whereas every structural choice above is K3's verbatim.
    """
    import dataclasses as _dc

    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

    cfg = _flavor_trainer_config("k3mini", "block_attn_res")
    cfg.model_spec.flavor = "kimi_k3_mini_block_attn_res"
    m = cfg.model_spec.model
    m.kimi_config = _dc.replace(
        build_kimi_linear_config("k3mini", vocab_size=2016),
    )
    cfg.loss = CrossEntropyLoss.Config(global_vocab_size=2016)
    cfg.hf_assets_path = "./tests/assets/tokenizer"
    return cfg


def kimi_k3_mini_diag_dense_mla() -> Trainer.Config:
    """DIAGNOSTIC: k3mini with no KDA and no MoE -- dense MLA only.

    The control that separates a TP BACKWARD bug from forward divergence. MoE
    top-k is discrete, so TP's different reduction order can flip an expert
    assignment and make the two runs genuinely different models from step one;
    that alone produces mismatched gradients with no bug anywhere. Removing both
    MoE and KDA leaves a path where TP is pure tensor sharding of dense matmuls,
    which MUST be numerically equivalent. A ratio above 1 here is a real backward
    defect. Not a training configuration.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_diag_no_kda()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_dense_mla"
    kc = cfg.model_spec.model.kimi_config
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=_dc.replace(kc, first_k_dense_replace=kc.num_hidden_layers),
    )
    return cfg


def _diag_single_layer(name: str, *, kda: bool, moe: bool) -> Trainer.Config:
    """DIAGNOSTIC builder: one layer, so amplification cannot run.

    One layer removes cross-layer amplification, so whatever difference remains
    is what TP introduces in a single forward/backward.

    The amplification is NOT a general property of this model, contrary to what
    this docstring used to claim: 21 dense layers with 8 AttnRes blocks sit at
    4e-4 under pure TP with no growth over depth. It appears only with MoE, which
    injects a ~1e-4 difference into the gradient stream when the reduction order
    changes, which AttnRes's uniform-at-init softmax then amplifies about 15x per
    layer. See TP_GRAD_FINDING_2026-07-29.

    Note the flip side, since it cost time to learn: one layer also pins
    num_blocks=1, which makes block_attn_res nearly degenerate. Anything about
    AttnRes verified here says nothing about the multi-block case -- use the
    _diag_multi_layer flavors for that.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = name
    kc = cfg.model_spec.model.kimi_config
    kc = _dc.replace(
        kc,
        num_hidden_layers=1,
        kda_layers=[1] if kda else [],
        full_attn_layers=[] if kda else [1],
        first_k_dense_replace=0 if moe else 1,
    )
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model, kimi_config=kc, num_blocks=1
    )
    return cfg


def _diag_multi_layer(
    name: str, *, num_layers: int, num_blocks: int | None, moe: bool = False
) -> Trainer.Config:
    """DIAGNOSTIC builder: N dense MLA layers with a real AttnRes block count.

    The single-layer builders pin ``num_blocks=1``, which makes block_attn_res
    nearly degenerate: the softmax runs over one block, so the pseudo-query
    gradient path that a real model exercises is barely touched. Anything
    verified only at one layer says nothing about the multi-block case. These
    keep every other knob identical and vary only the layer/block count, so an
    AttnRes defect that needs several blocks has room to appear.

    ``num_blocks=None`` disables AttnRes entirely -- the control leg.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = name
    kc = cfg.model_spec.model.kimi_config
    kc = _dc.replace(
        kc,
        num_hidden_layers=num_layers,
        kda_layers=[],
        full_attn_layers=list(range(1, num_layers + 1)),
        first_k_dense_replace=0 if moe else num_layers,
    )
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model, kimi_config=kc, num_blocks=num_blocks
    )
    return cfg


def kimi_k3_mini_diag_4l_mla() -> Trainer.Config:
    """Four dense MLA layers, 2 AttnRes blocks -- the smallest multi-block case."""
    return _diag_multi_layer("kimi_k3_mini_diag_4l_mla", num_layers=4, num_blocks=2)


def kimi_k3_mini_diag_4l_mla_noattnres() -> Trainer.Config:
    """Four dense MLA layers with AttnRes disabled -- control for the above."""
    return _diag_multi_layer(
        "kimi_k3_mini_diag_4l_mla_noattnres", num_layers=4, num_blocks=None
    )


def kimi_k3_mini_diag_8l_mla() -> Trainer.Config:
    """Eight dense MLA layers, 4 AttnRes blocks -- does the effect scale?"""
    return _diag_multi_layer("kimi_k3_mini_diag_8l_mla", num_layers=8, num_blocks=4)


def kimi_k3_mini_diag_1l_moe_depth() -> Trainer.Config:
    """Depth curve leg: 1 MLA+MoE layer. See _diag_multi_layer."""
    return _diag_multi_layer(
        "kimi_k3_mini_diag_1l_moe_depth", num_layers=1, num_blocks=1, moe=True
    )


def kimi_k3_mini_diag_4l_moe_depth() -> Trainer.Config:
    """Depth curve leg: 4 MLA+MoE layers, 2 AttnRes blocks."""
    return _diag_multi_layer(
        "kimi_k3_mini_diag_4l_moe_depth", num_layers=4, num_blocks=2, moe=True
    )


def kimi_k3_mini_diag_8l_moe_depth() -> Trainer.Config:
    """Depth curve leg: 8 MLA+MoE layers, 4 AttnRes blocks.

    With 1, 4 and 8 the deviation-vs-depth curve separates a per-layer defect
    (roughly linear in depth) from this model's ~1.6x per-layer amplification of
    any perturbation, bf16 included (geometric).
    """
    return _diag_multi_layer(
        "kimi_k3_mini_diag_8l_moe_depth", num_layers=8, num_blocks=4, moe=True
    )


def kimi_k3_mini_diag_4l_moe_8h() -> Trainer.Config:
    """Four MLA+MoE layers widened to 8 attention heads, so tp8 is possible.

    k3mini has 4 heads, which caps tp at 4 -- tp8 fails structurally with
    "Cannot unflatten unevenly sharded tensor", not because of a defect. The real
    2.8T config has far more heads, so the tp8 code path is reachable there and
    ought to be exercised somewhere.

    hidden_size is also widened to 1024: at 512 the per-rank shard under tp8 is
    small enough to trip "strides should be multiple of 16 bytes" inside the
    kernels, which is an alignment constraint of the shard width rather than a
    parallelism defect.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_diag_4l_moe_depth()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_4l_moe_8h"
    kc = cfg.model_spec.model.kimi_config
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=_dc.replace(
            kc,
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_size=1024,
            intermediate_size=2048,
        ),
    )
    return cfg


def kimi_k3_mini_pp8vp4() -> Trainer.Config:
    """32 layers, sized so PP8 admits VP=4.

    With first/last_stage_less_layers=0 the virtual-stage count equals
    n_layers // layers_per_stage; the AttnRes tail modules ride along with the
    stage owning their layers rather than counting separately. 32 layers at
    lps=1 gives 32 stages -- exactly 4 per rank. 21 layers admits no VP>=2
    split at all (21 is not divisible by 8 at any lps).

    Everything else matches kimi_k3_mini_block_attn_res.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_pp8vp4"
    kc = cfg.model_spec.model.kimi_config
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=_dc.replace(
            kc,
            num_hidden_layers=32,
            # All-MLA: pp8 needs dp_shard=1, and without FSDP's
            # mixed-precision cast the KDA params stay fp32 and fla's kernel
            # asks for 108,160 B of shared memory against this GPU's 101,376 B.
            # The PP/VP machinery under test is attention-type agnostic.
            kda_layers=[],
            full_attn_layers=list(range(1, 33)),
        ),
    )
    return cfg


def kimi_k3_mini_diag_21l_mla() -> Trainer.Config:
    """21 dense MLA layers, 8 AttnRes blocks -- full depth, AttnRes on."""
    return _diag_multi_layer("kimi_k3_mini_diag_21l_mla", num_layers=21, num_blocks=8)


def kimi_k3_mini_diag_21l_mla_noattnres() -> Trainer.Config:
    """21 dense MLA layers, AttnRes disabled -- the depth control.

    Separates "AttnRes is broken at depth" from "this model amplifies any
    perturbation ~1.6x per layer, so bf16 saturates by layer 21".
    """
    return _diag_multi_layer(
        "kimi_k3_mini_diag_21l_mla_noattnres", num_layers=21, num_blocks=None
    )


def kimi_k3_mini_diag_1l_mla_nogate() -> Trainer.Config:
    """One dense MLA layer with the Gated-MLA output gate DISABLED.

    attn_gate_proj is ColwiseParallel(use_local_output=True) and its INPUT is the
    replicated residual x, so the gradient it contributes back into x is Partial
    and has to be all-reduced across the tp axis. That is the same shape as the
    block_attn_res bug fixed earlier, where a bare to_local() defaulted the
    backward placement to Replicate and silently skipped the all-reduce.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_diag_1l_mla()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_1l_mla_nogate"
    kc = cfg.model_spec.model.kimi_config
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model, kimi_config=_dc.replace(kc, mla_gated=False)
    )
    return cfg


def kimi_k3_mini_diag_1l_mla_noattnres() -> Trainer.Config:
    """One dense MLA layer with AttnRes DISABLED.

    block_attn_res reads proj.weight directly and hand-rolls the backward grad
    placements (Partial on the tp axis, to force an all-reduce the default
    Replicate would skip). That code runs in every layer and is ours, which makes
    it the first thing to rule in or out for the ~6.5% per-layer TP gap.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_diag_1l_mla()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_1l_mla_noattnres"
    cfg.model_spec.model = _dc.replace(cfg.model_spec.model, num_blocks=None)
    return cfg


def kimi_k3_mini_diag_1l_mla() -> Trainer.Config:
    """One dense MLA layer: pure tensor sharding, must be TP-exact."""
    return _diag_single_layer("kimi_k3_mini_diag_1l_mla", kda=False, moe=False)


def kimi_k3_mini_diag_1l_mla_moe() -> Trainer.Config:
    """One MLA layer with MoE: adds discrete routing."""
    return _diag_single_layer("kimi_k3_mini_diag_1l_mla_moe", kda=False, moe=True)


def kimi_k3_mini_diag_1l_kda() -> Trainer.Config:
    """One KDA layer: adds the recurrence."""
    return _diag_single_layer("kimi_k3_mini_diag_1l_kda", kda=True, moe=False)


def kimi_k3_mini_diag_no_kda() -> Trainer.Config:
    """DIAGNOSTIC: k3mini with every layer full-attention (no KDA).

    Exists to isolate which module carries the TP gradient attenuation measured
    on 2026-07-29 (see TP_GRAD_FINDING). Everything else -- MoE, latent, AttnRes,
    FSDP, bf16 -- is held identical, so a ratio that returns to 1.0 here points
    at the KDA layers under TP. Not a training configuration.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_no_kda"
    kc = cfg.model_spec.model.kimi_config
    n = kc.num_hidden_layers
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=_dc.replace(
            kc, kda_layers=[], full_attn_layers=list(range(1, n + 1))
        ),
    )
    return cfg


def kimi_k3_mini_k3recipe() -> Trainer.Config:
    """K3-faithful structure AND K3's training recipe: Muon + Quantile Balancing.

    Structure alignment was verified module by module against the released
    reference; these two were the remaining recipe gaps. Kimi K3 trains with Muon
    on its matrix parameters (report sec 2.5) and with Quantile Balancing on the
    router (sec 2.3.3), while this repo defaulted to AdamW and core's sign rule.

    Deliberately a SEPARATE flavor rather than a change to
    kimi_k3_mini_block_attn_res. That flavor carries the cross-parallelism
    numerical baselines (PARALLEL_NUMERIC_BASELINE / PP_VP_REEXAMINATION), and
    changing its optimizer or router rule would invalidate every one of those
    recorded numbers. The baseline flavor stays a fixed reference; faithfulness
    lives here and in the 2p8t flavor.
    """
    import dataclasses as _dc

    from torchtitan.models.kimi_k3.muon import default_muon
    from torchtitan.models.kimi_k3.quantile_balance import register_quantile_balancing

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_k3recipe"
    cfg.model_spec.model = _dc.replace(cfg.model_spec.model, per_head_muon=True)
    cfg.optimizer = default_muon()
    cfg.model_spec.post_optimizer_build_fn = register_quantile_balancing
    return cfg


def kimi_k3_mini_muon() -> Trainer.Config:
    """K3-faithful structure trained with Per-Head Muon (report sec 2.5).

    Kimi K3 uses Muon for its matrix parameters, refined per attention head:
    instead of orthogonalizing the full Q/K/V projection, each head's block of the
    momentum matrix is orthogonalized separately, which equalizes the update scale
    across heads. Non-matrix parameters stay on AdamW.

    This is the last of the two training-recipe items that were implemented but
    unused -- the Muon optimizer and its tagger existed with nothing selecting
    them, which is the inert-feature pattern this phase spent days removing.
    """
    import dataclasses as _dc

    from torchtitan.models.kimi_k3.muon import default_muon

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_muon"
    cfg.model_spec.model = _dc.replace(cfg.model_spec.model, per_head_muon=True)
    cfg.optimizer = default_muon()
    return cfg


def kimi_k3_mini_qat_mxfp4() -> Trainer.Config:
    """K3-faithful QAT: MXFP4 routed-expert weights, MXFP8 expert activations.

    Report sec 4.1.4 runs QAT through the whole post-training stage (SFT and
    RL), quantizing only the MoE expert weights while attention projections,
    latent MoE projections, shared experts and routers stay in higher
    precision. The scope comes from quant_scope.py, which derives it from the
    released quantization_config rather than a hand-maintained name list.

    Fake-quant (dequant(quant(w)) with an STE) so this runs on any GPU; FP4
    hardware speeds deployment, not QAT.
    """
    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_qat_mxfp4"
    cfg.model_spec.model.mxfp4_qat = True
    return cfg


def kimi_k3_mini_qlora() -> Trainer.Config:
    """K3-faithful structure + LoRA rank 8 on the K3 module set.

    Exercises the updated target set: the compressed-Q pair (q_a_proj /
    q_b_proj), the Gated MLA output gate, and the latent MoE projections --
    none of which existed when DEFAULT_LORA_TARGETS was written.
    """
    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_qlora"
    cfg.model_spec.model.lora_rank = 8
    return cfg


def kimi_k3_debugmodel_pr_4025() -> Trainer.Config:
    """Architectural twin of pytorch/torchtitan#4025's kimi_k3_debugmodel.

    Same model on both sides, so the comparison is our parallelism against
    theirs rather than two different debug models. Every extent is read off
    that PR's _debugmodel: 13 layers at dim 256, 4 heads, q_lora 128 /
    kv_lora 64, qk_nope 32 / qk_rope 16 / v 32, full attention on layers
    {4, 8, 12} and KDA (head_dim 32, conv 4) elsewhere, AttnRes block size 12,
    LatentMoE with latent 128 / expert hidden 128 / 8 experts top-2 / 2 shared,
    dense FFN hidden 1024, vocab 163840, and a 4-layer 3-head MoonViT at
    dim 256 / qkv 384 / hidden 1024 with spatial merge 2.

    #4025 raises NotImplementedError on tensor, context and pipeline parallel
    ("Kimi K3 eager reference supports FSDP2 data parallelism only"), so on
    that side this config has exactly one runnable cell. Here it runs the
    whole matrix.
    """
    import dataclasses as _dc

    from torchtitan.components.tokenizer import MultiModalTokenizer
    from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
    from torchtitan.models.kimi_k3.moonvit import MoonViTConfig
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalSpec

    cfg = kimi_k3_mini_vl()
    cfg.model_spec.flavor = "kimi_k3_debugmodel_pr_4025"
    kc = _dc.replace(
        cfg.model_spec.model.kimi_config,
        num_hidden_layers=13,
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        vocab_size=163840,
        full_attn_layers=[4, 8, 12],
        # Must be derived, not inherited: k3mini's list has 15 entries and this
        # model has 13 layers, so carrying it over leaves the two descriptions
        # of the same stack contradicting each other.
        kda_layers=[i for i in range(1, 14) if i not in (4, 8, 12)],
    )
    vision = MoonViTConfig(
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=3,
        qkv_hidden_size=384,
        intermediate_size=1024,
        text_hidden_size=256,
    )
    cfg.model_spec.model = KimiK3MultimodalSpec(
        kimi_config=kc,
        vision_config=vision,
        num_blocks=cfg.model_spec.model.num_blocks,
        vision_token_id=cfg.model_spec.model.vision_token_id,
    )
    cfg.loss = _dc.replace(cfg.loss, global_vocab_size=163840)
    cfg.tokenizer = MultiModalTokenizer.Config(
        image_token="<|media_pad|>",
        video_token="<|media_pad|>",
        vision_start_token="<|media_begin|>",
        vision_end_token="<|media_end|>",
        pad_token="[PAD]",
    )
    cfg.dataloader = MMDataLoader.Config(
        dataset="cc12m-test",
        max_images_per_batch=8,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        patch_order="raster",
        resize_fn=resize_to_patch_budget,
        max_patches=1024,
        max_patches_per_side=64,
        min_pixels=56 * 56,
        max_pixels=1048576,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return cfg


def kimi_k3_mini_diag_4l_mla_lora() -> Trainer.Config:
    """Dense (no MoE) + AttnRes + LoRA rank 8 -- the LoRA gradient control.

    Every LoRA parallelism measurement so far used kimi_k3_mini_qlora, which has
    MoE, and MoE top-k routing flips under any numerical perturbation: a
    cross-layout gradient comparison there measures route divergence, not
    correctness. This flavor removes the confound so a LoRA gradient defect can
    be told apart from routing.
    """
    cfg = kimi_k3_mini_diag_4l_mla()
    cfg.model_spec.flavor = "kimi_k3_mini_diag_4l_mla_lora"
    cfg.model_spec.model.lora_rank = 8
    return cfg


def kimi_k3_mini_quantile_balance() -> Trainer.Config:
    """K3-faithful structure with Quantile Balancing driving the router bias.

    Replaces the auxiliary-loss-free sign rule with the solved-bias rule of
    report sec 2.3.3 (Eqs. 13-14). The hook goes on via post_optimizer_build_fn,
    the same extension point upstream models use for their own load-balancing
    hook; core's sign-rule hook stays registered because it is what keeps the
    expert_bias_E buffer allocated, and QB overwrites the bias afterwards.
    """
    from torchtitan.models.kimi_k3.quantile_balance import register_quantile_balancing

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_quantile_balance"
    cfg.model_spec.post_optimizer_build_fn = register_quantile_balancing
    return cfg


def kimi_k3_mini_kcp() -> Trainer.Config:
    """K3-faithful structure with KDA Context Parallelism (report sec 5.1.2).

    The sequence stays sharded across CP ranks end to end: a fixed-size halo for
    the short convolutions plus fla's prefix scan for the delta-rule state. The
    default kda_cp_mode is "ulysses", which all-to-alls the head axis instead
    and is kept as the A/B; see kcp.py for why both exist.

    Requires local_batch_size 1, because fla's cp_context packs a single varlen
    sequence.
    """
    import dataclasses as _dc

    cfg = kimi_k3_mini_block_attn_res()
    cfg.model_spec.flavor = "kimi_k3_mini_kcp"
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=_dc.replace(cfg.model_spec.model.kimi_config, kda_cp_mode="kcp"),
    )
    return cfg


def kimi_k3_2p8t_block_attn_res() -> Trainer.Config:
    """Kimi K3 at full scale, from the official config.json (2026-07-27).

    93 layers / hidden 7168 / 96 heads (head_dim 128) / 896 experts, top-16, 2
    shared / moe_intermediate 3072 in a 3584 latent / q_lora 1536 / kv_lora 512
    / vocab 163840 / 1M positions / dense FFN 33792 at layer 0 / full attention
    on [4, 8, ..., 88, 92, 93]. All 29 structural fields are asserted against
    the stored artifact in tests/test_k3_official_config.py.

    Needs >= 16 ranks and real hardware; it exists so scale-up is a config
    selection rather than a code change.
    """
    return _flavor_trainer_config("2p8t", "block_attn_res")


def kimi_k3_debugmodel_latentmoe() -> Trainer.Config:
    """Debug flavor with K3's Stable LatentMoE (report Eq. 11).

    Routed experts run in a latent of width ``routed_expert_hidden_size``
    (half of hidden here, mirroring K3's 3584-of-7168), entered/left through
    the shared down/up pair with an RMSNorm on the aggregate; the router
    still reads the full-width token. Shared experts stay full width.
    Carrier for the latent path at CI scale.
    """
    import dataclasses as _dc

    cfg = kimi_k3_debugmodel()
    cfg.model_spec.flavor = "kimi_k3_debugmodel_latentmoe"
    m = cfg.model_spec.model
    m.kimi_config = _dc.replace(
        m.kimi_config,
        routed_expert_hidden_size=m.kimi_config.hidden_size // 2,
        latent_moe_use_norm=True,
        num_shared_experts=2,  # K3 fixes Ns = 2
    )
    return cfg


def kimi_k3_debugmodel8h() -> Trainer.Config:
    """8-head debug flavor (d=512, H=8) for deep tp x cp meshes.

    The 4-head debugmodel binds at tp*cp=4 (MLA heads must divide
    tp*cp); this flavor enables tp2cp4 / tp4cp2 cells on 8 ranks.
    """
    import dataclasses as _dc

    cfg = kimi_k3_debugmodel()
    cfg.model_spec.flavor = "kimi_k3_debugmodel8h"
    kimi_config = build_kimi_linear_config(
        "debugmodel8h",
        num_experts=8,
        vocab_size=2016,
    )
    cfg.model_spec.model = _dc.replace(
        cfg.model_spec.model,
        kimi_config=kimi_config,
        num_blocks=resolve_num_blocks("debugmodel8h", "block_attn_res"),
    )
    return cfg


def kimi_k3_debugmodel_gated_qlora_mxfp4() -> Trainer.Config:
    """Debug QLoRA: gated_lora with the frozen base packed to MXFP4.

    Meta-first trainer flow: the model builds with the PACKED layout
    (base_qdata/base_scale, no base.weight), FSDP shards the packed
    bytes, and the quantized values load from a DCP checkpoint produced
    by an offline streaming quantizer from a bf16 run. CI-scale
    rehearsal of 48B QLoRA on small-VRAM fleets (no rank ever holds the
    full bf16 model).
    """
    cfg = kimi_k3_debugmodel_gated_lora()
    cfg.model_spec.flavor = "kimi_k3_debugmodel_gated_qlora_mxfp4"
    cfg.model_spec.model.lora_quantize_base = "mxfp4"
    return cfg


def kimi_linear_48b_block_attn_res_gated_lora() -> Trainer.Config:
    """48B graft + LoRA rank-16: the 5090-feasible post-training target
    (frozen 48B base sharded at ~12GB/card; only adapters + AttnRes
    params train)."""
    cfg = kimi_linear_48b_block_attn_res_gated()
    cfg.model_spec.flavor = "kimi_linear_48b_block_attn_res_gated_lora"
    cfg.model_spec.model.lora_rank = 16
    return cfg


def kimi_linear_48b_block_attn_res_gated() -> Trainer.Config:
    """48B Block AttnRes with the alpha graft gate enabled.

    The post-training graft flavor: load the official
    Kimi-Linear-48B-A3B weights into the backbone, keep the AttnRes
    params (pseudo-queries + alphas) zero-init -- at step 0 the model
    function EXACTLY equals the original checkpoint (alpha=0 identity);
    alpha then trains away from identity. Use the ungated
    kimi_linear_48b_block_attn_res for from-scratch pretraining.
    """
    cfg = _flavor_trainer_config("48b", "block_attn_res")
    cfg.model_spec.flavor = "kimi_linear_48b_block_attn_res_gated"
    cfg.model_spec.model.attn_res_gated = True
    return cfg


def kimi_k3_debugmodel() -> Trainer.Config:
    """Tiny CI flavor: 4 layers (3 KDA + 1 MLA), d=256, 8 experts,
    Block AttnRes, 2016-token bundled test tokenizer, c4_test dataset.

    Runs a few-step train smoke in seconds on 1 GPU (or a CPU forward
    via the fla fallback); meant for CI and quick regression checks,
    not a training target.
    """
    from torchtitan.models.kimi_k3 import (
        KimiK3Spec,
        parallelize_kimi_k3,
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiLinearStateDictAdapter
    from torchtitan.protocols.model_spec import ModelSpec

    kimi_config = build_kimi_linear_config(
        "debugmodel",
        num_experts=8,
        vocab_size=2016,
    )
    spec_config = KimiK3Spec(
        kimi_config=kimi_config,
        num_blocks=resolve_num_blocks("debugmodel", "block_attn_res"),
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=2016),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=ModelSpec(
            name="kimi_linear",
            flavor="kimi_k3_debugmodel",
            model=spec_config,
            parallelize_fn=parallelize_kimi_k3,
            pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
            post_optimizer_build_fn=None,
            state_dict_adapter=KimiLinearStateDictAdapter,
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        checkpoint=CheckpointManager.Config(interval=100),
        activation_checkpoint=None,
        # See _base_trainer_config: kimi CP requires contiguous seq shards.
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
    )


def kimi_k3_2p8t_block_attn_res_provisional() -> Trainer.Config:
    """PROVISIONAL K3 2.8T-A50B flavor (896 experts / 16 active, Block
    AttnRes). Config-level construction target only -- multi-node + EP
    to materialize; dims are placeholders pending the 7.27 config. Used
    for the 'scale-out is config-level' claim and EP@896 mesh checks.
    """
    return _flavor_trainer_config("2p8t", "block_attn_res")


def kimi_linear_528m_baseline() -> Trainer.Config:
    return _flavor_trainer_config("528m", "baseline")


def kimi_linear_528m_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "block_attn_res")


def kimi_linear_528m_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("528m", "full_attn_res")


# ----- Full Kimi Linear 48B-A3B carriers ---------------------------------- #
# Paper §"Training recipe": 27 transformer-blocks = 54 paper-layers,
# Block AttnRes N=9 (= 6 paper-layers per AttnRes-block = 3
# transformer-blocks per AttnRes-block). 48B total / 3B activated.
# Construction-only: requires multi-node + EP to actually train.
# Single-node use case is meta-device build / param-count sanity / PP
# layout planning, NOT actual gradient steps.


def kimi_linear_48b_baseline() -> Trainer.Config:
    return _flavor_trainer_config("48b", "baseline")


def kimi_linear_48b_block_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("48b", "block_attn_res")


def kimi_linear_48b_full_attn_res() -> Trainer.Config:
    return _flavor_trainer_config("48b", "full_attn_res")


# ----- 48B downscale variants (single-node feasibility sweep) ------------ #
# Paper 48B (256 experts × dim=2304) doesn't fit 8×32 GiB. These variants
# reduce num_experts (and optionally dim) while keeping n_layers=27 and
# N=9 (paper sweet spot, 3 t-blocks per AttnRes-block). Used to find the
# largest single-node-feasible carrier with paper-aligned architecture.


def _kimi_linear_48b_attnres_downscale(
    *,
    num_experts: int,
    dim: int | None = None,
    n_layers: int | None = None,
    num_blocks: int | None = None,
) -> Trainer.Config:
    """48B Block AttnRes with overridden num_experts (and optionally dim,
    n_layers, num_blocks).

    Defaults: n_layers=27, num_blocks=9 (paper sweet spot 3 t-blocks per
    AttnRes-block), seq_len=4096 (paper). Pass n_layers / num_blocks to
    deviate (e.g. n_layers=24, num_blocks=8 keeps the paper 3:1 ratio
    while making the depth divisible by PP=8 × VP=3 = 24 chunks).
    """
    from torchtitan.models.kimi_k3 import KimiK3Spec, parallelize_kimi_k3
    from torchtitan.models.kimi_k3.pipeline_adapter import (
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kwargs = {"num_experts": num_experts}
    kcfg = build_kimi_linear_config("48b", **kwargs)
    if dim is not None:
        kcfg.hidden_size = dim
        H = kcfg.num_attention_heads
        head_dim_aligned = max(32, (dim // H) & ~15)
        kcfg.qk_nope_head_dim = head_dim_aligned
        kcfg.qk_rope_head_dim = max(16, head_dim_aligned // 2)
        kcfg.v_head_dim = head_dim_aligned
        kcfg.kda_head_dim = head_dim_aligned
        kcfg.kv_lora_rank = (dim // 2) & ~63
        # Paper 48B dense FFN intermediate (layer 0 only) = 4 × dim.
        kcfg.intermediate_size = 4 * dim
    if n_layers is not None:
        kcfg.num_hidden_layers = n_layers
        # Re-derive KDA/MLA pattern with 3:1 ratio.
        kda_layers, full_attn_layers = _alternating_kda_mla_layers(
            n_layers,
            kda_mla_ratio=3,
        )
        kcfg.kda_layers = kda_layers
        kcfg.full_attn_layers = full_attn_layers

    final_num_blocks = num_blocks if num_blocks is not None else 9
    if n_layers is not None and n_layers % final_num_blocks != 0:
        raise ValueError(
            f"num_blocks={final_num_blocks} must divide n_layers={n_layers}"
        )
    spec_config = KimiK3Spec(kimi_config=kcfg, num_blocks=final_num_blocks)
    cfg = _base_trainer_config("48b")
    cfg.training.seq_len = 4096
    cfg.training.local_batch_size = 1  # single-node aggressive
    flavor_name = f"kimi_linear_48b_attnres_e{num_experts}"
    if dim is not None:
        flavor_name += f"_d{dim}"
    if n_layers is not None:
        flavor_name += f"_L{n_layers}"
    if num_blocks is not None:
        flavor_name += f"_N{num_blocks}"
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor=flavor_name,
        model=spec_config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_48b_block_attn_res_e32() -> Trainer.Config:
    """48B carrier, paper dim=2304, num_experts=32 (vs paper 256).
    First feasibility step.
    """
    return _kimi_linear_48b_attnres_downscale(num_experts=32)


def kimi_linear_48b_block_attn_res_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16)


def kimi_linear_48b_block_attn_res_e8() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=8)


def kimi_linear_48b_block_attn_res_d1280_e32() -> Trainer.Config:
    """48B layout (L=27, N=9) at narrower dim=1280, num_experts=32.
    Fallback if paper-dim variants don't fit.
    """
    return _kimi_linear_48b_attnres_downscale(num_experts=32, dim=1280)


def kimi_linear_48b_block_attn_res_d1280_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16, dim=1280)


def kimi_linear_48b_block_attn_res_d1024_e32() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=32, dim=1024)


def kimi_linear_48b_block_attn_res_d1024_e16() -> Trainer.Config:
    return _kimi_linear_48b_attnres_downscale(num_experts=16, dim=1024)


def kimi_linear_48b_block_attn_res_d1280_e32_L24_N8() -> Trainer.Config:
    """48B-layout carrier shrunk to L=24 (vs paper 27) so PP=8 × VP=3 = 24
    chunks divides cleanly. N=8 keeps paper sweet spot 3 transformer-blocks
    per AttnRes-block (24/8 = 3). dim=1280, num_experts=32. seq=2048.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=32,
        dim=1280,
        n_layers=24,
        num_blocks=8,
    )


def kimi_linear_48b_block_attn_res_d1280_e32_L32_N8() -> Trainer.Config:
    """48B-layout at L=32 N=8 (4 transformer-blocks per AttnRes-block,
    1.33× paper sweet spot). Allows PP=8 × VP=4 = 32 chunks × 1 layer.
    dim=1280, num_experts=32.

    NOTE: OOM at step 2 on 8×32 GiB (rank 7 hit 31.34 GiB after cache
    accumulation). Use the e16 variant below instead.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=32,
        dim=1280,
        n_layers=32,
        num_blocks=8,
    )


def kimi_linear_48b_block_attn_res_d1280_e16_L32_N8() -> Trainer.Config:
    """L=32 N=8 carrier with num_experts=16 (vs e32 OOM). Fits PP=8 ×
    VP=4 = 32 chunks paper-aligned, paper-sweet-spot t-blocks/AttnRes-block
    ratio off by 1.33×.
    """
    return _kimi_linear_48b_attnres_downscale(
        num_experts=16,
        dim=1280,
        n_layers=32,
        num_blocks=8,
    )


# ----- PP=4 V=2 lps=2 compatibility variant -------------------------------- #
# Paper's 528M has n_layers=17 (prime), which doesn't divide the 8 virtual
# stages needed by Interleaved1F1B PP=4 V=2 with lps=2. Drop to n_layers=16
# (one fewer layer) so the PP cache adapter layout tables build cleanly.
# All other 528M paper hyperparameters retained (d=1264, d_ff=560,
# lr=2.02e-3, batch=432). The KDA/MLA 3:1 alternation is re-derived for
# L=16 so 4 MLA layers land at the same relative positions.


def _build_528m_l16_config():
    """528M-like Kimi Linear config with n_layers=16 for PP=4 V=2 lps=2
    divisibility. d_model / d_ff / num_heads / LR all match paper's 528M.
    """
    cfg = build_kimi_linear_config("528m")
    cfg.num_hidden_layers = 16
    # Re-derive KDA:MLA = 3:1 pattern for 16 layers
    # (1-indexed). Period 4 → MLA at {4, 8, 12, 16}, KDA at the rest.
    period = 4
    cfg.kda_layers = [i for i in range(1, 17) if i % period != 0]
    cfg.full_attn_layers = [i for i in range(1, 17) if i % period == 0]
    return cfg


def kimi_linear_528m_l16_block_attn_res() -> Trainer.Config:
    """528M-scale Kimi Linear AttnRes with n_layers=16, Block AttnRes N=8.

    PP=4 V=2 lps=2 compatible (8 virtual stages on 4 ranks, 2 layers per
    stage). Every stage is a block boundary → cross-stage cache adapter
    exercised at every stage transition. Paper 528M d/d_ff/heads/LR
    retained; only depth reduced by 1 to satisfy the Interleaved1F1B
    divisibility requirement.
    """
    from torchtitan.models.kimi_k3 import KimiK3Spec, parallelize_kimi_k3
    from torchtitan.models.kimi_k3.pipeline_adapter import (
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiK3Spec(kimi_config=kcfg, num_blocks=8)
    cfg = _base_trainer_config("528m")  # paper 528M lr / batch template
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_block_attn_res",
        model=spec,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_528m_l16_full_attn_res() -> Trainer.Config:
    """528M-scale Kimi Linear Full AttnRes (num_blocks = n_layers = 16)."""
    from torchtitan.models.kimi_k3 import KimiK3Spec, parallelize_kimi_k3
    from torchtitan.models.kimi_k3.pipeline_adapter import (
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiK3Spec(kimi_config=kcfg, num_blocks=16)
    cfg = _base_trainer_config("528m")
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_full_attn_res",
        model=spec,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg


def kimi_linear_528m_l16_baseline() -> Trainer.Config:
    """528M-scale Kimi Linear baseline (no AttnRes) with n_layers=16.
    Paired control for the two AttnRes variants above.
    """
    from torchtitan.models.kimi_k3 import KimiK3Spec, parallelize_kimi_k3
    from torchtitan.models.kimi_k3.pipeline_adapter import (
        pipeline_kimi_k3_with_cache_adapter,
    )
    from torchtitan.protocols.model_spec import ModelSpec

    kcfg = _build_528m_l16_config()
    spec = KimiK3Spec(kimi_config=kcfg, num_blocks=None)
    cfg = _base_trainer_config("528m")
    cfg.model_spec = ModelSpec(
        name="kimi_linear",
        flavor="kimi_linear_528m_l16_baseline",
        model=spec,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
    return cfg
