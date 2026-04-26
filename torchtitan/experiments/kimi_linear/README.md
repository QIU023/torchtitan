# Kimi Linear (Phase 4)

Torchtitan-idiom port of [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) —
a MoE Transformer that interleaves Kimi Delta Attention (KDA) and Multi-head
Latent Attention (MLA) layers in a 3:1 ratio. Used as the platform for the
Block Attention Residuals (AttnRes) scaling-law sweep (paper Table 2:
194M → 528M activated params) and the eventual 48B-A3B target.

> **Status (2026-04-26).** Phases 4a → 4c complete. 436M FSDP overnight
> baseline ran 12,500 steps; PP=4 cache-adapter 12,500-step run completed
> against the same flavor. The 12.5 k ckpt is **architecture-validation
> grade**, not pretraining-quality (~0.35% of the paper's 119 B-token Table 2
> budget).

## Files

- `reference/` — verbatim fork from HF `moonshotai/Kimi-Linear-48B-A3B-Base`:
  - `modeling_kimi.py` (1028 lines)
  - `configuration_kimi.py` (140 lines)
  - `config.json` (48B-A3B reference config)

  These are the blueprint. **Not imported** by torchtitan code — they assume
  HF Transformers, HF Cache, HF generation utilities. We port the essential
  forward paths into torchtitan's `Module.Config` pattern below. Keep as-is
  for future correctness diffs.

- `model.py` — torchtitan-idiom port:
  - `KimiDeltaAttention` (KDA via `fla-core` ops)
  - `KimiMLAAttention` (NoPE MLA, faithful to Kimi spec — not reusing DSv3 MLA;
    see `phase4/README.md` for rationale)
  - `KimiMLP` (SwiGLU dense FFN for layer 0 + shared-experts)
  - `KimiMoE` (wraps torchtitan's `TokenChoiceTopKRouter` + `GroupedExperts`
    for Kimi's sigmoid-gated grouped-topk routing)
  - `KimiDecoderLayer` (block orchestration: layernorm → attn → residual →
    layernorm → MoE/MLP → residual)
  - `KimiLinearModel` (embedding + stack of layers + final norm + LM head)

- `attn_res_model.py` *(Phase 4c)* — `KimiLinearAttnResModel` subclass that
  adds per-block-start RMSNorm + zero-init pseudo-query (the AttnRes pattern
  from [`../attn_res/`](../attn_res/)). Exposes `_return_only_new_blocks` so
  the Phase-3 PP cache adapter drives it unchanged.

- `multimodal_model.py` *(Phase 4e scaffold)* — `KimiLinearMultimodalModel`
  + `KimiVisionProjector`, using `vision_token_id=-200` (LLaVA convention)
  to splice image-token embeddings into the text stream. Architecture-only:
  not pretraining-quality (the LLM ckpt is too weak; see logbook).

- `parallelize.py` — `parallelize_kimi_linear`: FSDP2 + EP for Kimi's MoE
  layers, with the AttnRes block boundary respected for FSDP unit splits.

- `pipeline_adapter.py` — same cross-stage cache adapter as
  [`../attn_res/`](../attn_res/), wired to Kimi Linear's block boundaries.

- `config_registry.py` *(Phase 4c)* — registered flavors:
  - `kimi_linear_debug` / `kimi_linear_debug_attn_res` (small test config)
  - `kimi_linear_194m_attn_res` through `kimi_linear_528m_attn_res`
    (paper Table 2 scaling-law sizes)
  - `kimi_linear_436m_baseline` / `kimi_linear_436m_attn_res` (paper-native
    L=16, used for the overnight runs)
  - `kimi_linear_48b_attn_res` (faithful to 48B-A3B ref config; flavor only,
    not run on this hardware)

- `tests/` — CPU smoke tests:
  - `test_layers.py` — KDA / MLA / MoE / decoder-layer forward shapes
  - `test_attn_res_model.py` — AttnRes wrapper forward+backward
  - `test_model_spec.py` — registry dispatch to `parallelize_kimi_linear`
  - `test_pipeline_adapter.py` — same-rank cache lifecycle
  - `test_multimodal_model.py` — projector + token-splice forward

## Dependencies

```bash
pip install fla-core  # provides KDA kernel
```

Provides: `fla.ops.kda.chunk_kda`, `fla.ops.kda.fused_recurrent_kda`,
`fla.ops.kda.gate.fused_kda_gate`, `fla.modules.ShortConvolution`,
`fla.modules.FusedRMSNormGated`. Confirmed importable on this box
(fla-core 0.5.0).

## Reproduction

Launchers live in the outer logbook repo:

- `phase4/launch_fsdp_small.sh` — 4-GPU FSDP, AttnRes wrapper on/off
- `phase4/launch_pp4_kimi.sh` — 4-GPU PP=4 V=2 with the cache adapter

The 436M overnight runs use `--training.steps 12500`, `LOCAL_BS=4`,
`GLOBAL_BS=12`, seqlen 2048 → ~307 M training tokens. Sustained throughput
on 4× RTX 5090 PCIe: ~3.2 k tps / 47.5 k tflops / ~78% memory.

## Why ckpt is "architecture-grade", not "pretraining-grade"

- AttnRes paper's 436M scaling-law row (Table 2): **119 B tokens** — about
  14× Chinchilla-optimal.
- Chinchilla-optimal for 432M params: 432M × 20 ≈ **8.6 B tokens**.
- 4× RTX 5090 sustained: ~280 M tokens per 12 h overnight.
- Reaching even Chinchilla ≈ 15-20 days continuous; reaching paper-spec ≈ ~150
  days. Neither is feasible on this hardware.

So the contribution this folder makes is **the integration scaffolding** —
KDA + MLA + MoE + AttnRes + PP cache adapter + FSDP2/EP parallelize plan +
multimodal token-splicing — all running end-to-end inside torchtitan, ready
for whoever has the H100-days to actually pretrain it.

## Phase log

- **Phase 4a** — skeleton + reference files + phase4 plan doc
- **Phase 4b** — port `model.py`, CPU smoke for KDA/MLA forward shapes,
  17 k-step single-shape FSDP smoke (`init_weights` fix)
- **Phase 4c** — `attn_res_model.py` + scaling-law flavors + PP adapter
  integration tests; `grouped_mm` + `torch.compile` carve-out tightening
- **Phase 4d** — 436M FSDP baseline + PP=4 cache-adapter overnight runs
  (12,500 steps, both completed); `MLA cuDNN attention backend + TF32` bump
  later reverted (default cutlass is correct)
- **Phase 4e (current)** — multimodal scaffolding (`multimodal_model.py`)
  + realistic-target writeup
