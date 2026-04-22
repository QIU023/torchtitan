# Kimi Linear experiment (Phase 4)

Torchtitan-idiom port of [MoonshotAI/Kimi-Linear](
https://github.com/MoonshotAI/Kimi-Linear), a MoE Transformer that
interleaves Kimi Delta Attention (KDA) and Multi-head Latent Attention
(MLA) layers in a 3:1 ratio. Used as the platform for the AttnRes
scaling-law sweep (paper Table 2: 194M → 528M activated params) and
the 48B-A3B upscale target.

## Files

- `reference/` — verbatim fork from HF
  `moonshotai/Kimi-Linear-48B-A3B-Base`:
  - `modeling_kimi.py` (1028 lines)
  - `configuration_kimi.py` (140 lines)
  - `config.json` (48B-A3B reference config)

  These are the blueprint. **Not imported** by torchtitan code — they
  assume HF Transformers, HF Cache, HF generation utilities. We port
  the essential forward paths into torchtitan's `Module.Config`
  pattern below. Keep as-is for future correctness diffs.

- `model.py` — torchtitan-idiom port:
  - `KimiDeltaAttention` (KDA via fla-core ops)
  - `KimiMLAAttention` (NoPE MLA, faithful to Kimi spec — not reusing
    DSv3 MLA, see `phase4/README.md` for rationale)
  - `KimiMLP` (SwiGLU dense FFN for layer 0 + shared-experts)
  - `KimiMoE` (wraps torchtitan's `TokenChoiceTopKRouter` +
    `GroupedExperts` for Kimi's sigmoid-gated grouped-topk routing)
  - `KimiDecoderLayer` (block orchestration: layernorm → attn →
    residual → layernorm → MoE/MLP → residual)
  - `KimiLinearModel` (embedding + stack of layers + final norm +
    LM head)

- `attn_res_model.py` *(Phase 4c)* — `KimiLinearAttnResModel` subclass
  that adds per-block-start RMSNorm + zero-init pseudo-query
  (AttnRes pattern from `attn_res/`). Exposes `_return_only_new_blocks`
  so the Phase-3 PP cache adapter can drive it unchanged.

- `config_registry.py` *(Phase 4c)* — registered flavors:
  - `kimi_linear_debug` / `kimi_linear_debug_attn_res` (small test config)
  - `kimi_linear_194m_attn_res` through `kimi_linear_528m_attn_res`
    (paper Table 2 scaling-law sizes)
  - `kimi_linear_48b_attn_res` (faithful to 48B-A3B ref config)

- `tests/` — CPU smoke tests for layer-level correctness.

## Dependencies

```bash
pip install fla-core  # provides KDA kernel
```

Provides: `fla.ops.kda.chunk_kda`, `fla.ops.kda.fused_recurrent_kda`,
`fla.ops.kda.gate.fused_kda_gate`, `fla.modules.ShortConvolution`,
`fla.modules.FusedRMSNormGated`. Confirmed importable on this box
(fla-core 0.5.0).

## Reproduction

*(Populated in Phase 4c/4d.)* The launcher scripts will mirror
`phase3/launch_4gpu_*.sh` structure, wiring
`pipeline_llm_with_cache_adapter` as the `pipelining_fn` on the
AttnRes flavors.

## Status

- **Phase 4a** (current): skeleton + reference files + phase4 plan doc.
- **Phase 4b**: port `model.py`, CPU smoke for KDA/MLA forward shapes.
- **Phase 4c**: `attn_res_model.py` + scaling-law flavors + PP adapter
  integration tests.
- **Phase 4d**: 4-GPU PP smoke of debug flavor.
- **Phase 4e** (far future): rent multi-node, run scaling-law sweep.
