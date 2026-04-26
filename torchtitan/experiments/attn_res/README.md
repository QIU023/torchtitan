# Block Attention Residuals (AttnRes)

Reference implementation of **Block Attention Residuals** from
[*Attention Residuals* (Kimi Team, 2026), arXiv:2603.15031](https://arxiv.org/abs/2603.15031),
inside torchtitan.

> **Status (2026-04-26).** RFC [pytorch/torchtitan#3029](https://github.com/pytorch/torchtitan/issues/3029)
> filed; reviewers asked to gate upstream merge on the **Kimi K3** release. Until
> then this fork is the canonical reference implementation. The dense Llama3 A/B
> evidence (paper Table 1 reproduction) and the cross-stage caching adapter for
> `Interleaved1F1B` PP (paper §4.1) are both functional and tested here.

## Motivation

Standard residuals `h_{l+1} = h_l + f_l(h_l)` accumulate layer contributions with
equal weight, causing hidden-state magnitude to grow with depth and diluting
shallow-layer signal. AttnRes replaces the fixed add with softmax attention over
previous layer outputs, using a per-layer learned pseudo-query. **Block AttnRes**
is the practical variant: partition the layer stack into `N` blocks, use standard
residuals inside each block, attention only at block boundaries — keeping memory
and cross-stage communication at `O(N d)` instead of `O(L d)`.

Empirically (paper Table 1 / Figure 3) AttnRes ≈ baseline × 1.25 effective
compute at matched param count, with PP-friendly `N ≈ 8`.

## What's in this folder

| File | Role |
| --- | --- |
| [`attn_res.py`](./attn_res.py) | `block_attn_res()` primitive, `AttnResConfig`, `AttnResProjection` (pseudo-query, zero-initialized), `stack_blocks` / `unstack_blocks` |
| [`model.py`](./model.py) | `AttnResTransformerBlock` and `AttnResModel` — standalone classes inheriting only from the shared `torchtitan.models.common.decoder` bases (no Llama3 coupling); supports both dense GQA FFN and per-layer MoE-MLA via the DSv3 pattern |
| [`__init__.py`](./__init__.py) | Flavor registry. Dense flavors → GQA + `parallelize_llama`; MoE flavors → MLA + `parallelize_deepseekv3`. Routing is automatic from the flavor name. |
| [`config_registry.py`](./config_registry.py) | Trainer configs (shared hyperparameters per family, only flavor name differs) |
| [`pipeline_adapter.py`](./pipeline_adapter.py) | Cross-stage caching adapter + custom `pipelining_fn`. Activates on any flavor when `TORCHTITAN_ATTNRES_CACHE=1`. |
| [`layout.py`](./layout.py) | Static block-delta layout tables consumed by the PP adapter |
| [`tests/`](./tests/) | CPU unit tests: primitive / projection / stack-unstack / dense model / DSv3 MoE model / model-registry dispatch |

### Flavor families

| Family | Attention | FFN | `parallelize_fn` | Flavors |
| --- | --- | --- | --- | --- |
| Dense + GQA (Llama3 shape) | GQA | dense SwiGLU | `parallelize_llama` | `debugmodel_attn_res`, `175M_attn_res` (default N=6), `175M_attn_res_n{2,3,4,12}`, `175M_attn_res_L16_n8` |
| MoE + MLA (DSv3 shape) | MLA (DSv3) | first-N dense + rest MoE | `parallelize_deepseekv3` | `dsv3_debugmodel_attn_res`, `dsv3_16b_attn_res` (default N=9), `dsv3_16b_attn_res_n{3,27}` |

No MoE / MLA code is duplicated under this folder. The MoE module is
torchtitan's shared `torchtitan.models.common.moe.MoE`, and MLA reuses DSv3's
own `torchtitan.models.deepseek_v3.model.Attention` class verbatim — via the
one-way `experiments → core` dependency rule documented in
`torchtitan/experiments/README.md`.

Core torchtitan files are not modified: `AttnResModel` inherits from the shared
`Decoder` base in `torchtitan/models/common/decoder.py`, not from any
model-family-specific class.

## Running

```bash
# Unit tests (all flavors, CPU)
pytest torchtitan/experiments/attn_res/tests/ -v

# Dense single-GPU: A/B baseline vs AttnRes at matched shape.
bash run_train.sh --module attn_res --config llama3_175m_baseline \
    --training.steps 100
bash run_train.sh --module attn_res --config llama3_175m_attn_res \
    --training.steps 100

# MoE A/B: upstream DSv3 as the baseline (same shape, no AttnRes),
# this experiment's dsv3_attn_res_16b as the variant.
bash run_train.sh --module deepseek_v3 --config deepseek_v3_16b \
    --training.steps 100
bash run_train.sh --module attn_res --config dsv3_attn_res_16b \
    --training.steps 100

# 4-GPU PP=4 V=2 with the cross-stage caching adapter.
# Launchers in the outer logbook:
#   <logbook>/phase3/launch_4gpu_naive.sh   (PP, no cache)
#   <logbook>/phase3/launch_4gpu_adapter.sh (PP, with cache adapter)
```

## Design notes

- **Zero-init pseudo-queries.** Each `AttnResProjection.weight` is
  zero-initialized so softmax weights are uniform at step 0 and the model is
  numerically equivalent to standard residuals on the first forward. Training
  stability depends on this.
- **FSDP dispatch.** `AttnResTransformerBlock.forward` routes through
  `__call__` (not `forward_attn_res` directly) when AttnRes kwargs are
  provided, so FSDP2's pre-forward `all_gather` hook fires on the block unit
  and AttnRes sub-params unshard before the `rms_norm` mul.
- **PP intermediate stage.** `AttnResModel.forward` returns
  `(partial_block, stack_blocks(blocks))` at non-last stages so `PipelineStage`
  sends both tensors via P2P. The last stage (identified by
  `self.output is not None`) applies a final cross-block aggregation before
  `norm` and `output`.
- **Cross-stage caching adapter.** Producer stages publish each completed
  block's output once via `_LocalCacheAugment`'s detached-leaf cache; consumer
  stages on the same rank read it back through a hook + `register_hook` bridge
  so the second-order backward pass through the cached tensor doesn't double-
  accumulate gradients into the producer stage.

## Evidence

Three tracks share this folder.

### 1. Single-GPU dense correctness — paper Table 1 reproduction

174M Llama3 dense, FSDP, 20 k steps on C4-en, identical hyperparameters; only
`model_spec` differs:

| step | baseline | attn_res | Δ |
|---:|---:|---:|---:|
| 500   | 6.1412 | 6.0146 | −0.1265 |
| 5000  | 4.3575 | 4.2696 | −0.0879 |
| 10000 | 4.3235 | 4.2192 | −0.1043 |
| 15000 | 3.7368 | 3.6861 | −0.0507 |

AttnRes is consistently below baseline at every logged milestone — consistent
with the paper's "≈ baseline × 1.25 compute" range.

### 2. PP cross-stage caching adapter — 4-GPU PP=4 V=2, 1000 steps

`llama3_175m_attn_res_L16_n8` (16 layers / 8 blocks), `layers_per_stage=2` —
every stage boundary is a block boundary, so the cache adapter fires at every
transition.

- `|Δ_naive→adapter|` max **0.06** at step 1000
- `|Δ_naive→naive|` (seed-vs-seed nondeterminism band) max **0.13**
- Memory accounting matches design: +260 MB cache on rank 3 for 175M at M=4 mb
- 41 / 41 CPU unit tests green

What is **not** yet shown:

- ≥ 5 k step PP horizon stability,
- PP=8 scale-up,
- AttnRes-vs-baseline delta preservation under PP,
- the 1.5–2 B PCIe-overhead headline plot.

These are the natural next experiments — gated on multi-node access, not on the
algorithm or adapter.

### 3. MoE + MLA + AttnRes (DSv3-shape) — early scaffolding, superseded

The DSv3-shape flavors here (`dsv3_*_attn_res`) were the first MoE+MLA+AttnRes
integration pass: CPU forward+backward tests pass on the debug flavor, and the
flavors register cleanly through `parallelize_deepseekv3`. Track 3 was always a
stepping-stone toward the Kimi-production-aligned shape.

**The actual end-to-end MoE+MLA+AttnRes results live in the sibling
[`../kimi_linear/`](../kimi_linear/) experiment** — KDA + MLA + sigmoid-gated
MoE + AttnRes wrapper, with the cross-stage cache adapter wired through.
Phase 4 ran a 436M FSDP overnight baseline + a PP=4 cache-adapter overnight
run (both 12,500 steps). For Kimi-K3-shape work, start there; the DSv3-shape
flavors here remain useful as the minimal MoE smoke target.

## Why this lives here as a "reference implementation"

- The single-GPU evidence (Track 1) reproduces paper Table 1 numbers
  independently inside torchtitan.
- The PP adapter (Track 2) is an integration story that's hard to reconstruct
  from the paper alone — the cache lifecycle and the second-order backward fix
  are non-obvious.
- The MoE+MLA scaffolding (Track 3) is the production-aligned shape Kimi will
  almost certainly ship with K3.

Reviewers on RFC [#3029](https://github.com/pytorch/torchtitan/issues/3029)
asked to defer upstream merge until K3 lands. Until then, anyone who wants
AttnRes inside torchtitan can pull this fork instead of re-implementing from
the paper.

## Ownership

- Owner: [@QIU023](https://github.com/QIU023) — yiqiao.lbj23@gmail.com
