# Kimi K3 (KDA + MLA + MoE + Block Attention Residuals)

Torchtitan implementation of the **Kimi K3 architecture family**: the
[Kimi-Linear](https://arxiv.org/pdf/2510.26692) backbone (Kimi Delta
Attention + MLA + sigmoid-gated MoE) with **Block Attention Residuals**
([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) woven in.
[Kimi K3](https://www.kimi.com/blog/kimi-k3) (2026-07-16) confirmed
AttnRes + KDA as production architecture components; open weights and the
tech report are due 2026-07-27, and this experiment's configs will be
aligned to the official release (structure details currently pending hold
placeholder interfaces).

> **Status (2026-07-18).** RFC
> [pytorch/torchtitan#3029](https://github.com/pytorch/torchtitan/issues/3029)
> was gated by reviewers on the Kimi K3 release -- that gate is now met. A
> follow-up RFC proposing this experiment is in preparation.

## What's in this folder

| File | Role |
| --- | --- |
| [`model.py`](./model.py) | K3 backbone: `KimiDeltaAttention` (KDA via `fla-core`), `KimiMLAAttention`, `KimiMoE`, `KimiDecoderLayer`, `KimiK3Model` |
| [`attn_res_model.py`](./attn_res_model.py) | `KimiK3AttnResModel`: AttnRes weave over the backbone (per-block-start RMSNorm + zero-init pseudo-queries) |
| [`attn_res.py`](./attn_res.py) | `block_attn_res()` primitive, `AttnResConfig`, `AttnResProjection`, `stack_blocks` / `unstack_blocks` |
| [`multimodal_model.py`](./multimodal_model.py) | `KimiK3LlavaMultimodalModel` + `KimiVisionProjector` (SigLIP-splice scaffold for the vision-native path) |
| [`parallelize.py`](./parallelize.py) | `parallelize_kimi_k3`: FSDP2/HSDP + TP + EP (CP blocked on fla-core `chunk_kda`) |
| [`pipeline_adapter.py`](./pipeline_adapter.py) | Cross-stage caching adapter + `pipelining_fn` (Interleaved1F1B), private to this experiment. Opt-in via `TORCHTITAN_ATTNRES_CACHE=1`. |
| [`layout.py`](./layout.py) | Static block-delta layout tables consumed by the PP adapter |
| [`model_configs.py`](./model_configs.py) | Architecture-side builders: AttnRes tech-report Table 2 scaling-law table (194m..528m), the SGLang-aligned 447m carrier, the 48B-A3B layout, `build_kimi_linear_config` |
| [`config_registry.py`](./config_registry.py) | Trainer configs for every `kimi_linear_<size>_<variant>` flavor (variants: baseline / block_attn_res / full_attn_res; + fp8 rowwise) |
| [`__init__.py`](./__init__.py) | `model_registry` -> `ModelSpec` (fla-core guarded) |
| [`tests/`](./tests/) | CPU unit tests: AttnRes primitive, KDA/MLA/MoE layers, AttnRes model, multimodal splice, pipeline-adapter wiring, all-flavor registry sweep |

## Running

```bash
# Unit tests (CPU; KDA falls back to fla-core's CPU path)
pytest torchtitan/models/kimi_k3/tests/ -v

# Single-node FSDP, 447M carrier
bash run_train.sh --module kimi_k3 --config kimi_linear_447m_aligned_block_attn_res_n4     --training.steps 100

# PP with the cross-stage cache adapter
TORCHTITAN_ATTNRES_CACHE=1 torchrun --nproc_per_node=4 ...     --module kimi_k3 --config kimi_linear_436m_block_attn_res     --parallelism.pipeline_parallel_degree 4     --parallelism.pipeline_parallel_schedule Interleaved1F1B
```

Dependencies: `pip install fla-core` (KDA kernels; CPU fallback exists for
tests, training needs the triton path).

## Design notes

- **Zero-init pseudo-queries.** AttnRes projections are zero-initialized so
  softmax weights are uniform at step 0 and the model is numerically
  equivalent to standard residuals on the first forward -- also the anchor
  for grafting AttnRes onto the released Kimi-Linear-48B checkpoint.
- **PP cross-stage cache adapter.** Producer stages publish each committed
  block once; consumers on the same rank read it back through a
  detached-leaf cache + gradient bridge, so backward through cached
  tensors does not double-accumulate into the producer. Delta mode sends
  only newly committed blocks.
- **CP is out of scope**: KDA needs Ulysses-style head sharding or
  LASP-style cross-rank state passing, neither of which fla-core provides
  today (same blank as `qwen3_5`).

## Evidence

Development history, pretraining runs, and PP pressure tests live in the
companion logbook repo
[QIU023/torchtitan_attention_residual](https://github.com/QIU023/torchtitan_attention_residual):

- **PP adapter numerics**: naive-vs-adapter |dLoss| <= 0.011 across PPxVP
  shapes up to PP=8 x VP=4 (32 virtual stages), incl. a 48B-layout
  carrier -- [pressure-test report](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md).
- **12.5K-step pretraining** on the 436M/447M shapes --
  [phase-4 log](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase4_kimi_attnres_lm_pretrain/README.md).
- **Dense A/B + adapter test grid**: the Llama3-shape/DSv3-shape AttnRes
  test carrier (paper Table 1 reproduction; 1460-line PP adapter test
  grid) was developed here and now lives at
  [phase3 `dense_carrier/`](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase3_attnres_pp_integration/dense_carrier)
  (runnable against fork history <= `666cf7ad6`).
- **HF reference blueprint** (`modeling_kimi.py`, for correctness diffs):
  [phase4 `hf_reference/`](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase4_kimi_attnres_lm_pretrain/hf_reference).

## Ownership

- Owner: [@QIU023](https://github.com/QIU023) -- open issues on the fork
  repo for technical questions.
