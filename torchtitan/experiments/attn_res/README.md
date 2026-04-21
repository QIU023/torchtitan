# Block Attention Residuals (AttnRes)

Implementation of **Block Attention Residuals** from *"Attention Residuals"*
(Kimi Team, 2026), [arXiv:2603.15031](https://arxiv.org/abs/2603.15031).

## Motivation

Standard residuals `h_{l+1} = h_l + f_l(h_l)` accumulate layer contributions
with equal weight, causing hidden-state magnitude to grow with depth and
diluting shallow-layer signal. AttnRes replaces the fixed add with softmax
attention over previous layer outputs, using a per-layer learned
pseudo-query vector. Block AttnRes is the practical variant: it partitions
the layer stack into `N` blocks, uses standard residuals inside each block,
and uses attention only at block boundaries — keeping memory and cross-stage
communication at `O(N d)` instead of `O(L d)`.

Empirically (paper Table 1 / Figure 3) this is equivalent to
baseline × 1.25 compute at matched size, with PP-friendly `N ≈ 8`.

## What's in this folder

| File | Role |
| --- | --- |
| [`attn_res.py`](./attn_res.py) | `block_attn_res()` primitive, `AttnResConfig`, `AttnResProjection` (pseudo-query, zero-initialized), `stack_blocks` / `unstack_blocks` |
| [`model.py`](./model.py) | `AttnResTransformerBlock` and `AttnResModel` (standalone dense classes inheriting only from the shared `torchtitan.models.common.decoder` bases — no Llama3 coupling) |
| [`__init__.py`](./__init__.py) | Model flavors: `debugmodel_attn_res`, `175M_attn_res`; `model_registry(flavor)` |
| [`config_registry.py`](./config_registry.py) | Trainer configs: `llama3_175m_baseline`, `llama3_175m_attn_res` (shared hyperparameters, only model flavor differs) |
| [`tests/`](./tests/) | CPU unit tests for the primitive, projection, stack/unstack, and end-to-end debug model |

Core torchtitan files are not modified: subclasses extend
`torchtitan.models.llama3.model.Llama3{Model,TransformerBlock}` and override
`forward` to route through `block_attn_res` when AttnRes kwargs are provided.

## Running

```bash
# Unit tests
pytest torchtitan/experiments/attn_res/tests/ -v

# Single-GPU debug forward
bash run_train.sh --module attn_res --config llama3_175m_baseline \
    --training.steps 100

bash run_train.sh --module attn_res --config llama3_175m_attn_res \
    --training.steps 100
```

## Design notes

- **Zero-init pseudo-queries.** Each `AttnResProjection.weight` is
  zero-initialized so that softmax weights are uniform at step 0 and the
  model is numerically equivalent to standard residuals for the first
  forward. Training stability depends on this.
- **FSDP dispatch.** `AttnResTransformerBlock.forward` routes through
  `__call__` (not `forward_attn_res` directly) when AttnRes kwargs are
  provided, so FSDP2's pre-forward `all_gather` hook fires on the block
  unit and AttnRes sub-params unshard before the `rms_norm` mul.
- **PP intermediate stage.** `AttnResModel.forward` returns
  `(partial_block, stack_blocks(blocks))` at non-last stages so
  `PipelineStage` sends both tensors via P2P. The last stage (identified
  by `self.output is not None`) applies a final cross-block aggregation
  before `norm` and `output`.

## Scope

This experiment is **Phase 2** of a larger effort. It demonstrates
algorithm correctness on a single GPU under FSDP. The core payoff — the
cross-stage caching adapter that hides AttnRes communication inside
interleaved 1F1B steady-state over PP + VP — is out of scope here and will
land in a follow-up.

## Ownership

- Owner: @QIU023
- Upstream contacts: @fegin, @wconstab (routing; not yet confirmed)
