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
| [`model.py`](./model.py) | `AttnResTransformerBlock` and `AttnResModel` (standalone classes inheriting only from the shared `torchtitan.models.common.decoder` bases — no Llama3 coupling; support both dense and per-layer MoE FFN via the DSv3 pattern) |
| [`__init__.py`](./__init__.py) | Flavor registry. Dense flavors use GQA + Llama3 parallelize; MoE flavors use MLA + DSv3 parallelize. See table below. |
| [`config_registry.py`](./config_registry.py) | Trainer configs (shared hyperparameters per family, only flavor name differs) |
| [`pipeline_adapter.py`](./pipeline_adapter.py) | Cross-stage caching adapter + custom `pipelining_fn`. Applies to any flavor when `TORCHTITAN_ATTNRES_CACHE=1`. |
| [`layout.py`](./layout.py) | Static block-delta layout tables used by the PP adapter. |
| [`tests/`](./tests/) | CPU unit tests: primitive / projection / stack-unstack / dense model / DSv3 MoE model / model-registry dispatch |

### Flavor families

| Family | Attention | FFN | `parallelize_fn` | Flavors |
| --- | --- | --- | --- | --- |
| Dense + GQA (Llama3 shape) | GQA | dense SwiGLU | `parallelize_llama` | `debugmodel_attn_res`, `175M_attn_res` (default N=6), `175M_attn_res_n{2,3,4,12}`, `175M_attn_res_L16_n8` |
| MoE + MLA (DSv3 shape) | MLA (DSv3) | first-N dense + rest MoE | `parallelize_deepseekv3` | `dsv3_debugmodel_attn_res`, `dsv3_16b_attn_res` (default N=9), `dsv3_16b_attn_res_n{3,27}` |

No MoE / MLA code is duplicated under this folder: the MoE module is
torchtitan's shared `torchtitan.models.common.moe.MoE`, and the MLA
attention reuses DSv3's own `torchtitan.models.deepseek_v3.model.Attention`
class verbatim (via the one-way `experiments → core` dependency rule in
`torchtitan/experiments/README.md`).

Core torchtitan files are not modified: `AttnResModel` inherits from the
shared `Decoder` base in `torchtitan/models/common/decoder.py`, not from
any model-family-specific class.

## Running

```bash
# Unit tests (all flavors, CPU)
pytest torchtitan/experiments/attn_res/tests/ -v

# Dense single-GPU: A/B baseline vs AttnRes at the same shape.
bash run_train.sh --module attn_res --config llama3_175m_baseline \
    --training.steps 100
bash run_train.sh --module attn_res --config llama3_175m_attn_res \
    --training.steps 100

# MoE A/B: use upstream DSv3 as the baseline (same shape, no AttnRes),
# this experiment's dsv3_attn_res_16b as the variant.
bash run_train.sh --module deepseek_v3 --config deepseek_v3_16b \
    --training.steps 100
bash run_train.sh --module attn_res --config dsv3_attn_res_16b \
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

Three tracks share this folder:

1. **Single-GPU dense correctness** (Llama3-proportioned flavors).
   `175M_attn_res` has a recorded 20 k-step loss curve vs baseline; see
   the RFC on [pytorch/torchtitan#3029](https://github.com/pytorch/torchtitan/issues/3029).
2. **PP cross-stage caching adapter**. Forward-delta correctness + 41/41
   CPU tests pass; 8-GPU end-to-end backward is still being
   stabilized — see the Status block in the RFC.
3. **MoE + MLA + AttnRes** (DSv3-shape flavors). Architectural match for
   Kimi's production line (paper §5: "MoE Transformer following the
   Moonlight / DeepSeek-V3 design"). CPU tests cover forward + backward
   on the debug flavor; 8-GPU MoE training validation is a follow-up.

## Ownership

- Owner: @QIU023
- Upstream contacts: @fegin, @wconstab (routing; not yet confirmed)
