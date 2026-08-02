# K3 5D refactor: imperative TP plan -> declarative sharding_config

## Scope: notation only

This is a **standardization**, not a redesign. Every module keeps its current
mathematics, its current placements and its current numerical behaviour. What
changes is where the sharding is *declared*: today it lives in an imperative
`parallelize_module(plan={...})` call in parallelize.py; upstream's convention is
a `sharding_config` on each `Linear.Config`, resolved declaratively.

**Explicitly out of scope -- do not touch:**

- `pipeline_adapter.py`. The PP adapter is verified per-parameter at 0.00000 over
  548 parameters across pp2, pp4, pp2xvp2 and pp4xvp2
  (PP_PERPARAM_VERIFICATION). It is the one piece of this stack with a clean
  bill of health at that resolution. It does not move.
- `attn_res.py`'s forward mathematics.
- `moonvit.py`, `kcp.py`, `muon.py`, `packed_mxfp4.py`, the MoE body.
- Anything in `torchtitan/models/common/` or `torchtitan/distributed/` beyond the
  one already-committed moe_sharding fix.

## Why this unblocks four open problems

Upstream's `LoRAConverter` derives adapter placements from the base linear's
`sharding_config` and subclasses the base linear, so its adapters never leave
DTensor. Our `KimiLoRALinear` wraps instead, hand-unwrapping both adapters, which
is what produced the o_proj defect. The converter cannot be adopted while K3's
linears have no `sharding_config` -- so the same refactor that standardizes the
notation is also the prerequisite for replacing the LoRA path.

Open problems this is meant to reach:

| problem | current state |
|---|---|
| LoRA under TP | o_proj.lora_b off by 1.34 |
| LoRA under PP | 0.105 weighted |
| LoRA under CP | 0.150 weighted |
| LoRA under FSDP / EP | already exact, must stay exact |
| full-parameter TP/PP/CP/EP | already <= 0.010, must stay there |

## Method: one module per step, smoke each, combine last

Each step migrates ONE module's sharding declaration and is gated on the
parallelism numbers for that module's axis not moving. A step that changes a
number is reverted, not explained away.

Gates, in order of what each step must reproduce:

1. per-parameter, warm checkpoint, one step, against the reference sharing the
   accumulation structure -- the instrument in MEASUREMENT_REGIMES
2. the full-parameter combination smoke (8-step loss curves, <= 0.010)
3. the LoRA matrix (FSDP and EP+FSDP exact; TP/PP/CP no worse than recorded)

Only after every module is migrated individually does the combined 5D smoke run.
Combining first would make a regression impossible to attribute, which is the
mistake that cost the most time in the TP investigation.

## Order

Attention linears (q/kv/o) -> FFN linears -> AttnRes projections -> embedding and
lm_head -> LoRA path last, since it consumes the sharding_config the earlier
steps introduce.
