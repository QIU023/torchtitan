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

## Step 1 mapping: attention linears

Derived from the imperative plan in parallelize.py (lines ~640-690). The
declarative form must reproduce these exactly -- this table is the contract, and
any step that changes a placement is a bug in the migration, not an improvement.

| module | imperative today | declarative sharding_config (weight) |
|---|---|---|
| `q_proj` (no q_lora) | `ColwiseParallel(use_local_output=False)` | `tp=spmd.S(0)` |
| `q_a_proj` | `NoParallel()` | `tp=spmd.R` |
| `q_a_layernorm` | `NoParallel()` | `tp=spmd.R` |
| `q_b_proj` | `ColwiseParallel(use_local_output=False)` | `tp=spmd.S(0)` |
| `kv_a_proj_with_mqa` | `NoParallel()` | `tp=spmd.R` |
| `kv_a_layernorm` | `NoParallel()` | `tp=spmd.R` |
| `kv_b_proj` | `ColwiseParallel(use_local_output=False)` | `tp=spmd.S(0)` |
| `o_proj` | `RowwiseParallel(output_layouts=Replicate(), use_local_output=True)` | `tp=spmd.S(1)` |
| `attn_gate_proj` | `ColwiseParallel(use_local_output=True)` | `tp=spmd.S(0)` |

Two details the table alone does not carry, both load-bearing:

- `use_local_output` differs across these. q/kv expansions keep `False` so the
  downstream split into `[kv_lora, qk_rope]`, the `kv_a_layernorm` and the `cat`
  with `k_pass_expanded` all stay in DTensor space. `o_proj` and
  `attn_gate_proj` use `True` because the model's boundary convention is plain
  tensors (PP P2P, AttnRes stacking, fla kernels). The declarative form has to
  preserve which is which, not just the placements.
- `attn_gate_proj` shards on the head axis under BOTH parameterizations -- the
  per-head variant is `[num_heads]` and K3's full-rank variant is
  `[num_heads * v_head_dim]`. Colwise keeps the local gate matched to the local
  attention output either way.

## Step 1 prerequisite discovered

K3's attention linears are built with POSITIONAL arguments
(`self.q_a_proj = Linear(in, out, bias=False)` at model.py:467-507) against a
local `Linear(_TTLinear)` subclass, not with `Linear.Config(...).build()`.
`sharding_config` lives on the Config, so step 1 is really two changes: convert
the construction to the Config form first, then attach the sharding.

That first half is pure notation with no placement content, so it can be gated
on a stricter condition than the rest of the refactor: the parallelism numbers
must be BIT-IDENTICAL, not merely within tolerance. If converting a constructor
moves any digit, the conversion is wrong.

## Step 2 status: driver wired, boundary convention not yet expressed

Done and working:

- `_apply_declarative_sharding` walks the tree and parallelizes every leaf that
  declares a `sharding_config`. Upstream calls `model.parallelize(parallel_dims)`
  directly, but K3's top-level `KimiLinearModel` is a plain `nn.Module` rather
  than the protocol `Module`; making it one pulls in Config/Configurable and
  reaches past "standardize the notation", so the walk does the same work.
- Leaves only, and the `_moe` subtree is excluded because `moe.parallelize()`
  owns it -- its shared experts are `KimiMLP` and carry the dense-FFN
  declaration, so they would be parallelized twice. Same exclusion the
  imperative plan already makes.
- 37 modules parallelized declaratively; the run completes.

Blocking the rest:

With the migrated entries removed from the imperative plan, tp2 fails with
`aten.mul.Tensor got mixed torch.Tensor and DTensor`. That is the boundary
convention, and it is the substance the mapping table warned about rather than a
loose end: K3 keeps module boundaries as PLAIN tensors so PP P2P, AttnRes's
`torch.stack` and fla's triton kernels never meet a DTensor. The imperative plan
encodes that per module via `use_local_output=True` / `output_layouts=Replicate()`.
A `sharding_config` with only `state_shardings` says how the WEIGHT is placed and
says nothing about the output type, so the wrapped forward now returns DTensors
into plain-tensor code.

Expressing it declaratively means `out_src_shardings` / `out_dst_shardings` on
each config, matching what each module's imperative entry specified -- and those
differ within the attention block (q/kv expansions keep DTensor outputs, o_proj
and attn_gate_proj go plain), so it is per-module transcription, not one global
setting.

Numbers at this checkpoint, for the record. With both mechanisms active on the
same modules (before removing the plan entries) tp2 gave
7.69966 / 6.98268 / 6.10026 against the imperative baseline
7.70006 / 7.04439 / 6.16003 -- i.e. double application changes the numerics, as
expected, which is why the plan entries have to come out in the same step that
the declarations go in.

## Step 3: the declarative driver is reverted, and why that is the right answer

The declarative vocabulary has no `use_local_output`. `Module.parallelize()`
works entirely in DTensor space, and `ShardingConfig` carries
`state_shardings` / `in_*` / `out_*` / `local_map` -- none of which can say "hand
the next module a plain tensor". K3's boundary convention is exactly that, and it
exists for three reasons that cannot be given up: PP P2P sends raw tensors,
`block_attn_res` stacks plain and DTensor operands, and fla's triton kernels
dispatch on data pointers. The `aten.mul.Tensor got mixed torch.Tensor and
DTensor` failure in step 2 is that convention, not a loose end.

So the plan's premise -- replace the imperative plan wholesale -- does not hold
against this model. Three ways out were available: change the boundary convention
(gives up the three things it exists for), add `use_local_output` to
ShardingConfig (modifying core, out of scope), or keep the imperative plan as the
parallelization mechanism while the declarations stand alongside it.

The third is taken, and it turns out to cost nothing that mattered. What
`LoRAConverter` needs is `sharding_config` PRESENT on the base linear -- it reads
`config.sharding_config` to derive adapter placements. It does not need the
declarations to be the thing that parallelizes the model. Step 1b already
satisfies that, and the declarations are inert without a driver, so nothing about
the current numerics changes.

Driver reverted. tp2 back to 7.70006 / 7.04439 / 6.16003, bit-identical to the
imperative baseline.

## Revised plan

The blocker recorded in LORA_CONVERTER_BLOCKER is cleared by step 1b alone. What
remains is the LoRA work itself, not a whole-model migration:

1. Adopt `LoRAConverter` for the K3 LoRA path, deriving adapter placements from
   the declarations step 1b added.
2. Require it to clear the recorded LoRA numbers: TP's o_proj.lora_b defect
   (1.33869), PP at 0.105 and CP at 0.150 weighted, with FSDP and EP+FSDP staying
   exact at 0.00000.
3. Keep the packed-MXFP4 base and MXFP8 activation path, which upstream has no
   equivalent for -- the conversion is adapters-to-upstream, quantized-base
   stays.

The full-model imperative-to-declarative migration is NOT part of this. It needs
`use_local_output` in the declarative vocabulary, which is a core change and
belongs upstream as its own proposal if it is wanted at all.
