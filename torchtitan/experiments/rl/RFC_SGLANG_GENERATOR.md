# RFC: Engine-Agnostic Generator for TorchTitan RL

**Status:** draft  
**Author:** AttnRes contributor (yiqiao.lbj23@gmail.com)  
**Tracking issue:** TBD  

## Motivation

The current `experiments/rl/` module is hard-coded to vLLM as the
rollout engine: `__init__.py` imports `vllm_wrapper` at module load,
and `actors/generator.py` is named `VLLMGenerator` and imports vllm
unconditionally.

This couples the RL framework to vLLM's release cycle and ABI.
Concrete consequences:

1. **PyTorch nightly required.** vLLM's pre-built wheels target
   `cu128` on torch nightly. Users on torch stable + cu129 (e.g.
   anyone who depends on `sgl_kernel`'s ABI) cannot import
   `experiments.rl` at all.
2. **No path for SGLang users.** SGLang is a peer of vLLM with
   competitive throughput, simpler dependency tree on stable torch,
   and better support for some inference fabrics (e.g., the
   sequence-dim TP shard + reduce-scatter+all-gather fusion path
   used by Block AttnRes / Hyper-Connections / mHC residual
   families).
3. **The Generator's contract is engine-independent.** The RL loop
   only needs `generate(prompts) → Episodes` and
   `pull_model_state_dict(version)`. Both vLLM and SGLang expose
   these primitives.

This RFC proposes adding `SGLangGenerator` parallel to
`VLLMGenerator`, with the framework's import path made lazy so users
can install only the engine they need.

## Design

### Lazy package imports

`experiments/rl/__init__.py` exports two registration functions
(`register_model_to_{vllm,sglang}_model_registry`). Both are defined
in `plugin.py` and import their engine's libraries lazily inside the
function body, not at module load.

This means:

```python
# Works on torch stable + sgl_kernel, even without vLLM installed:
import torchtitan.experiments.rl
from torchtitan.experiments.rl import register_model_to_sglang_model_registry
```

vLLM-only users see no behavior change. SGLang-only users get a
working framework.

### `SGLangGenerator` actor

Mirrors `VLLMGenerator`'s API exactly:

| API | VLLMGenerator | SGLangGenerator |
|---|---|---|
| `__init__(config, *, model_spec, model_path)` | ✅ | ✅ |
| `generate(prompt_texts, expected_answers) -> list[Episode]` | ✅ | ✅ + optional `images=` for VLM rollouts |
| `pull_model_state_dict(version)` | torchstore RDMA | torchstore RDMA OR DCP-disk fallback |

Same `Episode` shape, same `group_id` convention, same monarch
`@endpoint` decorator. **Any RLHF method (PPO, GRPO, DPO, online-DPO,
RLHF-V) that already uses `VLLMGenerator` can swap to
`SGLangGenerator` by changing one config line.**

### Config

`SGLangGenerator.Config` mirrors `VLLMGenerator.Config` field-for-
field, with engine-specific extras:

* `compile: SGLangCompileConfig` — `cuda_graph` + `piecewise_cuda_graph`
  flags (matching SGLang's own engine flags)
* `backend: SGLangBackendConfig` — `attention_backend` +
  `linear_attn_backend` selectors
* `weight_sync_method: Literal["torchstore", "disk"]` — direct
  RDMA pull vs DCP→HF disk reload (the latter works without
  monarch RDMA but is slower)

### Multimodal

SGLang already supports VLM rollouts (LLaVA, Qwen-VL, Kimi-VL via
its `image_data` request field). `SGLangGenerator.generate` exposes
an optional `images=` argument so multimodal RLHF doesn't need to
fork this file. The RL loop's training side is multimodal-agnostic
(text token ids → log-probs).

## Open questions

1. **`torchstore` weight sync requires SGLang to expose its inner
   model state dict.** vLLM does this via
   `model_executor.driver_worker.get_model()`; SGLang's `Engine`
   keeps the model in a worker subprocess and doesn't expose it.
   Until SGLang adds an equivalent (or we run the SGLang model in-
   process via `ModelRunner`), the disk fallback is the recommended
   path. Filing this as a separate SGLang-side proposal.

2. **Bitwise log-prob parity.** vLLM's batch-invariant mode (Triton
   replacement of `mm`/`addmm`/`log_softmax`/`mean`) is wired through
   `set_batch_invariance`. SGLang's batch-invariance story is less
   developed. For text RLHF this is a gap; for multimodal RLHF the
   image projector usually dominates noise so the gap matters less.
   Documented as a deferred limitation.

3. **Cross-engine fabric trace differences.** SGLang and vLLM
   schedule their KV-cache and attention differently; we expect the
   NCCL trace patterns from a GRPO run on vLLM vs SGLang to differ
   in proportions (more `AllGather` in SGLang's seq-shard path,
   different chunked-prefill collective sequencing). This is itself
   useful research output — captured by running the same task with
   `engine=vllm` vs `engine=sglang` and dumping NCCL traces.

## Backward compatibility

* `VLLMGenerator` and its config are unchanged.
* `register_model_to_vllm_model_registry` is unchanged.
* `__init__.py` no longer eagerly imports `vllm_wrapper`, so any
  user code that did `from torchtitan.experiments.rl.models import
  vllm_wrapper` directly continues to work; the module is still
  loadable via its full path.

## Files changed

* `experiments/rl/__init__.py` — lazy exports
* `experiments/rl/plugin.py` — adds `register_model_to_sglang_model_registry` + `SGLANG_MODEL_NAME`
* `experiments/rl/models/sglang_wrapper.py` — new
* `experiments/rl/actors/sglang_generator.py` — new

No core torchtitan files modified (per `experiments/` rules).

## Reference impl

This RFC is co-developed with
[`yiqiao-lbj/torchtitan_attention_residual`](https://github.com/QIU023/torchtitan_attention_residual)
where the `SGLangGenerator` is integrated with a Block-AttnRes-aware
SGLang inference path. See `phase11/PHASE11_SGLANG_REPORT.md` for
the inference-side measurements (decode-tps +27% from a Phase-2
fused Triton kernel; AR fabric -58% under seq-shard).
