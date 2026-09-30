# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the standard Kimi training engine with packed import and DistMuon."""
import argparse
import json
import math
import os
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from gradient_probes import changed_elements, select_probes
from step_watchdog import watch_step
from torch.distributed.tensor import DTensor

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import CrossEntropyLoss, LinearCrossEntropyLoss
from torchtitan.components.optimizer import DistMuon, LRSchedulersContainer
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import MXQATTransform
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common.moe import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.config_registry import _dist_muon_optimizer
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan.observability.profiler import Profiler
from torchtitan.training_engine import TrainingEngine


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def log_memory(rank, phase):
    free, total = torch.cuda.mem_get_info()
    print(
        json.dumps(
            {
                "rank": rank,
                "phase": phase,
                "allocated_cuda_bytes": torch.cuda.memory_allocated(),
                "reserved_cuda_bytes": torch.cuda.memory_reserved(),
                "device_used_bytes": total - free,
                "device_total_bytes": total,
            }
        ),
        flush=True,
    )


def save_debug_state(engine, output, rank, phase):
    tensors = {}
    for model in engine.model_parts:
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            name = name.replace("._checkpoint_wrapped_module", "")
            tensors[name] = local(value).detach().cpu()
            if isinstance(value, torch.nn.Parameter) and value.grad is not None:
                tensors[name + ".grad"] = local(value.grad).detach().cpu()
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save(tensors, Path(output, f"debug-{phase}-rank-{rank}.pt"))


def configs(args):
    model = model_registry(
        args.flavor,
        enable_sp=False,
        seq_len=args.seq_len,
        moe_comm_backend=args.dispatcher,
    )
    if args.balanced_routing:
        for _, router, parent, attr in model.traverse(
            QuantileBalancedTopKRouter.Config
        ):
            values = {
                f.name: getattr(router, f.name)
                for f in fields(TokenChoiceTopKRouter.Config)
            }
            values["_debug_force_load_balance"] = True
            setattr(parent, attr, TokenChoiceTopKRouter.Config(**values))
    adapter = KimiK3StateDictAdapter(model, None)
    policy = adapter.mxfp4_policy(args.checkpoint)
    selected = adapter.qat_weight_fqns(policy)
    model = MXQATTransform.from_weight_fqns(model, selected).transform(model)
    world = int(os.environ.get("WORLD_SIZE", "8"))
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=world // args.pp,
        expert_parallel_degree=args.ep,
        pipeline_parallel_degree=args.pp,
        num_pp_microbatches=args.microbatches,
        enable_sequence_parallel=False,
        fsdp_reshard_after_forward="always",
    )
    optimizer = _dist_muon_optimizer(
        model, muon_lr=args.lr, adamw_lr=args.lr, parallelism=parallelism
    )
    # Zero decay makes a changed matrix positive evidence of a gradient update.
    for config in optimizer.optimizers:
        config.weight_decay = 0.0
    training = TrainingConfig(
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        num_tokens_per_microbatch_per_dp_rank=args.seq_len * args.microbatch_size,
        max_context_length=args.seq_len,
        steps=args.steps,
        disable_cuda_graphs=True,
        max_norm=1.0,
    )
    from pipeline_memory import RecomputeMatmulsSelectiveAC

    ac_class = (
        RecomputeMatmulsSelectiveAC
        if args.activation_checkpoint == "recompute-matmuls"
        else SelectiveAC
    )
    config = TrainingEngine.Config(
        debug=DebugConfig(seed=42, deterministic=args.deterministic),
        optimizer=optimizer,
        parallelism=parallelism,
        training=training,
        activation_checkpoint=(
            ac_class.Config() if args.activation_checkpoint != "none" else None
        ),
        loss=(
            LinearCrossEntropyLoss.Config(batch_chunk_size=128)
            if args.fused_loss
            else CrossEntropyLoss.Config(
                global_vocab_size=model.tok_embeddings.num_embeddings
            )
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0, decay_ratio=0.0),
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=2,
            profiler_skip_first=args.profile_skip_steps,
            profiler_warmup=1,
            profiler_active=1,
            profiler_repeat=1,
        ),
    )
    model.update_from_config(config=config)
    return model, config, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flavor", default="debugmodel")
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile-skip-steps", type=int, default=0)
    parser.add_argument(
        "--activation-checkpoint",
        choices=["none", "selective", "recompute-matmuls"],
        default="none",
    )
    parser.add_argument("--reshard-after-backward", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--dispatcher", default="standard")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fused-loss", action="store_true")
    parser.add_argument("--balanced-routing", action="store_true")
    parser.add_argument("--diagnostic-state", action="store_true")
    parser.add_argument("--diagnose-first-step", action="store_true")
    parser.add_argument("--diagnose-steps", type=int, default=0)
    parser.add_argument("--pipeline-startup-timeout", type=int, default=0)
    parser.add_argument("--pipeline-steady-timeout", type=int, default=600)
    parser.add_argument("--release-cache-before-first-update", action="store_true")
    args = parser.parse_args()
    args.diagnose_steps = max(args.diagnose_steps, int(args.diagnose_first_step))
    if args.diagnostic_state and args.flavor != "debugmodel":
        raise ValueError(
            "Full-state diagnostics are restricted to the small debug model"
        )
    model_config, config, selected = configs(args)
    if args.preflight:
        with torch.device("meta"):
            model = model_config.build()
        names = dict(model.named_parameters())
        assert selected <= names.keys()
        report = {
            "qat_masters": len(selected),
            "qat_modules": sum(
                getattr(type(m), "_mx_qat", False) for m in model.modules()
            ),
            "parameters": sum(p.numel() for p in names.values()),
            "optimizer_types": [
                type(c).__qualname__ for c in config.optimizer.optimizers
            ],
        }
        print(json.dumps(report), flush=True)
        return
    engine = TrainingEngine(
        config,
        model_config=model_config,
        max_num_documents=args.microbatch_size,
        output_dir=args.output,
    )
    rank = dist.get_rank()
    print(f"RANK {rank} ENGINE_INITIALIZE", flush=True)
    engine.initialize(compile_config=None, hf_assets_path="")
    if args.reshard_after_backward:
        if args.pp <= 1:
            raise ValueError("Pipeline backward resharding requires pp > 1")
        from pipeline_memory import enable_backward_resharding

        engine._campaign_reshard_handles = enable_backward_resharding(
            engine.model_parts
        )
        print(f"RANK {rank} PIPELINE_BACKWARD_RESHARD_ENABLED", flush=True)
    print(f"RANK {rank} ENGINE_READY", flush=True)
    log_memory(rank, "engine_ready")
    adapter = KimiK3StateDictAdapter(model_config, None)
    state = {
        k: v for model in engine.model_parts for k, v in model.state_dict().items()
    }
    hf_state = adapter.to_hf(state)
    # Checkpoint planning exchanges CPU objects; keep training collectives on NCCL.
    checkpoint_group = dist.new_group(backend="gloo")
    try:
        dcp.load(
            hf_state,
            storage_reader=adapter.get_hf_storage_reader(args.checkpoint, True),
            process_group=checkpoint_group,
        )
    finally:
        dist.destroy_process_group(checkpoint_group)
    native = adapter.from_hf(hf_state)
    for model in engine.model_parts:
        model.load_state_dict({k: native[k] for k in model.state_dict()}, strict=True)
    del state, native, hf_state
    print(f"RANK {rank} CHECKPOINT_LOADED", flush=True)
    log_memory(rank, "checkpoint_loaded")
    pipeline_startup = None
    if args.pp > 1 and args.pipeline_startup_timeout:
        from pipeline_startup import PipelineStartupTimeout

        pipeline_startup = PipelineStartupTimeout(
            engine.pp_schedule,
            args.pipeline_startup_timeout,
            args.pipeline_steady_timeout,
        )
    diagnostic_handles = []
    diagnostic_step = 0
    if args.diagnose_steps:
        from finite_diagnostics import install

        diagnostic_handles = install(
            engine.model_parts, args.output, rank, step=lambda: diagnostic_step
        )
    if args.diagnostic_state:
        save_debug_state(engine, args.output, rank, "initial")
    muons = [o for o in engine.optimizers if isinstance(o, DistMuon)]
    assert muons, "DistMuon not engaged"
    muon_params = {id(p) for o in muons for g in o.param_groups for p in g["params"]}
    candidates = {}
    for model in engine.model_parts:
        for name, param in model.named_parameters():
            if id(param) in muon_params and local(param).numel():
                kind = "expert" if "routed_experts" in name else "dense"
                candidates.setdefault(kind, []).append((name, param))
    assert candidates
    import torchao.prototype.qat as qat

    original = qat.mx_fake_quantized_grouped_mm
    counts = {"calls": 0}

    def counted(*a, **kw):
        counts["calls"] += 1
        return original(*a, **kw)

    qat.mx_fake_quantized_grouped_mm = counted
    report = {
        "rank": rank,
        "world": dist.get_world_size(),
        "steps": [],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "qat_masters": len(selected),
        "distmuon_instances": len(muons),
        "settings": vars(args),
    }
    engine.start_profiler()
    try:
        for step in range(args.steps):
            with watch_step(90):
                diagnostic_step = step + 1
                print(f"RANK {rank} STEP {step + 1} START", flush=True)
                calls_before = counts["calls"]
                gen = torch.Generator().manual_seed(
                    20260928
                    + step * 1000
                    + engine.parallelism_context.get_mesh("dp").get_local_rank()
                )
                group = []
                n = args.seq_len * args.microbatch_size
                for mb in range(args.microbatches):
                    tokens = torch.randint(
                        0,
                        model_config.tok_embeddings.num_embeddings,
                        (n + 1,),
                        generator=gen,
                    )
                    group.append(
                        TokenizedTrainingMicrobatch(
                            input=tokens[:-1],
                            labels=tokens[1:],
                            positions=torch.arange(
                                args.seq_len, dtype=torch.int32
                            ).repeat(args.microbatch_size),
                            padding_mask=torch.zeros(n, dtype=torch.bool),
                            num_valid_tokens=n,
                        )
                    )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                valid = engine.prepare_step(
                    n * args.microbatches * (dist.get_world_size() // args.pp)
                )
                with torch.profiler.record_function("campaign_forward_backward"):
                    loss = engine.forward_backward_microbatch(
                        microbatch_group=group, global_valid_tokens=valid
                    )
                if step == 0:
                    log_memory(rank, "first_forward_backward_complete")
                    if args.release_cache_before_first_update:
                        # Lazy NCCL buffers allocate outside PyTorch's caching allocator.
                        # Release startup cache once before these first reductions; all
                        # live parameters/gradients remain allocated. Steady steps retain caching.
                        torch.cuda.empty_cache()
                        log_memory(rank, "first_update_cache_released")
                assert math.isfinite(float(loss)), f"Nonfinite loss at step {step + 1}"
                if args.diagnostic_state and step == 0:
                    save_debug_state(engine, args.output, rank, "backward")
                probe_error = None
                try:
                    probes = select_probes(candidates)
                except AssertionError as error:
                    probe_error = str(error)
                    print(f"RANK {rank} PROBE_FAILURE {probe_error}", flush=True)
                # Every rank must reach clipping; a local probe failure must not strand peers.
                probe_ok = torch.tensor(int(probe_error is None), device="cuda")
                dist.all_reduce(probe_ok, op=dist.ReduceOp.MIN)
                assert probe_ok.item(), (
                    probe_error or "Gradient probe failed on another rank"
                )
                gradients = {p["name"]: p["norm"] for p in probes}
                if step == 0:
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "phase": "gradient_probes",
                                "probes": [
                                    {
                                        k: p[k]
                                        for k in ("kind", "name", "offset", "norm")
                                    }
                                    for p in probes
                                ],
                            }
                        ),
                        flush=True,
                    )
                if step == 0:
                    log_memory(rank, "first_optimizer_start")
                with torch.profiler.record_function("campaign_optimizer"):
                    grad_norm = engine.optimizer_step()
                if step == 0:
                    log_memory(rank, "first_optimizer_complete")
                assert math.isfinite(
                    float(grad_norm)
                ), f"Nonfinite gradient norm at step {step + 1}"
                if step < args.diagnose_steps:
                    from finite_diagnostics import check_updated_state

                    checked = check_updated_state(
                        engine.model_parts, muons, args.output, rank, step + 1
                    )
                    state_ok = torch.tensor(
                        int(checked["finite"] and checked["bounded"]), device="cuda"
                    )
                    dist.all_reduce(state_ok, op=dist.ReduceOp.MIN)
                    assert (
                        state_ok.item()
                    ), f"Nonfinite or extreme parameter/momentum after step {step + 1}"
                if args.diagnostic_state and step == 0:
                    save_debug_state(engine, args.output, rank, "updated")
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                changes = changed_elements(probes)
                assert all(changes.values()), changes
                assert (
                    counts["calls"] > calls_before
                ), f"QAT inactive at step {step + 1}"
                device_free, device_total = torch.cuda.mem_get_info()
                process_rss = int(
                    Path("/proc/self/statm").read_text().split()[1]
                ) * os.sysconf("SC_PAGE_SIZE")
                record = {
                    "step": step + 1,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "seconds": seconds,
                    "probe_grad_norms": gradients,
                    "changed_elements": changes,
                    "probe_offsets": {p["name"]: p["offset"] for p in probes},
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "allocated_cuda_bytes": torch.cuda.memory_allocated(),
                    "reserved_cuda_bytes": torch.cuda.memory_reserved(),
                    "device_free_bytes": device_free,
                    "device_total_bytes": device_total,
                    "process_rss_bytes": process_rss,
                    "profile_phase": (
                        "warmup"
                        if step == args.profile_skip_steps
                        else "active"
                        if step == args.profile_skip_steps + 1
                        else "off"
                    ),
                }
                report["steps"].append(record)
                print(json.dumps(record), flush=True)
                if step == 0 and pipeline_startup is not None:
                    pipeline_startup.finish()
                engine.step_profiler()
                if step + 1 == args.diagnose_steps:
                    for handle in diagnostic_handles:
                        handle.remove()
                    diagnostic_handles.clear()
        assert counts["calls"] > 0, "QAT not engaged"
        report["qat_calls"] = counts["calls"]
        report["muon_momentum_entries"] = sum(
            "momentum_buffer" in state for o in muons for state in o.state.values()
        )
        assert report["muon_momentum_entries"] > 0
        report["status"] = "PASS"
    finally:
        engine.close()
        Path(args.output).mkdir(parents=True, exist_ok=True)
        Path(args.output, f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier()
    print(f"KIMI_DISTMUON_PASS rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
