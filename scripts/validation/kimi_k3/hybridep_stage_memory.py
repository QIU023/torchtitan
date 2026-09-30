# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Released first-stage memory control: FSDP32/EP32, eight live forwards.

Uses eleven released layers plus the normal embedding/vision/output-norm path.
Manual 1F1B-like ordering isolates live activation/gradient memory, without PP
transport or optimizer updates. The hidden-state squared-mean objective is only
a diagnostic. This is not a training/performance or full-model validation run.
"""
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pipeline_memory import enable_backward_resharding
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.optimizer import DistMuon
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan.training_engine import TrainingEngine

from validate_distmuon import configs, log_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        flavor="Kimi-K3",
        seq_len=2048,
        dispatcher="hybridep",
        balanced_routing=True,
        checkpoint=args.checkpoint,
        ep=32,
        pp=1,
        microbatches=1,
        lr=8e-4,
        microbatch_size=2,
        steps=1,
        deterministic=False,
        activation_checkpoint="recompute-matmuls",
        fused_loss=True,
        profile_skip_steps=0,
    )
    model_config, config, _ = configs(settings)
    reader_config = copy.deepcopy(model_config)
    model_config.layers = model_config.layers[:11]
    prefixes = tuple(f"layers.{i}." for i in range(11))
    for opt in config.optimizer.optimizers:
        if isinstance(opt, DistMuon.Config):
            opt.compute_sharding_by_fqn = {
                n: v
                for n, v in opt.compute_sharding_by_fqn.items()
                if n.startswith(prefixes)
            }
            opt.bucket_configs = tuple(
                replace(b, patterns=p)
                for b in opt.bucket_configs
                if (p := tuple(n for n in b.patterns if n.startswith(prefixes)))
            )
    engine = TrainingEngine(
        config, model_config=model_config, max_num_documents=2, output_dir=str(out)
    )
    rank = dist.get_rank()
    engine.initialize(compile_config=None, hf_assets_path="")
    model = engine.model_parts[0]
    handles = enable_backward_resharding([model])
    adapter = KimiK3StateDictAdapter(reader_config, None)
    state = model.state_dict()
    hf = adapter.to_hf(state)
    # DCP exchanges CPU planning objects; avoid persistent NCCL peer buffers.
    log_memory(rank, "before_checkpoint_load")
    checkpoint_group = dist.new_group(backend="gloo")
    try:
        dcp.load(
            hf,
            storage_reader=adapter.get_hf_storage_reader(settings.checkpoint, True),
            process_group=checkpoint_group,
        )
    finally:
        dist.destroy_process_group(checkpoint_group)
    log_memory(rank, "after_checkpoint_load")
    native = adapter.from_hf(hf)
    model.load_state_dict({k: native[k] for k in state}, strict=True)
    del state, hf, native
    report = {
        "rank": rank,
        "status": "INCOMPLETE",
        "torch": torch.__version__,
        "scope": "released 11-layer prefix, EP32/FSDP32, 8 live forwards; no optimizer or PP transport",
        "phases": [],
    }
    live = {}
    torch.cuda.reset_peak_memory_stats()

    def record(phase):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        row = {
            "phase": phase,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak": torch.cuda.max_memory_allocated(),
            "device_used": total - free,
            "device_total": total,
        }
        report["phases"].append(row)
        print(json.dumps({"rank": rank, **row}), flush=True)
        (out / f"stage-memory-rank-{rank}.json").write_text(
            json.dumps(report, indent=2)
        )

    record("checkpoint_loaded")

    def forward(i):
        gen = torch.Generator().manual_seed(999 + rank * 100 + i)
        tokens = torch.randint(
            0, model_config.tok_embeddings.num_embeddings, (4097,), generator=gen
        )
        batch = TokenizedTrainingMicrobatch(
            input=tokens[:-1],
            labels=tokens[1:],
            positions=torch.arange(2048, dtype=torch.int32).repeat(2),
            padding_mask=torch.zeros(4096, dtype=torch.bool),
            num_valid_tokens=4096,
        )
        with engine.parallelism_context.activate_spmd(typechecking=False):
            inputs, _, kwargs = model.preprocess_inputs(
                batch.to_input_dict(engine.device),
                parallelism_context=engine.parallelism_context,
                parallelism=config.parallelism,
                max_num_documents=2,
                max_context_length=2048,
            )
            pred = model(inputs, **kwargs)
            assert tuple(pred.shape) == (4096, 7168), pred.shape
            loss = pred.float().square().mean()
        assert bool(torch.isfinite(loss))
        live[i] = loss
        record(f"forward_{i}")

    def backward(i):
        model.set_is_last_backward(False)
        model.set_reshard_after_backward(False)  # Actual pipeline override.
        model.set_requires_gradient_sync(False)
        with engine.parallelism_context.activate_spmd(
            typechecking=False
        ), spmd.no_typecheck():
            live.pop(i).backward()
        record(f"backward_{i}")

    try:
        for i in range(8):
            forward(i)
        for i in range(10):
            backward(i)
            if i + 8 < 10:
                forward(i + 8)
        grad_bytes = 0
        active = {"dense": 0, "expert": 0}
        unsharded_bytes = 0
        for module in model.modules():
            if not isinstance(module, FSDPModule):
                continue
            for group in module._get_fsdp_state()._fsdp_param_groups:
                for p in group.fsdp_params:
                    unsharded = getattr(p, "_unsharded_param", None)
                    if unsharded is not None:
                        if hasattr(unsharded, "to_local"):
                            unsharded = unsharded.to_local()
                        unsharded_bytes += unsharded.untyped_storage().nbytes()
                    if p.unsharded_accumulated_grad is None:
                        continue
                    grad = p.unsharded_accumulated_grad_data.detach().reshape(-1)
                    assert grad.dtype == torch.float32, grad.dtype
                    grad_bytes += grad.numel() * grad.element_size()
                    positive = False
                    for start in range(0, grad.numel(), 1 << 20):
                        chunk = grad[start : start + (1 << 20)]
                        assert bool(torch.isfinite(chunk).all()), p._param_fqn
                        positive |= bool(torch.count_nonzero(chunk))
                    category = (
                        "expert" if "routed_experts" in str(p._param_fqn) else "dense"
                    )
                    active[category] += int(positive)
        assert all(active.values()), active
        assert unsharded_bytes == 0, unsharded_bytes
        report.update(
            status="PASS",
            active_gradients=active,
            fp32_gradient_bytes=grad_bytes,
            remaining_unsharded_bytes=unsharded_bytes,
        )
        record("verified")
        dist.barrier()
        print("HYBRIDEP_STAGE_MEMORY_PASS", flush=True)
    finally:
        (out / f"stage-memory-rank-{rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        for handle in handles:
            handle.remove()
        engine.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
