# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Temporary checks for local parameters, optimizer state and module outputs."""
import json
import math
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def check_updated_state(models, optimizers, output, rank, step, absolute_limit=1e6):
    """Check all local parameters and Muon momentum without gathering shards."""
    names = {}
    report = {
        "phase": "updated_state",
        "step": step,
        "finite": True,
        "bounded": True,
        "absolute_limit": absolute_limit,
        "parameters": 0,
        "momentum_buffers": 0,
        "failures": [],
        "largest_tensors": [],
    }

    def check(tensor, kind, name):
        value = _local(tensor.detach())
        # Slice before allocating masks, including for noncontiguous tensors.
        rows = max(1, 1048576 // max(1, math.prod(value.shape[1:])))
        chunks = value.split(rows) if value.ndim else (value,)
        counts = {"nan": 0, "posinf": 0, "neginf": 0}
        maximum = 0.0
        for chunk in chunks:
            if chunk.numel():
                maximum = max(maximum, float(chunk.abs().max()))
            if not bool(torch.isfinite(chunk).all()):
                counts["nan"] += int(torch.isnan(chunk).sum())
                counts["posinf"] += int(torch.isposinf(chunk).sum())
                counts["neginf"] += int(torch.isneginf(chunk).sum())
        report[kind] += 1
        report["largest_tensors"].append(
            {"kind": kind, "name": name, "max_abs": maximum}
        )
        report["largest_tensors"].sort(key=lambda item: item["max_abs"], reverse=True)
        del report["largest_tensors"][8:]
        if any(counts.values()):
            report["finite"] = False
        if maximum > absolute_limit:
            report["bounded"] = False
        if any(counts.values()) or maximum > absolute_limit:
            report["failures"].append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(value.shape),
                    "max_abs": maximum,
                    **counts,
                }
            )

    for model in models:
        for name, parameter in model.named_parameters():
            names[id(parameter)] = name
            check(parameter, "parameters", name)
    for optimizer in optimizers:
        for parameter, state in optimizer.state.items():
            if "momentum_buffer" in state:
                check(
                    state["momentum_buffer"], "momentum_buffers", names[id(parameter)]
                )
    Path(output).mkdir(parents=True, exist_ok=True)
    with Path(output, f"finite-diagnostics-rank-{rank}.jsonl").open("a") as handle:
        handle.write(json.dumps(report) + "\n")
    print(json.dumps({"rank": rank, **report}), flush=True)
    return report


def install(models, output, rank, step=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"finite-diagnostics-rank-{rank}.jsonl"

    def record(row):
        if step is not None:
            row = {"step": step(), **row}
        print(json.dumps({"rank": rank, **row}), flush=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    count = 0
    for model in models:
        for name, parameter in model.named_parameters():
            value = _local(parameter.detach())
            if not bool(torch.isfinite(value).all()):
                record(
                    {
                        "phase": "loaded_parameter",
                        "name": name,
                        "finite": False,
                        "shape": list(value.shape),
                    }
                )
                raise AssertionError(f"Nonfinite loaded parameter: {name}")
            count += 1
    record({"phase": "loaded_parameters", "finite": True, "parameters": count})

    def hook(name):
        def check(module, inputs, result):
            values = [
                _local(t.detach()) for t in _tensors(result) if t.is_floating_point()
            ]
            finite = all(bool(torch.isfinite(t).all()) for t in values)
            if not finite:
                record(
                    {
                        "phase": "first_nonfinite_output",
                        "module": name,
                        "shapes": [list(t.shape) for t in values],
                        "input_finite": [
                            bool(torch.isfinite(_local(t.detach())).all())
                            for t in _tensors(inputs)
                            if t.is_floating_point()
                        ],
                    }
                )
                # Bound diagnostic storage, including modules with weight tensors as inputs.
                snapshots = [
                    {
                        "shape": list(t.shape),
                        "values": _local(t.detach()).flatten()[:1048576].cpu(),
                    }
                    for t in list(_tensors(inputs))[:16]
                ]
                torch.save(snapshots, output / f"first-nonfinite-inputs-rank-{rank}.pt")
                raise AssertionError(f"First nonfinite output: {name}")
            if name.startswith("layers.") and name.count(".") == 1:
                record(
                    {
                        "phase": "layer_output",
                        "module": name,
                        "finite": True,
                        "max_abs": [
                            float(t.abs().max()) if t.numel() else 0 for t in values
                        ],
                    }
                )

        return check

    return [
        module.register_forward_hook(hook(name))
        for model in models
        for name, module in model.named_modules()
    ]
