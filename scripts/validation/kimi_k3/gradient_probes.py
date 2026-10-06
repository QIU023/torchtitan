# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded validation samples selected from active local DistMuon gradients."""
import torch
from torch.distributed.tensor import DTensor


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def select_probes(candidates, window=4096, scan_chunk=1048576):
    probes = []
    for kind, parameters in candidates.items():
        selected = None
        for name, parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = local(parameter.grad).detach().view(-1)
            for start in range(0, gradient.numel(), scan_chunk):
                chunk = gradient[start : start + scan_chunk]
                assert torch.isfinite(chunk).all(), f"Nonfinite gradient: {name}"
                magnitude, index = chunk.abs().max(dim=0)
                if magnitude.item() == 0:
                    continue
                offset = (start + int(index)) // window * window
                sample = gradient[offset : offset + window]
                selected = {
                    "kind": kind,
                    "name": name,
                    "parameter": parameter,
                    "offset": offset,
                    "norm": float(sample.float().norm()),
                    "before": local(parameter)
                    .detach()
                    .view(-1)[offset : offset + window]
                    .clone(),
                }
                break
            if selected is not None:
                break
        assert selected is not None, f"No active local {kind} DistMuon gradient"
        probes.append(selected)
    return probes


def changed_elements(probes):
    return {
        p["name"]: int(
            torch.count_nonzero(
                p["before"]
                != local(p["parameter"])
                .detach()
                .view(-1)[p["offset"] : p["offset"] + p["before"].numel()]
            )
        )
        for p in probes
    }
