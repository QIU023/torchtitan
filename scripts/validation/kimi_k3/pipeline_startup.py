# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded pipeline startup budget for the pinned September 28 runtime.

Torchcomms directed P2P children default to 600s instead of inheriting the
parent timeout. Dynamic metadata inference serializes first-use forward and
backward compilation across all stages, exceeding that budget on full K3.
Use the public timeout setter after those children exist, before inference.
This harness adapter touches no tensor, collective order, or optimizer math.
"""
import json
from datetime import timedelta

import torch
import torch.distributed as dist


class PipelineStartupTimeout:
    def __init__(self, schedule, startup_seconds, steady_seconds):
        if not 0 < steady_seconds <= startup_seconds:
            raise ValueError("Require 0 < steady timeout <= startup timeout")
        if not callable(
            getattr(schedule, "_initialize_pipeline_distributed_state", None)
        ):
            raise RuntimeError("Pinned pipeline initialization API is unavailable")
        self.schedule = schedule
        self.original = schedule._initialize_pipeline_distributed_state
        self.startup = timedelta(seconds=startup_seconds)
        self.steady = timedelta(seconds=steady_seconds)
        self.groups = set()
        schedule._initialize_pipeline_distributed_state = self.initialize

    def initialize(self, stages, has_backward, initialize_p2p):
        result = self.original(stages, has_backward, initialize_p2p)
        for stage in stages:
            self.groups.add(stage.group)
            self.groups.update(stage._p2p_edge_groups.values())
        for group in self.groups:
            dist.set_timeout(self.startup, group)
        print(
            json.dumps(
                {
                    "phase": "pipeline_startup_timeout",
                    "seconds": self.startup.total_seconds(),
                    "groups": len(self.groups),
                }
            ),
            flush=True,
        )
        return result

    def finish(self):
        # All ranks must finish the first update before anyone lowers a timeout.
        dist.barrier()
        torch.cuda.synchronize()
        for group in self.groups:
            dist.set_timeout(self.steady, group)
        self.schedule._initialize_pipeline_distributed_state = self.original
        print(
            json.dumps(
                {
                    "phase": "pipeline_steady_timeout",
                    "seconds": self.steady.total_seconds(),
                    "groups": len(self.groups),
                }
            ),
            flush=True,
        )
