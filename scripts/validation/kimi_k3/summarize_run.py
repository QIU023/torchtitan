# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize rank evidence and sampled Kineto traces without conflating GPU and wall time."""
import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("result", type=Path)
parser.add_argument("--trace-ranks", nargs="+", type=int, default=[0])
parser.add_argument("--warmup-steps", type=int, default=10)
args = parser.parse_args()
reports = [json.loads(p.read_text()) for p in sorted(args.result.glob("rank-*.json"))]
assert reports and all(r.get("status") == "PASS" for r in reports)
assert len(reports) == reports[0]["world"]
assert {r["rank"] for r in reports} == set(range(len(reports)))
assert all(len(r["steps"]) == r["settings"]["steps"] for r in reports)
assert len({len(r["steps"]) for r in reports}) == 1
steps = []
for index in range(min(len(r["steps"]) for r in reports)):
    rows = [r["steps"][index] for r in reports]
    steps.append(
        {
            "step": index + 1,
            "max_rank_seconds": max(s["seconds"] for s in rows),
            "summed_last_stage_loss": sum(s["loss"] for s in rows if s["loss"] >= 0),
            "peak_cuda_bytes": max(s["peak_cuda_bytes"] for s in rows),
            "min_changed_probe_elements": min(
                min(s["changed_elements"].values()) for s in rows
            ),
        }
    )
summary = {"passed_ranks": len(reports), "steps": steps, "traces": []}
steady = [
    s
    for i, s in enumerate(steps)
    if s["step"] > args.warmup_steps
    and all(r["steps"][i].get("profile_phase", "off") == "off" for r in reports)
]
if steady:
    durations = [s["max_rank_seconds"] for s in steady]
    settings = reports[0]["settings"]
    tokens = (
        settings["seq_len"]
        * settings["microbatch_size"]
        * settings["microbatches"]
        * len(reports)
        // settings["pp"]
    )
    summary["steady_state"] = {
        "steps": [s["step"] for s in steady],
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": statistics.mean(durations),
        "step_time_cv": statistics.pstdev(durations) / statistics.mean(durations),
        "tokens_per_second": tokens * len(durations) / sum(durations),
        "timing_scope": "max per-rank training and gradient-validation time; excludes trace export and input creation",
    }
    if all("allocated_cuda_bytes" in row for r in reports for row in r["steps"]):
        summary["memory_by_rank"] = []
        measured_steps = {s["step"] for s in steady}
        for r in reports:
            rows = [row for row in r["steps"] if row["step"] in measured_steps]
            summary["memory_by_rank"].append(
                {
                    "rank": r["rank"],
                    "allocated_start": rows[0]["allocated_cuda_bytes"],
                    "allocated_end": rows[-1]["allocated_cuda_bytes"],
                    "allocated_growth": rows[-1]["allocated_cuda_bytes"]
                    - rows[0]["allocated_cuda_bytes"],
                    "reserved_start": rows[0]["reserved_cuda_bytes"],
                    "reserved_end": rows[-1]["reserved_cuda_bytes"],
                    "per_step": [
                        {
                            k: row[k]
                            for k in (
                                "step",
                                "allocated_cuda_bytes",
                                "reserved_cuda_bytes",
                                "peak_cuda_bytes",
                            )
                        }
                        for row in rows
                    ],
                }
            )
            if all(
                "device_free_bytes" in row and "process_rss_bytes" in row
                for row in rows
            ):
                summary["memory_by_rank"][-1].update(
                    {
                        "driver_used_growth": rows[0]["device_free_bytes"]
                        - rows[-1]["device_free_bytes"],
                        "process_rss_growth": rows[-1]["process_rss_bytes"]
                        - rows[0]["process_rss_bytes"],
                        "external_memory_per_step": [
                            {
                                k: row[k]
                                for k in (
                                    "step",
                                    "device_free_bytes",
                                    "device_total_bytes",
                                    "process_rss_bytes",
                                )
                            }
                            for row in rows
                        ],
                    }
                )
for rank in args.trace_ranks:
    for path in args.result.glob(
        f"profiling/traces/iteration_*/rank{rank}_trace.json.gz"
    ):
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        kernels = [
            e
            for e in trace["traceEvents"]
            if e.get("cat") == "kernel" and e.get("ph") == "X"
        ]
        totals = defaultdict(float)
        for event in kernels:
            totals[event["name"]] += event["dur"]
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
        union = 0
        if intervals:
            start, end = intervals[0]
            for a, b in intervals[1:]:
                if a > end:
                    union += end - start
                    start, end = a, b
                else:
                    end = max(end, b)
            union += end - start
        summary["traces"].append(
            {
                "rank": rank,
                "file": str(path),
                "kernel_events": len(kernels),
                "gpu_active_union_ms": union / 1000,
                "kernel_duration_sum_ms": sum(totals.values()) / 1000,
                "nccl_kernel_duration_sum_ms": sum(
                    v for k, v in totals.items() if "nccl" in k.lower()
                )
                / 1000,
                "top_kernels_by_summed_ms": [
                    (k, v / 1000)
                    for k, v in sorted(
                        totals.items(), key=lambda kv: kv[1], reverse=True
                    )[:10]
                ],
            }
        )
(args.result / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
