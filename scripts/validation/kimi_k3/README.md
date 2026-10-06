# Kimi K3 MX-QAT and DistMuon working branch

This branch combines the MX checkpoint/QAT work with the independent vision-QKV
checkpoint fix and the validation infrastructure used to investigate full-model
training. It does not modify the draft PR branches.

## Included changes

| Component | Included behavior |
| --- | --- |
| TorchTitan base | `aa9916b7ab9447cee186e9f24b8cb6b2d61db096`: packed import, manifest-selected QAT, released tensor shape handling, grouped linears, pipeline checkpoint support, fused linear CE, and vision-QKV placement |
| TorchAO dependency | `6352062064145e9ca0d37b05a677c420ba5a7bb6`: stateless MX fake quantization and bounded eager quantization scratch memory |
| Checkpoint loading | These validation entrypoints use a temporary Gloo group for DCP metadata; training stays on NCCL |
| Pipeline memory | Selective AC recomputes dense matmuls; a public backward pre-hook restores FSDP backward resharding after pipeline runtime overrides it |
| Profiling | CPU/CUDA traces remain enabled; this working branch sets `record_shapes=False` in the shared profiler, matching the successful full128 experiment |
| Diagnostics | Active-gradient probes, safe Python step watchdog, pipeline startup timeout restoration, and inspection of every FSDP parameter group |
| Runtime | Launcher sets `NCCL_RUNTIME_CONNECT=0` and `NCCL_NET_PLUGIN=none`, matching the validated environment |

Gloo loading and the pipeline policies live in these experiment scripts. They
are not silently enabled in the normal Trainer/checkpointer path. The profiler
shape setting is the one additional shared-source change. Its original hang
mechanism remains unproven; it is a validated experimental workaround.

## Runtime

The GPU experiments used PyTorch `2.15.0.dev20260928+cu130`, CUDA 13.0,
NCCL 2.30.7, Triton `3.8.0+gitc01b6774`, CUTLASS DSL 4.7.1,
and AttentionGym 0.0.13 on GB200. Use the repository's existing dependency
installation and a compatible Linux GPU environment. HybridEP additionally
requires a working DeepEP hybrid-ep build (tested commit
`42144303752422ade37f24bca9e2dde12df70e09`) and NVSHMEM 3.7.2;
the campaign build included the previously validated site-specific build patch.
This branch does not claim to reproduce that binary from an unpatched checkout.

Install the separate TorchAO implementation into that environment:

```bash
python -m pip install --no-deps -r scripts/validation/kimi_k3/requirements-torchao.txt
```

The full128 result used the earlier TorchAO revision
`88b69eabda136ab9ac7da4ca2f9f4c56aa017bf4`. The new pin adds the separately
validated bounded-scratch fix; the combined branch has not itself completed a
new 50-step full-model run.

## Launch

Use an allocated job with four GB200 GPUs per node, shared checkpoint/output
paths, and the environment above. Run the command on every node, supplying its
own `NODE_RANK` and the same reachable `MASTER_ADDR` and `MASTER_PORT`:

```bash
torchrun --nnodes="$NNODES" --nproc-per-node=4 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" --no-python \
  scripts/validation/kimi_k3/run.sh "$MODE" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT"
```

| Mode | Nodes | Workload |
| --- | --- | --- |
| `full128` | 32 | Full K3; PP1/EP64/dense-FSDP128/expert-FSDP2; sequence16; global batch128; standard dispatcher; selective AC; 50 updates and profiler |
| `memory32` | 8 | Released 11-layer prefix; EP32/FSDP32; HybridEP; sequence2048; microbatch2; eight live forwards and ten backwards; no optimizer update or PP transport |
| `full256` | 64 | Full K3; TP1/PP8/CP1/EP32/DP32; sequence2048; global batch4096; microbatch2 x64; HybridEP; fixed balanced routing IDs with learned scores; fused CE; 50 updates and profiler |

Keep each EP64 or EP32 group inside one physical NVLink domain. Use block rank
ordering. For Slurm, `validate_ep_placement.py` checks the allocation's actual
NVLink features before launch; the full128 placement check uses `--ep 64 --pp 2`
to describe its two EP groups (the model still uses PP1). The full256 check uses
`--ep 32 --pp 8`. The launcher does not allocate nodes or bypass placement checks.

Both full modes use BF16 master parameters, MXFP4-weight/MXFP8-activation fake
quantization with BF16 grouped GEMMs, DistMuon plus AdamW, LR `8e-4`, zero weight
decay, no LR warmup, FP32 gradient reduction, and clipping norm 1.0. They use
synthetic tokens. CPU/CUDA profiling warms up on step11 and captures step12.
Per-rank timings include gradient-probe overhead. These are functional tests,
not a reproduction of the published pretraining recipe or convergence evidence.

## Validation evidence

| Experiment | Result |
| --- | --- |
| Full128 `3199574` | All128 ranks completed50 finite updates with nonzero dense/expert updates and populated Muon momentum; all128 CPU/CUDA traces independently parsed |
| Full128 memory/timing | Maximum training-step PyTorch allocation150.65 GB/GPU; steady live allocation stable; host RSS increased242-276 MB/rank; median maximum-rank step13.73 seconds at sequence16 |
| DCP comparison `3199945` | Default NCCL planning added12.717 GB outside PyTorch on rank0; Gloo added0; both32-rank arms loaded exact tensors and passed a subsequent NCCL reduction |
| Released memory32 `3200251` | All32 reports PASS: ten backwards, finite FP32 accumulated gradients, active dense/expert gradients, zero retained unsharded parameter storage; peak allocation174.37 GB |
| Full256 | Pending; no 50-step throughput, optimizer-memory-stability, or full256 profile claim |

The safe watchdog cannot report a Python stack while an extension permanently
holds the GIL. Do not reintroduce the native timed faulthandler that crashed an
earlier diagnostic. Runtime-specific transport flags are not general model fixes.

Run the focused helper regressions in the matching environment:

```bash
PYTHONPATH="$PWD/scripts/validation/kimi_k3:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest scripts/validation/kimi_k3/test_*.py
```
