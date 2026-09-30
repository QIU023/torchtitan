#!/usr/bin/env bash
# Invoke through torchrun --no-python, or once per Slurm rank.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_RUNTIME_CONNECT=0 NCCL_NET_PLUGIN=none
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if [[ -n ${SLURM_PROCID:-} && -z ${RANK:-} ]]; then
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
fi
MODE=${1:?Usage: run.sh full128|full256|memory32 --checkpoint PATH --output PATH}
shift
case "$MODE" in
    full128)
        test "${WORLD_SIZE:?Distributed launcher required}" = 128
        exec python "$SCRIPT_DIR/validate_distmuon.py" \
            --flavor Kimi-K3 --ep 64 --pp 1 --seq-len 16 --steps 50 \
            --profile-skip-steps 10 --dispatcher standard \
            --activation-checkpoint selective --diagnose-steps 2 "$@"
        ;;
    full256)
        test "${WORLD_SIZE:?Distributed launcher required}" = 256
        exec python "$SCRIPT_DIR/validate_distmuon.py" \
            --flavor Kimi-K3 --ep 32 --pp 8 --seq-len 2048 \
            --microbatch-size 2 --microbatches 64 --steps 50 \
            --profile-skip-steps 10 --dispatcher hybridep --fused-loss \
            --balanced-routing --activation-checkpoint recompute-matmuls \
            --reshard-after-backward --diagnose-steps 2 \
            --pipeline-startup-timeout 1800 --pipeline-steady-timeout 600 "$@"
        ;;
    memory32)
        test "${WORLD_SIZE:?Distributed launcher required}" = 32
        exec python "$SCRIPT_DIR/hybridep_stage_memory.py" "$@"
        ;;
    *)
        echo "Unknown validation mode: $MODE" >&2
        exit 2
        ;;
esac
