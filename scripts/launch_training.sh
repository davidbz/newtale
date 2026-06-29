#!/usr/bin/env bash
# Launch training with DeepSpeed or torchrun.
#
# Single-node:
#   bash scripts/launch_training.sh --config configs/3b.yaml --backend deepspeed --num_gpus 8
#
# Multi-node (set MASTER_ADDR, MASTER_PORT, NNODES, NODE_RANK in environment):
#   MASTER_ADDR=node0 NNODES=8 bash scripts/launch_training.sh \
#       --config configs/3b.yaml --backend deepspeed --num_gpus 8

set -euo pipefail

CONFIG=""
BACKEND="deepspeed"
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";   shift 2 ;;
        --backend)  BACKEND="$2";  shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 --config <yaml> [--backend deepspeed|fsdp] [--num_gpus N]"
    exit 1
fi

NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

case "$BACKEND" in
    deepspeed)
        HOSTFILE_ARG=""
        if [[ "$NNODES" -gt 1 ]] && [[ -f hostfile ]]; then
            HOSTFILE_ARG="--hostfile hostfile"
        fi
        # shellcheck disable=SC2086
        deepspeed \
            --num_gpus "$NUM_GPUS" \
            ${HOSTFILE_ARG} \
            train.py --config "$CONFIG"
        ;;
    fsdp)
        torchrun \
            --nproc_per_node "$NUM_GPUS" \
            --nnodes "$NNODES" \
            --rdzv_backend c10d \
            --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
            train.py --config "$CONFIG"
        ;;
    *)
        echo "Unknown backend: $BACKEND (expected deepspeed or fsdp)"
        exit 1
        ;;
esac
