#!/usr/bin/env bash
# SLURM launcher for multi-node NewTale pretraining.
#
# Usage:
#   sbatch scripts/launch_slurm.sh --config configs/3b.yaml
#
# Override defaults with env vars before sbatch:
#   NNODES=4 GPUS_PER_NODE=8 PARTITION=gpu sbatch scripts/launch_slurm.sh ...
#
#SBATCH --job-name=newtale-pretrain
#SBATCH --nodes=${NNODES:-1}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE:-8}
#SBATCH --gres=gpu:${GPUS_PER_NODE:-8}
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# srun fills SLURM_NODELIST; pick the first node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT

# NCCL tuning for InfiniBand / RoCE clusters
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------
srun \
    --label \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.run \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --nnodes="${NNODES}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        train.py "$@"
