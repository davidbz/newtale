#!/usr/bin/env bash
# Download and shard datasets to local disk before training.
# Requires HF_TOKEN and sufficient disk space (~500 GB for full FineWeb-Edu + StarCoder).
#
# Usage:
#   HF_TOKEN=<token> bash scripts/prepare_data.sh --output_dir /data/newtale --num_shards 16

set -euo pipefail

OUTPUT_DIR=""
NUM_SHARDS=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --num_shards)  NUM_SHARDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --output_dir <dir> [--num_shards N]"
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: HF_TOKEN environment variable is required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python - <<EOF
import os
from datasets import load_dataset

token = os.environ["HF_TOKEN"]
output_dir = "$OUTPUT_DIR"
num_shards = $NUM_SHARDS

datasets_to_download = [
    ("HuggingFaceFW/fineweb-edu", "fineweb-edu"),
    ("bigcode/starcoderdata",     "starcoderdata"),
]

for hf_path, name in datasets_to_download:
    print(f"Downloading {hf_path} ...")
    ds = load_dataset(hf_path, split="train", token=token)
    for shard_idx in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=shard_idx)
        out_path = f"{output_dir}/{name}/shard-{shard_idx:04d}.parquet"
        shard.to_parquet(out_path)
        print(f"  Saved shard {shard_idx}/{num_shards}: {out_path}")

print("Done.")
EOF
