#!/usr/bin/env bash
# Train on the full 999-program corpus. Download it first (see README, "Data"):
# it must be at data/train/full/seed_index.jsonl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -f data/train/full/seed_index.jsonl ]] || {
  echo "missing data/train/full/seed_index.jsonl (see README: Data)" >&2; exit 1; }
scripts/train.sh configs/train/default.yaml --data data/train/full --out output/trees/trained.json
