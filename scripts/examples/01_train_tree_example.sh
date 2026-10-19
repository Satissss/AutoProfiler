#!/usr/bin/env bash
# Smoke-test the training loop on the 100-program example corpus shipped in git.
# Exercises the whole pipeline (GPU + NCU + LLM) on 20 programs; not meant to match the released tree.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
scripts/train.sh configs/train/default.yaml --limit 20 --out output/trees/example.json
