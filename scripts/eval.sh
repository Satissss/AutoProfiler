#!/usr/bin/env bash
# Run an experiment. One config per method, under configs/eval/.
#   scripts/eval.sh configs/eval/cudaforge.yaml                      # the full run
#   scripts/eval.sh configs/eval/cudaforge.yaml --limit 3 --rounds 2 # a quick check
#   scripts/eval.sh configs/eval/cudaforge.yaml --fresh              # ignore earlier results
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m autoprofiler eval "$@"
