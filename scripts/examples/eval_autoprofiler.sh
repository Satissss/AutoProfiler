#!/usr/bin/env bash
# AutoProfiler on KernelBench (see configs/eval/autoprofiler.yaml).
# Pass through extra flags, e.g. --limit 3 --rounds 2 for a short run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
scripts/eval.sh configs/eval/autoprofiler.yaml "$@"
