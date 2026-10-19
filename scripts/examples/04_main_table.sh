#!/usr/bin/env bash
# The main table: every method on all 250 KernelBench problems, one after another.
# Long and LLM-heavy. Interrupt-safe -- re-running resumes from per-task results.
# A single method:  scripts/eval.sh configs/eval/autoprofiler.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for m in autoprofiler cudaforge kernelmem kernelagent autokernel; do
  echo "=== $m ==="
  scripts/eval.sh "configs/eval/$m.yaml" "$@"
done
echo "Each method wrote its own table; output/results/<method>/results.md"
