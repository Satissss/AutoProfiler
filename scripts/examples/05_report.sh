#!/usr/bin/env bash
# Rebuild the results table (results.md / summary.json) from per-task results; needs no GPU or LLM.
#   scripts/examples/05_report.sh [RUN_DIR]     (default: output/results/autoprofiler)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m autoprofiler report "${1:-output/results/autoprofiler}"
