#!/usr/bin/env bash
# Does the evaluation stack work end to end? Three problems, two rounds, one method.
# Run this before anything long; it exercises seeding, profiling, optimizing and scoring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
scripts/eval.sh "configs/eval/${1:-autoprofiler}.yaml" --limit 3 --rounds 2
