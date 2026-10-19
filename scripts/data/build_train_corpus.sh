#!/usr/bin/env bash
# (Optional) rebuild the training corpus from the raw torch programs. The released corpus is
# already provided (data/train/full); you only need this to regenerate it with another LLM.
#
# It converts data/source/data.parquet
# into correctness-verified Triton programs; every later training run reads the
# frozen output instead of re-porting anything.
#
# Usage:
#   scripts/data/build_train_corpus.sh --limit 50          # pilot first, always
#   scripts/data/build_train_corpus.sh                     # full corpus
#   scripts/data/build_train_corpus.sh --report-only       # rebuild artifacts, no GPU/LLM
#   scripts/data/build_train_corpus.sh --dry-run --limit 5 # corpus sanity check only
#   OUT_DIR=output/seed_v2 scripts/data/build_train_corpus.sh
#
# Interrupting is safe: attempts.jsonl is append-only and the next run resumes
# from it, skipping rows already accepted or already out of repair budget.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-data/train/rebuilt}"
SOURCE="${SOURCE:-data/source/data.parquet}"
LLM="${LLM:-gpt}"
WAVE_SIZE="${WAVE_SIZE:-24}"
LLM_WORKERS="${LLM_WORKERS:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

RUN_DIR="output/runs"
mkdir -p "$RUN_DIR"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
RUN_LOG="$RUN_DIR/corpus_${STAMP}.log"

echo "[corpus] repo=$ROOT"
echo "[corpus] source=$SOURCE"
echo "[corpus] out=$OUT_DIR"
echo "[corpus] run log=$RUN_LOG"

"$PY" -m autoprofiler.corpus \
  --source "$SOURCE" \
  --out-dir "$OUT_DIR" \
  --llm "$LLM" \
  --wave-size "$WAVE_SIZE" \
  --llm-workers "$LLM_WORKERS" \
  --max-rounds "$MAX_ROUNDS" \
  "$@" 2>&1 | tee "$RUN_LOG"
exit "${PIPESTATUS[0]}"
