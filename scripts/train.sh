#!/usr/bin/env bash
# Train a profiling tree.
#
#   scripts/train.sh                                   # configs/train/default.yaml
#   scripts/train.sh configs/train/my.yaml             # another config
#   scripts/train.sh --out output/trees/my_tree.json   # extra `autoprofiler train` flags
#
# The GPU backend is detected automatically: this machine's GPU if it has one, else the
# k8s pod in configs/backends/k8s.yaml. Override with BACKEND=<path> or --backend <path>.
# The corpus, the models for each stage and where the tree is written all come from the config.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="configs/train/default.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"; shift
fi
[[ -f "$CONFIG" ]] || { echo "train: no such config: $CONFIG" >&2; exit 1; }

# ── Pick a GPU backend ───────────────────────────────────────────────────────
# Skipped when the caller passes --backend explicitly.
if [[ -z "${BACKEND:-}" ]] && [[ ! " $* " =~ " --backend " ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    BACKEND="configs/backends/local.yaml"
    GPU_DESC="local: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    command -v ncu >/dev/null 2>&1 || {
      echo "train: Nsight Compute (ncu) is not on PATH; profiling needs it." >&2
      echo "       Install the CUDA toolkit, or point --backend at a pod that has it." >&2
      exit 1
    }
  elif [[ -f configs/backends/k8s.yaml ]]; then
    BACKEND="configs/backends/k8s.yaml"
    GPU_DESC="k8s: $(grep -E '^\s*deployment:' configs/backends/k8s.yaml | head -1 | sed 's/.*: *//')"
  else
    echo "train: no GPU found on this machine and no configs/backends/k8s.yaml." >&2
    echo "       Training needs a CUDA GPU with Nsight Compute. Either run this where the GPU is," >&2
    echo "       or copy configs/backends/k8s.example.yaml to configs/backends/k8s.yaml and fill it in." >&2
    exit 1
  fi
  set -- --backend "$BACKEND" "$@"
else
  GPU_DESC="${BACKEND:-from --backend}"
  [[ -n "${BACKEND:-}" ]] && set -- --backend "$BACKEND" "$@"
fi

echo "[train] config : $CONFIG"
echo "[train] gpu    : $GPU_DESC"
exec python -m autoprofiler train "$CONFIG" "$@"
