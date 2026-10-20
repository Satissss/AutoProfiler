# AutoProfiler

<div align="center">

**Less is more: learn *what to profile* for LLM-driven GPU kernel optimization**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

![AutoProfiler Overview](assets/main.png)

</div>

---

## Overview

**The problem:** LLM-based kernel optimizers increasingly rely on hardware profiling, yet most systems still use a *fixed* profiling configuration for every kernel. Profiling is costly to acquire, and its utility is inherently kernel-dependent—adding more metrics does not monotonically improve optimization. A single static setup cannot be optimal for all kernels.

**Our approach:** **AutoProfiler** learns a deterministic *profiling tree* from optimization feedback: nodes measure a few counters, route by the readings, and deliver path-specific metrics to a single kernel rewrite. An LLM proposes **REWRITE** / **SPLIT** edits during training; empirical evaluation accepts or rejects them. At inference, routing is counter-driven only—**profiling reasoning** under a fixed budget, not a fixed metric list for every kernel.

### Key Features

- 🎯 **Learned profiling policy**: Feedback-driven training of a profiling tree tailored to downstream speedups
- 🧭 **Kernel-conditioned selection**: Sequential, counter-driven routing instead of one-size-fits-all metric sets
- ⚡ **Budget-aware measurement**: Acquire only the signals needed along each path—no redundant rounds of LLM-guided profiling at inference
- 🔬 **Reproducible experiments**: Config-driven runs with checkpoint/resume support

---

## Quick Start

### 1. Install

```bash
# Any machine
pip install -e ".[dev]"

# GPU machine: PyTorch + Triton
pip install -e ".[gpu]"
```

**Requirements:** Python 3.8+; CUDA workers need Nsight Compute (`ncu`) with performance-counter permissions.

Set an LLM API key (or use a `.env` file), then pick a profile under `configs/llm/` (`gpt`, `claude`, or `openai_compatible.example.yaml`):

```bash
export OPENAI_API_KEY=...
```

### 2. Run AutoProfiler & Evaluation

```bash
scripts/examples/eval_autoprofiler.sh
```

---

## Training a Tree

```bash
# Smoke test on 100-program example corpus (in repo)
scripts/examples/train_tree_example.sh

# Full training on 999-program corpus (download first)
scripts/examples/train_tree_full.sh

# Use default config
scripts/train.sh
```

---

## Project Layout

```
autoprofiler/
  core/        Task, Candidate, Trajectory; component interfaces; 
               registry; Evaluator; benchmark loader
  seeders/     Initial candidate generation: torch_baseline, llm_port
  profilers/   Profiling strategies: none, ncu_fixed, tree (AutoProfiler)
  optimizers/  Optimization backends: leaf (AutoProfiler), baselines
  baselines/   CudaForge, KernelMem, KernelAgent, AutoKernel implementations
  npt/         Tree structure, training loop, REWRITE/SPLIT, reward computation
  eval/        Unified runner and statistical reporting
  backends/    Execution: local GPU or Kubernetes pod (kubectl exec)
  llm/         Unified LLM client with call/token accounting

configs/       Configuration system
  llm/         Model endpoints and API keys
  backends/    GPU execution backends
  benchmarks/  Test suite definitions
  methods/     Method component specifications
  eval/        Complete experiment configs
  train/       Training hyperparameters

scripts/       Entry points: train.sh, eval.sh, examples/, data/
data/          Benchmarks and training corpora
output/        Results, trees, checkpoints
```

---

## Citation

```bibtex
@inproceedings{autoprofiler2026,
  title     = {AutoProfiler: Learning What to Profile for GPU Kernel Optimization},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

---

<div align="center">
Built with ❤️ for the GPU optimization community
</div>

