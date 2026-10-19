# Baselines

All baselines run through the same `IterativeMethod` loop, LLM, evaluator, backend and round budget as
AutoProfiler (see the `methods:` block of any config under `configs/`). They are re-implementations on the shared components, not the
upstream code, so that only the method differs. Each keeps the mechanism the original is about and
simplifies the rest; this page lists both so results are read correctly.

| method | seeder | profiler | optimizer | language |
|---|---|---|---|---|
| autoprofiler | `llm_port` | `tree` (learned, adaptive counters) | `leaf` | Triton |
| cudaforge | `llm_cuda` | `ncu_fixed` (24 counters) | `judge_coder` | CUDA |
| kernelmem | `llm_cuda` | `ncu_fixed` (extended set) | `judge_coder(memory)` | CUDA |
| kernelagent | `llm_port` | `ncu_fixed` (roofline set) | `judge_coder` (beam 2) | Triton |
| autokernel | `llm_port` | `ncu_fixed` (roofline set) | `edit_loop` | Triton |

## CudaForge
Kept: Coder + Judge, one issue and one fix per round, the fixed Nsight Compute metric set rendered as
a per-kernel JSON block, a separate auditor call on failures.
Simplified: the optimization step builds on the best kernel so far, not the most recent one; the GPU
spec table is a small built-in one.

## KernelMem
Written from the public README only (the upstream repo has no licence). Kept: the CudaForge-style loop
plus memory, i.e. a log of optimisation methods tried this run with outcomes, and a bottleneck-keyed
knowledge note, so the Judge avoids repeating failed methods; extended counter set.
Not reproduced: the curated memory bank, machine-check method matching, Nsight Systems metrics.

## KernelAgent
Kept: Triton generation with test-driven repair, then roofline-guided bottleneck analysis and
optimisation with a small beam (`n_samples: 2`).
Not reproduced: the parallel multi-worker orchestrator and its agent roles.

## AutoKernel
Kept: the experiment protocol: one focused edit per round, kept only if correct and at least 1% faster than the
kept kernel, otherwise reverted; full experiment log in the prompt; the optimisation playbook.
Not reproduced: model-level profiling and multi-kernel orchestration (KernelBench problems are single
programs); roofline data comes from Nsight Compute counters.

## Budget accounting
`rounds` caps optimisation rounds for every method. Best-of-N candidates (`n_samples`) share a round;
their LLM calls, tokens and GPU time are recorded per task and shown in the results table so the
cost of each method is visible next to its speedup.
