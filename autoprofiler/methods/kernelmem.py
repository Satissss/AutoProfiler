"""KernelMem: CudaForge plus memory.

Written from the public README only (the upstream repository carries no licence, so no code or
prompt text was copied). The memory has two parts: a log of the optimisation methods already tried
in this run with their outcomes (short-term), and bottleneck-keyed optimisation notes (long-term).
The Judge is told not to repeat a method that failed or slowed the kernel. The counter set is
extended with memory-system and launch-shape counters. Not reproduced: the curated memory bank,
machine-check method matching and Nsight Systems metrics (see docs/baselines.md).
"""
from __future__ import annotations

from autoprofiler.components.optimizers import JudgeCoder
from autoprofiler.components.profilers import NCUFixedProfiler
from autoprofiler.core.interfaces import Optimizer, Profiler
from autoprofiler.core.types import Step
from autoprofiler.methods.cudaforge import CudaForge

KNOWLEDGE = """\
- DRAM throughput high, SM throughput low -> memory-bound: vectorised/coalesced loads, fuse passes, cut bytes (fp16/bf16 where tolerance allows), reuse via shared memory.
- SM / tensor pipe low on a matmul-like op -> compute-bound with idle tensor cores: tiling + tensor-core MMA (tf32/fp16), larger tiles, double buffering.
- Low achieved occupancy with register or shared-memory limit -> shrink tile / registers per thread, adjust block size.
- Many tiny kernels, low per-kernel utilisation -> launch-bound: fuse kernels, CUDA graphs, persistent kernels.
- High long-scoreboard stall -> poor latency hiding: more loads in flight, prefetch, increase occupancy.
- High barrier stall / bank conflicts -> restructure shared-memory tiles (padding), reduce __syncthreads.
- L2/L1 hit rate low with strided access -> reorder loops or tile to improve locality."""

MEMORY_RULE = (
    "- You are given a log of optimisation methods already tried this run, with their outcomes. Do NOT repeat\n"
    "  a method that failed or slowed the kernel; prefer an untried method. Use the knowledge notes below.\n"
)


class MemoryJudgeCoder(JudgeCoder):
    name = "memory_judge_coder"

    def judge_extra_rules(self) -> str:
        return MEMORY_RULE

    def judge_extra_context(self, history: list[Step]) -> str:
        rows = []
        for s in history:
            method = s.candidate.meta.get("method")
            if method:
                r = s.result
                rows.append(f"- round {s.round}: {method} -> "
                            + (f"{r.speedup:.3f}x" if r.ok and r.speedup else f"failed ({r.stage})"))
        tried = "\n".join(rows) or "(nothing yet)"
        return f"\n# Methods tried so far\n{tried}\n\n# Knowledge notes\n{KNOWLEDGE}\n"


class KernelMem(CudaForge):
    name = "kernelmem"

    def make_profiler(self) -> Profiler:
        return NCUFixedProfiler(metric_set="kernelmem")

    def make_optimizer(self) -> Optimizer:
        return MemoryJudgeCoder(language="cuda", gpu=self.params.gpu, temperature=1.0)
