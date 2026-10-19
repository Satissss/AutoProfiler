"""KernelAgent (Meta, Apache-2.0): Triton generation, then roofline-guided analysis and optimisation.

Kept: Triton kernels, verified-then-repaired seeding, bottleneck analysis from speed-of-light
counters, and a small beam of candidates per round (``n_samples``). Not reproduced: the parallel
multi-worker orchestrator and its agent roles (see docs/baselines.md).
"""
from __future__ import annotations

from autoprofiler.components.optimizers import JudgeCoder
from autoprofiler.components.profilers import NCUFixedProfiler
from autoprofiler.components.seeders import LLMTritonSeeder
from autoprofiler.core.interfaces import Method, Optimizer, Profiler, Seeder
from autoprofiler.core.iterative import IterativeMethod


class KernelAgent(IterativeMethod):
    name = "kernelagent"

    class Params(IterativeMethod.Params):
        gpu: str = "RTX-4090"
        beam: int = 2           # candidates drafted per round

    def make_seeder(self) -> Seeder:
        return LLMTritonSeeder(attempts=4, temperature=0.2)

    def make_profiler(self) -> Profiler:
        return NCUFixedProfiler(metric_set="roofline")

    def make_optimizer(self) -> Optimizer:
        return JudgeCoder(language="triton", gpu=self.params.gpu,
                           n_samples=self.params.beam, temperature=0.7)
