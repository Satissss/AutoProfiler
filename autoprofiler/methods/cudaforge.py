"""CudaForge (Zhang et al., MIT): a CUDA Coder plus a Judge that reads a fixed Nsight Compute set.

Each round the Judge names the single most important bottleneck from 24 fixed counters (or, if
the kernel is broken, its error log) and the Coder applies one fix. Here it runs on the shared
components, so only the method differs from the others in a comparison.
"""
from __future__ import annotations

from autoprofiler.components.optimizers import JudgeCoder
from autoprofiler.components.profilers import NCUFixedProfiler
from autoprofiler.components.seeders import LLMCudaSeeder
from autoprofiler.core.interfaces import Method, Optimizer, Profiler, Seeder
from autoprofiler.core.iterative import IterativeMethod

class CudaForge(IterativeMethod):
    name = "cudaforge"

    class Params(IterativeMethod.Params):
        gpu: str = "RTX-4090"   # spec sheet quoted to the model

    def make_seeder(self) -> Seeder:
        return LLMCudaSeeder(gpu=self.params.gpu, attempts=4, temperature=1.0)

    def make_profiler(self) -> Profiler:
        return NCUFixedProfiler(metric_set="cudaforge")

    def make_optimizer(self) -> Optimizer:
        return JudgeCoder(language="cuda", gpu=self.params.gpu, temperature=1.0)
