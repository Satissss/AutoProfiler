"""AutoKernel (RightNow AI, MIT): an autonomous keep/revert experiment loop.

Each round is one experiment: hypothesise, make ONE focused edit to the current kept kernel,
benchmark it, KEEP it if correct and at least 1% faster, otherwise REVERT, and log it. Prompt
wording and the optimisation playbook are adapted from AutoKernel's program.md (see NOTICE).
Not reproduced: model-level profiling and multi-kernel orchestration (KernelBench problems are
single programs).
"""
from __future__ import annotations

from autoprofiler.codegen.text import extract_python_block
from autoprofiler.components.profilers import NCUFixedProfiler
from autoprofiler.components.seeders import LLMTritonSeeder
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Method, Optimizer, Profiler, Seeder
from autoprofiler.core.iterative import IterativeMethod
from autoprofiler.core.types import Candidate, Profile, Step, Task

KEEP_THRESHOLD = 1.01  # "improved" means at least a 1% gain

SYSTEM = """\
You are an autonomous GPU kernel optimisation researcher. You improve one kernel through a series of
experiments. Each experiment is ONE focused change (do not combine unrelated optimisations), so the
effect of the change is attributable. A change is kept only if the result is correct and at least 1%
faster than the current kept kernel; otherwise it is reverted.

Optimisation playbook (roughly in order of expected payoff):
1. Block/tile sizes and num_warps/num_stages tuning.
2. Memory access: coalescing, vectorised loads, avoid redundant reads, fuse adjacent elementwise ops.
3. Compute: use tensor cores (tf32/fp16 via tl.dot), reduce instruction count, fp32 accumulation.
4. Advanced: software pipelining/prefetch, L2-friendly tile ordering (swizzle), split-K, persistent kernels.
Anti-patterns that usually do not work: over-large tiles that spill registers, extra synchronisation,
changing numerics beyond tolerance.

Reply with a 1-2 sentence hypothesis, then the complete new file in one ```python block."""

USER = """\
# Reference (PyTorch)
```python
{arch}
```

# Current kept kernel: {speed}
```python
{code}
```

# Roofline / profiling data for the current kept kernel
{profile}

# Experiment log (oldest first)
{log}

Propose the next single focused experiment."""


def kept_and_log(history: list[Step]) -> tuple[Step | None, str]:
    """Replay the keep/revert rule over the history: the kernel kept so far, and the log."""
    kept: Step | None = None
    rows: list[str] = []
    for s in history:
        r = s.result
        hyp = (s.candidate.meta.get("hypothesis") or "initial kernel")[:160]
        if not (r.ok and r.speedup):
            rows.append(f"- exp {s.round}: {hyp} -> FAILED ({r.stage}) [REVERT]")
        elif kept is None or r.speedup >= kept.result.speedup * KEEP_THRESHOLD:
            kept = s
            rows.append(f"- exp {s.round}: {hyp} -> {r.speedup:.3f}x [KEEP]")
        else:
            rows.append(f"- exp {s.round}: {hyp} -> {r.speedup:.3f}x [REVERT]")
    return kept, "\n".join(rows) or "(none)"


class EditLoop(Optimizer):
    name = "edit_loop"

    class Params(Optimizer.Params):
        language: str = "triton"
        temperature: float = 0.7
        max_tokens: int = 16384

    def parent(self, history: list[Step]) -> Step | None:
        return kept_and_log(history)[0]

    def propose(self, task: Task, history: list[Step], profile: Profile, env: Env) -> list[Candidate]:
        kept, log = kept_and_log(history)
        if kept is None:  # nothing correct yet: experiment on the latest attempt, whose error is in the log
            code, speed = history[-1].candidate.code, f"does not run yet ({history[-1].result.error[:600]})"
        else:
            code, speed = kept.candidate.code, f"{kept.result.speedup:.3f}x vs PyTorch"
        raw = env.llm.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER.format(arch=task.ref_code.strip(), code=code, speed=speed,
                                                     profile=profile.text or "(none)", log=log)}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
        )
        try:
            new_code = extract_python_block(raw)
        except (ValueError, SyntaxError):
            return []
        hypothesis = raw.split("```", 1)[0].strip().replace("\n", " ")
        return [Candidate(new_code, self.params.language, self.name, {"hypothesis": hypothesis})]


class AutoKernel(IterativeMethod):
    name = "autokernel"

    def make_seeder(self) -> Seeder:
        return LLMTritonSeeder(attempts=4, temperature=0.2)

    def make_profiler(self) -> Profiler:
        return NCUFixedProfiler(metric_set="roofline")

    def make_optimizer(self) -> Optimizer:
        return EditLoop(language="triton", temperature=0.7)
