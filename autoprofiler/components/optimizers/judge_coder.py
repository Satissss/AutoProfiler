"""Judge-then-Coder optimization, the structure CudaForge introduced.

Each round a *Judge* LLM reads the profile (or, if the last attempt is broken, its error log)
and names exactly one issue and one fix; a *Coder* LLM then rewrites the kernel to apply it.
Prompt wording is adapted from CudaForge (MIT licence, see NOTICE).
"""
from __future__ import annotations

import json

from autoprofiler.codegen.text import extract_json, extract_python_block
from autoprofiler.components.gpu_specs import gpu_block
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Optimizer
from autoprofiler.core.types import Candidate, Profile, Step, Task, best_step

JUDGE_OPT_SYSTEM = """\
You are a senior {lang} performance engineer. Read the target GPU spec, the PyTorch reference,
the current candidate, and the Nsight Compute metrics. Identify exactly one highest-impact speed
bottleneck, propose exactly one optimisation method and a modification plan. Be surgical and
metrics-driven.

Rules:
- Return one and only one optimisation method: the one with the largest expected speedup.
- Prefer changes that directly address measured bottlenecks (occupancy limits, memory coalescing,
  shared-memory bank conflicts, register pressure, scoreboard stalls, tensor-core underutilisation).
- Keep fields brief; no alternatives, disclaimers or generic advice.
{extra_rules}
Output format (JSON):
```json
{{"bottleneck": "<max 30 words>", "optimisation method": "<max 35 words>", "modification plan": "<max 35 words>"}}
```"""

JUDGE_REPAIR_SYSTEM = """\
You are a senior {lang} + PyTorch correctness auditor. Read the error log, the PyTorch reference and
the candidate, and report exactly one most critical issue that causes the failure or a behavioural
mismatch. Prefer semantic/correctness issues over micro-optimisations. Be terse.

Output format (JSON):
```json
{{"critical_issue": "<max 20 words>", "why_it_matters": "<max 35 words>", "minimal_fix_hint": "<max 20 words>"}}
```"""

CODER_RULES = {
    "cuda": ("Write the whole file again as one ```python block: imports (`torch`, `torch.nn`, `load_inline`), "
             "`source` CUDA string(s), `cpp_src` prototypes, one `load_inline` call per kernel group, then "
             "`class ModelNew(nn.Module)` mirroring the original inputs/outputs. No test code, no prose."),
    "triton": ("Write the whole file again as one ```python block using `triton` / `triton.language` "
               "(`@triton.jit` kernels launched from `class ModelNew(nn.Module)`). Keep the computation in "
               "Triton kernels (do not just call the original torch ops). No test code, no prose."),
}

CODER_OPT = """\
# Target GPU
{gpu}

# Previously generated kernels
{history}

You are a GPU-kernel optimisation specialist. Analyse the architecture and kernel history and strictly
apply the following STRATEGY to produce an improved kernel.

[ARCHITECTURE FILE]
```python
{arch}
```

[CURRENT BEST KERNEL]
```python
{code}
```

[OPTIMISATION INSTRUCTIONS]
{suggestion}

GOAL: lower latency on the target GPU, keep outputs within atol/rtol 1e-4, preserve the public API
(same inputs, outputs, shapes, dtypes).
OUTPUT RULES: {rules}
"""

CODER_REPAIR = """\
# Target GPU
{gpu}

[ARCHITECTURE FILE]
```python
{arch}
```

[FAILING KERNEL]
```python
{code}
```

[ERROR LOG]
{error}

[AUDITOR'S DIAGNOSIS]
{suggestion}

Fix the kernel so it compiles, runs and matches the reference; preserve the public API.
OUTPUT RULES: {rules}
"""


def tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else "...\n" + text[-n:]


def history_block(steps: list[Step], limit: int = 3) -> str:
    """Recent attempts (code + outcome) for the 'previously generated kernels' section."""
    if not steps:
        return "(None)"
    parts = []
    for s in steps[-limit:]:
        r = s.result
        outcome = f"speedup {r.speedup:.3f}x" if r.ok and r.speedup else f"FAILED at {r.stage}"
        parts.append(f"## Attempt (round {s.round}), {outcome}\n```python\n{s.candidate.code}\n```")
    return "\n\n".join(parts)


class JudgeCoder(Optimizer):
    name = "judge_coder"

    class Params(Optimizer.Params):
        language: str = "cuda"  # cuda | triton
        gpu: str = "RTX-4090"
        n_samples: int = 1  # Coder drafts per round (best-of-N)
        temperature: float = 1.0
        max_tokens: int = 16384

    def __init__(self, **params) -> None:
        super().__init__(**params)
        if self.params.language not in CODER_RULES:
            raise ValueError(f"language must be one of {sorted(CODER_RULES)}, got {self.params.language!r}")

    # -- hooks for variants that add memory / extra context -------------------------
    def judge_extra_rules(self) -> str:
        return ""

    def judge_extra_context(self, history: list[Step]) -> str:
        return ""

    def strategy_label(self, strategy: dict) -> str | None:
        """Short name of the method the Judge chose (recorded on the candidate)."""
        return strategy.get("optimisation method")

    # -- Optimizer -----------------------------------------------------------------
    def parent(self, history: list[Step]) -> Step | None:
        # A broken last attempt is repaired from its error log; there is nothing to profile.
        return best_step(history) if history[-1].result.ok else None

    def propose(self, task: Task, history: list[Step], profile: Profile, env: Env) -> list[Candidate]:
        last = history[-1]
        repair = not last.result.ok
        strategy = self._judge(task, history, profile, env, repair)
        suggestion = json.dumps(strategy, indent=2) if strategy else "(no specific suggestion)"
        p = self.params
        rules = CODER_RULES[p.language]
        if repair:
            user = CODER_REPAIR.format(gpu=gpu_block(p.gpu), arch=task.ref_code.strip(), code=last.candidate.code,
                                       error=tail(last.result.error or last.result.stage),
                                       suggestion=suggestion, rules=rules)
        else:
            best = best_step(history)
            user = CODER_OPT.format(gpu=gpu_block(p.gpu), history=history_block(history), arch=task.ref_code.strip(),
                                    code=best.candidate.code, suggestion=suggestion, rules=rules)
        out: list[Candidate] = []
        for i in range(p.n_samples):
            tag = f"  coder {i + 1}/{p.n_samples}" if p.n_samples > 1 else "  coder"
            with env.step(f"{tag} ({'repair' if repair else 'optimize'})") as done:
                raw = env.llm.chat(
                    [{"role": "system", "content": f"You write {p.language} GPU kernels. Return one complete Python file."},
                     {"role": "user", "content": user}],
                    temperature=p.temperature, max_tokens=p.max_tokens,
                )
                try:
                    code = extract_python_block(raw)
                except (ValueError, SyntaxError) as e:
                    done(f"no usable code ({type(e).__name__})")
                    continue
                done(f"{len(code)} chars")
            out.append(Candidate(code, p.language, self.name,
                                 {"strategy": strategy, "method": self.strategy_label(strategy), "repair": repair}))
        return out

    def _judge(self, task: Task, history: list[Step], profile: Profile, env: Env, repair: bool) -> dict:
        p = self.params
        lang = "CUDA" if p.language == "cuda" else "Triton"
        last = history[-1]
        if repair:
            system = JUDGE_REPAIR_SYSTEM.format(lang=lang)
            user = (f"ERROR_LOG:\n{tail(last.result.error or last.result.stage)}\n\n"
                    f"PyTorch reference:\n{task.ref_code.strip()}\n\n{lang} candidate:\n{last.candidate.code}\n")
        else:
            best = best_step(history)
            system = JUDGE_OPT_SYSTEM.format(lang=lang, extra_rules=self.judge_extra_rules())
            user = (f"# Target GPU\n{gpu_block(p.gpu)}\n\n# PyTorch reference\n{task.ref_code.strip()}\n\n"
                    f"# {lang} candidate\n```python\n{best.candidate.code}\n```\n\n"
                    f"# Nsight Compute metrics\n{profile.text or '(no profiling data)'}\n"
                    f"{self.judge_extra_context(history)}\nFollow the rules exactly and return the JSON.")
        with env.step(f"  judge ({'repair' if repair else 'optimize'})") as done:
            raw = env.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                               temperature=p.temperature, max_tokens=2048)
            strategy = extract_json(raw)
            key = "critical_issue" if repair else "optimisation method"
            done(str(strategy.get(key) or "no verdict")[:70])
        return strategy
