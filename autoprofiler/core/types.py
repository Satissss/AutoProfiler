"""Plain data passed between components."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """One benchmark problem: a PyTorch reference defining Model/get_inputs/get_init_inputs."""

    id: str
    benchmark: str
    group: str  # e.g. "level1"
    ref_code: str
    ref_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _trim(text: str, head: int = 200, tail: int = 1200) -> str:
    """Keep both ends of a long error.

    nvcc puts the compile command first and the reason it failed last, so trimming
    to the first N characters preserves the flags and throws away the diagnosis.
    """
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n  … {len(text) - head - tail} chars elided …\n{text[-tail:]}"


@dataclass
class EvalResult:
    ok: bool  # compiled, ran, and matched the reference
    stage: str = ""  # where it failed (see backends/workers/seed_verify.py); "ok" on success
    error: str = ""
    baseline_ms: float | None = None  # PyTorch reference runtime
    candidate_ms: float | None = None
    speedup: float | None = None  # baseline_ms / candidate_ms

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["error"] = _trim(d["error"] or "")
        return d


@dataclass
class Candidate:
    """A program defining ``ModelNew`` that should match the task's ``Model``."""

    code: str
    language: str = "triton"  # torch | triton | cuda
    origin: str = ""  # which component produced it
    meta: dict[str, Any] = field(default_factory=dict)
    # Set when the producer already evaluated it (e.g. to drive its own repair turn); the
    # method then reuses it instead of paying for a second run.
    result: EvalResult | None = None


@dataclass
class Profile:
    """What a profiler learned about a candidate, as data and as prompt text."""

    metrics: dict[str, float] = field(default_factory=dict)
    text: str = ""  # rendered for the optimizer's prompt; empty means "no profiling"
    info: dict[str, Any] = field(default_factory=dict)  # e.g. tree path, leaf id


@dataclass
class Step:
    """One attempt: what was tried and how it scored."""

    round: int
    candidate: Candidate
    result: EvalResult
    profile: Profile | None = None
    usage: dict[str, Any] = field(default_factory=dict)  # cost of the round this step closed


@dataclass
class Trajectory:
    """Everything a method did on one task; ``best`` is what gets scored."""

    task_id: str
    method: str
    steps: list[Step] = field(default_factory=list)
    best: Candidate | None = None
    best_result: EvalResult | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def record(self, step: Step) -> None:
        """Append a step and keep ``best`` as the fastest correct candidate so far."""
        self.steps.append(step)
        r = step.result
        if r.ok and r.speedup is not None and (
            self.best_result is None or r.speedup > (self.best_result.speedup or 0.0)
        ):
            self.best, self.best_result = step.candidate, r

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "best_code": self.best.code if self.best else "",
            "best_language": self.best.language if self.best else "",
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "rounds": [
                {"round": s.round, "result": s.result.to_dict(), "usage": s.usage,
                 "profile": s.profile.info if s.profile else {}}
                for s in self.steps
            ],
            "usage": self.usage,
            "error": self.error[:500],
        }


def best_step(history: list[Step]) -> Step | None:
    """The fastest correct step so far (what profilers and optimizers build on)."""
    ok = [s for s in history if s.result.ok and s.result.speedup is not None]
    return max(ok, key=lambda s: s.result.speedup) if ok else None
