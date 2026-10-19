"""The four kinds of component. A *method* is a recipe over the other three.

    Seeder     task                            -> first candidate
    Profiler   candidate                       -> what to tell the optimizer about it
    Optimizer  history + profile               -> new candidate(s)
    Method     task, env                       -> Trajectory   (owns the loop and the budget)

Everything a component needs from the outside world arrives through :class:`Env`, so all
methods run on the same LLM, GPU backend and evaluator by construction.
"""
from __future__ import annotations

from abc import abstractmethod

from autoprofiler.core.component import Component
from autoprofiler.core.env import Env
from autoprofiler.core.types import Candidate, Profile, Step, Task, Trajectory, best_step


class Seeder(Component):
    """Produces the first candidate for a task, or ``None`` if it cannot."""

    kind = "seeder"

    @abstractmethod
    def seed(self, task: Task, env: Env) -> Candidate | None: ...


class Profiler(Component):
    """Measures a candidate. ``Profile.text`` is what the optimizer will read."""

    kind = "profiler"

    @abstractmethod
    def profile(self, task: Task, cand: Candidate, env: Env) -> Profile: ...


class Optimizer(Component):
    """Proposes new candidates given everything tried so far and a fresh profile."""

    kind = "optimizer"

    def parent(self, history: list[Step]) -> Step | None:
        """The step this round builds on, i.e. what gets profiled. ``None`` means nothing is
        worth profiling (e.g. the last attempt failed and is about to be repaired)."""
        return best_step(history)

    @abstractmethod
    def propose(self, task: Task, history: list[Step], profile: Profile, env: Env) -> list[Candidate]: ...


class Method(Component):
    """A complete kernel-optimization method: what the evaluation framework compares.

    ``run`` takes the benchmark problem, not a kernel, because producing the first
    kernel is part of the method (CudaForge writes CUDA, AutoProfiler ports to Triton).
    Pass ``start`` to skip that and optimize a kernel you already have.
    """

    kind = "method"

    @abstractmethod
    def run(self, task: Task, env: Env, start: Candidate | None = None) -> Trajectory: ...
