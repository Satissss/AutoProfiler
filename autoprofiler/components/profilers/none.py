from __future__ import annotations

from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Profiler
from autoprofiler.core.types import Candidate, Profile, Task


class NoProfiler(Profiler):
    """No measurement: the optimizer sees only code and past results ("blind")."""

    name = "none"

    def profile(self, task: Task, cand: Candidate, env: Env) -> Profile:
        return Profile()
