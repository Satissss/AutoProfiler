"""Core abstractions: data types, component base, interfaces, Env.

Public surface: Component, Seeder, Profiler, Optimizer, Method, IterativeMethod,
Env, and the data types (Task, Candidate, Trajectory, …). The registry is in
:mod:`autoprofiler.core.component`; use :func:`~autoprofiler.core.component.get`
to look up and instantiate components by name.
"""
from autoprofiler.core.types import (
    Candidate, EvalResult, Profile, Step, Task, Trajectory, best_step,
)
from autoprofiler.core.env import Env
from autoprofiler.core.component import get, available, load_builtin
from autoprofiler.core.interfaces import Seeder, Profiler, Optimizer, Method
from autoprofiler.core.iterative import IterativeMethod

__all__ = [
    "Candidate", "EvalResult", "Profile", "Step", "Task", "Trajectory", "best_step",
    "Env",
    "get", "available", "load_builtin",
    "Seeder", "Profiler", "Optimizer", "Method", "IterativeMethod",
]
