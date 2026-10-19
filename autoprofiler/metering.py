"""Cost accounting that stays correct when tasks run concurrently.

Anything that spends LLM tokens or GPU time calls :func:`charge_llm` / :func:`charge_gpu`;
the charge lands on every :func:`metering` scope active in the *calling* thread, so a
per-task (or per-round) meter sees only its own spending even with many tasks in flight.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


class Meter:
    """Thread-safe counters for LLM calls / tokens and GPU evaluations."""

    __slots__ = ("calls", "prompt_tokens", "completion_tokens", "gpu_seconds", "n_evals", "_lock")

    def __init__(self) -> None:
        self.calls = self.prompt_tokens = self.completion_tokens = self.n_evals = 0
        self.gpu_seconds = 0.0
        self._lock = threading.Lock()

    def add(self, prompt: int = 0, completion: int = 0) -> None:
        """Alias kept for backward compat (tests call .add())."""
        return self.add_llm(prompt, completion)

    def add_llm(self, prompt: int = 0, completion: int = 0) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(prompt or 0)
            self.completion_tokens += int(completion or 0)

    def add_gpu(self, seconds: float) -> None:
        with self._lock:
            self.gpu_seconds += float(seconds)
            self.n_evals += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "gpu_seconds": round(self.gpu_seconds, 2),
                "n_evals": self.n_evals,
            }


_active: ContextVar[tuple[Meter, ...]] = ContextVar("autoprofiler_meters", default=())


@contextmanager
def metering() -> Iterator[Meter]:
    """Count everything charged inside the ``with`` block (in this thread/context)."""
    meter = Meter()
    token = _active.set(_active.get() + (meter,))
    try:
        yield meter
    finally:
        _active.reset(token)


def charge_llm(prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    for m in _active.get():
        m.add_llm(prompt_tokens, completion_tokens)


def charge_gpu(seconds: float) -> None:
    for m in _active.get():
        m.add_gpu(seconds)
