"""The single source of truth for "is this candidate correct, and how fast".

Methods use it inside their loops (cheap timing); the runner uses it again for the final
score (strict timing). Both go through the same GPU worker, so a Triton or CUDA candidate
is judged by exactly the same code no matter which method produced it.
"""
from __future__ import annotations

import itertools
import os
import re
import threading
import time
import uuid

from autoprofiler import metering
from autoprofiler.backends import Backend
from autoprofiler.backends.verify import RemoteVerifier, VerifyRequest, VerifyResult
from autoprofiler.config import ExecConfig
from autoprofiler.core.types import Candidate, EvalResult, Task

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class Evaluator:
    def __init__(
        self,
        backend: Backend,
        cfg: ExecConfig,
        *,
        fast: tuple[int, int] = (3, 10),  # (warmup, repeats) used inside method loops
        final: tuple[int, int] = (10, 100),  # used for the reported score
        row_timeout_s: int = 300,
    ) -> None:
        self.backend = backend
        self._verifiers = {
            False: RemoteVerifier(backend, cfg, warmup=fast[0], repeats=fast[1], row_timeout_s=row_timeout_s),
            True: RemoteVerifier(backend, cfg, warmup=final[0], repeats=final[1], row_timeout_s=row_timeout_s),
        }
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self.gpu_seconds = 0.0  # cumulative over all tasks
        self.n_evals = 0
        # Remote job directories are named from this uid. Without a process-unique
        # prefix, two Evaluators evaluating the same task (two methods on the same
        # benchmark, run as separate processes against the same pod -- the natural
        # way to run several methods at once) produce the IDENTICAL uid on their
        # Nth call each, since _next_id starts at 0 in every process. They then
        # collide on the same remote directory: one process's write can land
        # mid-read of another's, surfacing as FileNotFoundError, truncated
        # candidate.py, or a syntax error at a line the model never wrote.
        self._run_tag = f"{os.getpid()}{uuid.uuid4().hex[:6]}"

    def _next_id(self) -> int:
        with self._lock:
            return next(self._ids)

    def verify_batch(self, requests: list[VerifyRequest], *, tag: str, final: bool = False) -> dict[str, VerifyResult]:
        """Raw batch verification (same call shape as :class:`RemoteVerifier`), metered."""
        t0 = time.monotonic()
        out = self._verifiers[final].verify_batch(requests, tag=tag)
        dt = time.monotonic() - t0
        with self._lock:
            self.gpu_seconds += dt
            self.n_evals += len(requests)
        metering.charge_gpu(dt)
        return out

    def evaluate(self, task: Task, cand: Candidate, *, tag: str = "", final: bool = False) -> EvalResult:
        uid = f"{_SAFE.sub('_', task.id)}_{self._run_tag}_{self._next_id()}"
        res = self.verify_batch(
            [VerifyRequest(uid=uid, ref_code=task.ref_code, cand_code=cand.code)],
            tag=_SAFE.sub("_", f"{tag or task.id}_{uid}"), final=final,
        )
        return to_result(res[uid])

    def baseline_ms(self, task: Task, *, final: bool = False) -> float | None:
        """Runtime of the original PyTorch program (a row with no candidate)."""
        uid = f"{_SAFE.sub('_', task.id)}_{self._run_tag}_{self._next_id()}"
        res = self.verify_batch([VerifyRequest(uid=uid, ref_code=task.ref_code)], tag=f"baseline_{uid}", final=final)
        return res[uid].baseline_ms


def to_result(r: VerifyResult) -> EvalResult:
    ok = bool(r.ok and r.correct is not False and r.candidate_ms)
    speedup = r.speedup
    if ok and speedup is None and r.baseline_ms and r.candidate_ms:
        speedup = r.baseline_ms / r.candidate_ms
    return EvalResult(ok=ok, stage=r.stage, error=r.error, baseline_ms=r.baseline_ms,
                      candidate_ms=r.candidate_ms, speedup=speedup if ok else None)
