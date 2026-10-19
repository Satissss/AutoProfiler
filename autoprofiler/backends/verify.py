"""Local driver for remote batch verification on the GPU pod.

Cost shape this is built around: a ``kubectl exec`` round-trip costs seconds of
pure overhead, and the pipeline needs 6k rows times up to a few repair rounds. So
everything is batched -- one source-bundle transfer plus one exec verifies a
whole batch -- while the pod isolates each row in its own subprocess so a kernel
that hard-crashes the interpreter only loses its own row.
"""

from __future__ import annotations

import json
import hashlib
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from autoprofiler.backends import Backend, SingleGPUQueue, gpu_flock
from autoprofiler.config import ExecConfig


@dataclass
class VerifyRequest:
    uid: str
    ref_code: str
    cand_code: str | None = None


@dataclass
class VerifyResult:
    uid: str
    ok: bool
    stage: str
    error: str = ""
    baseline_ms: float | None = None
    candidate_ms: float | None = None
    speedup: float | None = None
    correct: bool = False
    raw: dict | None = None

    @classmethod
    def from_raw(cls, raw: dict) -> "VerifyResult":
        return cls(
            uid=str(raw.get("uid", "")),
            ok=bool(raw.get("ok")),
            stage=str(raw.get("stage") or "unknown"),
            error=str(raw.get("error") or ""),
            baseline_ms=raw.get("baseline_ms"),
            candidate_ms=raw.get("candidate_ms"),
            speedup=raw.get("speedup"),
            correct=bool(raw.get("correct")),
            raw=raw,
        )

    @classmethod
    def transport_failure(cls, uid: str, error: str) -> "VerifyResult":
        return cls(uid=uid, ok=False, stage="transport", error=error)


class RemoteVerifier:
    """Runs batches of (ref, candidate) verifications on the pod."""

    def __init__(
        self,
        backend: Backend,
        cfg: ExecConfig,
        *,
        warmup: int = 3,
        repeats: int = 10,
        row_timeout_s: int = 300,
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.warmup = warmup
        self.repeats = repeats
        self.row_timeout_s = row_timeout_s
        self._script: str | None = None

    @property
    def remote_root(self) -> str:
        return self.cfg.remote_root

    def sync_script(self) -> str:
        """Path of seed_verify.py on the GPU machine (shipped once if remote)."""
        if self._script is None:
            self._script = self.backend.script("seed_verify.py")
        return self._script

    def verify_batch(
        self, requests: list[VerifyRequest], *, tag: str
    ) -> dict[str, VerifyResult]:
        """Verify a batch. Returns uid -> result for every requested uid."""
        if not requests:
            return {}
        script = self.sync_script()
        job = f"{self.remote_root}/jobs/seedgen/{tag}"
        bundle_path = f"{job}/bundle.json"
        out_path = f"{job}/results.jsonl"
        src_dir = f"{job}/src"

        self.backend.mkdir(job)
        bundle = {
            "rows": [
                {"uid": r.uid, "ref_code": r.ref_code, "cand_code": r.cand_code}
                for r in requests
            ]
        }
        self.backend.put_text(bundle_path, json.dumps(bundle))

        cmd = (
            f"{self.backend.python_bin} {script} --bundle {bundle_path} "
            f"--workdir {src_dir} --out {out_path} "
            f"--warmup {self.warmup} --repeats {self.repeats} "
            f"--timeout-s {self.row_timeout_s}"
        )
        # Budget: worst case every row hits its own timeout, plus import overhead.
        budget = min(21600, 180 + self.row_timeout_s * len(requests))
        # Verification and NCU share this cross-process single-GPU boundary.
        with gpu_flock():
            res = self.backend.run(cmd, timeout_s=budget)

        results: dict[str, VerifyResult] = {}
        try:
            text = self.backend.get_text(out_path)
        except Exception as e:
            text = ""
            transport_err = f"cannot read remote results: {e}"
        else:
            transport_err = ""

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            vr = VerifyResult.from_raw(raw)
            if vr.uid:
                results[vr.uid] = vr

        # A row missing from the results file never produced a verdict: the pod
        # died, the exec timed out, or the driver never reached it. Surface that
        # as a transport failure rather than silently dropping the row.
        if not res.ok and not results:
            transport_err = transport_err or (
                f"remote exec failed rc={res.returncode}: "
                f"{(res.stderr or res.stdout or '')[-800:]}"
            )
        for r in requests:
            if r.uid not in results:
                results[r.uid] = VerifyResult.transport_failure(
                    r.uid, transport_err or "no result line produced for this row"
                )
        return results

    def probe_baselines(
        self, rows: list[tuple[str, str]], *, tag: str
    ) -> dict[str, VerifyResult]:
        """Step 0: check the original torch programs run, and time them."""
        reqs = [VerifyRequest(uid=uid, ref_code=code) for uid, code in rows]
        return self.verify_batch(reqs, tag=tag)


@dataclass
class _PendingVerify:
    requests: list[VerifyRequest]
    tag: str
    done: threading.Event
    result: dict[str, VerifyResult] | None = None
    error: BaseException | None = None


class MicrobatchRemoteVerifier:
    """Coalesce concurrent ``verify_batch`` calls onto one RemoteVerifier.

    A wrapper instance represents one verifier parameter set (backend, warmup,
    repeats and timeout), so requests with unlike parameters cannot be mixed.
    The public verification methods retain RemoteVerifier's synchronous API.
    """

    def __init__(
        self,
        verifier: RemoteVerifier,
        *,
        window_s: float = 0.02,
        max_batch_size: int = 32,
        gpu_queue: SingleGPUQueue | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        if window_s < 0:
            raise ValueError("window_s must be non-negative")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.verifier = verifier
        self.window_s = window_s
        self.max_batch_size = max_batch_size
        self.gpu_queue = gpu_queue
        self.journal_path = Path(journal_path) if journal_path else None
        self._journal_lock = threading.Lock()
        self._completed: dict[str, VerifyResult] = {}
        if self.journal_path and self.journal_path.exists():
            for line in self.journal_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "done" and isinstance(row.get("result"), dict):
                    self._completed[str(row.get("job_id") or "")] = VerifyResult.from_raw(
                        row["result"]
                    )
        self._pending: queue.Queue[_PendingVerify | None] = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._batch_id = 0
        self._run_token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._thread = threading.Thread(
            target=self._run, name="verify-microbatch", daemon=True
        )
        self._thread.start()

    def verify_batch(
        self, requests: list[VerifyRequest], *, tag: str
    ) -> dict[str, VerifyResult]:
        if not requests:
            return {}
        with self._close_lock:
            if self._closed:
                raise RuntimeError("verification microbatcher is closed")
        ready: dict[str, VerifyResult] = {}
        missing: list[VerifyRequest] = []
        with self._journal_lock:
            for request in requests:
                cached = self._completed.get(self._job_id(request))
                if cached is None:
                    missing.append(request)
                else:
                    ready[request.uid] = replace(cached, uid=request.uid)
        if not missing:
            return ready
        call = _PendingVerify(missing, tag, threading.Event())
        with self._close_lock:
            if self._closed:
                raise RuntimeError("verification microbatcher is closed")
            self._pending.put(call)
        call.done.wait()
        if call.error is not None:
            raise call.error
        ready.update(call.result or {})
        return ready

    def probe_baselines(
        self, rows: list[tuple[str, str]], *, tag: str
    ) -> dict[str, VerifyResult]:
        return self.verify_batch(
            [VerifyRequest(uid=uid, ref_code=code) for uid, code in rows],
            tag=tag,
        )

    def close(self, *, wait: bool = True) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self._pending.put(None)
        if wait:
            self._thread.join()

    def __getattr__(self, name: str) -> Any:
        # Keep access to backend/cfg/timing fields compatible with RemoteVerifier.
        return getattr(self.verifier, name)

    def _run(self) -> None:
        backlog: _PendingVerify | None = None
        while True:
            first = backlog if backlog is not None else self._pending.get()
            backlog = None
            if first is None:
                return
            calls = [first]
            request_count = len(first.requests)
            deadline = time.monotonic() + self.window_s
            stop_after_dispatch = False
            while request_count < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    # Finish already accepted work before shutting down.
                    stop_after_dispatch = True
                    break
                if request_count + len(nxt.requests) > self.max_batch_size:
                    # Preserve FIFO without splitting an individual API call.
                    backlog = nxt
                    break
                calls.append(nxt)
                request_count += len(nxt.requests)
            self._dispatch(calls)
            if stop_after_dispatch:
                return

    def _dispatch(self, calls: list[_PendingVerify]) -> None:
        self._batch_id += 1
        combined: list[VerifyRequest] = []
        identities: list[tuple[_PendingVerify, str, str]] = []
        for call_idx, call in enumerate(calls):
            for request_idx, request in enumerate(call.requests):
                transport_uid = (
                    f"mb{self._batch_id}_{call_idx}_{request_idx}_{request.uid}"
                )
                combined.append(replace(request, uid=transport_uid))
                identities.append((call, request.uid, transport_uid))
        try:
            kwargs = {
                "tag": (
                    f"microbatch_{self._run_token}_"
                    f"{self._batch_id}_{calls[0].tag}"
                )
            }
            if self.gpu_queue is None:
                raw = self.verifier.verify_batch(combined, **kwargs)
            else:
                raw = self.gpu_queue.run(
                    self.verifier.verify_batch, combined, **kwargs
                )
            for call in calls:
                call.result = {}
            for call, original_uid, transport_uid in identities:
                result = raw.get(transport_uid)
                if result is None:
                    result = VerifyResult.transport_failure(
                        original_uid, "no result returned for microbatched request"
                    )
                else:
                    result = replace(result, uid=original_uid)
                assert call.result is not None
                call.result[original_uid] = result
                if result.stage != "transport":
                    request = next(
                        request for request in call.requests
                        if request.uid == original_uid
                    )
                    self._record_done(self._job_id(request), result)
        except BaseException as exc:
            for call in calls:
                call.error = exc
        finally:
            for call in calls:
                call.done.set()

    def __enter__(self) -> "MicrobatchRemoteVerifier":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        if self.gpu_queue is None:
            return {}
        m = self.gpu_queue.metrics()
        return {
            "submitted": m.submitted,
            "completed": m.completed,
            "failed": m.failed,
            "queue_depth": m.queue_depth,
            "active": m.active,
            "busy_time_s": m.busy_time_s,
            "utilization": m.utilization,
            "journal_done": len(self._completed),
        }

    def _job_id(self, request: VerifyRequest) -> str:
        payload = (
            f"{request.ref_code}\0{request.cand_code or ''}\0"
            f"{getattr(self.verifier, 'warmup', '')}\0"
            f"{getattr(self.verifier, 'repeats', '')}\0"
            f"{getattr(self.verifier, 'row_timeout_s', '')}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _record_done(self, job_id: str, result: VerifyResult) -> None:
        stored = replace(result, uid=job_id)
        with self._journal_lock:
            if job_id in self._completed:
                return
            self._completed[job_id] = stored
            if self.journal_path is None:
                return
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "job_id": job_id,
                "status": "done",
                "result": {
                    "uid": job_id,
                    "ok": stored.ok,
                    "stage": stored.stage,
                    "error": stored.error,
                    "baseline_ms": stored.baseline_ms,
                    "candidate_ms": stored.candidate_ms,
                    "speedup": stored.speedup,
                    "correct": stored.correct,
                    "raw": stored.raw,
                },
            }
            with self.journal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
