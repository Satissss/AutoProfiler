from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoprofiler.backends import Backend, gpu_flock
from autoprofiler.config import ExecConfig, REPO_ROOT


def remote_ncu_profile(
    backend: Backend,
    cfg: ExecConfig,
    program: dict[str, Any],
    metrics: list[str],
    *,
    cache_dir: Path | None = None,
    profile_script: str | None = None,
) -> dict[str, Any]:
    """Run ncu on remote pod; cache by program uid + metrics hash."""
    cache_dir = cache_dir or (REPO_ROOT / "output" / "cache" / "ncu")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # shorter stable key
    import hashlib

    h = hashlib.sha1(",".join(metrics).encode()).hexdigest()[:10]
    # A program may carry its own source instead of a KernelBench coordinate
    # (seed corpus rows, and kernels produced mid-optimization, exist nowhere in
    # KernelBench). When it does, the source is part of the cache identity: two
    # different kernels must never share a reading just because they share a uid.
    src = str(program.get("source") or "")
    src_tag = f"_{hashlib.sha1(src.encode()).hexdigest()[:8]}" if src else ""
    cache_path = cache_dir / f"{program['uid']}_{h}{src_tag}.json"
    cached = _successful_cache(cache_path)
    if cached is not None:
        return cached

    remote_root = cfg.remote_root
    uid = program["uid"]
    ref_code = str(program.get("reference_code") or src or "")
    candidate_code = src or ref_code
    job = f"{remote_root}/jobs/ncu/{uid}_{h}{src_tag}"
    backend.mkdir(job)
    ref_path = f"{job}/ref.py"
    candidate_path = f"{job}/candidate.py"
    out_path = f"{job}/profile.json"
    script = profile_script or backend.script("profile_ncu.py")
    backend.put_text(ref_path, ref_code)
    backend.put_text(candidate_path, candidate_code)

    metrics_arg = ",".join(metrics)
    # Per-job isolated caches (Triton / torch-extensions / TMPDIR) so that when
    # several NCU passes run at once (one per GPU) they never collide on a shared
    # compile cache or temp file. `job` is unique per uid+metrics+source.
    import shlex as _shlex

    cmd = (
        f"mkdir -p {_shlex.quote(job)}/.triton {_shlex.quote(job)}/.torch_ext "
        f"{_shlex.quote(job)}/.tmp && "
        f"TRITON_CACHE_DIR={_shlex.quote(job)}/.triton "
        f"TORCH_EXTENSIONS_DIR={_shlex.quote(job)}/.torch_ext "
        f"TMPDIR={_shlex.quote(job)}/.tmp "
        f"{backend.python_bin} {script} --ref {ref_path} "
        f"--candidate {candidate_path} "
        f"--metrics {metrics_arg} --out {out_path}"
    )
    print(f"[ncu] profiling {uid} (metrics={metrics_arg}) ...", flush=True)
    with gpu_flock():
        # Another local process may have populated the cache while we waited.
        cached = _successful_cache(cache_path)
        if cached is not None:
            return cached
        res = backend.run(cmd, timeout_s=1200)
        if not res.ok:
            return _ncu_failure(
                "remote_failed", stdout=res.stdout, stderr=res.stderr
            )
        try:
            payload = json.loads(backend.get_text(out_path))
        except Exception as exc:
            return _ncu_failure("transport", stderr=f"{type(exc).__name__}: {exc}")

        if not isinstance(payload, dict):
            return _ncu_failure(
                "invalid_output", stderr="profile output must be a JSON object"
            )
        # Failed and partial measurements are deliberately transient. In
        # particular, a transport timeout must be retryable on the next call.
        if payload.get("ok") is True:
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            cache_path.unlink(missing_ok=True)
        return payload


def _successful_cache(cache_path: Path) -> dict[str, Any] | None:
    """Return only durable successful NCU results; purge stale failures."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache_path.unlink(missing_ok=True)
        return None
    if isinstance(payload, dict) and payload.get("ok") is True:
        return payload
    cache_path.unlink(missing_ok=True)
    return None


def _ncu_failure(
    error: str, *, stdout: str = "", stderr: str = ""
) -> dict[str, Any]:
    return {
        "ok": False,
        "metrics": {},
        "kernel_count": 0,
        "metadata": {
            "error": error,
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
        },
    }


@dataclass
class _PendingNCU:
    program: dict[str, Any]
    metrics: list[str]
    done: threading.Event
    payload: dict[str, Any] | None = None
    error: BaseException | None = None


class MicrobatchNCUProfiler:
    """Batch concurrent equal-metric requests into one remote exec."""

    def __init__(
        self,
        backend: Backend,
        cfg: ExecConfig,
        *,
        gpu_queue=None,
        window_s: float = 0.05,
        max_batch_size: int = 8,
        cache_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.gpu_queue = gpu_queue
        self.window_s = max(0.0, float(window_s))
        self.max_batch_size = max(1, int(max_batch_size))
        self.cache_dir = cache_dir or (REPO_ROOT / "output" / "cache" / "ncu")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pending: queue.Queue[_PendingNCU | None] = queue.Queue()
        self._backlog: list[_PendingNCU] = []
        self._closed = False
        self._close_lock = threading.Lock()
        self._batch_id = 0
        self._run_token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._thread = threading.Thread(
            target=self._run, name="ncu-microbatch", daemon=True
        )
        self._thread.start()

    def profile(
        self, program: dict[str, Any], metrics: list[str]
    ) -> dict[str, Any]:
        cache_path = _ncu_cache_path(self.cache_dir, program, metrics)
        cached = _successful_cache(cache_path)
        if cached is not None:
            return cached
        call = _PendingNCU(dict(program), list(metrics), threading.Event())
        with self._close_lock:
            if self._closed:
                raise RuntimeError("NCU microbatcher is closed")
            self._pending.put(call)
        call.done.wait()
        if call.error is not None:
            raise call.error
        return call.payload or _ncu_failure("no_batch_result")

    def close(self, *, wait: bool = True) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self._pending.put(None)
        if wait:
            self._thread.join()

    def _next(self) -> _PendingNCU | None:
        if self._backlog:
            return self._backlog.pop(0)
        return self._pending.get()

    def _run(self) -> None:
        while True:
            first = self._next()
            if first is None:
                return
            calls = [first]
            signature = ",".join(sorted(first.metrics))
            deadline = time.monotonic() + self.window_s
            while len(calls) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._pending.put(None)
                    break
                if ",".join(sorted(nxt.metrics)) == signature:
                    calls.append(nxt)
                else:
                    self._backlog.append(nxt)
            self._dispatch(calls)

    def _dispatch(self, calls: list[_PendingNCU]) -> None:
        try:
            payloads = self._remote_batch(calls)
            for call in calls:
                uid = str(call.program["uid"])
                call.payload = payloads.get(uid) or _ncu_failure("missing_batch_row")
        except BaseException as exc:
            for call in calls:
                call.error = exc
        finally:
            for call in calls:
                call.done.set()

    def _remote_batch(self, calls: list[_PendingNCU]) -> dict[str, dict[str, Any]]:
        self._batch_id += 1
        remote_root = self.cfg.remote_root
        job = (
            f"{remote_root}/jobs/ncu_batch/"
            f"{self._run_token}_b{self._batch_id}"
        )
        bundle_path = f"{job}/bundle.json"
        out_path = f"{job}/results.jsonl"
        workdir = f"{job}/rows"
        profile_script = self.backend.script("profile_ncu.py")
        batch_script = self.backend.script("profile_ncu_batch.py")
        self.backend.mkdir(job)
        rows = []
        for call in calls:
            src = str(call.program.get("source") or "")
            reference_code = str(call.program.get("reference_code") or src or "")
            rows.append(
                {
                    "uid": str(call.program["uid"]),
                    "candidate_code": src or reference_code,
                    "reference_code": reference_code,
                    "metrics": list(call.metrics),
                }
            )
        self.backend.put_text(bundle_path, json.dumps({"rows": rows}))
        cmd = (
            f"{self.backend.python_bin} {batch_script} --bundle {bundle_path} "
            f"--workdir {workdir} --out {out_path} "
            f"--profile-script {profile_script}"
        )

        deadline = min(21600, 180 + 1200 * len(calls))

        def execute():
            with gpu_flock():
                return self.backend.run(
                    cmd, timeout_s=deadline,
                )

        metric_names = list(calls[0].metrics)
        print(
            f"[ncu-batch] start size={len(calls)} "
            f"metrics={metric_names}",
            flush=True,
        )
        started = time.monotonic()
        result = (
            self.gpu_queue.run(execute)
            if self.gpu_queue is not None
            else execute()
        )
        if not result.ok:
            return {
                str(call.program["uid"]): _ncu_failure(
                    "remote_failed",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                for call in calls
            }
        text = self.backend.get_text(out_path)
        payloads: dict[str, dict[str, Any]] = {}
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("uid") or "")
            payload = row.get("payload")
            if uid and isinstance(payload, dict):
                payloads[uid] = payload
        for call in calls:
            uid = str(call.program["uid"])
            payload = payloads.get(uid)
            cache_path = _ncu_cache_path(
                self.cache_dir, call.program, call.metrics
            )
            if payload and payload.get("ok") is True:
                cache_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            else:
                cache_path.unlink(missing_ok=True)
        n_ok = sum(
            1 for payload in payloads.values() if payload.get("ok") is True
        )
        print(
            f"[ncu-batch] done ok={n_ok}/{len(calls)} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
        return payloads


def _ncu_cache_path(
    cache_dir: Path, program: dict[str, Any], metrics: list[str]
) -> Path:
    import hashlib

    h = hashlib.sha1(",".join(metrics).encode()).hexdigest()[:10]
    src = str(program.get("source") or "")
    src_tag = f"_{hashlib.sha1(src.encode()).hexdigest()[:8]}" if src else ""
    return cache_dir / f"{program['uid']}_{h}{src_tag}.json"
