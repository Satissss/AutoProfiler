"""Process-local scheduling and cross-process locking for the single GPU pod.

The queue deliberately does not acquire the file lock itself: GPU operations
such as verification and NCU profiling own that boundary.  This keeps generic
queued work composable and avoids taking the non-reentrant OS lock twice.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

DEFAULT_GPU_LOCK_PATH = Path("/tmp/autoprofiler_seed_gpu_verify.lock")
GPU_LOCK_ENV = "AUTOPROFILER_GPU_VERIFY_LOCK"
# Set in the environment while a process holds the GPU flock, so that a child
# process (e.g. the eval task_worker, spawned by a parent that already holds the
# lock in pipeline.run_locked) can detect the ancestor's hold and skip taking the
# same cross-process exclusive lock again. Without this, parent-holds + child-
# waits on the same fcntl.flock(LOCK_EX) is a permanent deadlock.
_GPU_LOCK_HELD_ENV = "AUTOPROFILER_GPU_LOCK_HELD"

T = TypeVar("T")

_local_locks: dict[str, threading.RLock] = {}
_local_locks_guard = threading.Lock()

# Per-thread physical GPU binding, for in-process parallel harnesses that run
# several GPU jobs at once (one per device) as threads rather than subprocesses.
# The k8s backend reads this to pin each remote exec; a subprocess-based caller
# uses CUDA_VISIBLE_DEVICES in the child env instead.
_gpu_tls = threading.local()


def set_thread_gpu(gpu_id: int | None) -> None:
    _gpu_tls.gpu_id = gpu_id


def thread_gpu() -> int | None:
    return getattr(_gpu_tls, "gpu_id", None)


def gpu_lock_path(path: str | Path | None = None) -> Path:
    """Resolve the shared GPU lock path, retaining the historical default."""
    if path is not None:
        return Path(path)
    return Path(os.environ.get(GPU_LOCK_ENV, str(DEFAULT_GPU_LOCK_PATH)))


def _local_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _local_locks_guard:
        return _local_locks.setdefault(key, threading.RLock())


@contextmanager
def gpu_flock(path: str | Path | None = None) -> Generator[Path, None, None]:
    """Exclusively lock the shared GPU across threads and local processes."""
    lock_path = gpu_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(lock_path.resolve())

    # Re-entrant across the process tree: if an ancestor process already holds
    # this exact lock, acquiring it again here (in a child subprocess with its own
    # fd) would block forever. The ancestor's hold already guarantees exclusive
    # GPU access for the whole subtree, so we yield without re-locking.
    if os.environ.get(_GPU_LOCK_HELD_ENV) == resolved:
        yield lock_path
        return

    with _local_lock(lock_path):
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            prev = os.environ.get(_GPU_LOCK_HELD_ENV)
            os.environ[_GPU_LOCK_HELD_ENV] = resolved
            try:
                yield lock_path
            finally:
                if prev is None:
                    os.environ.pop(_GPU_LOCK_HELD_ENV, None)
                else:
                    os.environ[_GPU_LOCK_HELD_ENV] = prev
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class GPUQueueMetrics:
    submitted: int
    completed: int
    failed: int
    queue_depth: int
    active: bool
    busy_time_s: float
    uptime_s: float

    @property
    def utilization(self) -> float:
        return self.busy_time_s / self.uptime_s if self.uptime_s > 0 else 0.0


class SingleGPUQueue:
    """One-worker local queue with Future and asyncio-friendly APIs."""

    def __init__(self, *, name: str = "autoprofiler-gpu") -> None:
        self.name = name
        self._queue: queue.Queue[Any] = queue.Queue()
        self._guard = threading.Lock()
        self._started_at = time.monotonic()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._busy_time_s = 0.0
        self._active = False
        self._closed = False
        self._sentinel = object()
        self._worker = threading.Thread(
            target=self._work, name=name, daemon=True
        )
        self._worker.start()

    def submit(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> Future[T]:
        """Enqueue work immediately and return a concurrent Future."""
        future: Future[T] = Future()
        with self._guard:
            if self._closed:
                raise RuntimeError("GPU queue is closed")
            self._submitted += 1
            self._queue.put((future, fn, args, kwargs))
        return future

    def run(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T:
        """Enqueue work and wait synchronously for its result."""
        return self.submit(fn, *args, **kwargs).result()

    async def run_async(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T:
        """Enqueue work without blocking the caller's asyncio event loop."""
        return await asyncio.wrap_future(self.submit(fn, *args, **kwargs))

    def metrics(self) -> GPUQueueMetrics:
        """Return a consistent point-in-time queue telemetry snapshot."""
        with self._guard:
            return GPUQueueMetrics(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                queue_depth=self._queue.qsize(),
                active=self._active,
                busy_time_s=self._busy_time_s,
                uptime_s=time.monotonic() - self._started_at,
            )

    def close(self, *, wait: bool = True) -> None:
        with self._guard:
            if not self._closed:
                self._closed = True
                self._queue.put(self._sentinel)
        if wait:
            self._worker.join()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                self._queue.task_done()
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                self._queue.task_done()
                continue
            started = time.monotonic()
            with self._guard:
                self._active = True
            failed = False
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                failed = True
                future.set_exception(exc)
            finally:
                elapsed = time.monotonic() - started
                with self._guard:
                    self._active = False
                    self._busy_time_s += elapsed
                    self._completed += 1
                    self._failed += int(failed)
                self._queue.task_done()

    def __enter__(self) -> "SingleGPUQueue":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
