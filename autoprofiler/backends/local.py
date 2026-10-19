"""Run workers on this machine's own GPU."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from autoprofiler.backends.base import Backend, ExecResult
from autoprofiler.config import REMOTE_DIR, ExecConfig


class LocalBackend(Backend):
    def __init__(self, cfg: ExecConfig) -> None:
        self.cfg = cfg
        self.root = cfg.remote_root
        Path(self.root).mkdir(parents=True, exist_ok=True)

    @property
    def python_bin(self) -> str:
        return self.cfg.python

    def run(self, command: str, *, timeout_s: int = 600, use_env: bool = True) -> ExecResult:
        from autoprofiler.backends.gpu import thread_gpu

        env = os.environ.copy()
        tg = thread_gpu()
        if tg is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(tg)
        t0 = time.time()
        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout_s, env=env, encoding="utf-8", errors="replace",
            )
            return ExecResult(proc.returncode, proc.stdout, proc.stderr, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(124, out, f"local_timeout: {e}", time.time() - t0)

    def put_bytes(self, remote_path: str, data: bytes) -> None:
        p = Path(remote_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_text(self, remote_path: str) -> str:
        return Path(remote_path).read_text(encoding="utf-8")

    def mkdir(self, remote_path: str) -> None:
        Path(remote_path).mkdir(parents=True, exist_ok=True)

    def script(self, name: str) -> str:
        # Same machine: run the packaged script in place, no copy.
        return str(REMOTE_DIR / name)
