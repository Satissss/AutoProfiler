"""Backend protocol: where GPU work runs.

A backend is a thin "run a command / move a file" surface. Everything GPU-bound
(verification, timing, NCU) is a worker script under ``autoprofiler/backends/workers`` that
a backend launches; backends never know what the scripts do.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from autoprofiler.config import REMOTE_DIR, REMOTE_REL


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Backend(ABC):
    """Executes commands and moves files on the machine that owns the GPU."""

    root: str  # working root as seen by that machine

    @abstractmethod
    def run(self, command: str, *, timeout_s: int = 600, use_env: bool = True) -> ExecResult: ...

    @abstractmethod
    def put_bytes(self, remote_path: str, data: bytes) -> None: ...

    @abstractmethod
    def get_text(self, remote_path: str) -> str: ...

    @abstractmethod
    def mkdir(self, remote_path: str) -> None: ...

    @property
    @abstractmethod
    def python_bin(self) -> str: ...

    def put_text(self, remote_path: str, content: str) -> None:
        self.put_bytes(remote_path, content.encode("utf-8"))

    def put_file(self, local_path: Path, remote_path: str) -> None:
        self.put_bytes(remote_path, Path(local_path).read_bytes())

    def python(self, args: str, *, timeout_s: int = 600) -> ExecResult:
        return self.run(f"{self.python_bin} {args}", timeout_s=timeout_s)

    def script(self, name: str) -> str:
        """Path (on the GPU machine) of a worker script, shipping it first if needed."""
        raise NotImplementedError

    def _ship_script(self, name: str) -> str:
        remote = f"{self.root}/{REMOTE_REL}/{name}"
        self.put_file(REMOTE_DIR / name, remote)
        return remote
