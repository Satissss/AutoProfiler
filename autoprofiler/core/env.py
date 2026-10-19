"""The shared world every method runs in: one LLM, one evaluator, one GPU backend, one budget."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

from autoprofiler.backends import Backend, build_backend
from autoprofiler.config import REPO_ROOT, ExecConfig, load_exec_config
from autoprofiler.core.evaluator import Evaluator
from autoprofiler.llm import LLMClient
from autoprofiler.llm.config import resolve_llm


@dataclass(frozen=True)
class Env:
    llm: LLMClient
    evaluator: Evaluator
    backend: Backend
    exec_cfg: ExecConfig
    work_dir: Path
    rounds: int = 5  # hard cap on optimization rounds per task (the seed is round 0)
    # Where work that survives a run is kept (seed ports, so far). None means no
    # caching: an Env built by hand -- a test, a demo -- cannot write where a real
    # run will later read, which is a mistake that has already cost two debug
    # sessions chasing `class ModelNew: pass` seeds back to their source.
    cache_dir: Path | None = None
    log: Callable[[str], None] = field(default=lambda msg: None)

    def for_task(self, work_dir: Path) -> "Env":
        """Same resources, private scratch directory (so tasks can run concurrently)."""
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return replace(self, work_dir=Path(work_dir))

    @contextmanager
    def step(self, what: str) -> Iterator[Callable[[str], None]]:
        """Announce a slow step, then report how it went and how long it took.

        Everything inside a method is slow enough to look like a hang -- an LLM call,
        a kernel verification, an NCU pass all take tens of seconds -- so each one says
        what it is doing before it blocks::

            with env.step("seed") as done:
                ...
                done("ok, 1842 chars")      ->   seed: ok, 1842 chars [23.1s]
        """
        self.log(f"{what} …")
        t0 = time.monotonic()
        note: list[str] = []
        try:
            yield note.append
        except Exception as exc:
            note.append(f"FAILED {type(exc).__name__}: {str(exc)[:120]}")
            raise
        finally:
            tail = f": {note[0]}" if note else ""
            self.log(f"{what}{tail} [{time.monotonic() - t0:.1f}s]")


def build_env(
    *,
    llm: str | dict,   # a profile name, or the endpoint inline
    backend: str = "configs/backends/local.yaml",
    work_dir: str | Path = "output/runs",
    rounds: int = 5,
    fast: tuple[int, int] = (3, 10),
    final: tuple[int, int] = (10, 100),
    log: Callable[[str], None] = print,
) -> Env:
    """Assemble the shared resources every method in an evaluation runs against."""
    cfg = load_exec_config(backend)
    be = build_backend(cfg)
    wd = Path(work_dir)
    wd = wd if wd.is_absolute() else REPO_ROOT / wd
    wd.mkdir(parents=True, exist_ok=True)
    spec = {"profile": llm} if isinstance(llm, str) else llm
    return Env(llm=LLMClient(resolve_llm(spec)), evaluator=Evaluator(be, cfg, fast=fast, final=final),
               backend=be, exec_cfg=cfg, work_dir=wd, rounds=rounds, log=log,
               cache_dir=Path(os.environ.get("AUTOPROFILER_CACHE_DIR")
                              or REPO_ROOT / "output" / "_cache"))
