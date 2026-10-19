from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from autoprofiler.codegen.port import PortRejected
from autoprofiler.config import REPO_ROOT
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Seeder
from autoprofiler.core.types import Candidate, EvalResult, Task

_MISS = object()


class LLMSeeder(Seeder):
    """First draft by the LLM, then up to ``attempts - 1`` repairs driven by real errors.

    Every LLM seeder gets the same attempt budget, so methods differ in *what* they ask for
    (Triton or CUDA, which prompt) and not in how hard the seed is pushed to run. The result
    is cached per (task, reference, model, settings): the seed is generated once and every
    method that uses it starts from the identical program. A failure is cached too, unless
    it was infrastructure trouble (``transport``), which would otherwise poison the cache.
    """

    language: ClassVar[str]
    prompt_version: ClassVar[str] = "v1"

    class Params(Seeder.Params):
        attempts: int = 4  # first draft + repairs
        temperature: float = 0.2
        max_tokens: int = 16384

    @abstractmethod
    def draft(self, task: Task, env: Env, *, prev_code: str, error: str, stage: str) -> str:
        """One LLM turn: a first draft, or a repair when ``prev_code`` and ``error`` are given.
        Raise :class:`PortRejected` when the reply holds no usable code."""

    # -- cache -----------------------------------------------------------------
    def _cache_path(self, task: Task, env: Env) -> Path | None:
        """Where this seed is cached, or None when the run keeps no cache."""
        if env.cache_dir is None:
            return None
        model = getattr(getattr(env.llm, "config", None), "model", f"unknown_{id(env.llm)}")
        key = json.dumps([task.id, hashlib.sha1(task.ref_code.encode()).hexdigest(), model,
                          self.name, self.prompt_version, self.params.model_dump()], sort_keys=True)
        return Path(env.cache_dir) / "seeds" / f"{task.id}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def _load(self, path: Path):
        if not path.exists():
            return _MISS
        d = json.loads(path.read_text(encoding="utf-8"))
        if not d["code"]:
            return None
        res = EvalResult(**d["result"]) if d.get("result") else None
        return Candidate(d["code"], self.language, f"{self.name}(cached)", result=res)

    @staticmethod
    def _store(path: Path, cand: Candidate | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"code": cand.code if cand else "", "result": cand.result.to_dict() if cand and cand.result else None}
        path.write_text(json.dumps(payload), encoding="utf-8")

    # -- seeding ---------------------------------------------------------------
    def seed(self, task: Task, env: Env) -> Candidate | None:
        path = self._cache_path(task, env)
        cached = self._load(path) if path else _MISS
        if cached is not _MISS:
            env.log("  seed: cached" if cached else "  seed: cached (no usable port)")
            return cached

        prev_code = error = stage = ""
        last: Candidate | None = None
        infra_failure = False
        for attempt in range(self.params.attempts):
            what = f"  seed{f' {attempt + 1}' if attempt else ''}"
            try:
                with env.step(what) as done:
                    code = self.draft(task, env, prev_code=prev_code, error=error, stage=stage)
                    done(f"{len(code)} chars")
            except PortRejected as e:  # unusable reply: try again from scratch with the reason
                prev_code, error, stage = "", e.reason, e.stage
                continue
            cand = Candidate(code, self.language, self.name)
            with env.step(f"{what} verify") as done:
                cand.result = env.evaluator.evaluate(task, cand, tag=f"{task.id}_seed{attempt}")
                r = cand.result
                done(f"{r.speedup:.2f}x vs torch" if r.ok and r.speedup else f"{r.stage}: {r.error[:60]}")
            last = cand
            if cand.result.ok:
                break
            infra_failure |= cand.result.stage == "transport"
            prev_code, error, stage = code, cand.result.error or cand.result.stage, cand.result.stage
        if path and not infra_failure:
            self._store(path, last)
        return last  # a failed last attempt is returned too: a method's rounds may still repair it
