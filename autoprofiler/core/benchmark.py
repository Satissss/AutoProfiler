"""Benchmarks are plain directories of PyTorch reference programs plus a yaml.

Adding a benchmark = drop ``*.py`` problems (each defining ``Model``,
``get_inputs``, ``get_init_inputs``) under ``data/benchmarks/<name>/`` and write
``configs/benchmarks/<name>.yaml``; no code changes::

    name: kernelbench
    root: data/benchmarks/kernelbench
    id_prefix: kb
    groups:
      level1: {dir: level1, tag: l1}     # task id -> kb_l1_<leading number of file name>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from autoprofiler.config import REPO_ROOT
from autoprofiler.core.types import Task

_NUM = re.compile(r"^(\d+)_")


class GroupSpec(BaseModel):
    dir: str
    tag: str = ""


class BenchmarkSpec(BaseModel):
    name: str
    root: str
    id_prefix: str = ""
    groups: dict[str, GroupSpec]


@dataclass
class Benchmark:
    spec: BenchmarkSpec

    @classmethod
    def load(cls, name_or_path: str) -> "Benchmark":
        p = Path(name_or_path)
        if not (p.suffix in (".yaml", ".yml") and p.exists()):
            p = REPO_ROOT / "configs" / "benchmarks" / f"{name_or_path}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"benchmark config not found: {name_or_path} (looked for {p})")
        return cls(BenchmarkSpec.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))))

    @property
    def name(self) -> str:
        return self.spec.name

    def tasks(
        self,
        groups: list[str] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """Tasks of the selected groups (default all), optionally filtered by id / capped."""
        root = Path(self.spec.root)
        root = root if root.is_absolute() else REPO_ROOT / root
        unknown = set(groups or []) - set(self.spec.groups)
        if unknown:
            raise KeyError(f"unknown group(s) {sorted(unknown)}; available: {sorted(self.spec.groups)}")
        out: list[Task] = []
        for gname, g in self.spec.groups.items():
            if groups and gname not in groups:
                continue
            files = sorted((root / g.dir).glob("*.py"), key=_sort_key)
            for f in files:
                m = _NUM.match(f.name)
                key = m.group(1) if m else f.stem
                parts = [x for x in (self.spec.id_prefix, g.tag or gname, key) if x]
                out.append(Task(
                    id="_".join(parts), benchmark=self.name, group=gname,
                    ref_code=f.read_text(encoding="utf-8"), ref_path=str(f),
                    meta={"file": f.name},
                ))
        if ids:
            want = set(ids)
            out = [t for t in out if t.id in want]
            missing = want - {t.id for t in out}
            if missing:
                raise KeyError(f"task id(s) not found: {sorted(missing)}")
        return out[:limit] if limit else out


def _sort_key(p: Path) -> tuple[int, str]:
    m = _NUM.match(p.name)
    return (int(m.group(1)) if m else 10**9, p.name)
