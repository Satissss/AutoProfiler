"""One experiment, one file.

An eval config is self-contained: it says what to measure, which methods, on which
model and GPU, with what budget. Running a different experiment means writing another
file, not editing shared state that other experiments also read.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from autoprofiler.config import REPO_ROOT


class EvalConfig(BaseModel):
    """The settings below the ``methods`` entry are shared by every method in the run.

    That is deliberate: the same LLM, the same GPU, the same round budget and the same
    timing protocol, so a difference in the results is a difference between methods.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    # ── what to measure ──────────────────────────────────────────────────────
    benchmark: str = "kernelbench"       # configs/benchmarks/<name>.yaml
    groups: list[str] | None = None      # e.g. [level1]; null = every group
    ids: list[str] | None = None         # explicit task ids, overriding groups
    limit: int | None = None             # first N tasks only
    # false: start from the original PyTorch program and score against PyTorch.
    # true:  start from the Triton port of it (the "seed"), read from data/seeds/ or
    #        generated there when missing, and score against that seed instead.
    use_seed: bool = False

    # ── which methods ────────────────────────────────────────────────────────
    # A bare list uses each method's own defaults:   methods: [autoprofiler, cudaforge]
    # A mapping overrides them for this experiment:  methods: {autoprofiler: {n_samples: 4}}
    # `base` runs one method twice under different names, which is how an ablation
    # or a component swap becomes two rows of the same table.
    methods: list[str] | dict[str, dict]

    # ── how it runs (identical for every method) ─────────────────────────────
    llm: str | dict = "gpt"              # a profile in configs/llm/, or the endpoint inline
    backend: str = "local"               # a config in configs/backends/
    rounds: int = Field(5, ge=0)         # optimization rounds per task; the seed is round 0
    fast: tuple[int, int] = (3, 10)      # (warmup, repeats) for timing inside the method loop
    final: tuple[int, int] = (10, 100)   # (warmup, repeats) for the reported score
    output: str = "output/results"

    @field_validator("methods")
    @classmethod
    def _as_mapping(cls, v):
        return {name: {} for name in v} if isinstance(v, list) else v

    def method_specs(self) -> dict[str, dict]:
        """``{run name: settings}``, with ``base`` resolved to the method it runs."""
        out = {}
        for name, spec in self.methods.items():
            spec = dict(spec or {})
            out[name] = {"method": spec.pop("base", name), **spec}
        return out


def _resolve(value: str, kind: str) -> str:
    """A bare name means ``configs/<kind>/<name>.yaml``; a path is taken as given."""
    p = Path(value)
    if p.suffix in (".yaml", ".yml"):
        return value
    return f"configs/{kind}/{value}.yaml"


def load_eval_config(path: str | Path) -> EvalConfig:
    p = Path(path)
    p = p if p.is_absolute() or p.exists() else REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"eval config not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw.setdefault("name", p.stem)
    cfg = EvalConfig.model_validate(raw)
    cfg.backend = _resolve(cfg.backend, "backends")
    return cfg
