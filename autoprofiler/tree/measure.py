"""Run one node's Nsight Compute metric set on a kernel (through the backend, with on-disk caching)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoprofiler.backends.ncu import remote_ncu_profile
from autoprofiler.config import ExecConfig


@dataclass
class ProfileResult:
    ok: bool
    metrics: dict[str, float]
    kernel_count: int = 0
    kernel_names: list[str] = field(default_factory=list)
    # Measured wall time of collecting THIS metric set, in ms. Set-level by
    # construction (see backends/workers/profile_ncu.py): NCU cost goes by replay pass, so
    # summing per-metric costs would misprice every set.
    profile_ms: float = 0.0


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _coerce_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
    return {str(k): _to_float(v) for k, v in (raw_metrics or {}).items()}


def profile_leaf(
    backend,
    cfg: ExecConfig,
    program: dict[str, Any],
    ncu_metrics: list[str],
) -> ProfileResult:
    """Run a node's profiler: its NCU metric set on one kernel."""
    metrics = list(dict.fromkeys(m for m in ncu_metrics if m))
    if not metrics:
        return ProfileResult(ok=False, metrics={})
    payload = remote_ncu_profile(backend, cfg, program, metrics)
    return ProfileResult(
        ok=bool(payload.get("ok")),
        metrics=_coerce_metrics(payload.get("metrics")),
        kernel_count=int(payload.get("kernel_count") or 0),
        kernel_names=list(payload.get("kernel_names") or []),
        profile_ms=float(payload.get("profile_ms") or 0.0),
    )
