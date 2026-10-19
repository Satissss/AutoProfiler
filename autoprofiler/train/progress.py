"""Progress reporting during routing."""
from __future__ import annotations

from typing import Any

from autoprofiler.train.checkpoint import _checkpoint
from autoprofiler.train.log import _log
from autoprofiler.train.state import TrainState


def _log_routing_start(
    state: TrainState,
    *,
    bounds: list[str] | None,
    selected: int,
    mode: str,
    workers: int | None = None,
) -> None:
    bound_txt = ",".join(bounds) if bounds else "all"
    pool = len(state.store.active_pool())
    if workers is None:
        _log(
            f"[routing-start] {mode} bound={bound_txt} "
            f"available={selected} pool={pool}"
        )
    else:
        _log(
            f"[routing-start] {mode} bound={bound_txt} "
            f"available={selected} workers={workers} pool={pool}"
        )


def _log_step_progress(
    state: TrainState, rec: dict[str, Any], *, available: int
) -> None:
    steps = rec.get("steps") or []
    if not steps:
        return
    n = len(steps)
    n_adv = sum(1 for s in steps if s.get("advanced"))
    n_slow = sum(
        1 for s in steps
        if not s.get("advanced") and s.get("rel_speedup") is not None
    )
    n_failed = n - n_adv - n_slow
    last = steps[-1]
    _log(
        f"[progress] completed={n} available_at_start={available} "
        f"advanced={n_adv} slow={n_slow} failed={n_failed} "
        f"pool={len(state.store.active_pool())} "
        f"last={last.get('kernel_id', '?')} "
        f"outcome={last.get('outcome') or last.get('reason') or last.get('action') or '?'}"
    )


def _maybe_progress_checkpoint(
    state: TrainState, rec: dict[str, Any], *, interval: int
) -> None:
    n = len(rec.get("steps") or [])
    if n <= 0 or n % max(1, interval) != 0:
        return
    n_adv = sum(1 for step in rec["steps"] if step.get("advanced"))
    n_slow = sum(
        1 for step in rec["steps"]
        if not step.get("advanced") and step.get("rel_speedup") is not None
    )
    _log(
        f"[checkpoint] forwards={n} advanced={n_adv} "
        f"slow={n_slow} failed={n - n_adv - n_slow} "
        f"pool={len(state.store.active_pool())}; "
        "routing continues"
    )
    _checkpoint(state, reason="forward_interval")


def _log_routing_summary(rec: dict[str, Any]) -> None:
    diag = rec.get("diagnostics") or {}
    outcomes = diag.get("outcome_counts") or {}
    cov = diag.get("origin_coverage") or {}
    updates = rec.get("updates") or []
    _log(
        f"[routing-pause] steps={rec.get('n_steps', 0)} pool={rec.get('pool_size')} "
        f"updates={len(updates)} outcomes={outcomes} "
        f"seeds_tried={cov.get('attempted')}/{cov.get('total')}"
    )
    alerts = diag.get("branch_share_alerts") or []
    if alerts:
        _log(f"[routing-pause] branch alerts: {alerts[:3]}")
