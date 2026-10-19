"""One step of the Markov chain: route a kernel root->leaf, optimize at the leaf, record the outcome."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoprofiler.tree.leaf import OptimizationServiceError, OptimizeResult, optimize_kernel
from autoprofiler.tree.route import NCUProfileError, RouteOutcome, collect_route, new_route_id, record_route
from autoprofiler.tree.store import KernelState, STATUS_ADVANCED, STATUS_OPTIMIZATION_FAILED, STATUS_SLOW, STATUS_TERMINAL, STATUS_UPDATE_TEST
from autoprofiler.tree.structure import MetricTree
from autoprofiler.train.log import _log
from autoprofiler.train.state import TrainState, _sample_for


@dataclass
class _MarkovWork:
    kernel: KernelState
    route: RouteOutcome
    route_id: str
    result: OptimizeResult


def _compute_markov_work(
    state: TrainState,
    kernel: KernelState,
    routing_tree: MetricTree | None = None,
) -> _MarkovWork:
    """Expensive read/measure/generate part; it never changes kernel status."""
    sample = _sample_for(state, kernel)
    if sample is None:
        raise RuntimeError("origin sample not loaded")
    _log(
        f"  [forward-start] {kernel.kernel_id} hop={kernel.hop} "
        f"origin={kernel.origin_uid}"
    )
    route = collect_route(
        routing_tree or state.tree,
        state.store,
        kernel,
        state.profile_fn(),
        sample.reference(),
    )
    leaf = state.tree.node_by_id(route.leaf.node_id) or route.leaf
    route_id = new_route_id()
    _log(
        f"  [route] {kernel.kernel_id} hop={kernel.hop} -> leaf={leaf.node_id} "
        f"({leaf.bound}) path={'/'.join(route.node_path)} "
        f"measured={route.n_measured} cached={route.n_cache_hits} "
        f"profile_ms={route.profile_ms:.0f}"
    )
    res = optimize_kernel(
        state.verifier,
        state.train_cfg.optimize,
        state.train_cfg.reward,
        state.optimize_llm or state.llm,
        kernel_id=kernel.kernel_id,
        code=kernel.code,
        origin_ref_code=sample.reference(),
        metrics=route.path_metrics,
        parent_latency_ms=kernel.latency_ms,
        tag=f"e{state.epoch}_{leaf.node_id}",
        log=_log,
    )
    return _MarkovWork(kernel, route, route_id, res)


def _apply_markov_work(state: TrainState, work: _MarkovWork) -> dict[str, Any]:
    """Single-writer state transition for a completed pipeline item."""
    kernel, route, route_id, res = (
        work.kernel, work.route, work.route_id, work.result
    )
    leaf = route.leaf
    rec: dict[str, Any] = {
        "kernel_id": kernel.kernel_id,
        "origin_uid": kernel.origin_uid,
        "hop": kernel.hop,
        "leaf_id": leaf.node_id,
        "bound": leaf.bound,
        "route_id": route_id,
        "profile_ms": route.profile_ms,
        **res.to_dict(),
    }
    threshold = float(state.train_cfg.markov.speedup_threshold)
    scorable = bool(res.ok and res.rel_speedup is not None)
    optimizer_summary = str(res.meta.get("optimizer_summary") or "")
    if optimizer_summary:
        state.store.update_kernel(
            kernel.kernel_id,
            meta={
                **kernel.meta,
                "optimizer_summary": optimizer_summary,
                "optimizer_outcome": res.outcome,
            },
        )
    advanced = bool(
        scorable and res.rel_speedup >= threshold
    )
    child_id = ""
    if advanced:
        child = state.store.add_child(
            kernel, res.code, res.latency_ms,
            leaf_id=leaf.node_id, route_id=route_id,
            meta={"outcome": res.outcome, "rel_speedup": res.rel_speedup},
        )
        child_id = child.kernel_id
        # Tag the win to this leaf so it counts toward activation and joins the
        # proposal set: a better metric set is asked to push even the already-won
        # kernels faster. Acceptance still gates on the paired margin, so this
        # only accelerates activation, never relaxes the accept bar.
        state.store.update_kernel(
            kernel.kernel_id, status=STATUS_ADVANCED,
            assigned_leaf_id=leaf.node_id,
        )
        leaf.history_wins += 1
        if (res.rel_speedup or 1.0) - 1.0 < float(state.train_cfg.markov.min_rel_gain):
            state.store.update_kernel(child.kernel_id, status=STATUS_TERMINAL)
            rec["converged"] = True
        _log(
            f"  [advance] {kernel.kernel_id} -> {child_id} "
            f"spd={res.rel_speedup:.3f} wins={leaf.history_wins}"
        )
    elif scorable:
        # Once a leaf starts an update, later arrivals are isolated from the
        # proposal and kept out of the active pool as its unseen test set.
        status = (
            STATUS_UPDATE_TEST
            if leaf.node_id in state.leaf_updates
            else STATUS_SLOW
        )
        state.store.update_kernel(
            kernel.kernel_id, status=status, assigned_leaf_id=leaf.node_id
        )
        label = "held for update test" if status == STATUS_UPDATE_TEST else "accumulates"
        _log(
            f"  [slow] {kernel.kernel_id} outcome={res.outcome} "
            f"spd={res.rel_speedup} -> {label} at {leaf.node_id}"
        )
    else:
        # Compilation/correctness failures say nothing about whether this
        # profiler can improve speed. Keep them auditable but exclude them from
        # LeafSlowSet, update-test sets, rewards, and activation thresholds.
        state.store.update_kernel(
            kernel.kernel_id,
            status=STATUS_OPTIMIZATION_FAILED,
            assigned_leaf_id="",
            meta={
                **kernel.meta,
                "optimization_outcome": res.outcome,
                "optimization_stage": res.stage,
                "optimization_error": res.error,
            },
        )
        _log(
            f"  [opt-failed] {kernel.kernel_id} outcome={res.outcome} "
            f"stage={res.stage}; excluded from leaf activation"
        )
    rec["child_kernel_id"] = child_id
    rec["advanced"] = advanced
    record_route(
        state.store, kernel, route,
        route_id=route_id,
        child_kernel_id=child_id,
        result_outcome=res.outcome,
        rel_speedup=res.rel_speedup,
        error=res.error,
    )
    return rec


def _isolate_sample_ncu_failure(
    state: TrainState, kernel: KernelState, exc: NCUProfileError
) -> dict[str, Any]:
    """Remove one unprofileable sample without inventing routing evidence."""
    state.store.update_kernel(
        kernel.kernel_id,
        status=STATUS_TERMINAL,
        assigned_leaf_id="",
        meta={
            **kernel.meta,
            "ncu_profile_error": str(exc),
            "terminal_reason": "sample_ncu_error",
        },
    )
    _log(f"  [profile-skip] {kernel.kernel_id} sample error: {exc}")
    return {
        "kernel_id": kernel.kernel_id,
        "origin_uid": kernel.origin_uid,
        "hop": kernel.hop,
        "action": "terminal",
        "outcome": "profile_failed",
        "reason": "sample_ncu_error",
        "error": str(exc),
        "advanced": False,
    }


def _isolate_optimization_service_failure(
    state: TrainState, kernel: KernelState, exc: OptimizationServiceError
) -> dict[str, Any]:
    """Skip one sample after its optimizer request budget is exhausted."""
    state.store.update_kernel(
        kernel.kernel_id,
        status=STATUS_TERMINAL,
        assigned_leaf_id="",
        meta={
            **kernel.meta,
            "optimization_service_error": str(exc),
            "terminal_reason": "optimization_service_error",
        },
    )
    _log(f"  [opt-skip] {kernel.kernel_id} service error: {exc}")
    return {
        "kernel_id": kernel.kernel_id,
        "origin_uid": kernel.origin_uid,
        "hop": kernel.hop,
        "action": "terminal",
        "outcome": "service_error",
        "reason": "optimization_service_error",
        "error": str(exc),
        "advanced": False,
    }


def markov_step(state: TrainState, kernel: KernelState) -> dict[str, Any]:
    """Synchronous compatibility path for one complete Markov step."""
    sample = _sample_for(state, kernel)
    if sample is None:
        state.store.update_kernel(
            kernel.kernel_id, status=STATUS_TERMINAL,
            meta={**kernel.meta, "error": "origin sample not loaded"},
        )
        return {"kernel_id": kernel.kernel_id, "action": "drop", "reason": "no_origin"}
    if kernel.hop >= state.train_cfg.markov.max_hops:
        state.store.update_kernel(kernel.kernel_id, status=STATUS_TERMINAL)
        return {"kernel_id": kernel.kernel_id, "action": "terminal", "reason": "max_hops"}
    try:
        return _apply_markov_work(state, _compute_markov_work(state, kernel))
    except NCUProfileError as exc:
        if exc.sample_level:
            return _isolate_sample_ncu_failure(state, kernel, exc)
        raise
    except OptimizationServiceError as exc:
        return _isolate_optimization_service_failure(state, kernel, exc)
