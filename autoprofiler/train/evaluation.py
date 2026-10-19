"""Scoring a candidate metric set on a leaf's samples (the probe behind REWRITE / SPLIT)."""
from __future__ import annotations

from typing import Any

from autoprofiler.train.validate import SampleEval
from autoprofiler.tree.leaf import OptimizationServiceError, OptimizeResult, optimize_kernel
from autoprofiler.tree.reward import reward_from_speedup
from autoprofiler.tree.store import KernelState
from autoprofiler.tree.structure import TreeNode
from autoprofiler.train.log import _log
from autoprofiler.train.markov import _isolate_optimization_service_failure
from autoprofiler.train.state import TrainState, _metric_sig, _sample_for


def _route_to_sample_eval(
    state: TrainState, kernel_id: str, route: Any,
) -> SampleEval | None:
    kernel = state.store.get(kernel_id)
    if kernel is None or route.rel_speedup is None:
        return None
    scored = reward_from_speedup(route.rel_speedup, state.train_cfg.reward)
    metrics = {
        metric: value
        for values in route.metrics_by_node.values()
        for metric, value in values.items()
    }
    return SampleEval(
        kernel_id=kernel_id,
        outcome=route.outcome or scored.outcome,
        reward=scored.reward,
        rel_speedup=route.rel_speedup,
        latency_ms=(
            float(kernel.latency_ms) / float(route.rel_speedup)
            if kernel.latency_ms and route.rel_speedup > 0
            else None
        ),
        # The incumbent profiling and optimization already happened during
        # forward. Candidate validation is charged only for newly collected
        # evidence, rather than re-pricing work that is already sunk.
        profile_ms=0.0,
        kernel_ms=kernel.latency_ms,
        child_kernel_id=route.child_kernel_id,
        metrics=metrics,
    )


def _historical_incumbent_evals(
    state: TrainState,
    leaf: TreeNode,
    kernel_ids: list[str],
) -> dict[str, SampleEval]:
    """Recover each sample's persisted forward result as the incumbent arm.

    After a split, slow samples are reassigned to a child before they are
    re-routed. Their latest scorable route may therefore live on an ancestor
    leaf along the root->leaf path; prefer the deepest such route.
    """
    wanted = set(kernel_ids)
    chain = [n.node_id for n in state.tree.path_to(leaf.node_id)]
    latest_at_leaf: dict[tuple[str, str], Any] = {}
    for route in reversed(state.store.routes):
        if route.kernel_id not in wanted:
            continue
        key = (route.kernel_id, route.leaf_id)
        if key not in latest_at_leaf:
            latest_at_leaf[key] = route

    out: dict[str, SampleEval] = {}
    for kid in kernel_ids:
        route = None
        for lid in reversed(chain):
            candidate = latest_at_leaf.get((kid, lid))
            if candidate is not None and candidate.rel_speedup is not None:
                route = candidate
                break
        if route is None:
            continue
        ev = _route_to_sample_eval(state, kid, route)
        if ev is not None:
            out[kid] = ev
    return out


def _resolve_incumbent_evals(
    state: TrainState,
    leaf: TreeNode,
    kernel_ids: list[str],
    evaluate,
) -> dict[str, SampleEval]:
    """Historical incumbent baselines, backfilling any still-missing samples."""
    incumbent = _historical_incumbent_evals(state, leaf, kernel_ids)
    missing = [kid for kid in kernel_ids if kid not in incumbent]
    if not missing:
        return incumbent
    _log(
        f"  [incumbent-backfill] re-probe n={len(missing)}/{len(kernel_ids)} "
        f"(no scorable route on leaf path)"
    )
    backfilled = evaluate(list(leaf.profiler.ncu_metrics), missing)
    for kid, ev in backfilled.items():
        if ev.scorable:
            incumbent[kid] = ev
    return incumbent


def make_evaluator(
    state: TrainState,
    leaf: TreeNode,
    tag: str,
    *,
    absorb_wins: bool = True,
    progress=None,
):
    """Build ``evaluate(metrics, kernel_ids) -> {kernel_id: SampleEval}``.

    Runs the real optimizer with an OVERRIDE metric set (no routing) so the only
    difference between incumbent and candidate arms is the evidence provided.
    Ancestor metrics are always included: they are part of what the leaf actually
    shows the optimizer, and they cost nothing (already cached).
    """
    ancestors = state.tree.ancestor_metrics(leaf.node_id)
    service_failed: set[str] = set()

    def evaluate(metrics: list[str], kernel_ids: list[str]) -> dict[str, SampleEval]:
        full = list(dict.fromkeys(ancestors + list(metrics)))
        sig = _metric_sig(full)
        out: dict[str, SampleEval] = {}
        for kid in kernel_ids:
            if kid in service_failed:
                if progress is not None:
                    progress()
                continue
            cached = state.eval_cache.get((kid, sig))
            if cached is not None:
                out[kid] = cached
                if progress is not None:
                    progress()
                continue
            try:
                ev = _evaluate_one(
                    state, leaf, kid, full, tag, absorb_wins=absorb_wins
                )
            except OptimizationServiceError as exc:
                service_failed.add(kid)
                kernel = state.store.get(kid)
                if kernel is not None:
                    _isolate_optimization_service_failure(state, kernel, exc)
                if progress is not None:
                    progress()
                continue
            if ev is not None:
                state.eval_cache[(kid, sig)] = ev
                out[kid] = ev
            if progress is not None:
                progress()
        return out

    return evaluate


def _evaluate_one(
    state: TrainState,
    leaf: TreeNode,
    kernel_id: str,
    metrics: list[str],
    tag: str,
    *,
    absorb_wins: bool = True,
) -> SampleEval | None:
    kernel = state.store.get(kernel_id)
    if kernel is None:
        return None
    sample = _sample_for(state, kernel)
    if sample is None:
        return None

    values = state.store.cached_metrics(kernel_id)
    missing = [m for m in metrics if m not in values]
    profile_ms = state.store.profile_cost_ms(kernel_id, metrics) or 0.0
    if missing:
        fresh, cost_ms = state.profile_fn()(kernel, missing, sample.reference())
        if fresh:
            state.store.put_metrics(kernel_id, fresh, profile_ms=cost_ms)
            values.update(fresh)
        profile_ms = (profile_ms or 0.0) + cost_ms
    if not profile_ms:
        profile_ms = state.store.mean_profile_cost_ms(metrics) or 0.0

    res: OptimizeResult = optimize_kernel(
        state.verifier,
        state.train_cfg.optimize,
        state.train_cfg.reward,
        state.optimize_llm or state.llm,
        kernel_id=kernel_id,
        code=kernel.code,
        origin_ref_code=sample.reference(),
        metrics={m: values[m] for m in metrics if m in values},
        parent_latency_ms=kernel.latency_ms,
        tag=tag,
        log=_log,
    )
    # A validation trial that genuinely produced a faster kernel is a real
    # optimization, so it is banked before we return the score.
    if absorb_wins:
        _absorb_validation_win(state, leaf, kernel, res)
    return SampleEval(
        kernel_id=kernel_id,
        outcome=res.outcome,
        reward=res.reward,
        rel_speedup=res.rel_speedup,
        latency_ms=res.latency_ms,
        profile_ms=float(profile_ms),
        kernel_ms=kernel.latency_ms,
        child_kernel_id="",
        metrics={m: values[m] for m in metrics if m in values},
        candidate_code=res.code,
    )


def _absorb_validation_win(
    state: TrainState, leaf: TreeNode, kernel: KernelState, res: OptimizeResult
) -> None:
    """A validation run that produced a genuinely faster kernel is not overhead.

    The trial really did optimize the kernel, so discarding it would burn GPU time
    and throw away a real state. It enters the pool exactly like a normal advance.
    """
    threshold = float(state.train_cfg.markov.speedup_threshold)
    if not (res.ok and res.rel_speedup and res.rel_speedup >= threshold and res.code):
        return
    if kernel.hop + 1 > state.train_cfg.markov.max_hops:
        return
    state.store.add_child(
        kernel, res.code, res.latency_ms,
        leaf_id=leaf.node_id, route_id="validation",
        meta={"from_validation": True, "rel_speedup": res.rel_speedup},
    )
