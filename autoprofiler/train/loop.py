"""The training entry point: epochs until convergence, serial or per-bound parallel."""
from __future__ import annotations

import json
from typing import Any

from autoprofiler.tree.catalog import BOUND_KEYS
from autoprofiler.train.checkpoint import _checkpoint, _write_epoch
from autoprofiler.train.epoch import run_epoch
from autoprofiler.train.log import _log
from autoprofiler.train.state import TrainState, seed_pool


def train(state: TrainState) -> dict[str, Any]:
    try:
        seed_pool(state)
        # Make a run resumable before its first (potentially long) routing
        # interval completes.
        _checkpoint(state)
        mode = (state.train_cfg.schedule.bucket_mode or "serial").strip().lower()
        summary: dict[str, Any] = {
            "run_id": state.run_id,
            "bucket_mode": mode,
            "n_train": len(state.train_samples),
            "routing_rounds": list(state.history),
            "bounds": {},
        }
        if mode == "parallel":
            _train_parallel(state, summary)
        else:
            _train_serial(state, summary)
        return _finalize(state, summary)
    finally:
        _close_runtime(state)


def _all_converged(state: TrainState, bounds: list[str] | None) -> bool:
    leaves = [
        l for l in state.tree.leaves()
        if bounds is None or l.bound in bounds
    ]
    return bool(leaves) and all(l.converged for l in leaves)


def _train_serial(state: TrainState, summary: dict[str, Any]) -> None:
    """Continuously route until each bound converges or the pool is exhausted."""
    start_bound = min(max(0, state.bound_index), len(BOUND_KEYS))
    for bound_index in range(start_bound, len(BOUND_KEYS)):
        bk = BOUND_KEYS[bound_index]
        state.bound_index = bound_index
        routing_round = state.next_epoch if bound_index == start_bound else 0
        _log(f"\n########## SUBTREE {bk} (continuous routing) ##########")
        reason = "pool_exhausted"
        max_rounds = int(state.train_cfg.schedule.max_routing_rounds)
        while True:
            if max_rounds > 0 and routing_round >= max_rounds:
                reason = "max_routing_rounds"
                break
            if _all_converged(state, [bk]):
                reason = "all_leaves_converged"
                break
            state.epoch = routing_round
            state.next_epoch = routing_round
            _log(f"\n---------- {bk} routing round {routing_round} ----------")
            rec = run_epoch(state, bounds=[bk])
            rec["bound"] = bk
            state.history.append(rec)
            summary["routing_rounds"].append(rec)
            _write_epoch(state, bk, routing_round, rec)
            state.next_epoch = routing_round + 1
            _checkpoint(state)
            if _all_converged(state, [bk]):
                reason = "converged"
                break
            if not rec["updates"] and rec["n_steps"] == 0:
                reason = "pool_exhausted"
                break
            routing_round += 1
        summary["bounds"][bk] = {"stop_reason": reason}
        state.value[bk] = state.value.get(bk, 0.0) * state.train_cfg.schedule.value_decay
        state.bound_index = bound_index + 1
        state.next_epoch = 0
        _checkpoint(state)
    summary["stop_reason"] = "serial_done"


def _train_parallel(state: TrainState, summary: dict[str, Any]) -> None:
    routing_round = state.next_epoch
    max_rounds = int(state.train_cfg.schedule.max_routing_rounds)
    while True:
        if max_rounds > 0 and routing_round >= max_rounds:
            summary["stop_reason"] = "max_routing_rounds"
            return
        if _all_converged(state, BOUND_KEYS):
            summary["stop_reason"] = "all_leaves_converged"
            break
        state.epoch = routing_round
        state.next_epoch = routing_round
        _log(f"\n========== ROUTING ROUND {routing_round} (parallel) ==========")
        ordered = sorted(BOUND_KEYS, key=lambda bk: state.value.get(bk, 0.0), reverse=True)
        rec = run_epoch(state, bounds=ordered[: state.train_cfg.schedule.top_k])
        state.history.append(rec)
        summary["routing_rounds"].append(rec)
        _write_epoch(state, "all", routing_round, rec)
        for bk in BOUND_KEYS:
            state.value[bk] = state.value.get(bk, 0.0) * state.train_cfg.schedule.value_decay
        state.next_epoch = routing_round + 1
        _checkpoint(state)
        if _all_converged(state, None):
            summary["stop_reason"] = "converged"
            return
        if not rec["updates"] and rec["n_steps"] == 0:
            summary["stop_reason"] = "pool_exhausted"
            return
        routing_round += 1


def _finalize(state: TrainState, summary: dict[str, Any]) -> dict[str, Any]:
    origin_index = state.store.origin_index()
    best_speedups = [
        s for s in (
            state.store.abs_speedup(e["best_kernel_id"])
            for e in origin_index.values() if e.get("best_kernel_id")
        ) if s
    ]
    summary["final_leaf_count"] = state.tree.total_leaf_count()
    summary["tree_depth"] = state.tree.root.depth()
    summary["profilers"] = {
        leaf.node_id: {
            "bound": leaf.bound,
            "own": list(leaf.profiler.ncu_metrics),
            "path": state.tree.path_metrics(leaf.node_id),
            "history_wins": leaf.history_wins,
        }
        for leaf in state.tree.leaves()
    }
    summary["store"] = state.store.stats()
    summary["n_origins_improved"] = sum(1 for s in best_speedups if s > 1.0)
    summary["mean_best_abs_speedup"] = (
        sum(best_speedups) / len(best_speedups) if best_speedups else None
    )
    summary["route_spec"] = state.tree.route_spec()
    out = state.run_dir() / "train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[train] summary -> {out}")
    return summary


def _close_runtime(state: TrainState) -> None:
    close_verifier = getattr(state.verifier, "close", None)
    if callable(close_verifier):
        close_verifier()
    if state.ncu_batcher is not None:
        state.ncu_batcher.close()
    if state.gpu_queue is not None:
        state.gpu_queue.close()
