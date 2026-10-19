"""One forward pass: collect -> route -> collect -> ... -> optimize at the leaf.

At each node we measure that node's own metrics and use them, via the node's
threshold card, to pick the next hop. Because the kernel is not modified while
descending, an ancestor's readings stay valid and come from the cache, so each node
only pays for metrics no ancestor already measured.

No LLM is involved in routing. Routing is a deterministic function of measured
counters, which makes it cacheable, auditable, and grounded in hardware rather
than in a model's reading of source code.

At the leaf, the union of every metric measured along the path becomes the evidence
handed to the frozen optimizer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from autoprofiler.config import ExecConfig
from autoprofiler.tree.store import KernelState, KernelStore, RouteRecord
from autoprofiler.tree.measure import profile_leaf
from autoprofiler.tree.structure import MetricTree, TreeNode
from autoprofiler.backends.workers.profile_ncu import ncu_failure_scope

# Injected so this module needs no cluster:
# (kernel, metrics, original torch reference) -> (values, cost_ms).
ProfileFn = Callable[
    [KernelState, list[str], str], tuple[dict[str, float], float]
]


class NCUProfileError(RuntimeError):
    """The requested node profiler did not produce a complete metric set."""

    def __init__(self, message: str, *, sample_level: bool = False) -> None:
        super().__init__(message)
        self.sample_level = sample_level


@dataclass
class RouteOutcome:
    """Result of walking the tree for one kernel (no optimization yet)."""

    leaf: TreeNode
    node_path: list[str] = field(default_factory=list)
    metrics_by_node: dict[str, dict[str, float]] = field(default_factory=dict)
    path_metrics: dict[str, float] = field(default_factory=dict)
    profile_ms: float = 0.0
    n_measured: int = 0
    n_cache_hits: int = 0
    failed_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_id": self.leaf.node_id,
            "node_path": list(self.node_path),
            "metrics_by_node": self.metrics_by_node,
            "profile_ms": self.profile_ms,
            "n_measured": self.n_measured,
            "n_cache_hits": self.n_cache_hits,
            "failed_nodes": list(self.failed_nodes),
        }


def make_profile_fn(
    backend,
    cfg: ExecConfig,
    *,
    max_retries: int = 2,
    gpu_queue=None,
    ncu_batcher=None,
) -> ProfileFn:
    """Real-machine profiler: run NCU for a metric set on one kernel state."""

    def profile(
        kernel: KernelState, metrics: list[str], reference_code: str
    ) -> tuple[dict[str, float], float]:
        program = {
            "uid": kernel.kernel_id,
            "level": 0,
            "problem_id": 0,
            "scope_hint": "kernel",
            # The source travels with the state: a mid-chain kernel exists in no
            # corpus, so there is nothing to look it up by.
            "source": kernel.code,
            # ModelNew intentionally carries no input factories. NCU must use
            # the original torch program for get_inputs/get_init_inputs.
            "reference_code": reference_code,
        }
        requested = list(dict.fromkeys(m for m in metrics if m))

        max_attempts = 1 + max(0, int(max_retries))
        last_error = "incomplete_metrics"
        last_missing = list(requested)
        last_metadata: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            if ncu_batcher is not None:
                payload = ncu_batcher.profile(program, requested)
                values = {
                    str(name): float(value)
                    for name, value in (payload.get("metrics") or {}).items()
                }
                last_missing = [
                    metric for metric in requested if metric not in values
                ]
                metadata = payload.get("metadata") or {}
                last_metadata = dict(metadata)
                last_error = str(metadata.get("error", "incomplete_metrics"))
                ok = payload.get("ok") is True and not last_missing
                profile_ms = float(payload.get("profile_ms") or 0.0)
            else:
                if gpu_queue is None:
                    res = profile_leaf(backend, cfg, program, requested)
                else:
                    res = gpu_queue.run(
                        profile_leaf, backend, cfg, program, requested
                    )
                values = dict(res.metrics)
                last_missing = [
                    metric for metric in requested if metric not in values
                ]
                last_error = "profile_failed"
                ok = bool(res.ok) and not last_missing
                profile_ms = float(res.profile_ms)

            if ok:
                return values, profile_ms
            if attempt < max_attempts:
                print(
                    f"[ncu-retry] kernel={kernel.kernel_id} "
                    f"attempt={attempt}/{max_attempts} error={last_error}",
                    flush=True,
                )

        raise NCUProfileError(
            f"NCU failed for {kernel.kernel_id} after {max_attempts} attempts: "
            f"error={last_error} missing={last_missing}",
            sample_level=ncu_failure_scope(last_error, last_metadata) == "sample",
        )

    return profile


def collect_route(
    tree: MetricTree,
    store: KernelStore,
    kernel: KernelState,
    profile_fn: ProfileFn,
    reference_code: str,
    log=None,
) -> RouteOutcome:
    """Walk root->leaf, measuring each node's own metrics to pick the next hop.

    When ``log`` is given it emits one line per node visited: what was measured there
    and where that sent us, so the root->leaf trajectory is auditable as it happens.
    """
    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    def _fmt_metrics(vals: dict[str, float]) -> str:
        if not vals:
            return "{}"
        return ", ".join(
            f"{k.split('.')[0].split('__')[0]}={v:.3g}" for k, v in vals.items()
        )

    node = tree.root
    depth = 0
    node_path: list[str] = []
    metrics_by_node: dict[str, dict[str, float]] = {}
    path_values: dict[str, float] = {}
    total_ms = 0.0
    n_measured = 0
    n_hits = 0
    failed: list[str] = []

    while True:
        node_path.append(node.node_id)
        wanted = list(node.profiler.ncu_metrics)
        cached = store.cached_metrics(kernel.kernel_id)
        missing = [m for m in wanted if m not in cached]
        values = {m: cached[m] for m in wanted if m in cached}
        n_hits += len(values)

        took = ""
        if missing:
            # NCU blocks for tens of seconds with no output of its own; say so first.
            _emit(f"  route hop{depth}: measuring {len(missing)} counters …")
            import time as _t
            _t0 = _t.monotonic()
            fresh, cost_ms = profile_fn(kernel, missing, reference_code)
            took = f" [{_t.monotonic() - _t0:.1f}s]"
            if fresh:
                store.put_metrics(kernel.kernel_id, fresh, profile_ms=cost_ms)
                values.update({m: v for m, v in fresh.items() if m in wanted})
            total_ms += cost_ms
            n_measured += len(missing)
            if not fresh:
                # ProfileFn must raise on failed/incomplete NCU output. Keep this
                # defensive branch so an injected implementation cannot silently
                # turn missing hardware evidence into a LOW routing decision.
                failed.append(node.node_id)
                raise NCUProfileError(
                    f"NCU returned no metrics for node {node.node_id}"
                )

        metrics_by_node[node.node_id] = dict(values)
        path_values.update(values)

        if node.is_leaf:
            _emit(f"  route hop{depth}: {_fmt_metrics(values)} -> leaf {node.node_id} "
                  f"({node.bound or '-'}){took}")
            return RouteOutcome(
                leaf=node,
                node_path=node_path,
                metrics_by_node=metrics_by_node,
                path_metrics=path_values,
                profile_ms=total_ms,
                n_measured=n_measured,
                n_cache_hits=n_hits,
                failed_nodes=failed,
            )

        router = node.router
        if router is None or not node.children:
            # Structurally impossible if invariants hold; treat as a leaf rather
            # than dropping the sample.
            return RouteOutcome(
                leaf=node, node_path=node_path, metrics_by_node=metrics_by_node,
                path_metrics=path_values, profile_ms=total_ms,
                n_measured=n_measured, n_cache_hits=n_hits, failed_nodes=failed,
            )
        # Route on the accumulated path metrics, not just this node's own
        # readings. A node is forbidden from re-measuring an ancestor's counters
        # (see MetricTree.validate), so a router whose subset references an
        # ancestor-measured metric would otherwise see it as missing (=LOW) and
        # collapse every such decision to branch 0. path_values holds every
        # counter measured from the root to here, which is exactly the "path
        # cache" the tree invariant says routing reads from.
        idx = router.branch_index(path_values)
        child_idx = min(idx, len(node.children) - 1)
        child = node.children[child_idx]
        _emit(f"  route hop{depth}: {_fmt_metrics(values)} -> {router.branch_label(idx)} "
              f"-> {child.node_id}{took}")
        node = child
        depth += 1


def new_route_id() -> str:
    return uuid.uuid4().hex[:12]


def record_route(
    store: KernelStore,
    kernel: KernelState,
    outcome: RouteOutcome,
    *,
    route_id: str,
    child_kernel_id: str = "",
    result_outcome: str = "",
    rel_speedup: float | None = None,
    error: str = "",
) -> RouteRecord:
    """Persist one forward pass for audit and threshold calibration."""
    return store.add_route(
        RouteRecord(
            route_id=route_id,
            kernel_id=kernel.kernel_id,
            origin_uid=kernel.origin_uid,
            node_path=list(outcome.node_path),
            metrics_by_node=outcome.metrics_by_node,
            leaf_id=outcome.leaf.node_id,
            profile_ms=outcome.profile_ms,
            child_kernel_id=child_kernel_id,
            outcome=result_outcome,
            rel_speedup=rel_speedup,
            error=error[:500],
        )
    )


def branch_shares(tree: MetricTree, store: KernelStore) -> dict[str, Any]:
    """Per-decision-node share of samples landing in each branch.

    Thresholds are calibrated on the distribution present at split time. As
    optimized kernels enter the pool that distribution drifts and a branch can
    starve. We deliberately do NOT recalibrate (that would make routing jump
    around); we surface the drift so a human can decide.
    """
    counts: dict[str, dict[str, int]] = {}
    for rec in store.routes:
        for i, node_id in enumerate(rec.node_path[:-1]):
            child = rec.node_path[i + 1]
            counts.setdefault(node_id, {})
            counts[node_id][child] = counts[node_id].get(child, 0) + 1
    out: dict[str, Any] = {}
    for node_id, child_counts in counts.items():
        node = tree.node_by_id(node_id)
        if node is None or node.is_leaf:
            continue
        total = sum(child_counts.values()) or 1
        shares = {}
        for i, child in enumerate(node.children):
            label = node.router.branch_label(i) if node.router else str(i)
            shares[child.node_id] = {
                "label": label,
                "n": child_counts.get(child.node_id, 0),
                "share": child_counts.get(child.node_id, 0) / total,
            }
        out[node_id] = {"n_total": total, "branches": shares}
    return out


def starved_branches(shares: dict[str, Any], min_share: float) -> list[dict[str, Any]]:
    """Branches below the monitored share floor, for the diagnostic report."""
    alerts: list[dict[str, Any]] = []
    for node_id, info in shares.items():
        for child_id, b in info["branches"].items():
            if b["share"] < min_share:
                alerts.append(
                    {
                        "node_id": node_id,
                        "child_id": child_id,
                        "label": b["label"],
                        "share": round(b["share"], 4),
                        "n": b["n"],
                    }
                )
    return alerts


__all__ = [
    "NCUProfileError",
    "RouteOutcome",
    "collect_route",
    "make_profile_fn",
    "record_route",
    "new_route_id",
    "branch_shares",
    "starved_branches",
]
