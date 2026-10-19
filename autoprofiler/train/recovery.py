"""Repair of persisted state when a run is resumed."""
from __future__ import annotations

from autoprofiler.train.config import min_slow_samples_for_bound
from autoprofiler.tree.store import STATUS_ADVANCED, STATUS_OPTIMIZATION_FAILED, STATUS_PENDING, STATUS_SLOW, STATUS_TERMINAL, STATUS_UPDATE_TEST
from autoprofiler.train.state import TrainState


def _repair_persisted_failure_states(state: TrainState) -> dict[str, int]:
    """Repair legacy runs that incorrectly counted failed optimization as slow."""
    latest_route = {route.kernel_id: route for route in state.store.routes}
    repaired = {"service_error_to_pending": 0, "failed_to_excluded": 0}
    for kernel in list(state.store.kernels.values()):
        if kernel.status not in (STATUS_SLOW, STATUS_UPDATE_TEST):
            continue
        route = latest_route.get(kernel.kernel_id)
        if route is None:
            continue
        if route.outcome == "service_error":
            state.store.update_kernel(
                kernel.kernel_id,
                status=STATUS_PENDING,
                assigned_leaf_id="",
                meta={
                    **kernel.meta,
                    "recovered_service_error": route.error,
                },
            )
            repaired["service_error_to_pending"] += 1
        elif route.outcome in ("compile_fail", "incorrect"):
            state.store.update_kernel(
                kernel.kernel_id,
                status=STATUS_OPTIMIZATION_FAILED,
                assigned_leaf_id="",
                meta={
                    **kernel.meta,
                    "optimization_outcome": route.outcome,
                    "optimization_error": route.error,
                    "recovered_legacy_failure": True,
                },
            )
            repaired["failed_to_excluded"] += 1

    # A persisted proposal may have contained failures under the old state
    # machine. Filter it; if fewer than the activation minimum remain, cancel the
    # uncommitted cycle and return its valid held-out samples to the slow set.
    for leaf_id, cycle in list(state.leaf_updates.items()):
        leaf = state.tree.node_by_id(leaf_id)
        minimum = min_slow_samples_for_bound(
            state.train_cfg.activation,
            leaf.bound if leaf is not None else "",
        )
        cycle.proposal_ids = [
            kid
            for kid in cycle.proposal_ids
            if (
                (kernel := state.store.get(kid)) is not None
                and kernel.status in (STATUS_SLOW, STATUS_ADVANCED)
                and kernel.assigned_leaf_id == leaf_id
            )
        ]
        if len(cycle.proposal_ids) >= minimum:
            continue
        state.leaf_updates.pop(leaf_id, None)
        for kernel in list(state.store.kernels.values()):
            if (
                kernel.status == STATUS_UPDATE_TEST
                and kernel.assigned_leaf_id == leaf_id
            ):
                state.store.update_kernel(kernel.kernel_id, status=STATUS_SLOW)
    return repaired


def _recover_leaf_updates(state: TrainState) -> None:
    """Repair the small crash window between tree save and sample reassignment."""
    live_leaf_ids = {leaf.node_id for leaf in state.tree.leaves()}
    converged_leaf_ids = {
        leaf.node_id for leaf in state.tree.leaves() if leaf.converged
    }
    stale: set[str] = set()
    for leaf_id, cycle in state.leaf_updates.items():
        leaf = state.tree.node_by_id(leaf_id)
        if (
            leaf is None
            or not leaf.is_leaf
            or leaf.version != cycle.base_leaf_version
        ):
            stale.add(leaf_id)

    for ks in list(state.store.kernels.values()):
        if (
            ks.assigned_leaf_id in converged_leaf_ids
            and ks.status in (STATUS_SLOW, STATUS_UPDATE_TEST)
        ):
            state.store.update_kernel(
                ks.kernel_id, status=STATUS_TERMINAL, assigned_leaf_id=""
            )
            continue
        invalid_assignment = (
            ks.assigned_leaf_id
            and ks.assigned_leaf_id not in live_leaf_ids
            and ks.status in (STATUS_SLOW, STATUS_UPDATE_TEST)
        )
        orphan_test = (
            ks.status == STATUS_UPDATE_TEST
            and ks.assigned_leaf_id not in state.leaf_updates
        )
        stale_cycle = (
            ks.assigned_leaf_id in stale
            and ks.status in (STATUS_SLOW, STATUS_UPDATE_TEST)
        )
        if invalid_assignment or orphan_test or stale_cycle:
            state.store.update_kernel(
                ks.kernel_id, status=STATUS_PENDING, assigned_leaf_id=""
            )
    for leaf_id in stale:
        state.leaf_updates.pop(leaf_id, None)
