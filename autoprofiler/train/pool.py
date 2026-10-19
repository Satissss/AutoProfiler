"""The active kernel pool: ordering, size cap, which leaves are ready to update."""
from __future__ import annotations

import hashlib

from autoprofiler.train.config import min_slow_samples_for_bound
from autoprofiler.tree.store import STATUS_TERMINAL
from autoprofiler.tree.structure import TreeNode
from autoprofiler.train.checkpoint import _checkpoint
from autoprofiler.train.log import _log
from autoprofiler.train.state import LeafUpdateCycle, TrainState
from autoprofiler.train.update import choose_operator


def _pool_order(state: TrainState) -> list[str]:
    """Deterministic pool order: earlier hops first, then a stable hash shuffle.

    Hop-first keeps the corpus broad instead of chasing one origin down a deep
    chain while thousands of seeds are still untouched.
    """
    seed = int(state.train_cfg.data.holdout_seed)
    pending = [state.store.kernels[k] for k in state.store.active_pool()]
    return [
        ks.kernel_id
        for ks in sorted(
            pending,
            key=lambda k: (
                k.hop,
                hashlib.sha1(f"{seed}:{k.kernel_id}".encode()).hexdigest(),
            ),
        )
    ]


def _enforce_pool_cap(state: TrainState) -> int:
    """Evict the least promising pending kernels when the pool exceeds its cap.

    Rank by realized gain over the previous hop: a kernel whose last improvement
    was marginal is the least likely to repay another full profiling pass.
    """
    cap = int(state.train_cfg.schedule.max_active_pool)
    pool = state.store.active_pool()
    untouched_seeds = [
        kid for kid in pool
        if state.store.get(kid) and state.store.get(kid).hop == 0
    ]
    effective_cap = max(cap, len(untouched_seeds))
    if cap <= 0 or len(pool) <= effective_cap:
        return 0

    def gain(kid: str) -> float:
        spd = state.store.rel_speedup(kid)
        return spd if spd is not None else 1.0

    # A hop-0 pending state has never received its first training attempt. Do not
    # let a previously accelerated child displace untouched corpus coverage.
    ranked = sorted(
        pool,
        key=lambda kid: (
            0 if (state.store.get(kid) and state.store.get(kid).hop == 0) else 1,
            -gain(kid),
        ),
    )
    for kid in ranked[effective_cap:]:
        state.store.update_kernel(kid, status=STATUS_TERMINAL)
    n = len(ranked) - effective_cap
    _log(
        f"[budget] pool cap {cap} exceeded; retired {n} derived kernels "
        f"(protected {len(untouched_seeds)} untouched seeds)"
    )
    return n


def _activation_need(state: TrainState, leaf: TreeNode) -> int:
    """Samples required before this leaf may (re)propose.

    First activation needs `min_slow_samples`. After a rejected update the leaf
    must grow by `replenish_count` beyond the size it last proposed on, so a
    retry always sees fresh evidence rather than the identical stale set.
    """
    act = state.train_cfg.activation
    base = min_slow_samples_for_bound(act, leaf.bound)
    if leaf.last_proposal_size > 0:
        return max(base, leaf.last_proposal_size + int(act.replenish_count))
    return base


def _ready_leaves(state: TrainState, bounds: list[str] | None) -> list[tuple[TreeNode, list[str]]]:
    """Leaves whose sample set has reached the activation threshold."""
    slow = state.store.leaf_slow_set()
    out: list[tuple[TreeNode, list[str]]] = []
    for leaf in state.tree.leaves():
        if leaf.converged:
            continue
        if leaf.node_id in state.leaf_updates:
            continue
        if bounds is not None and leaf.bound not in bounds:
            continue
        ids = slow.get(leaf.node_id, [])
        if len(ids) >= _activation_need(state, leaf):
            out.append((leaf, ids))
    return out


def _start_update_cycles(
    state: TrainState, bounds: list[str] | None
) -> list[LeafUpdateCycle]:
    started: list[LeafUpdateCycle] = []
    slow = state.store.leaf_slow_set()
    for leaf in state.tree.leaves():
        if leaf.converged or leaf.node_id in state.leaf_updates:
            continue
        if bounds is not None and leaf.bound not in bounds:
            continue
        ids = list(slow.get(leaf.node_id, []))
        if len(ids) < _activation_need(state, leaf):
            continue
        cycle = LeafUpdateCycle(
            ids, choose_operator(state, leaf), base_leaf_version=leaf.version
        )
        state.leaf_updates[leaf.node_id] = cycle
        started.append(cycle)
        _log(
            f"[update-start] leaf={leaf.node_id} op={cycle.operator} "
            f"proposal={len(ids)}; later arrivals become update-test samples"
        )
    if started:
        _checkpoint(state, reason="leaf_activation")
    return started


def _in_bound_update_cycles(
    state: TrainState, bounds: list[str] | None
) -> list[tuple[TreeNode, LeafUpdateCycle]]:
    """Uncommitted leaf updates that belong to this routing round's bounds."""
    out: list[tuple[TreeNode, LeafUpdateCycle]] = []
    for leaf_id, cycle in list(state.leaf_updates.items()):
        leaf = state.tree.node_by_id(leaf_id)
        if leaf is None or not leaf.is_leaf:
            continue
        if bounds is not None and leaf.bound not in bounds:
            continue
        out.append((leaf, cycle))
    return out
