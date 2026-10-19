"""One training epoch: advance the chain, then update ready leaves."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any

from autoprofiler.tree.leaf import OptimizationServiceError
from autoprofiler.tree.route import NCUProfileError
from autoprofiler.tree.store import KernelState, STATUS_PENDING, STATUS_TERMINAL
from autoprofiler.tree.structure import MetricTree
from autoprofiler.train.checkpoint import _checkpoint, diagnostics
from autoprofiler.train.log import _log, _nonblocking_executor
from autoprofiler.train.markov import _apply_markov_work, _compute_markov_work, _isolate_optimization_service_failure, _isolate_sample_ncu_failure, markov_step
from autoprofiler.train.pool import _enforce_pool_cap, _in_bound_update_cycles, _pool_order, _ready_leaves, _start_update_cycles
from autoprofiler.train.progress import _log_routing_start, _log_routing_summary, _log_step_progress, _maybe_progress_checkpoint
from autoprofiler.train.recovery import _recover_leaf_updates
from autoprofiler.train.state import TrainState, _sample_for
from autoprofiler.train.update import _finish_update_test_samples, _propose, update_leaf


def _run_epoch_synchronous(
    state: TrainState, bounds: list[str] | None
) -> dict[str, Any]:
    """Original one-kernel-at-a-time execution, kept as a compatibility mode."""
    checkpoint_interval = int(
        state.train_cfg.schedule.checkpoint_interval_forwards
    )
    rec: dict[str, Any] = {
        "routing_round": state.epoch,
        "bounds": bounds,
        "steps": [],
        "updates": [],
        "n_retired_pool": 0,
    }
    rec["n_retired_pool"] = _enforce_pool_cap(state)

    pending = [
        kid for kid in _pool_order(state)
        if (ks := state.store.get(kid)) is not None and ks.status == STATUS_PENDING
    ]
    _log_routing_start(
        state,
        bounds=bounds,
        selected=len(pending),
        mode="sync",
    )

    n_steps = 0
    # A full leaf has priority over all further routing. This also makes resume
    # deterministic: if the threshold was persisted before interruption, update
    # starts immediately without spending one more kernel.
    if not _ready_leaves(state, bounds):
        for kid in pending:
            ks = state.store.get(kid)
            if ks is None or ks.status != STATUS_PENDING:
                continue
            rec["steps"].append(markov_step(state, ks))
            n_steps += 1
            _log_step_progress(state, rec, available=len(pending))
            _maybe_progress_checkpoint(
                state, rec, interval=checkpoint_interval
            )
            if _ready_leaves(state, bounds):
                break

    for leaf, slow_ids in _ready_leaves(state, bounds):
        rec["updates"].append(update_leaf(state, leaf, slow_ids))

    rec["n_steps"] = n_steps
    rec["pool_size"] = len(state.store.active_pool())
    rec["diagnostics"] = diagnostics(state, rec)
    _log_routing_summary(rec)
    return rec


def _run_epoch_pipeline(
    state: TrainState, bounds: list[str] | None
) -> dict[str, Any]:
    """Overlap different kernels while keeping every state transition single-writer."""
    checkpoint_interval = int(
        state.train_cfg.schedule.checkpoint_interval_forwards
    )
    workers = max(1, int(state.train_cfg.schedule.forward_concurrency))
    rec: dict[str, Any] = {
        "routing_round": state.epoch,
        "bounds": bounds,
        "steps": [],
        "updates": [],
        "n_retired_pool": _enforce_pool_cap(state),
        "execution_mode": "pipeline",
    }

    selected: list[KernelState] = []
    origins: set[str] = set()
    for kid in _pool_order(state):
        ks = state.store.get(kid)
        if ks is None or ks.status != STATUS_PENDING or ks.origin_uid in origins:
            continue
        if _sample_for(state, ks) is None:
            state.store.update_kernel(
                kid, status=STATUS_TERMINAL,
                meta={**ks.meta, "error": "origin sample not loaded"},
            )
            continue
        if ks.hop >= state.train_cfg.markov.max_hops:
            state.store.update_kernel(kid, status=STATUS_TERMINAL)
            continue
        selected.append(ks)
        origins.add(ks.origin_uid)

    _log_routing_start(
        state,
        bounds=bounds,
        selected=len(selected),
        mode="pipeline",
        workers=workers,
    )
    if selected:
        preview = ", ".join(ks.kernel_id[:12] for ks in selected[:4])
        suffix = "..." if len(selected) > 4 else ""
        _log(f"[pipeline] first batch: {preview}{suffix}")

    proposal_futures: dict[str, Any] = {}
    routing_tree = MetricTree.from_dict(state.tree.to_dict())
    with _nonblocking_executor(workers) as forward_pool, _nonblocking_executor(
        max(1, min(workers, 4))
    ) as proposal_pool:
        started = _start_update_cycles(state, bounds)
        started_ids = {
            leaf.node_id
            for leaf, cycle in _in_bound_update_cycles(state, bounds)
            if cycle in started
        }
        for leaf, cycle in _in_bound_update_cycles(state, bounds):
            if leaf.node_id not in started_ids:
                _log(
                    f"[update-resume] leaf={leaf.node_id} ({leaf.bound}) "
                    f"op={cycle.operator} proposal={len(cycle.proposal_ids)}; "
                    "restarting proposal and validation"
                )
            proposal_futures[leaf.node_id] = proposal_pool.submit(
                _propose, state, leaf, list(cycle.proposal_ids)
            )

        selected_iter = iter(selected)
        futures: dict[Any, KernelState] = {}
        closed_update_leaves: set[str] = set()

        def submit_until_full() -> None:
            while len(futures) < workers:
                try:
                    ks = next(selected_iter)
                except StopIteration:
                    return
                futures[
                    forward_pool.submit(
                        _compute_markov_work, state, ks, routing_tree
                    )
                ] = ks

        def process_forward(fut) -> None:
            ks = futures.pop(fut)
            try:
                work = fut.result()
                if work.route.leaf.node_id in closed_update_leaves:
                    state.store.update_kernel(
                        ks.kernel_id,
                        status=STATUS_TERMINAL,
                        assigned_leaf_id="",
                        meta={
                            **ks.meta,
                            "terminal_reason": "update_tail_abandoned",
                        },
                    )
                    step = {
                        "kernel_id": ks.kernel_id,
                        "origin_uid": ks.origin_uid,
                        "hop": ks.hop,
                        "action": "terminal",
                        "outcome": "update_tail_abandoned",
                        "reason": "update_tail_abandoned",
                        "advanced": False,
                    }
                    _log(
                        f"  [update-tail-skip] {ks.kernel_id} "
                        "finished after its queue closed"
                    )
                else:
                    step = _apply_markov_work(state, work)
            except OptimizationServiceError as exc:
                step = _isolate_optimization_service_failure(state, ks, exc)
            except NCUProfileError as exc:
                if exc.sample_level:
                    step = _isolate_sample_ncu_failure(state, ks, exc)
                else:
                    # System-level profiling failures still stop the run. Preserve
                    # the sample for resume and never route without evidence.
                    state.store.update_kernel(
                        ks.kernel_id,
                        status=STATUS_PENDING,
                        meta={**ks.meta, "ncu_profile_error": str(exc)},
                    )
                    _checkpoint(state)
                    _log(f"[fatal] {exc}")
                    raise
            except Exception as exc:
                state.store.update_kernel(
                    ks.kernel_id,
                    status=STATUS_TERMINAL,
                    meta={
                        **ks.meta,
                        "pipeline_error": f"{type(exc).__name__}: {exc}",
                    },
                )
                step = {
                    "kernel_id": ks.kernel_id,
                    "action": "terminal",
                    "reason": "pipeline_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rec["steps"].append(step)
            _log_step_progress(state, rec, available=len(selected))
            _maybe_progress_checkpoint(
                state, rec, interval=checkpoint_interval
            )

        def drain_ready_forwards() -> None:
            for fut in [f for f in futures if f.done()]:
                process_forward(fut)

        def start_new_cycles() -> None:
            for cycle in _start_update_cycles(state, bounds):
                leaf = next(
                    l for l in state.tree.leaves()
                    if state.leaf_updates.get(l.node_id) is cycle
                )
                proposal_futures[leaf.node_id] = proposal_pool.submit(
                    _propose, state, leaf, list(cycle.proposal_ids)
                )

        submit_until_full()
        stop_for_update = False
        while futures and not stop_for_update:
            watched = set(futures) | set(proposal_futures.values())
            done, _ = wait(watched, return_when=FIRST_COMPLETED)
            for fut in [f for f in done if f in futures]:
                process_forward(fut)
            start_new_cycles()
            if any(future.done() for future in proposal_futures.values()):
                # Proposal completion starts the frozen 20-sample validation
                # immediately. In-flight forwards may join its optional test tail
                # while validation runs, but are never a barrier in front of it.
                stop_for_update = True
            if not stop_for_update:
                submit_until_full()

        held = state.store.leaf_update_test_set()
        test_ids_by_leaf = {
            leaf_id: list(ids) for leaf_id, ids in held.items()
        }

        def validation_progress() -> None:
            """Append only forward results that finish while core validation runs."""
            drain_ready_forwards()
            fresh = state.store.leaf_update_test_set()
            for leaf_id, ids in fresh.items():
                queue = test_ids_by_leaf.setdefault(leaf_id, [])
                known = set(queue)
                queue.extend(kid for kid in ids if kid not in known)

        for leaf_id, cycle in list(state.leaf_updates.items()):
            leaf = state.tree.node_by_id(leaf_id)
            if leaf is None or not leaf.is_leaf:
                _recover_leaf_updates(state)
                continue
            proposal = None
            pf = proposal_futures.get(leaf_id)
            if pf is not None:
                # Waiting for the proposal is necessary; waiting for unrelated
                # Kimi forwards is not. Harvest any that finish in the meantime.
                while not pf.done():
                    watched = set(futures) | {pf}
                    wait(watched, return_when=FIRST_COMPLETED)
                    validation_progress()
                try:
                    proposal = pf.result()
                except Exception as exc:
                    _log(f"[update] proposal failed for {leaf_id}: {exc}")
            test_ids = test_ids_by_leaf.setdefault(leaf_id, [])
            try:
                update_rec = update_leaf(
                    state,
                    leaf,
                    list(cycle.proposal_ids),
                    test_ids,
                    operator=cycle.operator,
                    proposal=proposal,
                    validation_progress=validation_progress,
                )
                update_rec["proposal_n"] = len(cycle.proposal_ids)
                update_rec["update_test_n"] = len(test_ids)
            except Exception as exc:
                rec["updates"].append(
                    {
                        "leaf_id": leaf_id,
                        "action": cycle.operator,
                        "accepted": False,
                        "reason": "update_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rec["updates"].append(update_rec)
            # Remove the cycle durably before releasing held samples. A crash in
            # the following small window leaves orphan STATUS_UPDATE_TEST rows,
            # which resume safely re-routes instead of replaying the operation.
            state.leaf_updates.pop(leaf_id, None)
            closed_update_leaves.add(leaf_id)
            _checkpoint(state)
            _finish_update_test_samples(state, leaf_id, test_ids, update_rec)
            _checkpoint(state)

        # The core queue has closed. Running tail requests are deliberately not
        # awaited: their old-tree results cannot participate in a committed update.
        abandoned = 0
        for fut, ks in list(futures.items()):
            fut.cancel()
            current = state.store.get(ks.kernel_id)
            if current is not None and current.status == STATUS_PENDING:
                state.store.update_kernel(
                    ks.kernel_id,
                    status=STATUS_TERMINAL,
                    assigned_leaf_id="",
                    meta={
                        **current.meta,
                        "terminal_reason": "update_tail_abandoned",
                    },
                )
                abandoned += 1
                _log(f"  [update-tail-skip] {ks.kernel_id} result not awaited")
            futures.pop(fut, None)
        rec["n_abandoned_update_tail"] = abandoned

    rec["n_steps"] = len(rec["steps"])
    rec["pool_size"] = len(state.store.active_pool())
    rec["diagnostics"] = diagnostics(state, rec)
    _log_routing_summary(rec)
    return rec


def run_epoch(state: TrainState, bounds: list[str] | None = None) -> dict[str, Any]:
    """Run one epoch using either the original or pipelined scheduler."""
    mode = str(state.train_cfg.schedule.execution_mode or "synchronous").lower()
    if mode == "pipeline":
        return _run_epoch_pipeline(state, bounds)
    return _run_epoch_synchronous(state, bounds)
