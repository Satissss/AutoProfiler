"""The unified leaf update: propose new metrics, probe, accept an in-place REWRITE or fall back to a SPLIT."""
from __future__ import annotations

from typing import Any

from autoprofiler.train.config import effective_split_epsilon
from autoprofiler.train.operators import propose_rewrite, tiebreak_candidates
from autoprofiler.train.split_search import SpeedupSample, SplitSearchResult, near_ties, search_branch_condition_speedup
from autoprofiler.train.validate import SampleEval, select_guard_samples, validate_rewrite
from autoprofiler.tree.catalog import all_catalog_metric_entries
from autoprofiler.tree.store import STATUS_ADVANCED, STATUS_PENDING, STATUS_SLOW, STATUS_UPDATE_TEST
from autoprofiler.tree.structure import Profiler, TreeNode
from autoprofiler.train.checkpoint import _save_tree_checkpoint
from autoprofiler.train.evaluation import _resolve_incumbent_evals, make_evaluator
from autoprofiler.train.log import _log
from autoprofiler.train.state import TrainState


def _slow_sample_payload(
    state: TrainState, leaf: TreeNode, kernel_ids: list[str]
) -> list[dict[str, Any]]:
    """Evidence bundle for the REWRITE prompt: slow kernels + their readings."""
    path_metrics = state.tree.path_metrics(leaf.node_id)
    out: list[dict[str, Any]] = []
    for kid in kernel_ids:
        ks = state.store.get(kid)
        if ks is None:
            continue
        values = state.store.cached_metrics(kid)
        last = next(
            (r for r in reversed(state.store.routes) if r.kernel_id == kid), None
        )
        out.append(
            {
                "kernel_id": kid,
                "runtime_ms": ks.latency_ms,
                "rel_speedup": last.rel_speedup if last else None,
                "outcome": last.outcome if last else "",
                "metrics": {m: values[m] for m in path_metrics if m in values},
                "code_summary": _code_summary(ks.code),
                "optimizer_decision": str(
                    ks.meta.get("optimizer_summary")
                    or (
                        f"outcome={last.outcome}; speedup={last.rel_speedup}"
                        if last is not None
                        else ""
                    )
                ),
            }
        )
    return out


def _code_summary(code: str, max_len: int = 240) -> str:
    """Compact structural summary: kernel names and op mix, not the whole source."""
    import re

    kernels = re.findall(r"def\s+(\w+)\s*\(", code)
    jits = code.count("@triton.jit")
    torch_ops = sorted(set(re.findall(r"(?:torch|F)\.(\w+)", code)))[:8]
    return (
        f"{jits} triton kernel(s); funcs={kernels[:4]}; torch_ops={torch_ops}"
    )[:max_len]


def _guard_ids(state: TrainState, leaf: TreeNode) -> list[str]:
    """Three-tier regression guard drawn from this leaf's historical wins."""
    g = state.train_cfg.rewrite.guard
    wins = [
        (r.kernel_id, float(r.rel_speedup or 0.0))
        for r in state.store.leaf_wins(leaf.node_id)
        if r.rel_speedup
    ]
    return select_guard_samples(
        wins,
        top_k=g.top_k,
        marginal_k=g.marginal_k,
        random_k=g.random_k,
        threshold=float(state.train_cfg.markov.speedup_threshold),
        rng_seed=int(state.train_cfg.data.holdout_seed),
    )


def choose_operator(state: TrainState, leaf: TreeNode) -> str:
    """A single unified update operator.

    The old rewrite / split / free_carve / converge split is gone: an update
    always proposes new metrics, tries the in-place rewrite, falls back to a
    speedup-maximizing split, and on rejection waits for fresh samples. There is
    no preset convergence -- the leaf keeps retrying until it accepts or the pool
    is exhausted.
    """
    return "update"


def update_leaf(
    state: TrainState,
    leaf: TreeNode,
    slow_ids: list[str],
    test_ids: list[str] | None = None,
    *,
    operator: str | None = None,
    proposal: Any = None,
    validation_progress=None,
) -> dict[str, Any]:
    """Run one unified update on an activated leaf."""
    _log(
        f"[update] leaf={leaf.node_id} ({leaf.bound}) op=update "
        f"samples={len(slow_ids)} wins={leaf.history_wins}"
    )
    return do_update(
        state, leaf, slow_ids, proposal=proposal,
        validation_progress=validation_progress,
    )


def _propose(state: TrainState, leaf: TreeNode, slow_ids: list[str]):
    import json
    import time

    _log(
        f"[update-propose] leaf={leaf.node_id} ({leaf.bound}) "
        f"slow={len(slow_ids)}"
    )
    started_at = time.monotonic()
    proposal = propose_rewrite(
        leaf.node_id,
        inherited=state.tree.ancestor_metrics(leaf.node_id),
        owned=list(leaf.profiler.ncu_metrics),
        slow_samples=_slow_sample_payload(state, leaf, slow_ids),
        llm=state.llm,
        cfg=state.train_cfg.rewrite,
        unseparable_hint=leaf.unseparable_hint,
        catalog=all_catalog_metric_entries(),
        log=_log,
    )
    elapsed_s = time.monotonic() - started_at
    result = {
        "added": proposal.added,
        "blind_spot": proposal.blind_spot,
        "rationale": proposal.rationale,
        "rejected": proposal.rejected,
    }
    _log(
        f"[update-proposal-ready] leaf={leaf.node_id} ({leaf.bound}) "
        f"elapsed_s={elapsed_s:.1f} result="
        f"{json.dumps(result, ensure_ascii=False, separators=(',', ':'))}"
    )
    return proposal


def _adopt_rewrite(state: TrainState, leaf: TreeNode, prop: Any, val) -> None:
    """Commit an accepted in-place metric-set replacement."""
    leaf.profiler.ncu_metrics = list(prop.new_owned)
    leaf.version += 1
    leaf.evolution_depth += 1
    leaf.stall_counter = 0
    leaf.last_proposal_size = 0
    leaf.unseparable_hint = None
    _save_tree_checkpoint(state)
    _release_slow(state, leaf, val)
    _bump_value(state, leaf.bound, val.margin)


def _reject_replenish(
    state: TrainState, leaf: TreeNode, slow_ids: list[str],
    out: dict[str, Any], reason: str,
) -> dict[str, Any]:
    """A rejected update: require `replenish_count` NEW samples before retrying.

    The proposal's samples stay tagged to the leaf (their evals are cached and
    reused next round); the leaf simply may not re-propose until it has grown by
    replenish_count, so the next attempt sees fresh evidence rather than the same
    stale set. There is no convergence -- the leaf keeps retrying indefinitely.
    """
    leaf.stall_counter += 1  # retained for logging only; no longer gates anything
    leaf.last_proposal_size = len(slow_ids)
    out.update({"accepted": False, "reason": reason})
    _log(
        f"  [update] REJECT {reason}; replenish baseline={leaf.last_proposal_size} "
        f"(need +{state.train_cfg.activation.replenish_count} new samples)"
    )
    return out


def do_update(
    state: TrainState,
    leaf: TreeNode,
    slow_ids: list[str],
    *,
    proposal: Any = None,
    validation_progress=None,
) -> dict[str, Any]:
    """Unified leaf update: propose -> rewrite -> (fallback) split -> replenish.

    One proposal + one probe. The probe measures every activation sample under
    the NEW metric set (candidate) and reads its OLD-metric speedup from history
    (incumbent) -- both per-sample rewards are then reused by the split fallback
    with NO re-optimization. Acceptance is the paired margin (candidate mean minus
    incumbent mean) > epsilon, unchanged in spirit; only the operators are unified.
    """
    prop = proposal or _propose(state, leaf, slow_ids)
    out: dict[str, Any] = {
        "action": "update", "leaf_id": leaf.node_id, "proposal": prop.to_dict(),
    }
    if not prop.changed:
        return _reject_replenish(state, leaf, slow_ids, out, "no_new_metrics")

    guard = _guard_ids(state, leaf)
    sample_ids = list(dict.fromkeys(slow_ids + guard))
    evaluator = make_evaluator(
        state, leaf, f"e{state.epoch}_up", progress=validation_progress
    )
    incumbent = _resolve_incumbent_evals(state, leaf, sample_ids, evaluator)
    _log(f"  [update] add={prop.added} guard={len(guard)} samples={len(slow_ids)}")
    val = validate_rewrite(
        incumbent_metrics=list(leaf.profiler.ncu_metrics),
        candidate_metrics=list(prop.new_owned),
        slow_ids=slow_ids,
        guard_ids=guard,
        incumbent_evals=incumbent,
        evaluate=evaluator,
        accept=state.train_cfg.accept,
        rewrite_cfg=state.train_cfg.rewrite,
        depth=leaf.evolution_depth,
        speedup_threshold=float(state.train_cfg.markov.speedup_threshold),
    )
    out["validation"] = val.to_dict()

    # 1) In-place rewrite: new metrics beat old across the whole leaf. Gated by
    # rewrite.enabled for the E3 ablation (SPLIT-only arm) -- the probe above
    # still always runs, since _try_split below reuses its per-sample rewards
    # regardless of whether REWRITE itself is allowed to adopt.
    if val.accepted and state.train_cfg.rewrite.enabled:
        if validation_progress is not None:
            validation_progress()
        _adopt_rewrite(state, leaf, prop, val)
        out.update({"accepted": True, "resolution": "rewrite"})
        _log(f"  [rewrite] ACCEPT margin={val.margin} > eps={val.epsilon}")
        return out
    if val.accepted:
        _log(f"  [rewrite] accepted margin={val.margin} but rewrite.enabled=False -> try split")
    else:
        _log(f"  [rewrite] below margin={val.margin} eps={val.epsilon} -> try split")

    # 2) Split fallback: partition so each branch keeps its better metric set,
    #    reusing the per-sample rewards already measured above (no re-run).
    # Gated by split.enabled for the E3 ablation (REWRITE-only arm).
    if state.train_cfg.split.enabled:
        split_out = _try_split(
            state, leaf, slow_ids, prop, val.per_sample, incumbent,
            validation_progress,
        )
        out["split"] = split_out
        if split_out.get("accepted"):
            leaf.last_proposal_size = 0
            out.update({
                "accepted": True, "resolution": "split",
                "children": split_out.get("children"),
            })
            return out
        reject_reason = split_out.get("reason", "rejected")
    else:
        reject_reason = "split_disabled"

    # 3) Both failed -> replenish and retry later on fresh evidence.
    return _reject_replenish(state, leaf, slow_ids, out, reject_reason)


def _release_slow(state: TrainState, leaf: TreeNode, val) -> int:
    """After an accept, requeue the slow samples that the new set sped up.

    They are real optimizations, so they belong back in the chain rather than
    parked at a leaf whose evidence they already outgrew.
    """
    threshold = float(state.train_cfg.markov.speedup_threshold)
    released = 0
    for kid, ev in val.per_sample.items():
        if (ev.rel_speedup or 0.0) >= threshold:
            ks = state.store.get(kid)
            if ks is not None and ks.status == STATUS_SLOW:
                state.store.update_kernel(kid, status=STATUS_ADVANCED)
                released += 1
    return released


def _resolve_condition(
    state: TrainState, leaf: TreeNode, result: SplitSearchResult
) -> tuple[Any, str]:
    """Pick the branch condition, consulting the LLM only on a near-tie."""
    ties = near_ties(result, state.train_cfg.split)
    if len(ties) > 1:
        chosen, why = tiebreak_candidates(ties, state.llm, state.train_cfg.split)
        return chosen, f"llm_tiebreak:{why}"
    return result.best, "top_gain"


def _try_split(
    state: TrainState,
    leaf: TreeNode,
    slow_ids: list[str],
    prop: Any,
    per_sample_new: dict[str, SampleEval],
    incumbent: dict[str, SampleEval],
    validation_progress=None,
) -> dict[str, Any]:
    """Speedup-maximizing split, reusing the rewrite probe's per-sample rewards.

    For every candidate branch condition each branch keeps whichever metric set
    (new or old) gives it the higher mean reward; we take the branching whose
    sample-weighted chosen reward is largest and accept it if that objective beats
    the incumbent (all-old) baseline by more than epsilon. No optimization is
    re-run: ``per_sample_new`` (candidate) and ``incumbent`` (old) already hold
    every sample's reward under both metric sets.
    """
    out: dict[str, Any] = {"action": "split", "leaf_id": leaf.node_id}
    path_metrics = state.tree.path_metrics(leaf.node_id)
    samples: list[SpeedupSample] = []
    for kid in slow_ids:
        ev_new = per_sample_new.get(kid)
        ev_old = incumbent.get(kid)
        values = (ev_new.metrics if ev_new else None) or state.store.cached_metrics(kid)
        samples.append(
            SpeedupSample(
                kernel_id=kid,
                metrics={m: values[m] for m in path_metrics if m in values},
                reward_new=(ev_new.reward if ev_new and ev_new.scorable else None),
                reward_old=(ev_old.reward if ev_old and ev_old.scorable else None),
            )
        )

    result = search_branch_condition_speedup(
        samples, path_metrics, state.train_cfg.split
    )
    out["search_reason"] = result.reason
    if not result.ok or result.best is None:
        if result.unseparable_hint:
            leaf.unseparable_hint = result.unseparable_hint
        out.update({"accepted": False, "reason": result.reason})
        _log(f"  [split] no separating condition ({result.reason})")
        return out

    cand, how = _resolve_condition(state, leaf, result)
    eps = effective_split_epsilon(state.train_cfg.accept, leaf.evolution_depth)
    margin = cand.gain - result.baseline
    out.update({
        "condition": cand.to_dict(),
        "chosen_by": how,
        "objective": round(cand.gain, 4),
        "incumbent_mean": round(result.baseline, 4),
        "margin": round(margin, 4),
        "epsilon": round(eps, 4),
    })
    if margin <= eps:
        out.update({"accepted": False, "reason": "below_epsilon"})
        _log(f"  [split] REJECT below_epsilon margin={margin:.4f} eps={eps:.4f}")
        return out

    # Each branch owns its chosen metric set's increment over the path.
    branch_metrics: dict[int, list[str]] = {}
    for index in range(cand.n_branches):
        use_new = cand.branch_use_new.get(index, False)
        branch_metrics[index] = (
            list(prop.new_owned) if use_new else list(leaf.profiler.ncu_metrics)
        )
    profilers = [
        Profiler(_child_own_metrics(state, leaf, branch_metrics[i]))
        for i in range(cand.n_branches)
    ]
    if validation_progress is not None:
        validation_progress()
    router = cand.to_router(note=f"split:{how}")
    children = leaf.split(router, profilers)
    _save_tree_checkpoint(state)
    _reassign_after_split(state, leaf, children, cand.assignment)
    state.tree.assert_invariants()
    _bump_value(state, leaf.bound, margin)
    out.update({
        "accepted": True,
        "reason": "accepted",
        "children": [c.node_id for c in children],
    })
    _log(
        f"  [split] ACCEPT margin={margin:.4f} subset={router.subset} "
        f"-> {len(children)} branches "
        f"(new={[i for i in branch_metrics if cand.branch_use_new.get(i)]})"
    )
    return out


def _child_own_metrics(
    state: TrainState, leaf: TreeNode, full_metrics: list[str]
) -> list[str]:
    """Convert a branch's FULL evidence set into that child's OWN set.

    Validation reasons about full sets, because that is what the optimizer sees.
    The tree stores increments: after the split the parent keeps its own metrics
    (it needs them to route), so a child must only carry what the parent does not
    already measure. This is also precisely what "the branch inherits the parent's
    metric set" means -- such a child owns NOTHING and its evidence is the path.
    """
    on_path = set(state.tree.path_metrics(leaf.node_id))
    return [m for m in full_metrics if m not in on_path]


def _reassign_after_split(
    state: TrainState,
    leaf: TreeNode,
    children: list[TreeNode],
    assignment: dict[int, list[str]],
) -> dict[str, int]:
    """Redistribute the parent's slow samples onto the new branches."""
    counts: dict[str, int] = {}
    placed: set[str] = set()
    for index, kernel_ids in assignment.items():
        if index >= len(children):
            continue
        child = children[index]
        for kid in kernel_ids:
            ks = state.store.get(kid)
            if ks is None or ks.status != STATUS_SLOW:
                continue
            state.store.update_kernel(kid, assigned_leaf_id=child.node_id)
            counts[child.node_id] = counts.get(child.node_id, 0) + 1
            placed.add(kid)
    # Anything the split did not see goes back into the pool to be re-routed by
    # the new condition rather than being stranded at a node that no longer exists.
    for kid, ks in list(state.store.kernels.items()):
        if ks.status == STATUS_SLOW and ks.assigned_leaf_id == leaf.node_id and kid not in placed:
            state.store.update_kernel(kid, status=STATUS_PENDING, assigned_leaf_id="")
    return counts


def _finish_update_test_samples(
    state: TrainState,
    leaf_id: str,
    test_ids: list[str],
    update_rec: dict[str, Any],
) -> None:
    """Move held-out arrivals exactly once after their leaf update finishes."""
    accepted = bool(update_rec.get("accepted"))
    test = dict(update_rec.get("update_test") or {})
    per_sample = dict(test.get("per_sample") or {})
    assignment = dict(test.get("assignment") or {})
    children = list(update_rec.get("children") or [])
    branch_for: dict[str, int] = {}
    for index, ids in assignment.items():
        for kid in ids:
            branch_for[str(kid)] = int(index)
    threshold = float(state.train_cfg.markov.speedup_threshold)

    for kid in test_ids:
        ks = state.store.get(kid)
        if ks is None or ks.status != STATUS_UPDATE_TEST:
            continue
        ev = dict(per_sample.get(kid) or {})
        if accepted and (ev.get("rel_speedup") or 0.0) >= threshold:
            code = str(ev.get("candidate_code") or "")
            latency_ms = ev.get("latency_ms")
            if code and latency_ms:
                state.store.add_child(
                    ks,
                    code,
                    float(latency_ms),
                    leaf_id=leaf_id,
                    route_id="update_test",
                    meta={
                        "from_update_test": True,
                        "rel_speedup": ev.get("rel_speedup"),
                    },
                )
                state.store.update_kernel(kid, status=STATUS_ADVANCED)
                continue
        if accepted and children:
            index = branch_for.get(kid)
            if index is not None and 0 <= index < len(children):
                state.store.update_kernel(
                    kid, status=STATUS_SLOW, assigned_leaf_id=children[index]
                )
            else:
                state.store.update_kernel(
                    kid, status=STATUS_PENDING, assigned_leaf_id=""
                )
            continue
        state.store.update_kernel(
            kid, status=STATUS_SLOW, assigned_leaf_id=leaf_id
        )


def _bump_value(state: TrainState, bound: str, margin: float) -> None:
    a = state.train_cfg.schedule.value_ema_alpha
    prev = state.value.get(bound, 0.0)
    state.value[bound] = a * float(margin) + (1 - a) * prev
