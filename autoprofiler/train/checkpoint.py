"""Checkpoints, per-epoch snapshots and diagnostics."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from autoprofiler.tree.reward import reward_source_breakdown
from autoprofiler.tree.route import branch_shares, starved_branches
from autoprofiler.tree.store import KernelStore, STATUS_PENDING
from autoprofiler.train.log import _log
from autoprofiler.train.state import TrainState


def diagnostics(state: TrainState, rec: dict[str, Any]) -> dict[str, Any]:
    """Per-epoch health metrics, including the check on the original motivation.

    ``reward_sources`` is the direct evidence for whether "positive reward comes
    from compiles being fixed rather than from speedup" is actually gone.
    """
    steps = rec.get("steps") or []
    outcomes: dict[str, int] = {}
    for s in steps:
        outcomes[str(s.get("outcome") or "?")] = outcomes.get(str(s.get("outcome") or "?"), 0) + 1
    n = len(steps) or 1
    n_slow = sum(
        1 for s in steps
        if not s.get("advanced") and s.get("rel_speedup") is not None
    )
    n_compile_fail = sum(1 for s in steps if s.get("outcome") == "compile_fail")
    profile_ratios = [
        float(s["profile_ms"]) / float(s["parent_latency_ms"])
        for s in steps
        if s.get("profile_ms") and s.get("parent_latency_ms")
    ]
    shares = branch_shares(state.tree, state.store)
    seeds = [ks for ks in state.store.kernels.values() if ks.hop == 0]
    attempted = sum(1 for ks in seeds if ks.status != STATUS_PENDING)
    verifier_stats = getattr(state.verifier, "stats", None)
    queue_stats = verifier_stats() if callable(verifier_stats) else {}
    return {
        "outcome_counts": outcomes,
        "slow_share": n_slow / n,
        "compile_fail_rate": n_compile_fail / n,
        "mean_profile_time_ratio": (
            sum(profile_ratios) / len(profile_ratios) if profile_ratios else None
        ),
        "reward_sources": reward_source_breakdown(steps),
        "tree": {
            "n_leaves": state.tree.total_leaf_count(),
            "depth": state.tree.root.depth(),
            "branching": [
                len(nd.children) for nd in state.tree.root.walk() if not nd.is_leaf
            ],
        },
        "store": state.store.stats(),
        "origin_coverage": {
            "total": len(seeds),
            "attempted": attempted,
            "remaining": len(seeds) - attempted,
            "ratio": attempted / len(seeds) if seeds else 0.0,
        },
        "gpu_queue": queue_stats,
        "branch_share_alerts": starved_branches(
            shares, float(state.train_cfg.monitor.min_branch_share)
        ),
    }


def _save_tree_checkpoint(state: TrainState) -> None:
    """Persist a tree mutation before store records can reference its new leaves."""
    if state.train_cfg.checkpoint.enabled:
        state.tree.save(state.run_dir() / "tree.json")


def _checkpoint_progress(state: TrainState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "epoch": state.epoch,
        "next_epoch": state.next_epoch,
        "bound_index": state.bound_index,
        "history": state.history,
        "value": state.value,
        "leaf_updates": {
            leaf_id: {
                "proposal_ids": list(cycle.proposal_ids),
                "operator": cycle.operator,
                "base_leaf_version": cycle.base_leaf_version,
            }
            for leaf_id, cycle in state.leaf_updates.items()
        },
    }


def _resolve_checkpoint_dir(run_dir: Path, checkpoint_id: str) -> Path:
    root = run_dir / "checkpoints"
    if checkpoint_id == "latest":
        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.isdigit()
        ) if root.exists() else []
        if not candidates:
            raise RuntimeError(f"no historical checkpoints in {run_dir}")
        return candidates[-1]
    if not checkpoint_id.isdigit():
        raise RuntimeError(
            f"invalid checkpoint id {checkpoint_id!r}; use its numeric id or 'latest'"
        )
    path = root / checkpoint_id.zfill(6)
    if not path.is_dir():
        raise RuntimeError(f"checkpoint does not exist: {path}")
    return path


def _checkpoint(state: TrainState, *, reason: str = "periodic") -> str:
    if not state.train_cfg.checkpoint.enabled:
        return ""
    d = state.run_dir()
    d.mkdir(parents=True, exist_ok=True)
    checkpoints = d / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in checkpoints.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    checkpoint_id = f"{(max(existing, default=0) + 1):06d}"
    final = checkpoints / checkpoint_id
    tmp_dir = checkpoints / f".{checkpoint_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    progress = _checkpoint_progress(state)
    state.tree.save(tmp_dir / "tree.json")
    (tmp_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    source_store_manifest = state.store.checkpoint_manifest()
    store_manifest = KernelStore.archive_checkpoint(
        source_store_manifest, d / "checkpoint_store"
    )
    checkpoint_meta = {
        "format": 1,
        "checkpoint_id": checkpoint_id,
        "run_id": state.run_id,
        "reason": reason,
        "epoch": state.epoch,
        "next_epoch": state.next_epoch,
        "bound_index": state.bound_index,
        "leaf_updates": {
            leaf_id: {
                "operator": cycle.operator,
                "proposal_n": len(cycle.proposal_ids),
            }
            for leaf_id, cycle in state.leaf_updates.items()
        },
        "store": store_manifest,
    }
    (tmp_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_dir.replace(final)

    # Keep the fixed paths as a convenient crash-recovery alias, but historical
    # checkpoints above are immutable and remain independently resumable.
    state.tree.save(d / "tree.json")
    tmp = d / "progress.json.tmp"
    tmp.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(d / "progress.json")
    with (checkpoints / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(checkpoint_meta, ensure_ascii=False) + "\n")
    _log(
        f"[checkpoint-save] id={checkpoint_id} reason={reason} "
        f"resume='--resume {state.run_id} --checkpoint {checkpoint_id}'"
    )
    return checkpoint_id


def _write_epoch(state: TrainState, bound: str, epoch: int, rec: dict[str, Any]) -> None:
    if not (state.train_cfg.checkpoint.enabled and state.train_cfg.logging.per_epoch_json):
        return
    d = state.base_dir() / bound / state.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"epoch_{epoch}.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
