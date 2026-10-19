"""The mutable state of a training run (tree, kernel store, verifier, budgets)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoprofiler.backends import SingleGPUQueue
from autoprofiler.backends.ncu import MicrobatchNCUProfiler
from autoprofiler.config import ExecConfig
from autoprofiler.llm.client import LLMClient
from autoprofiler.train.config import TrainConfig
from autoprofiler.train.validate import SampleEval
from autoprofiler.tree.route import make_profile_fn
from autoprofiler.tree.store import KernelState, KernelStore
from autoprofiler.tree.structure import MetricTree
from autoprofiler.train.corpus import Sample
from autoprofiler.train.log import _log


@dataclass
class LeafUpdateCycle:
    proposal_ids: list[str]
    operator: str
    base_leaf_version: int = 0


@dataclass
class TrainState:
    cfg: ExecConfig
    train_cfg: TrainConfig
    backend: Any
    llm: LLMClient
    tree: MetricTree
    store: KernelStore
    # Kept separate from the tree-reasoning client so the frozen optimizer really
    # runs on optimize.llm rather than silently inheriting the propose model.
    optimize_llm: LLMClient | None = None
    gpu_queue: SingleGPUQueue | None = None
    ncu_batcher: MicrobatchNCUProfiler | None = None
    verifier: Any = None
    train_samples: list[Sample] = field(default_factory=list)
    holdout_samples: list[Sample] = field(default_factory=list)
    by_uid: dict[str, Sample] = field(default_factory=dict)
    value: dict[str, float] = field(default_factory=dict)
    # (kernel_id, metric signature) -> SampleEval, so a metric set is never
    # re-optimized on the same kernel. With the paired seed the result is
    # reproducible, which makes caching sound rather than merely convenient.
    eval_cache: dict[tuple[str, str], SampleEval] = field(default_factory=dict)
    epoch: int = 0
    next_epoch: int = 0
    bound_index: int = 0
    run_id: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    budget_used: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    # leaf_id -> proposal kernel ids frozen when that leaf reached its configured
    # activation threshold. Later arrivals are persisted as STATUS_UPDATE_TEST.
    leaf_updates: dict[str, LeafUpdateCycle] = field(default_factory=dict)

    def base_dir(self) -> Path:
        return Path(self.train_cfg.checkpoint.dir)

    def run_dir(self) -> Path:
        return self.base_dir() / "_run" / self.run_id

    def out_dir(self) -> Path:
        return self.run_dir()

    def profile_fn(self):
        return make_profile_fn(
            self.backend,
            self.cfg,
            max_retries=self.train_cfg.schedule.ncu_max_retries,
            gpu_queue=self.gpu_queue,
            ncu_batcher=self.ncu_batcher,
        )


def _metric_sig(metrics: list[str]) -> str:
    return ",".join(sorted(set(metrics)))


def seed_pool(state: TrainState) -> int:
    """Register every corpus row as a hop-0 kernel state."""
    n_new = 0
    for s in state.train_samples:
        before = len(state.store.kernels)
        state.store.add_seed(s.uid, s.source, latency_ms=s.latency_ms)
        n_new += len(state.store.kernels) - before
    _log(f"[seed] pool has {len(state.store.active_pool())} pending kernels ({n_new} new)")
    return n_new


def _sample_for(state: TrainState, kernel: KernelState) -> Sample | None:
    return state.by_uid.get(kernel.origin_uid)


def _restore_history_wins(state: TrainState) -> int:
    """Recover durable leaf win counts that occurred after the last checkpoint."""
    changed = 0
    for leaf in state.tree.leaves():
        durable_wins = len(state.store.leaf_wins(leaf.node_id))
        if durable_wins > leaf.history_wins:
            leaf.history_wins = durable_wins
            changed += 1
    return changed
