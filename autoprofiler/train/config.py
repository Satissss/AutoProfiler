"""Training configuration: one strict pydantic model. Unknown or mistyped keys are errors."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from autoprofiler.config import REPO_ROOT
from autoprofiler.llm.config import LLMSpec
from autoprofiler.tree.catalog import (
    BOUND_KEYS,
    DEFAULT_METRIC_CATALOG_PATH,
    _default_seed_metrics,
    load_metric_catalog,
    set_active_catalog,
)
from autoprofiler.tree.leaf import OptimizeConfig
from autoprofiler.tree.reward import RewardConfig


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(_Section):
    datasets: list[dict] = Field(default_factory=list)  # [{id: triton_seed, path: <dir>}]
    limit: int | None = None        # cap on programs loaded per slice; null = all
    uid_manifest: str = ""          # yaml with a `uids:` list, to train on a subset
    holdout_enabled: bool = False   # track a never-trained split as a generalization probe
    holdout_ratio: float = 0.15
    holdout_seed: int = 42          # also seeds pool ordering and guard sampling


class ProposeConfig(LLMSpec):
    """Tree reasoning: REWRITE metric proposals and SPLIT branch conditions."""

    temperature: float = 1.0
    max_tokens: int = 4096
    attempts: int = 5           # HTTP attempts per proposal, first included


class TrainingConfig(_Section):
    warmup: int = 5     # timing warmup iterations on the real GPU
    repeats: int = 30   # timing repeats; higher = steadier speedups, more GPU time


class BucketizeConfig(_Section):
    """The fixed root: two SOL metrics, whose high/low 2x2 gives the four subtrees."""

    mode: str = "rule"      # rule = SM%/Mem% thresholds; llm = one classify call
    sm_hi: float = 60.0     # above = SM saturated (percent)
    mem_hi: float = 60.0    # above = memory saturated
    sm_lo: float = 40.0     # below = SM idle
    mem_lo: float = 40.0    # below = memory idle


class AcceptConfig(_Section):
    """Accept an update when its paired reward margin beats epsilon."""

    rewrite_epsilon: float = 0.015
    rewrite_epsilon_depth_decay: float = 0.75   # per accept along the leaf lineage
    rewrite_epsilon_floor: float = 0.015
    split_epsilon: float = 0.015
    split_epsilon_depth_decay: float = 0.75
    split_epsilon_floor: float = 0.015
    metric_time_weight: float = 0.0   # NCU-time penalty retired; kept as a no-op
    metric_time_cap: float = 0.0


class MarkovConfig(_Section):
    """The kernel-as-state / profiler-as-action chain."""

    speedup_threshold: float = 1.03   # a hop counts as progress above this, vs the PREVIOUS hop
    max_hops: int = 6                 # anti-runaway guard per origin sample
    min_rel_gain: float = 0.01        # a hop gaining less is treated as converged


class ActivationConfig(_Section):
    """When a leaf has enough slow samples to propose an update.

    Kernels that sped up leave the leaf (they re-enter from the root), so what remains is
    exactly the slow set. At the threshold it is frozen for proposal and validation.
    """

    min_slow_samples: int = 12
    min_slow_samples_by_bound: dict[str, int] = Field(default_factory=dict)  # per-bound override
    replenish_count: int = 3          # new samples required before a rejected leaf re-proposes
    update_test_min_samples: int = 4  # legacy held-out gate; unused by the unified update
    update_test_min_margin: float = 0.0

    @field_validator("min_slow_samples_by_bound")
    @classmethod
    def _bounds(cls, v: dict[str, int]) -> dict[str, int]:
        out = {str(k).strip().lower(): int(n) for k, n in v.items()}
        bad = set(out) - set(BOUND_KEYS)
        if bad:
            raise ValueError(f"unknown bound(s) {sorted(bad)}; expected {BOUND_KEYS}")
        return out


def min_slow_samples_for_bound(activation: ActivationConfig, bound: str) -> int:
    """Return the slow-sample activation threshold for a leaf bound."""
    key = str(bound or "").strip().lower()
    override = activation.min_slow_samples_by_bound.get(key)
    if override is not None:
        return max(1, int(override))
    return max(1, int(activation.min_slow_samples))


class GuardConfig(_Section):
    """Anti-forgetting: re-run past wins at their state BEFORE this leaf optimized them."""

    top_k: int = 2       # biggest wins; symbolic, regressions rarely show here
    marginal_k: int = 6  # wins that only just cleared the threshold; the real detector
    random_k: int = 3    # unbiased draw, counters the bias of the two tiers above


class ScheduleConfig(_Section):
    bucket_mode: str = "serial"          # serial = one subtree to convergence; parallel = interleave
    execution_mode: str = "pipeline"     # pipeline overlaps NCU / LLM / verify across kernels
    forward_concurrency: int = 1
    verify_batch_size: int = 8
    verify_batch_wait_ms: int = 100
    ncu_batch_size: int = 8
    ncu_batch_wait_ms: int = 100
    ncu_max_retries: int = 2             # retries for transient NCU failures; then fail closed
    min_leaf_size: int = 16              # samples before a leaf becomes an update target
    max_active_pool: int = 6000          # cap on kernels awaiting routing; lowest-gain evicted
    checkpoint_interval_forwards: int = 120  # reporting interval only; never stops routing
    max_routing_rounds: int = 0          # hard round budget; 0 = until converged or pool empty
    top_k: int = 4                       # (parallel) subtrees advanced per round
    release_ratio: float = 0.5           # samples returned to the queue after an accept
    value_ema_alpha: float = 0.5
    value_decay: float = 0.9


class RewriteConfig(_Section):
    """REWRITE: additive-only metric edits."""

    enabled: bool = True
    n_max_new_metrics: int = 2           # metrics added per REWRITE
    compile_fail_noise_tol: float = 0.15  # reject if compile failures rise more than this
    guard: GuardConfig = Field(default_factory=GuardConfig)


class SplitConfig(_Section):
    """SPLIT: separate the two populations a leaf is serving."""

    enabled: bool = True
    n_min: int = 3              # min samples per non-empty branch
    max_subset_size: int = 2    # |S| <= 2 => at most 4 branches, like the root
    threshold_quantiles: list[float] = Field(default_factory=lambda: [25.0, 50.0, 75.0])
    min_gain: float = 0.02      # below this gini gain, emit UNSEPARABLE_HINT back to REWRITE
    llm_tiebreak_top_n: int = 3     # near-ties handed to the LLM for physical sense
    llm_tiebreak_rel_tol: float = 0.15


class MonitorConfig(_Section):
    """Branch thresholds are never recalibrated (routing would jump); starvation is alerted."""

    min_branch_share: float = 0.05


class LoggingConfig(_Section):
    to_file: bool = True
    dir: str = ""              # empty = checkpoint.dir
    per_epoch_json: bool = True  # dump each routing-round record
    level: str = "normal"      # quiet | normal | verbose


class CacheConfig(_Section):
    """Persist each forward's verdict so restarts reuse the real-machine trial."""

    forward: bool = True
    dir: str = ""   # empty = <checkpoint.dir>/forward_cache


class CheckpointConfig(_Section):
    enabled: bool = True
    dir: str = "./output"   # run dirs, logs, per-epoch trees; data/ holds corpora only
    tree_out: str = ""      # if set, the finished tree is copied here


def effective_rewrite_epsilon(accept: AcceptConfig, depth: int) -> float:
    d = max(0, int(depth))
    return max(
        accept.rewrite_epsilon_floor,
        accept.rewrite_epsilon * (accept.rewrite_epsilon_depth_decay ** d),
    )


def effective_split_epsilon(accept: AcceptConfig, depth: int) -> float:
    d = max(0, int(depth))
    return max(
        accept.split_epsilon_floor,
        accept.split_epsilon * (accept.split_epsilon_depth_decay ** d),
    )


class TrainConfig(_Section):
    metric_catalog: str = str(DEFAULT_METRIC_CATALOG_PATH)  # allow-list REWRITE may propose from
    seed_metrics: dict[str, list[str]] = Field(default_factory=_default_seed_metrics)
    data: DataConfig = Field(default_factory=DataConfig)
    propose: ProposeConfig = Field(default_factory=ProposeConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    bucketize: BucketizeConfig = Field(default_factory=BucketizeConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    accept: AcceptConfig = Field(default_factory=AcceptConfig)
    markov: MarkovConfig = Field(default_factory=MarkovConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("seed_metrics")
    @classmethod
    def _known_bounds(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        bad = set(v) - set(BOUND_KEYS)
        if bad:
            raise ValueError(f"seed_metrics keys must be among {BOUND_KEYS}, got {sorted(bad)}")
        return v


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    """Load and validate a training config; also activates its metric catalog.

    A missing file is an error (a mistyped path must not silently train with defaults).
    """
    if path is None:
        raw: dict[str, Any] = {}
        base = REPO_ROOT
    else:
        p = Path(path).expanduser()
        p = p if p.is_absolute() or p.exists() else REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"training config not found: {path}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        base = p.parent
    cfg = TrainConfig.model_validate(raw)
    cat = Path(cfg.metric_catalog).expanduser()
    for cand in (cat, base / cat, REPO_ROOT / cat):
        if cand.is_absolute() and cand.exists() or cand.exists():
            set_active_catalog(load_metric_catalog(cand))
            break
    else:
        raise FileNotFoundError(f"metric catalog not found: {cfg.metric_catalog}")
    return cfg
