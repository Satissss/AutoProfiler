"""Branch-condition search for the unified update operator's split fallback.

REWRITE's typical failure is polarization: the new metric set makes some kernels
faster and others slower. That is not a dead end — it is a signal that the leaf is
serving two populations. The split fallback finds, inside the already-measured metric
space, the high/low combination that best separates them by speedup.

The search is deterministic and exhaustive over metric subsets and quantile
thresholds. The LLM is consulted only to break a near-tie among top candidates.

When the current metric space cannot separate populations, the search emits
UNSEPARABLE_HINT so the next proposal knows to add discriminating metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from autoprofiler.train.config import SplitConfig
from autoprofiler.tree.structure import RouterRule, enumerate_subsets


@dataclass
class SpeedupSample:
    """One validation sample carrying its per-metric-set rewards for split search.

    ``reward_new`` / ``reward_old`` are the already-computed (speedup - 1) rewards
    under the candidate (new) and incumbent (old) metric sets; None when that arm
    was not scorable.
    """

    kernel_id: str
    metrics: dict[str, float]
    reward_new: float | None
    reward_old: float | None


@dataclass
class SplitCandidate:
    subset: tuple[str, ...]
    thresholds: dict[str, float]
    gain: float
    # branch index -> kernel_ids
    assignment: dict[int, list[str]] = field(default_factory=dict)
    # branch index -> whether that branch uses the NEW metric set (else incumbent).
    branch_use_new: dict[int, bool] = field(default_factory=dict)

    @property
    def n_branches(self) -> int:
        return 1 << len(self.subset)

    def nonempty_branches(self) -> int:
        return sum(1 for ids in self.assignment.values() if ids)

    def to_router(self, note: str = "") -> RouterRule:
        return RouterRule(
            subset=list(self.subset),
            thresholds=dict(self.thresholds),
            kind="threshold",
            gain=round(self.gain, 5),
            note=note,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset": list(self.subset),
            "thresholds": {k: round(v, 6) for k, v in self.thresholds.items()},
            "gain": round(self.gain, 5),
            "branch_sizes": {i: len(ids) for i, ids in sorted(self.assignment.items())},
        }


@dataclass
class SplitSearchResult:
    ok: bool
    best: SplitCandidate | None = None
    runners_up: list[SplitCandidate] = field(default_factory=list)
    reason: str = ""
    unseparable_hint: dict[str, Any] | None = None
    # Mean incumbent (old-metric) reward over the paired samples; the split's
    # accept margin is best.gain (objective) minus this baseline.
    baseline: float = 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; no numpy dependency."""
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    pos = (q / 100.0) * (len(vs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vs) - 1)
    frac = pos - lo
    return vs[lo] * (1.0 - frac) + vs[hi] * frac


def _branch_index(
    sample: SpeedupSample, subset: Sequence[str], thresholds: dict[str, float]
) -> int:
    """Same bit packing as RouterRule.branch_index; missing counters count LOW."""
    idx = 0
    for m in subset:
        idx <<= 1
        if float(sample.metrics.get(m, float("-inf"))) >= float(thresholds[m]):
            idx |= 1
    return idx


def _paired(samples: Sequence[SpeedupSample]) -> list[SpeedupSample]:
    return [
        s for s in samples if s.reward_new is not None and s.reward_old is not None
    ]


def search_branch_condition_speedup(
    samples: Sequence[SpeedupSample],
    candidate_metrics: Sequence[str],
    cfg: SplitConfig,
) -> SplitSearchResult:
    """Find the branch condition that maximizes total speedup.

    Each candidate branching lets every branch pick the metric set (new or old)
    with the higher mean reward on its own samples. All per-sample rewards under
    both metric sets were already measured during the rewrite probe, so this
    search re-runs nothing — it is pure arithmetic over cached rewards.

    The objective and the incumbent baseline are computed on the PAIRED samples
    (scorable under both sets), so ``best.gain - result.baseline`` is a clean
    paired margin directly comparable to the rewrite gate.
    """
    paired = _paired(samples)
    if len(paired) < 2 * max(1, cfg.n_min):
        return SplitSearchResult(False, reason=f"too_few_samples:{len(paired)}")
    baseline = sum(s.reward_old for s in paired) / len(paired)

    usable = [
        m
        for m in candidate_metrics
        if sum(1 for s in paired if m in s.metrics) >= len(paired) * 0.8
        and len({s.metrics.get(m) for s in paired}) > 1
    ]
    if not usable:
        return SplitSearchResult(
            False,
            reason="no_usable_metrics",
            baseline=baseline,
            unseparable_hint=build_unseparable_hint(paired, candidate_metrics),
        )

    scored: list[SplitCandidate] = []
    for subset in enumerate_subsets(list(usable), cfg.max_subset_size):
        grids = [
            [
                _quantile([s.metrics[m] for s in paired if m in s.metrics], q)
                for q in cfg.threshold_quantiles
            ]
            for m in subset
        ]
        for combo in _product(grids):
            thresholds = {m: t for m, t in zip(subset, combo)}
            groups: dict[int, list[SpeedupSample]] = {
                i: [] for i in range(1 << len(subset))
            }
            for s in paired:
                groups[_branch_index(s, subset, thresholds)].append(s)
            nonempty = [g for g in groups.values() if g]
            if len(nonempty) < 2:
                continue
            if any(len(g) < cfg.n_min for g in nonempty):
                continue
            total = 0.0
            count = 0
            use_new: dict[int, bool] = {}
            for i, g in groups.items():
                if not g:
                    use_new[i] = False
                    continue
                mean_new = sum(s.reward_new for s in g) / len(g)
                mean_old = sum(s.reward_old for s in g) / len(g)
                if mean_new > mean_old:
                    use_new[i] = True
                    total += sum(s.reward_new for s in g)
                else:
                    use_new[i] = False
                    total += sum(s.reward_old for s in g)
                count += len(g)
            objective = total / count if count else 0.0
            all_groups: dict[int, list[str]] = {
                i: [] for i in range(1 << len(subset))
            }
            for s in samples:
                all_groups[_branch_index(s, subset, thresholds)].append(s.kernel_id)
            scored.append(
                SplitCandidate(
                    subset=subset,
                    thresholds=thresholds,
                    gain=objective,
                    assignment=all_groups,
                    branch_use_new=use_new,
                )
            )

    if not scored:
        return SplitSearchResult(
            False,
            reason="no_candidate_meets_n_min",
            baseline=baseline,
            unseparable_hint=build_unseparable_hint(paired, candidate_metrics),
        )
    scored.sort(key=lambda c: (-c.gain, len(c.subset)))
    best = scored[0]
    return SplitSearchResult(
        ok=True,
        best=best,
        runners_up=scored[1 : max(1, cfg.llm_tiebreak_top_n)],
        reason="found",
        baseline=baseline,
    )


def near_ties(result: SplitSearchResult, cfg: SplitConfig) -> list[SplitCandidate]:
    """Candidates within the relative tolerance of the best, for LLM tiebreaking."""
    if not result.ok or result.best is None:
        return []
    best = result.best
    tol = float(cfg.llm_tiebreak_rel_tol)
    out = [best]
    for c in result.runners_up:
        if best.gain > 0 and (best.gain - c.gain) / best.gain <= tol:
            out.append(c)
    return out[: max(1, cfg.llm_tiebreak_top_n)]


def build_unseparable_hint(
    samples: Sequence[SpeedupSample], candidate_metrics: Sequence[str]
) -> dict[str, Any]:
    """Describe both outcome groups so the next proposal can add a metric."""
    improved = [
        s for s in samples
        if s.reward_new is not None and s.reward_old is not None
        and s.reward_new > s.reward_old
    ]
    stalled = [
        s for s in samples
        if s.reward_new is not None and s.reward_old is not None
        and s.reward_new <= s.reward_old
    ]

    def summarize(group: Sequence[SpeedupSample]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m in candidate_metrics:
            vals = [s.metrics[m] for s in group if m in s.metrics]
            if vals:
                out[m] = {
                    "n": len(vals),
                    "min": round(min(vals), 4),
                    "median": round(_quantile(vals, 50), 4),
                    "max": round(max(vals), 4),
                }
        return out

    return {
        "reason": "current metrics cannot separate improved from unimproved kernels",
        "n_improved": len(improved),
        "n_unimproved": len(stalled),
        "improved_metric_ranges": summarize(improved),
        "unimproved_metric_ranges": summarize(stalled),
        "improved_kernel_ids": [s.kernel_id for s in improved][:20],
        "unimproved_kernel_ids": [s.kernel_id for s in stalled][:20],
    }


def _product(grids: list[list[float]]) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = [()]
    for g in grids:
        out = [prev + (v,) for prev in out for v in sorted(set(g))]
    return out


__all__ = [
    "SpeedupSample",
    "SplitCandidate",
    "SplitSearchResult",
    "search_branch_condition_speedup",
    "near_ties",
    "build_unseparable_hint",
]
