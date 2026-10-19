"""Acceptance gates for the unified update operator's rewrite probe.

Four things changed relative to the old gate, each fixing a specific defect:

1. The margin uses ONLY scorable samples. Compile and correctness failures are
   excluded rather than scored at -2 / -1. Those failures are the frozen
   optimizer's problem, not the metric set's, and letting them into the margin was
   what made "the kernel finally compiled" look like profiler progress.

2. The cost of a metric set is its MEASURED collection time as a fraction of the
   kernel's own runtime, not its metric count. NCU bills by replay pass: one more
   counter can be free or can cost a whole extra pass, so counting metrics prices
   the wrong thing. Expressed as a time ratio it is commensurate with
   (speedup - 1), so margin = speedup gain - profiling cost is a real trade.
   (The rewrite gate currently sets this cost to zero because pod NCU timing is
   too unstable to price reliably.)

3. Compile failures still gate, but as a coarse NECESSARY CONDITION outside the
   margin: if a candidate set raises the compile-failure rate beyond a noise
   tolerance, reject outright. The signal is too noisy to differentiate on but
   too important to ignore.

4. Split acceptance lives in ``train._try_split``: it reuses the rewrite probe's
   per-sample rewards and compares the speedup-maximizing partition against the
   incumbent baseline — no second GPU pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from autoprofiler.train.config import (
    AcceptConfig,
    RewriteConfig,
    effective_rewrite_epsilon,
)
from autoprofiler.tree.reward import compile_fail_rate, mean_reward

# (metrics, kernel_ids) -> {kernel_id: SampleEval}
Evaluate = Callable[[list[str], list[str]], dict[str, "SampleEval"]]


@dataclass
class SampleEval:
    """One sample's outcome under one metric set."""

    kernel_id: str
    outcome: str
    reward: float | None = None
    rel_speedup: float | None = None
    latency_ms: float | None = None
    # Measured NCU collection time for the metric set used, and the kernel's own
    # runtime, so cost can be expressed as a ratio.
    profile_ms: float = 0.0
    kernel_ms: float | None = None
    child_kernel_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    # Kept in memory for update-test samples so a winning candidate can be
    # committed only after the leaf operation itself is accepted.
    candidate_code: str = ""

    @property
    def scorable(self) -> bool:
        return self.reward is not None

    @property
    def improved(self) -> bool:
        return bool(self.reward is not None and self.reward > 0)

    def to_dict(self) -> dict:
        return {
            "kernel_id": self.kernel_id,
            "outcome": self.outcome,
            "reward": self.reward,
            "rel_speedup": self.rel_speedup,
            "latency_ms": self.latency_ms,
            "profile_ms": self.profile_ms,
            "kernel_ms": self.kernel_ms,
            "child_kernel_id": self.child_kernel_id,
        }


@dataclass
class RewriteValidation:
    accepted: bool
    margin: float
    epsilon: float
    incumbent_mean: float | None
    candidate_mean: float | None
    time_cost: float = 0.0
    n_scorable_incumbent: int = 0
    n_scorable_candidate: int = 0
    compile_fail_delta: float = 0.0
    reason: str = ""
    # Per-sample outcomes under the candidate set; reused by the split fallback.
    per_sample: dict[str, SampleEval] = field(default_factory=dict)
    guard_ids: list[str] = field(default_factory=list)
    regressed_guards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "margin": self.margin,
            "epsilon": self.epsilon,
            "incumbent_mean": self.incumbent_mean,
            "candidate_mean": self.candidate_mean,
            "time_cost": self.time_cost,
            "n_scorable_incumbent": self.n_scorable_incumbent,
            "n_scorable_candidate": self.n_scorable_candidate,
            "compile_fail_delta": self.compile_fail_delta,
            "reason": self.reason,
            "guard_ids": list(self.guard_ids),
            "regressed_guards": list(self.regressed_guards),
            "per_sample": {k: v.to_dict() for k, v in self.per_sample.items()},
        }


# ── cost ────────────────────────────────────────────────────────────────────────


def profiling_time_cost(
    evals: dict[str, SampleEval], accept: AcceptConfig
) -> float:
    """Mean profiling cost as a fraction of kernel runtime, weighted and capped.

    Deliberately relative: 20 ms of NCU on a 0.05 ms kernel is a catastrophic
    overhead, while the same 20 ms on a 50 ms kernel is noise. Only the ratio makes
    the cost commensurate with (speedup - 1).
    """
    ratios: list[float] = []
    for ev in evals.values():
        if ev.kernel_ms and ev.kernel_ms > 0 and ev.profile_ms > 0:
            ratios.append(ev.profile_ms / ev.kernel_ms)
    if not ratios:
        return 0.0
    mean_ratio = sum(ratios) / len(ratios)
    return min(
        float(accept.metric_time_cap), float(accept.metric_time_weight) * mean_ratio
    )


def compile_fail_gate(
    incumbent: dict[str, SampleEval],
    candidate: dict[str, SampleEval],
    cfg: RewriteConfig,
) -> tuple[bool, float]:
    """Necessary condition: the candidate must not break builds much more often.

    Returns (passed, delta). Kept strictly outside the margin -- see module docs.
    """
    inc_rate = compile_fail_rate({k: v.outcome for k, v in incumbent.items()})
    cand_rate = compile_fail_rate({k: v.outcome for k, v in candidate.items()})
    delta = cand_rate - inc_rate
    return delta <= float(cfg.compile_fail_noise_tol), delta


# ── regression guard ────────────────────────────────────────────────────────────


def select_guard_samples(
    wins: list[tuple[str, float]],
    *,
    top_k: int,
    marginal_k: int,
    random_k: int,
    threshold: float,
    rng_seed: int = 0,
) -> list[str]:
    """Three-tier sample of a leaf's historical wins.

    ``wins`` is (kernel_id, achieved rel_speedup) for kernels this leaf previously
    sped up, where kernel_id is the state BEFORE that optimization.

    Tiering matters: regressions do not appear on the biggest wins, they appear on
    the ones that only just cleared the threshold. ``marginal_k`` is therefore the
    actual detector, ``top_k`` is symbolic, and ``random_k`` exists to counter the
    selection bias of the other two. If regressions still slip through, raise
    marginal_k -- raising top_k would not help.
    """
    import random

    if not wins:
        return []
    by_speed = sorted(wins, key=lambda w: -w[1])
    picked: list[str] = [kid for kid, _ in by_speed[: max(0, top_k)]]

    # Marginal tier: smallest positive distance above the acceptance threshold.
    marginal = sorted(
        (w for w in wins if w[1] >= threshold), key=lambda w: w[1] - threshold
    )
    n_marginal = 0
    for kid, _ in marginal:
        if n_marginal >= max(0, marginal_k):
            break
        if kid in picked:
            continue
        picked.append(kid)
        n_marginal += 1

    rest = [kid for kid, _ in wins if kid not in picked]
    if rest and random_k > 0:
        rng = random.Random(rng_seed)
        picked.extend(rng.sample(rest, min(random_k, len(rest))))
    return list(dict.fromkeys(picked))


# ── REWRITE ─────────────────────────────────────────────────────────────────────


def validate_rewrite(
    *,
    incumbent_metrics: list[str],
    candidate_metrics: list[str],
    slow_ids: list[str],
    guard_ids: list[str],
    incumbent_evals: dict[str, SampleEval],
    evaluate: Evaluate,
    accept: AcceptConfig,
    rewrite_cfg: RewriteConfig,
    depth: int,
    speedup_threshold: float,
) -> RewriteValidation:
    """Validate only the candidate set against persisted forward baselines."""
    eps = effective_rewrite_epsilon(accept, depth)
    ids = list(dict.fromkeys(list(slow_ids) + list(guard_ids)))

    if set(candidate_metrics) == set(incumbent_metrics):
        return RewriteValidation(
            False, 0.0, eps, None, None, reason="no_change", guard_ids=list(guard_ids)
        )
    if not ids:
        return RewriteValidation(
            False, 0.0, eps, None, None, reason="no_validation_samples",
            guard_ids=list(guard_ids),
        )

    inc = {sid: incumbent_evals[sid] for sid in ids if sid in incumbent_evals}
    eval_ids = list(inc.keys())
    if not eval_ids:
        return RewriteValidation(
            False, 0.0, eps, None, None,
            reason="no_incumbent_baselines",
            guard_ids=list(guard_ids),
        )
    cand = evaluate(candidate_metrics, eval_ids)

    passed, delta = compile_fail_gate(inc, cand, rewrite_cfg)
    if not passed:
        return RewriteValidation(
            False, 0.0, eps, None, None, compile_fail_delta=round(delta, 4),
            reason=f"compile_fail_rate_up:{delta:.3f}", per_sample=cand,
            guard_ids=list(guard_ids),
        )

    # Paired means over the samples scorable under BOTH sets: comparing different
    # sample subsets would let composition differences masquerade as a margin.
    paired = [
        sid for sid in eval_ids if inc.get(sid) and cand.get(sid)
        and inc[sid].scorable and cand[sid].scorable
    ]
    inc_mean = mean_reward({sid: inc[sid].reward for sid in paired})
    cand_mean = mean_reward({sid: cand[sid].reward for sid in paired})
    if inc_mean is None or cand_mean is None:
        return RewriteValidation(
            False, 0.0, eps, inc_mean, cand_mean, reason="no_scorable_pairs",
            per_sample=cand, guard_ids=list(guard_ids),
            compile_fail_delta=round(delta, 4),
        )

    # NCU-time penalty retired: pod NCU timing is too unstable to price, so the
    # margin is the raw paired reward difference between candidate and incumbent
    # metric sets.
    cost = 0.0
    margin = cand_mean - inc_mean

    regressed = [
        sid for sid in guard_ids
        if sid in paired
        and (cand[sid].rel_speedup or 0.0) < speedup_threshold
        and (inc[sid].rel_speedup or 0.0) >= speedup_threshold
    ]
    accepted = margin > eps and not regressed
    reason = "accepted"
    if not accepted:
        reason = "guard_regression" if regressed else "below_epsilon"

    return RewriteValidation(
        accepted=accepted,
        margin=round(margin, 4),
        epsilon=round(eps, 4),
        incumbent_mean=round(inc_mean, 4),
        candidate_mean=round(cand_mean, 4),
        time_cost=round(cost, 4),
        n_scorable_incumbent=sum(1 for v in inc.values() if v.scorable),
        n_scorable_candidate=sum(1 for v in cand.values() if v.scorable),
        compile_fail_delta=round(delta, 4),
        reason=reason,
        per_sample=cand,
        guard_ids=list(guard_ids),
        regressed_guards=regressed,
    )


__all__ = [
    "SampleEval",
    "RewriteValidation",
    "validate_rewrite",
    "select_guard_samples",
    "profiling_time_cost",
    "compile_fail_gate",
]
