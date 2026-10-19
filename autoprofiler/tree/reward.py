"""Speedup-only reward.

The old scheme scored a compile failure at -2 and a wrong answer at -1, which made
"the optimizer happened to produce something that builds" the dominant source of
positive reward. That measured the optimizer's luck, not the profiler's value.

Here compile / correctness failures are EXCLUDED rather than penalized:

  - ``reward`` is defined only for a trial that ran and matched the reference; it
    is ``speedup - 1.0`` (clamped), so 1.0x -> 0 and gains stay in speedup units.
  - A failed trial yields ``scorable=False`` and carries NO reward. Callers drop
    it from every mean and margin.
  - Build failures are still watched, but as a separate rate with its own coarse
    gate (see ``validate.compile_fail_gate``), never as a number inside the margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class RewardConfig(BaseModel):
    """Clamps for the speedup-only reward.

    Compile / correctness failures deliberately carry no weight: they are EXCLUDED from the margin,
    not punished by it. A failed build is a property of the frozen optimizer, not of the metric set
    under test, so letting it move the margin would inject the optimizer's noise into the profiler's
    gradient.
    """

    model_config = ConfigDict(extra="forbid")

    floor: float = -0.46  # clamp for (speedup - 1) of a correct-but-slow kernel (~0.54x)
    hi: float = 7.0  # clamp ceiling (~8x speedup)

# Outcome vocabulary. Only the last three are scorable.
OUTCOME_COMPILE_FAIL = "compile_fail"
OUTCOME_INCORRECT = "incorrect"
OUTCOME_WIN = "win"
OUTCOME_NO_SPEEDUP = "no_speedup"
OUTCOME_REGRESSED = "regressed"

SCORABLE_OUTCOMES = (OUTCOME_WIN, OUTCOME_NO_SPEEDUP, OUTCOME_REGRESSED)


@dataclass
class RewardOutcome:
    outcome: str
    # None exactly when the trial is not scorable. Deliberately not 0.0: a
    # non-scorable trial must be excluded from means, and a zero would quietly
    # drag them toward "no change" instead.
    reward: float | None = None
    speedup: float | None = None
    correct: bool = False
    error: str = ""

    @property
    def scorable(self) -> bool:
        return self.reward is not None

    @property
    def compile_failed(self) -> bool:
        return self.outcome == OUTCOME_COMPILE_FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reward": self.reward,
            "speedup": self.speedup,
            "correct": self.correct,
            "error": self.error,
        }


def reward_from_speedup(speedup: float | None, cfg: RewardConfig) -> RewardOutcome:
    """Score a measured speedup. ``speedup`` is always relative to the input kernel."""
    if speedup is None or speedup <= 0:
        return RewardOutcome(
            outcome=OUTCOME_COMPILE_FAIL, error="no speedup measurement"
        )
    spd = max(float(speedup), 1e-6)
    # reward = speedup - 1.0, clamped to [floor, hi] so a single extreme sample
    # (e.g. a 30x outlier) cannot dominate the paired margin the update gate uses.
    r = max(cfg.floor, min(cfg.hi, spd - 1.0))
    if spd > 1.02:
        outcome = OUTCOME_WIN
    elif spd < 0.98:
        outcome = OUTCOME_REGRESSED
    else:
        outcome = OUTCOME_NO_SPEEDUP
    return RewardOutcome(
        outcome=outcome, reward=round(r, 4), speedup=spd, correct=True
    )


def reward_from_trial(trial: Any, cfg: RewardConfig) -> RewardOutcome:
    """Derive the outcome from a TrialResult-like object.

    Never reads ``y``: classification comes from correct + speedup + error only.
    """
    meta = _attr(trial, "metadata", {}) or {}
    error = str(meta.get("error") or "")
    correct = bool(_attr(trial, "correct", False))
    speedup = _attr(trial, "speedup", None)
    err_low = error.lower()

    if not correct:
        numeric = any(
            tok in err_low for tok in ("correctness_failed", "allclose", "not close")
        )
        return RewardOutcome(
            outcome=OUTCOME_INCORRECT if numeric else OUTCOME_COMPILE_FAIL,
            error=error,
        )
    return reward_from_speedup(speedup, cfg)


def mean_reward(rewards: dict[str, float | None]) -> float | None:
    """Mean over SCORABLE entries only. None when nothing is scorable."""
    vals = [v for v in rewards.values() if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def compile_fail_rate(outcomes: dict[str, str]) -> float:
    """Share of trials that never produced a correctness verdict."""
    if not outcomes:
        return 0.0
    n_fail = sum(1 for o in outcomes.values() if o == OUTCOME_COMPILE_FAIL)
    return n_fail / len(outcomes)


def reward_source_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostic: where positive reward actually comes from.

    This is the direct check that the original defect is gone. Under the old
    scheme most positive reward came from a failing kernel becoming a working one;
    it must now come from real speedups. ``failure_to_success`` should be
    structurally zero here, because a non-scorable trial has no reward at all --
    it is reported so a regression back to the old behaviour would be visible
    rather than silent.
    """
    real, f2s, n_pos = 0, 0, 0
    for rec in records:
        r = rec.get("reward")
        if r is None or float(r) <= 0:
            continue
        n_pos += 1
        if rec.get("outcome") in SCORABLE_OUTCOMES:
            real += 1
        else:
            f2s += 1
    return {
        "n_positive_reward": n_pos,
        "from_real_speedup": real,
        "from_failure_to_success": f2s,
        "real_speedup_share": (real / n_pos) if n_pos else None,
    }


def target_reward(cfg: RewardConfig, bound: str) -> float:
    """Coarse per-bound reward ceiling, used only for headroom reporting."""
    targets = {
        "compute": 1.0,     # 2.0x
        "memory": 1.0,
        "latency": 0.4,     # 1.4x
        "algorithm": 1.0,
    }
    return min(cfg.hi, targets.get(bound, 1.0))


def headroom(reward: float | None, cfg: RewardConfig, bound: str) -> float:
    if reward is None:
        return target_reward(cfg, bound)
    return max(0.0, target_reward(cfg, bound) - float(reward))


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        v = getattr(obj, key)
        return default if v is None else v
    return default


__all__ = [
    "RewardOutcome",
    "reward_from_speedup",
    "reward_from_trial",
    "mean_reward",
    "compile_fail_rate",
    "reward_source_breakdown",
    "headroom",
    "target_reward",
    "SCORABLE_OUTCOMES",
    "OUTCOME_COMPILE_FAIL",
    "OUTCOME_INCORRECT",
    "OUTCOME_WIN",
    "OUTCOME_NO_SPEEDUP",
    "OUTCOME_REGRESSED",
]
