"""Leaf kernel optimization: the one step that turns a metric set into a reward.

This is the ``action`` half of the Markov chain. The tree chooses WHICH metrics to
measure; this module feeds those readings to a FROZEN model with a FROZEN prompt
and measures what comes out. Nothing here is trainable, by design.

Two properties are load-bearing:

- Correctness is always checked against the ORIGIN's original torch program, never
  against the parent kernel. Checking each hop against its parent at 1e-3 would let
  error accumulate along the chain: five hops of "correct" could land far from the
  original while every single hop passed.

- Speedup is always measured relative to the PARENT kernel (the previous hop),
  because that is what the tree's action is responsible for. The speedup against
  the original seed is derived separately for reporting.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from autoprofiler.llm.client import LLMClient
from autoprofiler.llm.config import LLMSpec
from autoprofiler.tree.leaf_prompts import (
    OPTIMIZE_PROMPT_VERSION,
    OPTIMIZE_SYSTEM,
    OPTIMIZE_SYSTEM_V2,
    build_optimize_repair,
    build_optimize_user,
)
from autoprofiler.tree.reward import RewardConfig, RewardOutcome, reward_from_speedup
from autoprofiler.codegen.port import PortRejected, check_candidate, extract_code
from autoprofiler.codegen.port_prompts import repair_hint
from autoprofiler.backends.verify import VerifyRequest


class OptimizeConfig(LLMSpec):
    """The frozen leaf kernel optimizer: profiling readings -> optimized kernel.

    Model and prompt are fixed and never trained. That is what confines the action space to
    "which metrics to collect"; a variable optimizer prompt would break the Markov formulation.
    """

    temperature: float = 0.2
    max_tokens: int = 6000
    attempts: int = 5            # HTTP attempts per generation, first included
    max_repair_rounds: int = 2   # repair turns after a failed compile / allclose
    n_samples: int = 1           # best-of-N: keep the fastest correct draft
    strong: bool = False         # stronger leaf prompt (tl.dot, combined moves); off while training
    paired_seed: bool = True     # seed from kernel id, so metric sets get the same draw


@dataclass
class OptimizeResult:
    """One leaf optimization attempt, all the way to a measured verdict."""

    ok: bool
    outcome: str
    code: str = ""
    latency_ms: float | None = None
    parent_latency_ms: float | None = None
    rel_speedup: float | None = None
    reward: float | None = None
    stage: str = ""
    error: str = ""
    rounds_used: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.reward is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
            "parent_latency_ms": self.parent_latency_ms,
            "rel_speedup": self.rel_speedup,
            "reward": self.reward,
            "stage": self.stage,
            "error": self.error[:500],
            "rounds_used": self.rounds_used,
            "meta": self.meta,
        }


class OptimizationServiceError(RuntimeError):
    """LLM/verifier infrastructure failed before producing a sample verdict."""


def solver_seed(kernel_id: str, salt: str = "") -> int:
    """Deterministic seed derived from the kernel being optimized.

    This is what makes incumbent-vs-candidate a PAIRED comparison. A large part of
    the margin's variance is the optimizer's own sampling noise; with the seed tied
    to the kernel, both metric sets get the same draw on the same sample, so the
    difference reflects the metric sets rather than the dice. Without this the
    noise can easily exceed the effect and every margin becomes meaningless.
    """
    h = hashlib.sha1(f"{kernel_id}|{salt}".encode()).hexdigest()[:8]
    return int(h, 16)


def _decision_summary(raw: str, max_chars: int = 700) -> str:
    """Keep the optimizer's analysis, excluding its generated source code."""
    analysis = raw.split("```", 1)[0]
    compact = " ".join(analysis.split())
    return compact[:max_chars]


def optimize_kernel(
    verifier,
    ocfg: OptimizeConfig,
    rcfg: RewardConfig,
    llm: LLMClient,
    *,
    kernel_id: str,
    code: str,
    origin_ref_code: str,
    metrics: dict[str, float],
    parent_latency_ms: float | None,
    tag: str,
    log=None,
) -> OptimizeResult:
    """Optimize one kernel using the readings the tree collected for it.

    Best-of-N (ocfg.n_samples > 1): draw N independent candidates and keep the
    FASTEST correct one. n_samples == 1 reproduces the original single-sample path
    byte-for-byte (same seed, same prompt). ocfg.strong selects the stronger leaf
    prompt (tl.dot / combined moves) for eval-time runs.
    """
    n = max(1, ocfg.n_samples)
    system_prompt = OPTIMIZE_SYSTEM_V2 if ocfg.strong else OPTIMIZE_SYSTEM

    best: OptimizeResult | None = None
    last_fail: OptimizeResult | None = None
    for si in range(n):
        # Per-sample seed: keep n==1 identical to the historical single draw; for
        # N>1 vary the draw per sample so candidates differ.
        base_salt = "" if ocfg.paired_seed else tag
        salt = base_salt + (f"#s{si}" if n > 1 else "")
        seed = solver_seed(kernel_id, salt)
        res = _optimize_once(
            verifier, ocfg, rcfg, llm, kernel_id=kernel_id, code=code,
            origin_ref_code=origin_ref_code, metrics=metrics,
            parent_latency_ms=parent_latency_ms, tag=f"{tag}#s{si}" if n > 1 else tag,
            seed=seed, system_prompt=system_prompt, log=log,
        )
        if res.ok and res.latency_ms and (res.rel_speedup or 0) >= 1.0:
            if best is None or (res.latency_ms < (best.latency_ms or float("inf"))):
                best = res
        else:
            last_fail = res
    if best is not None:
        best.meta = {**(best.meta or {}), "n_samples": n, "strong": ocfg.strong}
        return best
    return last_fail or OptimizeResult(
        ok=False, outcome="compile_fail", stage="exhausted",
        error="no correct sample", parent_latency_ms=parent_latency_ms,
    )


def _optimize_once(
    verifier,
    ocfg: OptimizeConfig,
    rcfg: RewardConfig,
    llm: LLMClient,
    *,
    kernel_id: str,
    code: str,
    origin_ref_code: str,
    metrics: dict[str, float],
    parent_latency_ms: float | None,
    tag: str,
    seed: int,
    system_prompt: str,
    log=None,
) -> OptimizeResult:
    """One optimization sample: initial draw + repair rounds."""

    prev_code, prev_error, prev_stage = "", "", ""
    optimizer_summary = ""
    last: OptimizeResult | None = None

    for rnd in range(max(1, ocfg.max_repair_rounds + 1)):
        if prev_code or prev_error:
            user = build_optimize_repair(
                prev_code or code,
                metrics,
                prev_error,
                prev_stage,
                hint=repair_hint(prev_error, prev_stage),
            )
        else:
            user = build_optimize_user(
                code, metrics, parent_latency_ms, seed=seed
            )
        cand_code = ""
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=ocfg.temperature,
                max_tokens=ocfg.max_tokens,
                retries=max(0, int(ocfg.attempts) - 1),
                log=log,
                request_label=f"opt kernel={kernel_id} round={rnd}",
            )
            if not optimizer_summary:
                optimizer_summary = _decision_summary(raw)
            # extract_code may reject before any code exists; check_candidate
            # rejects code we do have. Keeping cand_code assigned in between lets
            # the repair turn show the model what it actually produced.
            _, cand_code = extract_code(raw)
            cand_meta = {
                **check_candidate(cand_code),
                "optimizer_summary": optimizer_summary,
            }
        except PortRejected as e:
            last = OptimizeResult(
                ok=False, outcome="compile_fail", stage=e.stage, error=e.reason,
                parent_latency_ms=parent_latency_ms, rounds_used=rnd + 1,
                meta={
                    "local_reject": True,
                    "prompt_version": OPTIMIZE_PROMPT_VERSION,
                    "optimizer_summary": optimizer_summary,
                },
            )
            prev_code, prev_error, prev_stage = cand_code, e.reason, e.stage
            continue
        except Exception as e:
            raise OptimizationServiceError(
                f"LLM optimization failed: {type(e).__name__}: {e}"
            ) from e

        if cand_code.strip() == code.strip():
            ro = reward_from_speedup(1.0, rcfg)
            return OptimizeResult(
                ok=True, outcome=ro.outcome, code=code,
                latency_ms=parent_latency_ms,
                rel_speedup=1.0, reward=ro.reward, stage="identity",
                error="optimizer returned the input kernel unchanged",
                parent_latency_ms=parent_latency_ms, rounds_used=rnd + 1,
                meta=cand_meta,
            )

        # Correctness is judged against the ORIGIN torch program (see module docs).
        try:
            results = verifier.verify_batch(
                [
                    VerifyRequest(
                        uid=f"{kernel_id}_r{rnd}",
                        ref_code=origin_ref_code,
                        cand_code=cand_code,
                    )
                ],
                tag=f"{tag}_r{rnd}",
            )
        except Exception as e:
            raise OptimizationServiceError(
                f"verification service failed: {type(e).__name__}: {e}"
            ) from e
        vr = results.get(f"{kernel_id}_r{rnd}")
        if vr is None:
            raise OptimizationServiceError(
                f"verification service returned no result for {kernel_id}_r{rnd}"
            )
        if log is not None:
            log(
                f"    [opt] round {rnd} stage={vr.stage} ok={vr.ok} "
                f"cand_ms={vr.candidate_ms}"
            )

        if vr.ok and vr.candidate_ms:
            spd = (
                (parent_latency_ms / vr.candidate_ms)
                if parent_latency_ms and vr.candidate_ms > 0
                else None
            )
            ro: RewardOutcome = reward_from_speedup(spd, rcfg)
            return OptimizeResult(
                ok=True,
                outcome=ro.outcome,
                code=cand_code,
                latency_ms=vr.candidate_ms,
                parent_latency_ms=parent_latency_ms,
                rel_speedup=spd,
                reward=ro.reward,
                stage="ok",
                rounds_used=rnd + 1,
                meta={
                    **cand_meta,
                    "prompt_version": OPTIMIZE_PROMPT_VERSION,
                    "origin_baseline_ms": vr.baseline_ms,
                    "seed": seed,
                },
            )

        outcome = "incorrect" if vr.stage == "correctness" else "compile_fail"
        last = OptimizeResult(
            ok=False, outcome=outcome, stage=vr.stage, error=vr.error,
            parent_latency_ms=parent_latency_ms, rounds_used=rnd + 1,
            meta={**cand_meta, "prompt_version": OPTIMIZE_PROMPT_VERSION},
        )
        prev_code, prev_error, prev_stage = cand_code, vr.error, vr.stage

    return last or OptimizeResult(
        ok=False, outcome="compile_fail", stage="exhausted",
        error="repair budget exhausted", parent_latency_ms=parent_latency_ms,
    )


__all__ = [
    "OptimizationServiceError",
    "OptimizeResult",
    "optimize_kernel",
    "solver_seed",
]
