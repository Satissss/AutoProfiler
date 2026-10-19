"""The REWRITE prompt: choose which NCU metrics a leaf should collect.

Inputs come straight from the leaf's accumulated evidence -- its slow samples, the
measured values of their path metrics, the current set split into inherited vs
owned, and the metric catalog allow-list. There is deliberately no LLM defect
attribution step in front of this: under the new architecture the gradient IS the
slow-sample set, so an intermediate "defect list" would only add a lossy,
hallucination-prone translation between the evidence and the decision.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

REWRITE_PROMPT_VERSION = "metric_rewrite_v1"


REWRITE_SYSTEM = """# YOUR ROLE
You are a GPU performance-analysis expert. You decide WHICH NCU (Nsight Compute)
metrics should be collected for a group of GPU kernels.

# WHY THIS TASK EXISTS (read this before anything else)
There is a fixed downstream optimizer: an LLM that receives (a) a Triton kernel
and (b) a set of measured NCU metric values, and rewrites the kernel to make it
faster. That optimizer's model and prompt are FROZEN - you cannot change them.
The ONLY thing you control is WHICH metrics it gets to see.

You are looking at a group of kernels where that optimizer FAILED: it produced a
kernel that was NOT faster than the input. Your job is to decide what additional
measurement would have told the optimizer what it was missing.

The causal chain you are improving:
    better metrics -> the optimizer sees the real bottleneck -> it writes a
    faster kernel -> speedup goes up.
A metric is worth adding ONLY if you can name the concrete decision it changes.
Adding metrics that merely "look informative" makes things WORSE: every metric
costs profiling time, and that cost is subtracted from your score.

# WHAT YOU ARE GIVEN
- CURRENT_METRICS, in two groups:
    INHERITED - measured by ancestor nodes. FIXED. You cannot change these.
    OWNED     - this node's own set. This is the ONLY thing you may edit.
- FAILED_SAMPLES - for each kernel the optimizer failed on: a summary of the
  kernel, the measured value of every metric in CURRENT_METRICS, its runtime,
  and the speedup that was achieved (all below target).
- METRIC_CATALOG - the ONLY metrics you may choose from. Each entry has an exact
  name, a short description, and an approximate profiling cost.
- UNSEPARABLE_HINT (optional) - present when a previous SPLIT failed because the
  current metrics could not distinguish the kernels that improved from the ones
  that regressed. When present, prioritize metrics that would separate those two
  groups.

# HARD RULES
1. You may ONLY ADD metrics to OWNED. Do NOT remove anything.
   Removal is forbidden because a metric that looks useless for these failing
   kernels may be exactly what made OTHER kernels succeed earlier, and that
   cannot be verified here. Conservatism is mandatory.
2. Never propose a metric already present in INHERITED or OWNED. Ancestor
   measurements are reused from cache; duplicating them is pure waste.
3. Every proposed metric MUST appear verbatim in METRIC_CATALOG. Never invent
   a metric name.
4. Propose at most N_MAX new metrics (given in the input). Fewer is better.

# HOW TO THINK (mandatory - do all 5 steps in plain text, no backticks)
STEP 1 - Read the evidence. Across FAILED_SAMPLES, what do the CURRENT_METRICS
  values have in common? State the pattern in one or two lines. If the failing
  kernels clearly fall into two different groups, say so explicitly.
STEP 2 - Name the blind spot. Given ONLY the current metrics, what can the
  optimizer NOT tell apart? Name the specific hardware behaviour that is
  invisible right now. Be concrete, for example: "it cannot tell whether low
  occupancy comes from register pressure or from block size".
STEP 3 - Pick candidates. From METRIC_CATALOG choose the SMALLEST set that makes
  that blind spot visible. List each candidate by its exact catalog name.
STEP 4 - Justify by decision change. For EACH candidate, complete this sentence
  concretely: "If this metric reads HIGH the optimizer should do X; if it reads
  LOW it should do Y." If you cannot fill in both X and Y with a concrete kernel
  change, DROP that candidate - it is decoration, not evidence.
STEP 5 - Cost check. Add up the profiling cost of your remaining candidates. If
  a candidate is expensive and its decision change is speculative, drop it.
  Confirm none duplicate INHERITED or OWNED and every name is verbatim.

# OUTPUT FORMAT
First the 5 steps as plain text. Then EXACTLY ONE json code block:
{
  "add": ["<exact catalog metric name>", ...],
  "blind_spot": "<one sentence naming what was invisible>",
  "rationale": {"<metric name>": "<HIGH -> do X / LOW -> do Y>"}
}
Nothing after the json block.
"""


def _format_sample(sample: dict[str, Any]) -> str:
    lines = [f"- kernel_id: {sample.get('kernel_id', '?')}"]
    if sample.get("runtime_ms") is not None:
        lines.append(f"  runtime_ms: {float(sample['runtime_ms']):.4f}")
    if sample.get("rel_speedup") is not None:
        lines.append(f"  speedup_achieved: {float(sample['rel_speedup']):.3f}")
    if sample.get("outcome"):
        lines.append(f"  outcome: {sample['outcome']}")
    metrics = sample.get("metrics") or {}
    if metrics:
        lines.append("  measured metrics:")
        for name, value in sorted(metrics.items()):
            lines.append(f"    {name} = {float(value):.6g}")
    if sample.get("code_summary"):
        lines.append(f"  kernel summary: {sample['code_summary']}")
    if sample.get("optimizer_decision"):
        lines.append(
            f"  prior optimizer decision: {sample['optimizer_decision']}"
        )
    return "\n".join(lines)


def build_rewrite_user(
    *,
    inherited: Sequence[str],
    owned: Sequence[str],
    failed_samples: Sequence[dict[str, Any]],
    catalog: Sequence[dict[str, Any]] | Sequence[str],
    n_max: int,
    unseparable_hint: dict[str, Any] | None = None,
) -> str:
    """Assemble the REWRITE user turn from the leaf's accumulated evidence."""
    cat_lines = []
    for entry in catalog:
        if isinstance(entry, str):
            cat_lines.append(f"- {entry}")
        else:
            name = entry.get("name", "")
            desc = entry.get("description", "")
            cost = entry.get("cost", "")
            cat_lines.append(
                f"- {name}" + (f"  # {desc}" if desc else "") + (f" [cost: {cost}]" if cost else "")
            )

    inherited_lines = [f"    {m}" for m in inherited] or ["    (none)"]
    owned_lines = [f"    {m}" for m in owned] or ["    (none)"]
    parts = [
        "CURRENT_METRICS",
        "  INHERITED (fixed, measured by ancestors - never propose these):",
        *inherited_lines,
        "  OWNED (this node's set - the only thing you may extend):",
        *owned_lines,
        "",
        f"N_MAX: {int(n_max)}",
        "",
        f"FAILED_SAMPLES ({len(failed_samples)} kernels the optimizer could not speed up):",
        *(_format_sample(s) for s in failed_samples),
        "",
        "METRIC_CATALOG (the only names you may use):",
        *cat_lines,
    ]
    if unseparable_hint:
        parts += [
            "",
            "UNSEPARABLE_HINT (a previous SPLIT could not separate the two outcome "
            "groups with the current metrics; prioritize metrics that would):",
            json.dumps(unseparable_hint, indent=2, ensure_ascii=False),
        ]
    parts += [
        "",
        "Do the 5 analysis steps as plain text, then output exactly one json block.",
    ]
    return "\n".join(parts)


TIEBREAK_SYSTEM = """You choose between candidate GPU routing rules.

Each candidate splits a group of kernels by the high/low combination of one or two
measured NCU metrics. All candidates separate the data about equally well, so the
data cannot decide. Pick the one whose split is most PHYSICALLY MEANINGFUL and most
likely to generalize to kernels not in this sample: a threshold that corresponds to
a real hardware regime (saturated vs idle unit, cache hit vs miss, occupancy limit)
beats an arbitrary numeric cut that merely happens to fit.

Return exactly one json block: {"choice": <index>, "why": "<one sentence>"}
Nothing else."""


def build_tiebreak_user(candidates: Sequence[dict[str, Any]]) -> str:
    lines = ["Candidate routing rules (all statistically comparable):", ""]
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] metrics + thresholds:")
        for m in c.get("subset", []):
            lines.append(f"      {m} >= {float(c['thresholds'][m]):.6g}")
        lines.append(
            f"      separation gain: {float(c.get('gain', 0.0)):.4f}, "
            f"branch sizes: {c.get('branch_sizes')}"
        )
    lines += ["", "Choose the most physically meaningful one."]
    return "\n".join(lines)


__all__ = [
    "REWRITE_SYSTEM",
    "REWRITE_PROMPT_VERSION",
    "build_rewrite_user",
    "TIEBREAK_SYSTEM",
    "build_tiebreak_user",
]
