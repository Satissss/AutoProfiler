"""Update operators: REWRITE proposals and the SPLIT tiebreak.

Routing lives in ``forward.py`` now and uses measured metrics only, so the old
LLM-based ``classify_side`` / ``route_to_leaf`` are gone: a router that read source
code could not be cached, audited, or grounded in hardware.

What remains here is proposal logic:
  - ``propose_rewrite``  : additive-only metric proposals from a leaf's slow samples.
  - ``tiebreak_candidates``: pick among statistically-tied branch conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from autoprofiler.llm.client import LLMClient
from autoprofiler.train.config import RewriteConfig, SplitConfig
from autoprofiler.tree.catalog import all_catalog_metrics
from autoprofiler.train.llm_json import parse_json_object
from autoprofiler.train.rewrite_prompts import (
    REWRITE_PROMPT_VERSION,
    REWRITE_SYSTEM,
    TIEBREAK_SYSTEM,
    build_rewrite_user,
    build_tiebreak_user,
)
from autoprofiler.train.split_search import SplitCandidate


@dataclass
class RewriteProposal:
    leaf_id: str
    inherited: list[str]
    old_owned: list[str]
    new_owned: list[str]
    added: list[str] = field(default_factory=list)
    blind_spot: str = ""
    rationale: dict[str, str] = field(default_factory=dict)
    rejected: list[dict[str, str]] = field(default_factory=list)
    prompt_version: str = REWRITE_PROMPT_VERSION

    @property
    def changed(self) -> bool:
        return set(self.new_owned) != set(self.old_owned)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_id": self.leaf_id,
            "inherited": list(self.inherited),
            "old_owned": list(self.old_owned),
            "new_owned": list(self.new_owned),
            "added": list(self.added),
            "blind_spot": self.blind_spot,
            "rationale": self.rationale,
            "rejected": self.rejected,
            "prompt_version": self.prompt_version,
        }


def propose_rewrite(
    leaf_id: str,
    *,
    inherited: Sequence[str],
    owned: Sequence[str],
    slow_samples: Sequence[dict[str, Any]],
    llm: LLMClient,
    cfg: RewriteConfig,
    unseparable_hint: dict[str, Any] | None = None,
    catalog: Sequence[str] | Sequence[dict[str, Any]] | None = None,
    log=None,
) -> RewriteProposal:
    """Ask for additional metrics for this leaf. Additive only, catalog-checked.

    Every proposal is filtered locally before it can cost a single GPU second:
    invented names, duplicates of inherited/owned metrics, and any attempt to
    remove an incumbent metric are dropped here with a recorded reason.
    """
    catalog_entries = list(catalog) if catalog is not None else all_catalog_metrics()
    names = [
        str(entry.get("name") or "")
        if isinstance(entry, dict)
        else str(entry)
        for entry in catalog_entries
    ]
    names = [name for name in names if name]
    allowed = set(names)
    inherited = list(dict.fromkeys(inherited))
    owned = list(dict.fromkeys(owned))
    present = set(inherited) | set(owned)

    proposal = RewriteProposal(
        leaf_id=leaf_id,
        inherited=inherited,
        old_owned=owned,
        new_owned=list(owned),
    )
    if not slow_samples:
        proposal.rejected.append({"metric": "", "reason": "no_slow_samples"})
        return proposal

    user = build_rewrite_user(
        inherited=inherited,
        owned=owned,
        failed_samples=list(slow_samples),
        catalog=catalog_entries,
        n_max=int(cfg.n_max_new_metrics),
        unseparable_hint=unseparable_hint,
    )
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user},
            ],
            log=log,
            request_label=f"proposal leaf={leaf_id}",
        )
    except Exception as e:
        proposal.rejected.append({"metric": "", "reason": f"llm_failed: {e}"})
        return proposal

    parsed = parse_json_object(resp)
    raw_add = parsed.get("add")
    candidates = (
        [str(m).strip() for m in raw_add if str(m).strip()]
        if isinstance(raw_add, list)
        else []
    )

    accepted: list[str] = []
    for m in candidates:
        if m in present:
            proposal.rejected.append({"metric": m, "reason": "already_measured_on_path"})
            continue
        if m not in allowed:
            proposal.rejected.append({"metric": m, "reason": "not_in_catalog"})
            continue
        if m in accepted:
            continue
        accepted.append(m)
        if len(accepted) >= int(cfg.n_max_new_metrics):
            break

    proposal.added = accepted
    # Additive only: the incumbent set is preserved verbatim. Dropping a metric
    # that looks useless for these failing kernels could remove exactly what made
    # earlier kernels succeed, and that cannot be verified from this evidence.
    proposal.new_owned = owned + accepted
    proposal.blind_spot = str(parsed.get("blind_spot") or "")
    rationale = parsed.get("rationale")
    if isinstance(rationale, dict):
        proposal.rationale = {str(k): str(v) for k, v in rationale.items()}
    return proposal


def tiebreak_candidates(
    candidates: Sequence[SplitCandidate], llm: LLMClient, cfg: SplitConfig
) -> tuple[SplitCandidate, str]:
    """Pick among statistically-tied branch conditions on physical meaning.

    The data cannot separate these, and the purely data-optimal cut is frequently
    an overfit accident, so we prefer a threshold that maps onto a real hardware
    regime. Falls back to the top-gain candidate on any failure.
    """
    if not candidates:
        raise ValueError("no candidates to tiebreak")
    if len(candidates) == 1:
        return candidates[0], "single_candidate"
    payload = [c.to_dict() for c in candidates]
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": TIEBREAK_SYSTEM},
                {"role": "user", "content": build_tiebreak_user(payload)},
            ]
        )
        parsed = parse_json_object(resp)
        idx = int(parsed.get("choice", 0))
        why = str(parsed.get("why") or "")
        if 0 <= idx < len(candidates):
            return candidates[idx], why or "llm_choice"
    except Exception:
        pass
    return candidates[0], "fallback_top_gain"


__all__ = ["RewriteProposal", "propose_rewrite", "tiebreak_candidates"]
