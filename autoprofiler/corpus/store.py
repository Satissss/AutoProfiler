"""Persistence for the seed pre-experiment.

Three artifacts, each with a distinct job:

  attempts.jsonl   append-only log of every attempt (LLM turn + machine verdict).
                   This is the resume ledger and the failure-cause evidence; it is
                   written before anything else so a crash never loses paid work.
  seed_triton/     the accepted ports, one .py per row, plus seed_index.jsonl and
                   (when a parquet writer is available) seed.parquet.
  report.json      aggregate yield and failure-cause breakdown.

The frozen seed set is written LAST and is fully derivable from attempts.jsonl,
so it can always be rebuilt with --report-only without re-running anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Attempt:
    """One (LLM proposal -> machine verdict) pair for one row."""

    uid: str
    row_index: int
    round: int
    ok: bool
    stage: str
    error: str = ""
    baseline_ms: float | None = None
    candidate_ms: float | None = None
    speedup: float | None = None
    code: str = ""
    # The ORIGINAL torch program. Kept with every accepted port because it stays
    # the correctness reference for the whole downstream optimization chain: each
    # hop is checked against this, never against its parent kernel, so numeric
    # drift cannot accumulate along a multi-hop trajectory.
    ref_code: str = ""
    ops: str = ""
    prompt_version: str = ""
    model: str = ""
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class AttemptLog:
    """Append-only jsonl ledger with resume support."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def append(self, attempt: Attempt) -> None:
        self._fh.write(json.dumps(asdict(attempt), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    @staticmethod
    def read_all(path: Path) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        out: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def resume_state(path: Path, *, max_rounds: int) -> tuple[set[str], set[str]]:
        """Return (succeeded_uids, exhausted_uids) from a previous run.

        ``exhausted`` = failed and already spent its repair budget, so a resume
        must not pay for it again. Anything else is retried from scratch.
        """
        succeeded: set[str] = set()
        rounds: dict[str, int] = {}
        for rec in AttemptLog.read_all(path):
            uid = str(rec.get("uid") or "")
            if not uid:
                continue
            if rec.get("ok"):
                succeeded.add(uid)
            rounds[uid] = max(rounds.get(uid, 0), int(rec.get("round", 0)) + 1)
        exhausted = {
            uid
            for uid, n in rounds.items()
            if uid not in succeeded and n >= max_rounds
        }
        return succeeded, exhausted


# ── frozen seed set ──────────────────────────────────────────────────────────


def _best_per_uid(records: Iterable[dict]) -> dict[str, dict]:
    """Keep, per uid, the successful attempt with the best measured speedup.

    Speedup does not gate acceptance, but when a row succeeded more than once
    (e.g. across resumes) the faster port is the strictly better seed.
    """
    best: dict[str, dict] = {}
    for rec in records:
        if not rec.get("ok"):
            continue
        uid = str(rec.get("uid") or "")
        if not uid:
            continue
        cur = best.get(uid)
        if cur is None or float(rec.get("speedup") or 0.0) > float(
            cur.get("speedup") or 0.0
        ):
            best[uid] = rec
    return best


def freeze_seed_set(attempts_path: Path, out_dir: Path) -> dict[str, Any]:
    """Materialize the accepted ports as the frozen training corpus."""
    out_dir = Path(out_dir)
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    accepted = _best_per_uid(AttemptLog.read_all(attempts_path))
    index: list[dict[str, Any]] = []
    for uid in sorted(accepted):
        rec = accepted[uid]
        code = str(rec.get("code") or "")
        rel = f"kernels/{uid}.py"
        (out_dir / rel).write_text(code, encoding="utf-8")
        index.append(
            {
                "uid": uid,
                "row_index": rec.get("row_index"),
                "ops": rec.get("ops", ""),
                "code_path": rel,
                "code": code,
                "orig_code": str(rec.get("ref_code") or ""),
                "seed_baseline_ms": rec.get("baseline_ms"),
                "seed_triton_ms": rec.get("candidate_ms"),
                "seed_speedup_vs_torch": rec.get("speedup"),
                "rounds_used": int(rec.get("round", 0)) + 1,
                "prompt_version": rec.get("prompt_version", ""),
                "model": rec.get("model", ""),
            }
        )

    index_path = out_dir / "seed_index.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in index:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    parquet_path = out_dir / "seed.parquet"
    parquet_written = _write_parquet(index, parquet_path)

    return {
        "count": len(index),
        "index_path": str(index_path),
        "parquet_path": str(parquet_path) if parquet_written else "",
        "kernels_dir": str(kernels_dir),
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> bool:
    """Best-effort parquet export; jsonl remains the authoritative artifact."""
    if not rows:
        return False
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        pq.write_table(pa.Table.from_pylist(rows), path)
        return True
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_parquet(path)
        return True
    except ImportError:
        return False


# ── report ───────────────────────────────────────────────────────────────────


def build_report(
    attempts_path: Path,
    *,
    total_rows: int,
    rejected: list[dict] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Yield + failure-cause breakdown, the thing that decides go / no-go."""
    records = AttemptLog.read_all(attempts_path)
    accepted = _best_per_uid(records)
    attempted_uids = {str(r.get("uid")) for r in records if r.get("uid")}

    # Failure causes counted on each uid's LAST attempt: that is why the row was
    # finally lost, whereas counting all attempts over-weights rows that were
    # retried the most.
    last_by_uid: dict[str, dict] = {}
    for rec in records:
        uid = str(rec.get("uid") or "")
        if uid:
            last_by_uid[uid] = rec

    stage_counts: dict[str, int] = {}
    error_kinds: dict[str, int] = {}
    for uid, rec in last_by_uid.items():
        if uid in accepted:
            continue
        stage = str(rec.get("stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        kind = _error_kind(str(rec.get("error") or ""))
        error_kinds[kind] = error_kinds.get(kind, 0) + 1

    rounds_used = [int(r.get("round", 0)) + 1 for r in accepted.values()]
    speedups = [
        float(r["speedup"]) for r in accepted.values() if r.get("speedup") is not None
    ]

    return {
        "total_rows_in_corpus": total_rows,
        "rows_rejected_before_llm": len(rejected or []),
        "rows_attempted": len(attempted_uids),
        "rows_accepted": len(accepted),
        "yield_rate": (len(accepted) / len(attempted_uids)) if attempted_uids else 0.0,
        "attempts_total": len(records),
        "attempts_per_accepted": (
            len(records) / len(accepted) if accepted else None
        ),
        "first_round_accept": sum(1 for n in rounds_used if n == 1),
        "repair_rescued": sum(1 for n in rounds_used if n > 1),
        "failure_by_stage": dict(
            sorted(stage_counts.items(), key=lambda kv: -kv[1])
        ),
        "failure_by_kind": dict(sorted(error_kinds.items(), key=lambda kv: -kv[1])),
        "seed_speedup_vs_torch": _summary(speedups),
        **(extra or {}),
    }


_ERROR_KINDS: list[tuple[str, tuple[str, ...]]] = [
    ("triton_api_hallucination", ("has no attribute", "triton.language")),
    ("illegal_memory_access", ("illegal memory access", "out of bounds")),
    ("numerics_mismatch", ("correctness_failed", "outputs differ")),
    ("compile_error", ("compilationerror", "triton.compiler", "compileerror")),
    ("getsource", ("could not get source",)),
    ("constructor_mismatch", ("positional argument", "init failed")),
    ("shape_error", ("shape", "size mismatch", "dimension")),
    ("oom", ("out of memory", "cuda oom")),
    ("timeout", ("timeout",)),
    ("no_modelnew", ("no modelnew", "defines no modelnew")),
    ("local_static_reject", ("no top-level @triton.jit", "syntax error", "forbidden")),
    ("worker_crash", ("worker_crash", "worker exited")),
    ("transport", ("remote exec failed", "cannot read remote results", "no result line")),
]


def _error_kind(error: str) -> str:
    low = error.lower()
    if not low:
        return "unknown"
    for kind, needles in _ERROR_KINDS:
        if any(n in low for n in needles):
            return kind
    return "other"


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    import statistics

    vs = sorted(values)
    return {
        "n": len(vs),
        "min": vs[0],
        "p50": statistics.median(vs),
        "max": vs[-1],
        "mean": statistics.fmean(vs),
        "frac_faster_than_torch": sum(1 for v in vs if v > 1.0) / len(vs),
    }
