"""The training corpus: Triton programs paired with the PyTorch reference they must match."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoprofiler.config import ExecConfig
from autoprofiler.train.config import TrainConfig


@dataclass
class Sample:
    """One corpus row: a Triton seed kernel plus its original torch reference."""

    uid: str
    level: int
    problem_id: int
    source: str
    # The ORIGINAL torch program. Correctness of every kernel derived from this
    # sample is judged against this, never against the immediately preceding hop,
    # so 1e-3 tolerances cannot compound along a long chain.
    ref_code: str = ""
    latency_ms: float | None = None
    # Runtime of the original torch program, measured once by the seed pipeline.
    # Only used for end-to-end reporting; the tree is scored against latency_ms.
    torch_latency_ms: float | None = None

    def reference(self) -> str:
        return self.ref_code or self.source

    def program(self) -> dict[str, Any]:
        scope = "kernel" if self.level <= 1 else "graph"
        return {
            "uid": self.uid,
            "level": self.level,
            "problem_id": self.problem_id,
            "scope_hint": scope,
            # Carried so the remote side never resolves source from a KernelBench
            # coordinate: seed-corpus rows have none.
            "source": self.source,
        }


def _seed_level(uid: str) -> int:
    """Recover a KernelBench level from a seed uid, else 0 for the synthetic corpus.

    The same porting pipeline builds both corpora, so the frozen index can hold
    either. Level 0 marks "not from KernelBench", which keeps the held-out set
    identifiable in any mixed pool.
    """
    m = re.match(r"^kb_l(\d+)_", uid)
    return int(m.group(1)) if m else 0


def load_triton_seed_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    uids: set[str] | None = None,
) -> list[Sample]:
    """Load a frozen Triton seed set built by ``scripts/build_triton_seed.sh``.

    Accepts the seed dir or its ``seed_index.jsonl``. These rows carry their own
    source, so training never needs the original corpus files again.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "seed_index.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Triton seed index not found: {p}. Run scripts/build_triton_seed.sh first."
        )
    out: list[Sample] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        uid = str(rec["uid"])
        if uids is not None and uid not in uids:
            continue
        code = str(rec.get("code") or "")
        if not code and rec.get("code_path"):
            code = (p.parent / str(rec["code_path"])).read_text(encoding="utf-8")
        if not code:
            continue
        seed_ms = rec.get("seed_triton_ms", rec.get("candidate_ms"))
        out.append(
            Sample(
                uid=uid,
                level=_seed_level(str(rec["uid"])),
                problem_id=int(rec.get("row_index") or 0),
                source=code,
                ref_code=str(rec.get("orig_code") or ""),
                # Measured during seed generation; reused so we never re-time 6k
                # kernels just to establish hop-0 latency.
                latency_ms=float(seed_ms) if seed_ms else None,
                torch_latency_ms=(
                    float(rec["seed_baseline_ms"]) if rec.get("seed_baseline_ms") else None
                ),
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def load_uid_manifest(path: str | Path) -> set[str]:
    """Load an explicit list of Triton seed ``uid`` values for curriculum runs."""
    import yaml

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"uid manifest not found: {p}")
    if p.suffix.lower() in {".txt", ".lst"}:
        return {
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    uids = raw.get("uids") if isinstance(raw, dict) else raw
    if not isinstance(uids, list) or not uids:
        raise RuntimeError(f"uid manifest {p} must contain a non-empty uids list")
    return {str(u) for u in uids}


def load_datasets(
    cfg: ExecConfig,
    datasets: list[dict],
    *,
    limit: int | None = None,
    uids: set[str] | None = None,
) -> list[Sample]:
    """Load the training corpus spec (a list of ``{id: triton_seed, path: <dir>}``).

    KernelBench is the held-out TEST set and is deliberately not loadable here.
    """
    out: list[Sample] = []
    for d in datasets:
        did = str(d.get("id", "")).lower()
        if did not in ("triton_seed", "seed", "triton"):
            raise NotImplementedError(
                f"training dataset id {did!r} is not supported; use "
                "{id: triton_seed, path: <dir with seed_index.jsonl>} "
                "(KernelBench is test-only and must not be trained on)"
            )
        if not d.get("path"):
            raise ValueError("triton_seed dataset needs a `path`")
        out.extend(load_triton_seed_samples(d["path"], limit=limit, uids=uids))
    return out


def split_holdout(
    samples: list[Sample], train_cfg: TrainConfig
) -> tuple[list[Sample], list[Sample]]:
    if not train_cfg.data.holdout_enabled or len(samples) < 4:
        return samples, []
    rng = random.Random(train_cfg.data.holdout_seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_hold = max(1, int(len(samples) * train_cfg.data.holdout_ratio))
    hold = {idx[i] for i in range(n_hold)}
    return (
        [s for i, s in enumerate(samples) if i not in hold],
        [s for i, s in enumerate(samples) if i in hold],
    )
