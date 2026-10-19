"""Read the raw torch corpus (``data/source/data.parquet``) into seed rows.

The corpus is already KernelBench-shaped: every row's ``code`` defines
``Model`` / ``get_inputs`` / ``get_init_inputs``, so it can be handed to the
existing trial harness unchanged. Only the parquet read needs a backend, and we
accept either pyarrow or pandas so this works in whichever env is available.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from autoprofiler.config import REPO_ROOT

DEFAULT_SOURCE = REPO_ROOT / "data" / "source" / "data.parquet"

_REQUIRED_MARKERS = ("class Model", "get_inputs", "get_init_inputs")


@dataclass(frozen=True)
class SeedRow:
    """One raw torch program awaiting a Triton port."""

    uid: str
    row_index: int
    ops: str
    data_source: str
    code: str


def _read_parquet(path: Path) -> list[dict]:
    """Return the parquet rows as dicts, via pyarrow or pandas."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path).to_dict("records")
    except ImportError as e:
        raise RuntimeError(
            "reading the seed corpus needs pyarrow or pandas; "
            "install one of them into the env running seedgen"
        ) from e


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def load_seed_rows(
    source: str | Path | None = None,
    *,
    limit: int | None = None,
    sample_seed: int | None = None,
    skip_malformed: bool = True,
) -> tuple[list[SeedRow], list[dict]]:
    """Load rows from the raw corpus.

    ``source`` is the parquet path of the raw PyTorch programs.

    Returns ``(rows, rejected)``. A row is rejected before any GPU or LLM work
    when its code cannot possibly be evaluated (missing Model / get_inputs),
    which is far cheaper than discovering it on the pod.
    """
    path = Path(source) if source else DEFAULT_SOURCE
    if not path.exists():
        raise FileNotFoundError(f"seed corpus not found: {path}")

    raw = _read_parquet(path)
    indexed = list(enumerate(raw))
    if limit:
        if sample_seed is None:
            indexed = indexed[:limit]
        else:
            indexed = random.Random(sample_seed).sample(
                indexed, min(limit, len(indexed))
            )
            indexed.sort(key=lambda pair: pair[0])
    rows: list[SeedRow] = []
    rejected: list[dict] = []
    for i, rec in indexed:
        code = _stringify(rec.get("code"))
        uid = f"syn_{i:05d}_{hashlib.sha1(code.encode()).hexdigest()[:8]}"
        missing = [m for m in _REQUIRED_MARKERS if m not in code]
        if missing:
            rejected.append(
                {
                    "uid": uid,
                    "row_index": i,
                    "stage": "malformed_source",
                    "ok": False,
                    "error": f"source missing {missing}",
                }
            )
            if skip_malformed:
                continue
        rows.append(
            SeedRow(
                uid=uid,
                row_index=i,
                ops=_stringify(rec.get("ops")),
                data_source=_stringify(rec.get("data_source")),
                code=code,
            )
        )
    return rows, rejected
