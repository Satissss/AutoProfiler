"""Kernel states, metric cache and route records: the Markov chain's memory.

Three design rules, all aimed at making inconsistency structurally impossible:

1. The metric cache is scoped to a KERNEL, not to an origin sample. Optimizing a
   kernel produces a brand-new state whose every NCU reading must be re-collected
   from the root. Since ``kernel_id`` is a hash of the code, the cache invalidates
   itself: new code cannot collide with an old key, so there is no invalidation
   logic to get wrong.

2. Only absolute ``latency_ms`` is stored. Every speedup (vs the previous hop, vs
   the origin seed) is derived on demand. Storing a speedup as well would create a
   second source of truth that can drift from the latencies.

3. Trajectories are not stored. Each kernel records ``parent_id`` and the
   trajectory is the derived walk back to the seed. Because a kernel can be routed
   again after the tree changes and thus produce a second child, this naturally
   forms one kernel EVOLUTION TREE per origin sample, which a parent pointer
   already expresses exactly.

Persistence is append-only jsonl. All three indices (OriginIndex / ActivePool /
LeafSlowSet) are in-memory projections rebuilt from the log, which makes resume
and audit fall out for free.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Kernel lifecycle:
#   pending  -> awaiting routing (it is in the active pool)
#   slow     -> was routed, the leaf's optimization missed the threshold; it now
#               accumulates at that leaf as gradient + validation material
#   optimization_failed -> optimizer did not produce a correct executable result;
#                          excluded from LeafSlowSet and every speedup calculation
#   advanced -> optimization cleared the threshold and produced a child
#   terminal -> hit the hop cap or was judged converged
STATUS_PENDING = "pending"
STATUS_SLOW = "slow"
# Reached a leaf after that leaf's proposal set was frozen. It is deliberately
# absent from both ActivePool and LeafSlowSet until the update accepts or rejects.
STATUS_UPDATE_TEST = "update_test"
STATUS_OPTIMIZATION_FAILED = "optimization_failed"
STATUS_ADVANCED = "advanced"
STATUS_TERMINAL = "terminal"


def kernel_id_for(code: str) -> str:
    """Content address of a kernel: sha1 of normalized source, first 16 hex.

    Normalizing whitespace means a reformatted but identical kernel hits the same
    node, so the store dedupes LLM output for free and never re-measures it.
    """
    norm = "\n".join(line.rstrip() for line in code.strip().splitlines())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


@dataclass
class KernelState:
    """One immutable state of the Markov chain."""

    kernel_id: str
    origin_uid: str
    code: str
    parent_id: str | None = None
    hop: int = 0
    latency_ms: float | None = None
    status: str = STATUS_PENDING
    assigned_leaf_id: str = ""
    # {leaf_id, route_id} of the optimization that produced this state.
    created_by: dict[str, str] = field(default_factory=dict)
    # Free-form: correctness verdict, optimizer metadata, error strings.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KernelState":
        return cls(
            kernel_id=str(d["kernel_id"]),
            origin_uid=str(d["origin_uid"]),
            code=str(d.get("code") or ""),
            parent_id=(str(d["parent_id"]) if d.get("parent_id") else None),
            hop=int(d.get("hop", 0)),
            latency_ms=(float(d["latency_ms"]) if d.get("latency_ms") is not None else None),
            status=str(d.get("status", STATUS_PENDING)),
            assigned_leaf_id=str(d.get("assigned_leaf_id", "")),
            created_by=dict(d.get("created_by") or {}),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class RouteRecord:
    """One root-to-leaf forward pass, for audit and for threshold calibration."""

    route_id: str
    kernel_id: str
    origin_uid: str
    node_path: list[str] = field(default_factory=list)
    metrics_by_node: dict[str, dict[str, float]] = field(default_factory=dict)
    leaf_id: str = ""
    profile_ms: float = 0.0
    # child kernel_id on success, else "".
    child_kernel_id: str = ""
    outcome: str = ""
    rel_speedup: float | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RouteRecord":
        error = str(d.get("error") or "")
        outcome = str(d.get("outcome") or "")
        # Older runs mislabeled LLM/verification infrastructure failures as
        # compile_fail. Normalize the in-memory projection so resume can repair
        # affected kernel states without rewriting the append-only audit log.
        error_lower = error.lower()
        if outcome == "compile_fail" and (
            "llm timeout" in error_lower
            or "llm http" in error_lower
            or "llm optimization failed" in error_lower
            or "verification service" in error_lower
            or "no verification result" in error_lower
        ):
            outcome = "service_error"
        return cls(
            route_id=str(d.get("route_id") or ""),
            kernel_id=str(d.get("kernel_id") or ""),
            origin_uid=str(d.get("origin_uid") or ""),
            node_path=list(d.get("node_path") or []),
            metrics_by_node={
                str(k): {str(m): float(v) for m, v in (vals or {}).items()}
                for k, vals in (d.get("metrics_by_node") or {}).items()
            },
            leaf_id=str(d.get("leaf_id") or ""),
            profile_ms=float(d.get("profile_ms") or 0.0),
            child_kernel_id=str(d.get("child_kernel_id") or ""),
            outcome=outcome,
            rel_speedup=(
                float(d["rel_speedup"]) if d.get("rel_speedup") is not None else None
            ),
            error=error,
        )

    @property
    def succeeded(self) -> bool:
        return bool(self.child_kernel_id)


class KernelStore:
    """Append-only kernel/metric/route log plus its in-memory projections."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.kernels_path = self.root / "kernels.jsonl"
        self.routes_path = self.root / "routes.jsonl"
        self.metrics_path = self.root / "metrics.jsonl"

        self.kernels: dict[str, KernelState] = {}
        self.routes: list[RouteRecord] = []
        # (kernel_id, metric) -> value, plus per-(kernel, set) measured cost.
        self._metrics: dict[str, dict[str, float]] = {}
        self._profile_ms: dict[tuple[str, str], float] = {}
        self._write_lock = threading.RLock()

    # ── load / persist ────────────────────────────────────────────
    def load(self) -> "KernelStore":
        """Rebuild every projection from the append-only logs.

        Later kernel records overwrite earlier ones for the same id, so the log
        doubles as an update journal without needing rewrites in place.
        """
        for rec in _read_jsonl(self.kernels_path):
            ks = KernelState.from_dict(rec)
            self.kernels[ks.kernel_id] = ks
        for rec in _read_jsonl(self.routes_path):
            self.routes.append(RouteRecord.from_dict(rec))
        for rec in _read_jsonl(self.metrics_path):
            kid = str(rec.get("kernel_id") or "")
            if not kid:
                continue
            vals = {str(k): float(v) for k, v in (rec.get("metrics") or {}).items()}
            self._metrics.setdefault(kid, {}).update(vals)
            sig = _signature(vals.keys())
            if rec.get("profile_ms") is not None:
                self._profile_ms[(kid, sig)] = float(rec["profile_ms"])
        return self

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        with self._write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def checkpoint_manifest(self) -> dict[str, Any]:
        """Return an immutable boundary in the append-only journals.

        The byte offsets are sufficient to reconstruct this exact store state
        even after later training appends more records to the same files.
        """
        with self._write_lock:
            return {
                "format": 1,
                "root": str(self.root.resolve()),
                "files": {
                    path.name: path.stat().st_size if path.exists() else 0
                    for path in (
                        self.kernels_path,
                        self.routes_path,
                        self.metrics_path,
                    )
                },
            }

    @classmethod
    def archive_checkpoint(
        cls, manifest: dict[str, Any], archive_root: str | Path
    ) -> dict[str, Any]:
        """Append a journal boundary into a run-owned checkpoint archive."""
        source_root = Path(str(manifest.get("root") or ""))
        files = dict(manifest.get("files") or {})
        archive_root = Path(archive_root)
        archive_root.mkdir(parents=True, exist_ok=True)
        archived: dict[str, int] = {}
        for name in ("kernels.jsonl", "routes.jsonl", "metrics.jsonl"):
            target_size = int(files.get(name, 0))
            source = source_root / name
            target = archive_root / name
            current_size = target.stat().st_size if target.exists() else 0
            actual_size = source.stat().st_size if source.exists() else 0
            if target_size < current_size:
                raise RuntimeError(
                    f"checkpoint archive is ahead of source boundary for {name}: "
                    f"{current_size} > {target_size}"
                )
            if actual_size < target_size:
                raise RuntimeError(
                    f"checkpoint source was truncated: {source} "
                    f"needs {target_size} bytes, has {actual_size}"
                )
            if target_size > current_size:
                with source.open("rb") as src, target.open("ab") as dst:
                    src.seek(current_size)
                    remaining = target_size - current_size
                    while remaining:
                        chunk = src.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError(
                                f"checkpoint source ended early: {source}"
                            )
                        dst.write(chunk)
                        remaining -= len(chunk)
                    dst.flush()
                    os.fsync(dst.fileno())
            elif not target.exists():
                target.touch()
            archived[name] = target_size
        return {
            "format": 1,
            "root": str(archive_root.resolve()),
            "files": archived,
        }

    @classmethod
    def fork_checkpoint(
        cls, manifest: dict[str, Any], destination: str | Path
    ) -> "KernelStore":
        """Materialize a checkpoint journal prefix into an independent store."""
        source_root = Path(str(manifest.get("root") or ""))
        files = dict(manifest.get("files") or {})
        destination = Path(destination)
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(
                f"checkpoint destination is not empty: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("kernels.jsonl", "routes.jsonl", "metrics.jsonl"):
            size = int(files.get(name, 0))
            source = source_root / name
            target = destination / name
            if size < 0:
                raise RuntimeError(f"invalid checkpoint size for {name}: {size}")
            actual = source.stat().st_size if source.exists() else 0
            if actual < size:
                raise RuntimeError(
                    f"checkpoint source was truncated: {source} "
                    f"needs {size} bytes, has {actual}"
                )
            with source.open("rb") as src, target.open("wb") as dst:
                remaining = size
                while remaining:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(
                            f"checkpoint source ended early: {source}"
                        )
                    dst.write(chunk)
                    remaining -= len(chunk)
                dst.flush()
                os.fsync(dst.fileno())
        return cls(destination).load()

    # ── kernels ───────────────────────────────────────────────────
    def add_kernel(self, ks: KernelState) -> KernelState:
        """Insert a state, or return the existing one on a content-hash hit."""
        with self._write_lock:
            existing = self.kernels.get(ks.kernel_id)
            if existing is not None:
                return existing
            self.kernels[ks.kernel_id] = ks
            self._append(self.kernels_path, ks.to_dict())
            return ks

    def add_seed(self, origin_uid: str, code: str, latency_ms: float | None = None) -> KernelState:
        return self.add_kernel(
            KernelState(
                kernel_id=kernel_id_for(code),
                origin_uid=origin_uid,
                code=code,
                parent_id=None,
                hop=0,
                latency_ms=latency_ms,
                status=STATUS_PENDING,
            )
        )

    def add_child(
        self,
        parent: KernelState,
        code: str,
        latency_ms: float | None,
        *,
        leaf_id: str,
        route_id: str,
        meta: dict[str, Any] | None = None,
    ) -> KernelState:
        return self.add_kernel(
            KernelState(
                kernel_id=kernel_id_for(code),
                origin_uid=parent.origin_uid,
                code=code,
                parent_id=parent.kernel_id,
                hop=parent.hop + 1,
                latency_ms=latency_ms,
                status=STATUS_PENDING,
                created_by={"leaf_id": leaf_id, "route_id": route_id},
                meta=dict(meta or {}),
            )
        )

    def update_kernel(self, kernel_id: str, **changes: Any) -> KernelState:
        with self._write_lock:
            ks = self.kernels[kernel_id]
            for k, v in changes.items():
                setattr(ks, k, v)
            self._append(self.kernels_path, ks.to_dict())
            return ks

    def get(self, kernel_id: str) -> KernelState | None:
        return self.kernels.get(kernel_id)

    # ── metric cache ──────────────────────────────────────────────
    def cached_metrics(self, kernel_id: str) -> dict[str, float]:
        with self._write_lock:
            return dict(self._metrics.get(kernel_id, {}))

    def missing_metrics(self, kernel_id: str, metrics: Iterable[str]) -> list[str]:
        with self._write_lock:
            have = dict(self._metrics.get(kernel_id, {}))
            return [m for m in metrics if m not in have]

    def put_metrics(
        self, kernel_id: str, values: dict[str, float], *, profile_ms: float = 0.0
    ) -> None:
        if not values:
            return
        with self._write_lock:
            self._metrics.setdefault(kernel_id, {}).update(
                {str(k): float(v) for k, v in values.items()}
            )
            sig = _signature(values.keys())
            self._profile_ms[(kernel_id, sig)] = float(profile_ms)
            self._append(
                self.metrics_path,
                {
                    "kernel_id": kernel_id,
                    "metrics": {str(k): float(v) for k, v in values.items()},
                    "profile_ms": float(profile_ms),
                },
            )

    def profile_cost_ms(self, kernel_id: str, metrics: Iterable[str]) -> float | None:
        """Measured collection time for exactly this metric set, if we have it."""
        with self._write_lock:
            return self._profile_ms.get((kernel_id, _signature(metrics)))

    def mean_profile_cost_ms(self, metrics: Iterable[str]) -> float | None:
        """Mean measured cost of this metric set across every kernel that ran it.

        Used when pricing a candidate set whose cost on a specific kernel is not
        yet known. Set-level by construction, never a per-metric sum.
        """
        sig = _signature(metrics)
        with self._write_lock:
            vals = [v for (_, s), v in self._profile_ms.items() if s == sig and v > 0]
        return (sum(vals) / len(vals)) if vals else None

    # ── routes ────────────────────────────────────────────────────
    def add_route(self, rec: RouteRecord) -> RouteRecord:
        with self._write_lock:
            self.routes.append(rec)
            self._append(self.routes_path, rec.to_dict())
            return rec

    def routes_for_leaf(self, leaf_id: str) -> list[RouteRecord]:
        return [r for r in self.routes if r.leaf_id == leaf_id]

    def leaf_wins(self, leaf_id: str) -> list[RouteRecord]:
        """Successful optimizations at this leaf.

        One query serves two purposes: its LENGTH reflects the leaf's historical
        value (``history_wins``) and its ``kernel_id`` fields are the
        regression-guard pool -- those ids are the states as they were BEFORE the
        leaf optimized them, which is exactly the decision we need to replay.
        """
        return [r for r in self.routes_for_leaf(leaf_id) if r.succeeded]

    # ── derived quantities (never stored) ─────────────────────────
    def rel_speedup(self, kernel_id: str) -> float | None:
        ks = self.kernels.get(kernel_id)
        if ks is None or ks.parent_id is None or not ks.latency_ms:
            return None
        parent = self.kernels.get(ks.parent_id)
        if parent is None or not parent.latency_ms:
            return None
        return parent.latency_ms / ks.latency_ms

    def abs_speedup(self, kernel_id: str) -> float | None:
        ks = self.kernels.get(kernel_id)
        if ks is None or not ks.latency_ms:
            return None
        seed = self.seed_of(ks.origin_uid)
        if seed is None or not seed.latency_ms:
            return None
        return seed.latency_ms / ks.latency_ms

    def trajectory(self, kernel_id: str) -> list[KernelState]:
        """Seed-to-kernel chain, derived by walking parent pointers."""
        chain: list[KernelState] = []
        cur = self.kernels.get(kernel_id)
        seen: set[str] = set()
        while cur is not None and cur.kernel_id not in seen:
            chain.append(cur)
            seen.add(cur.kernel_id)
            cur = self.kernels.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def seed_of(self, origin_uid: str) -> KernelState | None:
        for ks in self.kernels.values():
            if ks.origin_uid == origin_uid and ks.hop == 0:
                return ks
        return None

    # ── indices (projections) ─────────────────────────────────────
    def origin_index(self) -> dict[str, dict[str, Any]]:
        """origin_uid -> {seed_kernel_id, all_kernel_ids, best_kernel_id}.

        ``best`` is the lowest-latency state for that origin, i.e. "the best
        version of this sample so far".
        """
        out: dict[str, dict[str, Any]] = {}
        for ks in self.kernels.values():
            entry = out.setdefault(
                ks.origin_uid,
                {"seed_kernel_id": "", "all_kernel_ids": [], "best_kernel_id": ""},
            )
            entry["all_kernel_ids"].append(ks.kernel_id)
            if ks.hop == 0:
                entry["seed_kernel_id"] = ks.kernel_id
        for uid, entry in out.items():
            timed = [
                self.kernels[kid]
                for kid in entry["all_kernel_ids"]
                if self.kernels[kid].latency_ms
            ]
            if timed:
                entry["best_kernel_id"] = min(
                    timed, key=lambda k: float(k.latency_ms or 1e18)
                ).kernel_id
            entry["all_kernel_ids"].sort()
        return out

    def active_pool(self) -> list[str]:
        return sorted(
            kid for kid, ks in self.kernels.items() if ks.status == STATUS_PENDING
        )

    def leaf_slow_set(self) -> dict[str, list[str]]:
        """leaf_id -> proposal-sample kernel ids.

        This is the activation counter and becomes the frozen proposal set when
        an update starts. Later arrivals live in ``leaf_update_test_set``.

        Includes BOTH slow (correct-but-not-fast-enough) and advanced (already
        sped up) kernels tagged to the leaf: the metric-selection action is asked
        to improve the whole population the leaf serves, not only the ones the
        current metrics failed on, so a win can still be pushed faster by a better
        metric set. The accept gate (paired margin >= epsilon) is unchanged, so a
        win contributes to the margin only if the candidate metrics genuinely make
        it faster; otherwise it nets ~0. This only speeds up activation and widens
        the evidence, it does not relax acceptance.
        """
        out: dict[str, list[str]] = {}
        for kid, ks in self.kernels.items():
            if ks.status in (STATUS_SLOW, STATUS_ADVANCED) and ks.assigned_leaf_id:
                out.setdefault(ks.assigned_leaf_id, []).append(kid)
        for ids in out.values():
            ids.sort()
        return out

    def leaf_update_test_set(self) -> dict[str, list[str]]:
        """Samples held out because they arrived while their leaf was updating."""
        out: dict[str, list[str]] = {}
        for kid, ks in self.kernels.items():
            if ks.status == STATUS_UPDATE_TEST and ks.assigned_leaf_id:
                out.setdefault(ks.assigned_leaf_id, []).append(kid)
        for ids in out.values():
            ids.sort()
        return out

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for ks in self.kernels.values():
            by_status[ks.status] = by_status.get(ks.status, 0) + 1
        hops = [ks.hop for ks in self.kernels.values()]
        return {
            "n_kernels": len(self.kernels),
            "n_origins": len({ks.origin_uid for ks in self.kernels.values()}),
            "n_routes": len(self.routes),
            "by_status": by_status,
            "max_hop": max(hops) if hops else 0,
        }


def _signature(metrics: Iterable[str]) -> str:
    return ",".join(sorted(set(str(m) for m in metrics)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


__all__ = [
    "KernelState",
    "RouteRecord",
    "KernelStore",
    "kernel_id_for",
    "STATUS_PENDING",
    "STATUS_SLOW",
    "STATUS_UPDATE_TEST",
    "STATUS_OPTIMIZATION_FAILED",
    "STATUS_ADVANCED",
    "STATUS_TERMINAL",
]
