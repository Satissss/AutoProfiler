"""The profiler tree: one tree, a profiler on every node, threshold routing.

Structure (this replaces the four independent per-bucket trees):

    root                      profiler = the 2 roofline SOL metrics
     |-- 2x2 high/low  ->     4 top-level nodes (latency/memory/compute/algorithm)
                                each with its own metric set, then splitting further

Three invariants hold everywhere and everything else depends on them:

1. EVERY node carries a Profiler that must be executed, not just leaves. A forward
   pass is therefore "collect -> route -> collect": each node measures its own
   metrics purely to decide the next hop, and only the leaf optimizes the kernel.

2. A node's metric set is DISJOINT from all of its ancestors'. The kernel is not
   modified while walking down, so an ancestor's readings are still valid and are
   reused from cache. Re-measuring them would be pure cost.

3. Routing is a deterministic function of already-measured metrics. A RouterRule
   names a subset S of the node's OWN measured metrics (|S| <= 2) plus one
   threshold each, and the high/low bits form a branch index into 2^|S| children.
   No LLM participates in online routing, so routing is cacheable and auditable.

Also load-bearing: an internal node's metric set is FROZEN at the moment a split
creates it. REWRITE only ever edits a leaf's own set. Once a node's metrics carry
routing decisions, editing them would silently invalidate every branch condition
beneath it.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

from autoprofiler.tree.catalog import BOUND_KEYS, MEM_METRIC, ROOT_METRICS, SM_METRIC


@dataclass
class Profiler:
    """A node's payload: the NCU metrics IT measures (excluding ancestors').

    For a leaf this is the only thing the update operator may change in place. The
    interpreter is a frozen global prompt and is deliberately not stored here.
    """

    ncu_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ncu_metrics": list(self.ncu_metrics)}

    @classmethod
    def from_dict(cls, d: dict) -> "Profiler":
        return cls(ncu_metrics=list(d.get("ncu_metrics", [])))

    def metric_signature(self) -> str:
        return ",".join(sorted(self.ncu_metrics))


@dataclass
class RouterRule:
    """A multi-metric high/low decision card producing ``2**len(subset)`` branches.

    ``subset`` must be a subset of the node's OWN measured metrics. Each metric
    contributes one bit (1 = value >= threshold, "high"), and the bits are packed
    most-significant-first in ``subset`` order to give the branch index. A binary
    split is just ``len(subset) == 1``.
    """

    subset: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    # Provenance, for auditing and reporting only; never used for routing.
    kind: str = "threshold"
    gain: float = 0.0
    note: str = ""

    @property
    def n_branches(self) -> int:
        return 1 << len(self.subset)

    def branch_index(self, metrics: dict[str, float]) -> int:
        """Branch index for a sample's measured metrics.

        A metric missing from the reading counts as LOW. Missing values mean the
        NCU pass failed for that counter; treating that as low keeps routing total
        (never raises, never drops a sample) at the price of a biased hop, which
        is strictly better than losing the sample.
        """
        idx = 0
        for m in self.subset:
            idx <<= 1
            if float(metrics.get(m, float("-inf"))) >= float(self.thresholds.get(m, 0.0)):
                idx |= 1
        return idx

    def branch_label(self, index: int) -> str:
        """Human-readable branch description, e.g. "sm=high,dram=low"."""
        bits = []
        for pos, m in enumerate(self.subset):
            high = bool((index >> (len(self.subset) - 1 - pos)) & 1)
            short = m.split(".")[0].split("__")[0]
            bits.append(f"{short}={'high' if high else 'low'}")
        return ",".join(bits)

    def to_dict(self) -> dict:
        return {
            "subset": list(self.subset),
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "kind": self.kind,
            "gain": float(self.gain),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RouterRule":
        return cls(
            subset=list(d.get("subset", [])),
            thresholds={str(k): float(v) for k, v in (d.get("thresholds") or {}).items()},
            kind=str(d.get("kind", "threshold")),
            gain=float(d.get("gain", 0.0) or 0.0),
            note=str(d.get("note", "")),
        )


def root_router(sm_hi: float = 60.0, mem_hi: float = 60.0) -> RouterRule:
    """The fixed root card: the roofline 2x2 over compute vs memory throughput.

    Thresholds come from ``BucketizeConfig.*_hi`` so the four top-level nodes
    coincide with the roofline buckets the corpus statistics were built on.
    """
    return RouterRule(
        subset=[SM_METRIC, MEM_METRIC],
        thresholds={SM_METRIC: float(sm_hi), MEM_METRIC: float(mem_hi)},
        kind="roofline_root",
        note="fixed 2x2 roofline root: compute throughput x memory throughput",
    )


# Branch index (sm_high, mem_high) -> bound key. Index bits are packed
# most-significant-first over [SM, MEM], so index = 2*sm_high + mem_high.
ROOT_BRANCH_BOUNDS: tuple[str, ...] = (
    "latency",    # 0b00 both low  -> nothing saturated: latency / launch bound
    "memory",     # 0b01 mem high  -> memory bound
    "compute",    # 0b10 sm high   -> compute bound
    "algorithm",  # 0b11 both high -> balanced/saturated: algorithmic change needed
)


@dataclass
class TreeNode:
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # Every node has one. Non-null even for internal nodes (invariant 1).
    profiler: Profiler = field(default_factory=Profiler)
    router: RouterRule | None = None
    children: list["TreeNode"] = field(default_factory=list)
    # Which bound subtree this node belongs to; "" for the root.
    bound: str = ""
    # Training bookkeeping.
    stall_counter: int = 0
    version: int = 0
    evolution_depth: int = 0
    split_attempts: int = 0
    converged: bool = False
    # Activation-sample count at the last rejected update. The leaf may re-propose
    # only once its sample count grows by activation.replenish_count beyond this,
    # so a rejected proposal is retried on fresh evidence, not the identical set.
    last_proposal_size: int = 0
    # Count of kernels this leaf has actually sped up past the threshold. Retained
    # for logging/audit only (no longer gates an operator choice), persisted with
    # the tree rather than recomputed from logs.
    history_wins: int = 0
    # Set when a SPLIT could not separate the outcome groups; fed to the next
    # REWRITE so it knows to add discriminating metrics.
    unseparable_hint: dict[str, Any] | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    # ── traversal ─────────────────────────────────────────────────
    def leaves(self) -> list["TreeNode"]:
        if self.is_leaf:
            return [self]
        out: list[TreeNode] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def walk(self) -> Iterator["TreeNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, node_id: str) -> "TreeNode | None":
        for node in self.walk():
            if node.node_id == node_id:
                return node
        return None

    def path_to(self, node_id: str) -> list["TreeNode"] | None:
        """Root-to-node path, inclusive. None when the node is not in this subtree."""
        if self.node_id == node_id:
            return [self]
        for c in self.children:
            sub = c.path_to(node_id)
            if sub is not None:
                return [self] + sub
        return None

    def leaf_count(self) -> int:
        return len(self.leaves())

    def depth(self) -> int:
        return 0 if self.is_leaf else 1 + max(c.depth() for c in self.children)

    # ── mutation ──────────────────────────────────────────────────
    def split(
        self, router: RouterRule, branch_profilers: list[Profiler],
    ) -> list["TreeNode"]:
        """Turn this leaf into a decision node with ``router.n_branches`` children.

        Every branch index MUST get a node. A missing branch would be a routing
        hole: a sample whose metrics select it would have nowhere to go. A branch
        that gained nothing does not vanish -- it inherits this node's incumbent
        metric set (the caller passes it in), so the tree stays a total partition.
        """
        if not self.is_leaf:
            raise ValueError("cannot split a non-leaf node")
        if len(branch_profilers) != router.n_branches:
            raise ValueError(
                f"router needs {router.n_branches} branch profilers, "
                f"got {len(branch_profilers)}"
            )
        self.router = router
        self.children = [
            TreeNode(
                profiler=Profiler(list(p.ncu_metrics)),
                bound=self.bound,
                evolution_depth=self.evolution_depth + 1,
            )
            for p in branch_profilers
        ]
        self.stall_counter = 0
        self.split_attempts = 0
        self.unseparable_hint = None
        return self.children

    # ── serialization ─────────────────────────────────────────────
    def route_spec(self) -> dict[str, Any]:
        if self.is_leaf:
            return {
                "type": "leaf",
                "leaf_id": self.node_id,
                "bound": self.bound,
                "metrics": list(self.profiler.ncu_metrics),
            }
        r = self.router or RouterRule()
        return {
            "type": "decision",
            "node_id": self.node_id,
            "bound": self.bound,
            "metrics": list(self.profiler.ncu_metrics),
            "subset": list(r.subset),
            "thresholds": dict(r.thresholds),
            "branches": [
                {"index": i, "label": r.branch_label(i), "child": c.route_spec()}
                for i, c in enumerate(self.children)
            ],
        }

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "profiler": self.profiler.to_dict(),
            "bound": self.bound,
            "stall_counter": self.stall_counter,
            "version": self.version,
            "evolution_depth": self.evolution_depth,
            "split_attempts": self.split_attempts,
            "converged": self.converged,
            "last_proposal_size": self.last_proposal_size,
            "history_wins": self.history_wins,
        }
        if self.router is not None:
            d["router"] = self.router.to_dict()
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.unseparable_hint is not None:
            d["unseparable_hint"] = self.unseparable_hint
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        node = cls(
            node_id=d.get("node_id", uuid.uuid4().hex[:8]),
            profiler=Profiler.from_dict(d.get("profiler") or {}),
            bound=str(d.get("bound", "")),
            stall_counter=int(d.get("stall_counter", 0)),
            version=int(d.get("version", 0)),
            evolution_depth=int(d.get("evolution_depth", 0)),
            split_attempts=int(d.get("split_attempts", 0)),
            converged=bool(d.get("converged", False)),
            last_proposal_size=int(d.get("last_proposal_size", 0)),
            history_wins=int(d.get("history_wins", 0)),
        )
        if d.get("router") is not None:
            node.router = RouterRule.from_dict(d["router"])
        node.children = [TreeNode.from_dict(c) for c in (d.get("children") or [])]
        if d.get("unseparable_hint") is not None:
            node.unseparable_hint = dict(d["unseparable_hint"])
        return node


class MetricTree:
    """The single profiler tree, with its fixed roofline root."""

    def __init__(self, root: TreeNode) -> None:
        self.root = root

    # ── lookup ────────────────────────────────────────────────────
    def leaves(self) -> list[TreeNode]:
        return self.root.leaves()

    def node_by_id(self, node_id: str) -> TreeNode | None:
        return self.root.find(node_id)

    def path_to(self, node_id: str) -> list[TreeNode]:
        path = self.root.path_to(node_id)
        if path is None:
            raise KeyError(f"node not in tree: {node_id}")
        return path

    def path_metrics(self, node_id: str) -> list[str]:
        """Union of the metrics measured along the root->node path, in order.

        This is exactly the evidence a leaf hands to the optimizer.
        """
        out: list[str] = []
        for node in self.path_to(node_id):
            for m in node.profiler.ncu_metrics:
                if m not in out:
                    out.append(m)
        return out

    def ancestor_metrics(self, node_id: str) -> list[str]:
        """Metrics fixed by ancestors: the part of a leaf's evidence REWRITE cannot touch."""
        path = self.path_to(node_id)
        out: list[str] = []
        for node in path[:-1]:
            for m in node.profiler.ncu_metrics:
                if m not in out:
                    out.append(m)
        return out

    def top_level_nodes(self) -> dict[str, TreeNode]:
        """The four bound subtree roots, keyed by bound."""
        return {c.bound: c for c in self.root.children}

    def subtree_root(self, bound: str) -> TreeNode | None:
        return self.top_level_nodes().get(bound)

    def leaves_of(self, bound: str) -> list[TreeNode]:
        node = self.subtree_root(bound)
        return node.leaves() if node is not None else []

    def total_leaf_count(self) -> int:
        return self.root.leaf_count()

    # ── invariants ────────────────────────────────────────────────
    def check_invariants(self) -> list[str]:
        """Return a list of invariant violations; empty means the tree is sound.

        Called after every accepted update: a violation here silently corrupts
        every downstream routing decision, so it must fail loudly and early.
        """
        problems: list[str] = []

        def visit(node: TreeNode, ancestors: set[str]) -> None:
            own = list(node.profiler.ncu_metrics)
            overlap = sorted(set(own) & ancestors)
            if overlap:
                problems.append(
                    f"node {node.node_id} re-measures ancestor metrics {overlap}"
                )
            if len(set(own)) != len(own):
                problems.append(f"node {node.node_id} has duplicate metrics")
            if not node.is_leaf:
                r = node.router
                if r is None:
                    problems.append(f"internal node {node.node_id} has no router")
                    return
                if not r.subset:
                    problems.append(f"node {node.node_id} routes on an empty subset")
                # Routing reads from the path cache: ancestors' counters are already
                # measured and available here without re-profiling on this node.
                available = ancestors | set(own)
                missing = [m for m in r.subset if m not in available]
                if missing:
                    problems.append(
                        f"node {node.node_id} routes on {missing}, which it does not measure"
                    )
                if len(node.children) != r.n_branches:
                    problems.append(
                        f"node {node.node_id} has {len(node.children)} children "
                        f"but the router defines {r.n_branches} branches"
                    )
            nxt = ancestors | set(own)
            for c in node.children:
                visit(c, nxt)

        visit(self.root, set())
        return problems

    def assert_invariants(self) -> None:
        problems = self.check_invariants()
        if problems:
            raise RuntimeError("profiler tree invariants violated: " + "; ".join(problems))

    # ── serialization ─────────────────────────────────────────────
    def route_spec(self) -> dict[str, Any]:
        return self.root.route_spec()

    def to_dict(self) -> dict:
        return {"version": 2, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "MetricTree":
        return cls(root=TreeNode.from_dict(d["root"]))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "MetricTree":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def seed_tree(
    seed_metrics: dict[str, list[str]],
    *,
    sm_hi: float = 60.0,
    mem_hi: float = 60.0,
) -> MetricTree:
    """Build the initial tree: fixed roofline root + four top-level leaves.

    The top-level seed sets have the root's SOL metrics stripped out, so the
    disjointness invariant holds from the very first forward pass.
    """
    from autoprofiler.tree.catalog import top_level_seed_metrics

    trimmed = top_level_seed_metrics(seed_metrics)
    root = TreeNode(
        node_id="root",
        profiler=Profiler(ncu_metrics=list(ROOT_METRICS)),
        router=root_router(sm_hi, mem_hi),
        bound="",
    )
    for index in range(4):
        bound = ROOT_BRANCH_BOUNDS[index]
        root.children.append(
            TreeNode(
                profiler=Profiler(ncu_metrics=list(trimmed.get(bound, []))),
                bound=bound,
                evolution_depth=0,
            )
        )
    tree = MetricTree(root=root)
    tree.assert_invariants()
    return tree


def enumerate_subsets(metrics: list[str], max_size: int) -> list[tuple[str, ...]]:
    """All non-empty metric subsets up to ``max_size``, for branch-condition search."""
    out: list[tuple[str, ...]] = []
    for size in range(1, max(1, int(max_size)) + 1):
        out.extend(combinations(metrics, size))
    return out


__all__ = [
    "Profiler",
    "RouterRule",
    "TreeNode",
    "MetricTree",
    "seed_tree",
    "root_router",
    "enumerate_subsets",
    "ROOT_BRANCH_BOUNDS",
    "BOUND_KEYS",
]
