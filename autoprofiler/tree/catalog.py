"""The vocabulary of the profiling tree: bottleneck classes, the root's counters, and the allow-list of valid
Nsight Compute counters.

The allow-list (``configs/train/metric_catalog.yaml``) is what stops an LLM from inventing counter names
when it proposes new metrics for a node.
"""
from __future__ import annotations

from pathlib import Path

# The four roofline bottleneck states (buckets). Each grows its own routing tree.
BOUND_KEYS = ("latency", "memory", "compute", "algorithm")

# Baseline speed-of-light metric names. These two ARE the root node's profiler:
# their high/low 2x2 combination produces the four top-level subtrees.
SM_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
MEM_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

# The root's own metric set. Every descendant must exclude these: ancestors'
# readings are reused from cache, so re-measuring them is pure waste and would
# violate the path-disjointness invariant.
ROOT_METRICS: tuple[str, ...] = (SM_METRIC, MEM_METRIC)

# Seed NCU metric sets per bound. This is the *initial* profiler payload for each
# bucket's root leaf; REWRITE/SPLIT evolve it from here. All names are standard
# NCU section metrics valid on Ampere/Ada.
SEED_METRICS: dict[str, list[str]] = {
    "compute": [
        SM_METRIC,
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
    ],
    "memory": [
        MEM_METRIC,
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    ],
    "latency": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        SM_METRIC,
        MEM_METRIC,
    ],
    "algorithm": [
        SM_METRIC,
        MEM_METRIC,
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
}

# Controlled vocabulary of VALID NCU section counters lives in an external file
# (configs/train/metric_catalog.yaml), pointed to by the training config's `metric_catalog`.
# It is the allow-list that stops an LLM from inventing counter names. The active
# catalog is set by ``load_train_config``; helpers below fall back to loading the
# default file lazily so they work even without an explicit config load.
DEFAULT_METRIC_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "train" / "metric_catalog.yaml"
)

_ACTIVE_CATALOG: dict[str, list[str]] | None = None


def load_metric_catalog(path: str | Path) -> dict[str, list[str]]:
    """Load the NCU counter allow-list from a YAML file.

    Accepts either {catalog: {group: [...]}} or a flat {group: [...]} mapping.
    """
    import yaml

    p = Path(path).expanduser()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cat = raw.get("catalog", raw) if isinstance(raw, dict) else {}
    return {str(k): [str(m) for m in (v or [])] for k, v in cat.items()}


def set_active_catalog(catalog: dict[str, list[str]] | None) -> None:
    global _ACTIVE_CATALOG
    _ACTIVE_CATALOG = catalog


def _current_catalog() -> dict[str, list[str]]:
    if _ACTIVE_CATALOG is not None:
        return _ACTIVE_CATALOG
    try:
        return load_metric_catalog(DEFAULT_METRIC_CATALOG_PATH)
    except Exception:
        return {}


def all_catalog_metrics() -> list[str]:
    """Flat, de-duplicated, sorted list of every valid catalog counter."""
    seen: dict[str, None] = {}
    for group in _current_catalog().values():
        for m in group:
            seen.setdefault(m, None)
    # Seed metrics are authoritative-valid too; make sure none are missed.
    for group in SEED_METRICS.values():
        for m in group:
            seen.setdefault(m, None)
    return sorted(seen)


_CATALOG_GROUP_INFO: dict[str, tuple[str, str]] = {
    "speed_of_light": (
        "Top-level utilization/throughput counter used to identify the saturated GPU resource.",
        "medium replay risk",
    ),
    "compute": (
        "Compute-pipeline utilization counter used to identify which arithmetic pipe is busy.",
        "medium replay risk",
    ),
    "instruction": (
        "Instruction issue/execution counter used to distinguish low issue rate from excess work.",
        "medium replay risk",
    ),
    "memory": (
        "Memory-system traffic, cache, coalescing, or bank-conflict counter.",
        "medium replay risk",
    ),
    "scheduler_warp_state": (
        "Warp scheduler or stall-reason counter used to explain why resident warps cannot issue.",
        "medium replay risk",
    ),
    "occupancy_launch": (
        "Launch geometry or occupancy-limit counter used to diagnose insufficient parallelism.",
        "low replay risk",
    ),
    "timing": (
        "GPU duration counter used to anchor profiling measurements.",
        "low replay risk",
    ),
}

_METRIC_DESCRIPTIONS: dict[str, str] = {
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (
        "Share of active-warp cycles stalled on long-latency scoreboard dependencies, "
        "commonly global/local-memory dependencies."
    ),
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": (
        "Share of active-warp cycles stalled on short-latency scoreboard dependencies."
    ),
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": (
        "Share of active-warp cycles waiting at synchronization barriers."
    ),
    "smsp__warps_eligible.avg.per_cycle_active": (
        "Average number of warps eligible to issue per active scheduler cycle."
    ),
    "launch__waves_per_multiprocessor": (
        "Number of grid waves available per SM; values below one indicate insufficient "
        "grid-level parallelism."
    ),
    "launch__registers_per_thread": "Registers allocated per launched thread.",
    "launch__block_size": "Threads launched per thread block.",
    "launch__grid_size": "Total thread blocks in the launch grid.",
    "lts__t_sector_hit_rate.pct": "L2 cache sector hit rate.",
    "l1tex__t_sector_hit_rate.pct": "L1/TEX cache sector hit rate.",
    "dram__bytes.sum": "Total bytes transferred to and from device DRAM.",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": (
        "Scheduler issue activity as a percentage of peak during active cycles."
    ),
    "sm__warps_active.avg.pct_of_peak_sustained_active": (
        "Resident active warps as a percentage of the architecture limit."
    ),
}


def all_catalog_metric_entries() -> list[dict[str, str]]:
    """Catalog entries enriched for REWRITE reasoning.

    NCU cost is replay-group dependent, so this deliberately reports a coarse
    risk rather than pretending each counter has a stable millisecond price.
    Validation still measures the actual complete-set cost on hardware.
    """
    entries: dict[str, dict[str, str]] = {}
    for group, metrics in _current_catalog().items():
        group_desc, cost = _CATALOG_GROUP_INFO.get(
            group,
            ("NCU hardware performance counter.", "unknown replay risk"),
        )
        for metric in metrics:
            name = str(metric)
            entries.setdefault(
                name,
                {
                    "name": name,
                    "description": _METRIC_DESCRIPTIONS.get(name, group_desc),
                    "cost": cost,
                },
            )
    for name in all_catalog_metrics():
        entries.setdefault(
            name,
            {
                "name": name,
                "description": _METRIC_DESCRIPTIONS.get(
                    name, "NCU hardware performance counter."
                ),
                "cost": "unknown replay risk",
            },
        )
    return [entries[name] for name in sorted(entries)]


def is_valid_metric(name: str) -> bool:
    return name in set(all_catalog_metrics())


def _default_seed_metrics() -> dict[str, list[str]]:
    return {k: list(v) for k, v in SEED_METRICS.items()}


def top_level_seed_metrics(seed_metrics: dict[str, list[str]]) -> dict[str, list[str]]:
    """Seed sets for the four top-level nodes, with the root's metrics removed.

    The configured seed sets were written for standalone bucket roots and so still
    list the SOL metrics. Under the single tree those are measured at the root, so
    keeping them here would double-measure and break path disjointness.
    """
    root = set(ROOT_METRICS)
    return {
        key: [m for m in dict.fromkeys(metrics) if m not in root]
        for key, metrics in seed_metrics.items()
    }
