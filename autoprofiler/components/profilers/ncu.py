from __future__ import annotations

import csv
import hashlib
import io
import json

import autoprofiler.backends.ncu as _ncu_mod
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Profiler
from autoprofiler.core.types import Candidate, Profile, Task


# Compact speed-of-light view (time, SM/DRAM/tensor/cache throughput, launch shape).
ROOFLINE_METRICS = (
    "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_active", "launch__registers_per_thread",
    "launch__grid_size", "launch__block_size",
)


def ncu_block(csv_text: str, metrics: list[str], digits: int = 3) -> str:
    """Per-kernel metric table as JSON (``{kernel name: {metric: value}}``), last launch per kernel."""
    lines = csv_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"') or ln.startswith("ID,")), None)
    if start is None:
        return ""
    out: dict[str, dict[str, float | str]] = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        kernel = row.get("Kernel Name") or row.get("launch__kernel_name") or "kernel"
        vals: dict[str, float | str] = {}
        for m in metrics:
            v = (row.get(m) or "").replace(",", "")
            if v not in ("", "n/a"):
                try:
                    vals[m] = round(float(v), digits)
                except ValueError:
                    vals[m] = v
        out[kernel] = vals  # later launches overwrite earlier ones
    return json.dumps(out, indent=2) if out else ""


class NCUProfiler(Profiler):
    """Always measure the same Nsight Compute counters, whatever the kernel looks like.

    This is the profiling used by the fixed-metric baselines. AutoProfiler's tree profiler
    differs only in that it *chooses* the counters per kernel.
    """

    name = "ncu"

    class Params(Profiler.Params):
        metrics: tuple[str, ...]

    def profile(self, task: Task, cand: Candidate, env: Env) -> Profile:
        metrics = list(self.params.metrics)
        uid = f"{task.id}_{hashlib.sha1(cand.code.encode()).hexdigest()[:8]}"
        with env.step(f"  profile ({len(metrics)} counters)") as done:
            payload = _ncu_mod.remote_ncu_profile(
                env.backend, env.exec_cfg,
                {"uid": uid, "source": cand.code, "reference_code": task.ref_code}, metrics,
            )
            done(f"{payload.get('kernel_count', 0)} kernels" if payload.get("ok")
                 else f"FAILED {(payload.get('metadata') or {}).get('error', 'unknown')}")
        if not payload.get("ok"):
            err = (payload.get("metadata") or {}).get("error", "unknown")
            return Profile(text=f"(Nsight Compute profiling failed: {err})", info={"ncu_ok": False})
        values = {k: float(v) for k, v in (payload.get("metrics") or {}).items()}
        text = ncu_block(payload.get("csv") or "", metrics) or json.dumps(values, indent=2)
        return Profile(metrics=values, text=text, info={"ncu_ok": True, "n_kernels": payload.get("kernel_count")})


METRIC_SETS: dict[str, tuple[str, ...]] = {
    "cudaforge": (
        "sm__cycles_active.avg", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem", "launch__registers_per_thread",
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        "dram__bytes_read.sum", "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum.per_second", "l1tex__t_sector_hit_rate.pct",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__t_sector_hit_rate.pct", "lts__throughput.avg.pct_of_peak_sustained_active",
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    ),
    "roofline": ROOFLINE_METRICS,
    "kernelmem": (),  # extended at import time below
}
METRIC_SETS["kernelmem"] = METRIC_SETS["cudaforge"] + (
    "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "launch__grid_size", "launch__block_size",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
)


class NCUFixedProfiler(NCUProfiler):
    """Same as NCUProfiler but registered under the name configs use (``ncu_fixed``).

    Accepts ``metric_set`` (a key into METRIC_SETS) OR the explicit ``metrics`` tuple.
    """

    name = "ncu_fixed"

    class Params(NCUProfiler.Params):
        metric_set: str = "cudaforge"
        metrics: tuple[str, ...] = ()  # overrides metric_set when non-empty

    def __init__(self, **params) -> None:  # type: ignore[override]
        # Resolve metric_set -> metrics before calling super
        raw_metrics = params.get("metrics") or ()
        if not raw_metrics:
            ms = params.get("metric_set", "cudaforge")
            raw_metrics = METRIC_SETS.get(str(ms), METRIC_SETS["cudaforge"])
        params["metrics"] = tuple(raw_metrics)
        params.setdefault("metric_set", params.get("metric_set", "cudaforge"))
        super().__init__(**params)
