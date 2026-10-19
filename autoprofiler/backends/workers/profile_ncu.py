#!/usr/bin/env python3
"""Profile a KernelBench reference Model under ncu and emit aggregated metrics JSON.

Usage:
  python profile_ncu.py --ref ref.py --metrics m1,m2 --out profile.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


# NVTX range name that brackets the candidate's measured forward. ncu is told to
# profile only kernels launched inside this range, so reference-model and warmup
# kernels never enter the metrics.
NVTX_RANGE = "ap_candidate"


def _run_group(cmd, *, timeout, **kwargs):
    """subprocess.run, but the child leads its own process group and, on
    timeout OR any exit, the WHOLE group is SIGKILLed. ncu spawns a
    TreeLauncherSubreaper -> python runner.py grandchild that is NOT killed
    when only the direct child is signalled; without the group kill those
    grandchildren orphan on the GPU (holding a CUDA context + ~0.5GB each),
    serialize against later profiles, and are the physical source of the
    "715s / CSV-header-not-found" failures. start_new_session=True puts the
    child in a fresh session/group so os.killpg reaps the entire subtree."""
    proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            cmd, proc.returncode, stdout, stderr
        )
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _load_ref(ref_src: str):
    ns: dict = {}
    exec(ref_src, ns)
    get_init = ns.get("get_init_inputs", lambda: [])
    return ns["Model"], ns["get_inputs"], get_init


def _write_runner(
    path: Path, ref_path: Path, candidate_path: Path
) -> None:
    path.write_text(
        f"""
import importlib.util
import sys
import torch

# The original torch program supplies the constructor and input factories.
ref_path = {str(ref_path)!r}
ref = open(ref_path, encoding="utf-8").read()
ns = {{}}
exec(compile(ref, ref_path, "exec"), ns)
if "Model" not in ns or "get_inputs" not in ns:
    raise RuntimeError("reference must define Model and get_inputs")
ReferenceModel = ns["Model"]
get_inputs = ns["get_inputs"]
get_init = ns.get("get_init_inputs", lambda: [])

# Import the candidate from a real file. Triton's @jit calls inspect.getsource,
# which cannot work when the candidate is loaded with exec().
candidate_path = {str(candidate_path)!r}
module_name = "autoprofiler_ncu_candidate"
spec = importlib.util.spec_from_file_location(module_name, candidate_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import candidate module from {{candidate_path}}")
candidate_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = candidate_module
spec.loader.exec_module(candidate_module)
CandidateModel = getattr(
    candidate_module, "ModelNew", getattr(candidate_module, "Model", None)
)
if CandidateModel is None:
    raise RuntimeError("candidate must define ModelNew or Model")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
init_args = list(get_init())
reference_model = ReferenceModel(*init_args).cuda().eval()
inputs = [x.cuda() if hasattr(x, "cuda") else x for x in get_inputs()]
# Materialize LazyModule parameters before inspecting state_dict shapes.
with torch.no_grad():
    reference_model(*inputs)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
model = CandidateModel(*init_args).cuda().eval()
src_state = reference_model.state_dict()
dst_state = model.state_dict()
compatible = {{
    name: value
    for name, value in src_state.items()
    if name in dst_state and tuple(value.shape) == tuple(dst_state[name].shape)
}}
if compatible:
    model.load_state_dict(compatible, strict=False)
del reference_model

# Warm up the candidate OUTSIDE the profiled range: this triggers Triton JIT
# compilation and any lazy allocation, so the measured launch reflects steady
# state, not first-call compile artifacts.
with torch.no_grad():
    model(*inputs)
    torch.cuda.synchronize()

# Only the kernels launched inside this NVTX range are profiled (ncu is called
# with --nvtx --nvtx-include "{NVTX_RANGE}/"). Everything above — the reference
# model's forward (run only to source weights) and the candidate warmup — is
# deliberately excluded, so the metrics describe the CANDIDATE alone.
torch.cuda.nvtx.range_push("{NVTX_RANGE}")
with torch.no_grad():
    y = model(*inputs)
    torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("profile_runner_ok")
""",
        encoding="utf-8",
    )


def _parse_ncu_csv(text: str, metrics: list[str]) -> dict:
    # ncu prints logs before CSV; find header line starting with "ID"
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            start = i
            break
    if start is None:
        raise RuntimeError("ncu CSV header not found")

    buf = io.StringIO("\n".join(lines[start:]))
    reader = csv.DictReader(buf)
    rows = list(reader)
    if not rows:
        raise RuntimeError("ncu CSV has no kernel rows")

    kernel_names = []
    for r in rows:
        name = r.get("Kernel Name") or r.get("launch__kernel_name") or ""
        if name:
            kernel_names.append(name)

    # Prefer compute kernels over fill/normal init: pick rows with max gpu__time_duration.sum if present
    def row_weight(r: dict) -> float:
        for k in ("gpu__time_duration.sum", "gpu__time_duration.avg"):
            if k in r and r[k] not in (None, "", "n/a"):
                try:
                    return float(r[k])
                except ValueError:
                    pass
        return 1.0

    # Aggregate metric as duration-weighted mean across rows
    agg: dict[str, float] = {}
    for m in metrics:
        num = 0.0
        den = 0.0
        for r in rows:
            if m not in r or r[m] in (None, "", "n/a"):
                continue
            try:
                val = float(r[m])
            except ValueError:
                continue
            w = row_weight(r)
            num += val * w
            den += w
        if den > 0:
            agg[m] = num / den

    unique_kernels = sorted(set(kernel_names))
    return {
        "metrics": agg,
        "kernel_count": len(unique_kernels),
        "kernel_names": unique_kernels[:20],
        "num_rows": len(rows),
    }


def _failure_scope(returncode: int, stderr: str) -> str:
    """Classify runner exceptions separately from NCU/host failures."""
    if returncode == 0 or "Traceback (most recent call last)" not in stderr:
        return "system"
    infrastructure_markers = (
        "OSError: [Errno 5]",
        "Input/output error",
        "CUDA driver",
        "CUDA-capable device",
        "connection reset",
    )
    lowered = stderr.lower()
    if any(marker.lower() in lowered for marker in infrastructure_markers):
        return "system"
    return "sample"


def ncu_failure_scope(
    error: str, metadata: dict | None = None
) -> str:
    """Return ``sample`` or ``system`` for an NCU failure payload."""
    metadata = metadata or {}
    explicit = metadata.get("failure_scope")
    if explicit in ("sample", "system"):
        return str(explicit)

    err = str(error or metadata.get("error") or "")
    if err.startswith("ncu_exec:") or err == "batch_worker":
        return "sample"
    if err in ("no_kernels_profiled", "cuda_unavailable"):
        return "sample"
    if err.startswith("parse:"):
        return _failure_scope(
            int(metadata.get("ncu_returncode") or 1),
            str(metadata.get("stderr_tail") or metadata.get("stderr") or ""),
        )
    if err == "remote_failed":
        return _failure_scope(
            int(metadata.get("ncu_returncode") or 1),
            str(metadata.get("stderr") or metadata.get("stderr_tail") or ""),
        )
    if err in (
        "transport",
        "invalid_output",
        "missing_batch_row",
        "no_batch_result",
        "incomplete_metrics",
    ):
        return "system"
    return "system"


def profile(
    ref_path: Path, candidate_path: Path, metrics: list[str]
) -> dict:
    import torch

    out: dict = {"ok": False, "metrics": {}, "kernel_count": 0, "metadata": {}}
    if not torch.cuda.is_available():
        out["metadata"]["error"] = "cuda_unavailable"
        return out

    metrics_arg = ",".join(metrics)
    with tempfile.TemporaryDirectory(prefix="ap_ncu_") as td:
        td_path = Path(td)
        runner = td_path / "runner.py"
        _write_runner(runner, ref_path, candidate_path)
        cmd = [
            "ncu",
            "--target-processes",
            "all",
            # Restrict profiling to the candidate's measured forward (see runner).
            "--nvtx",
            "--nvtx-include",
            f"{NVTX_RANGE}/",
            "--metrics",
            metrics_arg,
            "--csv",
            "--page",
            "raw",
            # This interpreter, not whatever "python" is on PATH: the one that can
            # import torch is the one running this script.
            sys.executable,
            str(runner),
        ]
        env = os.environ.copy()
        # ensure conda python if caller activated env
        # A CUDA candidate's load_inline extension is named by the model, the same
        # name across repair rounds whose code actually changed; torch's default
        # build cache keys on that name and reloads a previous round's stale .so,
        # failing with "undefined symbol" instead of compiling the current code.
        # td is already unique per call (TemporaryDirectory above), so pointing the
        # extension cache there is free and makes every profile pass a real build.
        # A candidate that passes load_inline an explicit relative build_directory
        # (e.g. "./build") bypasses this env var and resolves against the runner's
        # cwd instead -- run the runner from inside td so that path is fresh too.
        env["TORCH_EXTENSIONS_DIR"] = str(td_path / "torch_ext")
        # Wall time of the whole ncu invocation is the profiling cost of this
        # metric SET. It must be measured at set level, not summed per metric:
        # one more counter can be free (same pass) or force an extra kernel
        # replay, and only the measured set cost reflects that.
        t0 = time.perf_counter()
        try:
            proc = _run_group(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900,
                env=env,
                cwd=str(td_path),
            )
        except Exception as e:
            out["profile_ms"] = (time.perf_counter() - t0) * 1000.0
            out["metadata"]["error"] = f"ncu_exec: {e}"
            out["metadata"]["failure_scope"] = "sample"
            out["metadata"]["traceback"] = traceback.format_exc()
            return out
        out["profile_ms"] = (time.perf_counter() - t0) * 1000.0

        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        out["metadata"]["ncu_returncode"] = proc.returncode
        if "No kernels were profiled" in combined:
            out["metadata"]["error"] = "no_kernels_profiled"
            out["metadata"]["failure_scope"] = "sample"
            out["metadata"]["tail"] = combined[-1500:]
            return out
        try:
            parsed = _parse_ncu_csv(proc.stdout or "", metrics)
        except Exception as e:
            out["metadata"]["error"] = f"parse: {e}"
            out["metadata"]["tail"] = (proc.stdout or "")[-1500:]
            out["metadata"]["stderr_tail"] = (proc.stderr or "")[-1500:]
            out["metadata"]["failure_scope"] = _failure_scope(
                proc.returncode, proc.stderr or ""
            )
            return out

        out["ok"] = True
        out["metrics"] = parsed["metrics"]
        out["kernel_count"] = parsed["kernel_count"]
        out["kernel_names"] = parsed["kernel_names"]
        out["num_rows"] = parsed["num_rows"]
        # Raw ncu CSV so callers can re-parse with an upstream method's own
        # loader (e.g. CudaForge load_ncu_metrics) for byte-faithful metric blocks.
        out["csv"] = proc.stdout or ""
        missing = [m for m in metrics if m not in out["metrics"]]
        if missing:
            out["metadata"]["missing_metrics"] = missing
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--candidate", default="")
    ap.add_argument("--metrics", required=True, help="comma-separated ncu metric names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    candidate = Path(args.candidate) if args.candidate else Path(args.ref)
    result = profile(Path(args.ref), candidate, metrics)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": result.get("ok"),
                "kernel_count": result.get("kernel_count"),
                "metrics": result.get("metrics"),
            }
        )
    )


if __name__ == "__main__":
    main()
