#!/usr/bin/env python3
"""Remote batch verifier for the Triton seed pre-experiment (runs on the GPU pod).

Two modes:

  driver (default)
      Read a jsonl manifest of rows and verify each one in an ISOLATED
      subprocess, appending one result line per row. Process isolation is not
      optional here: a bad Triton kernel can hard-crash the interpreter with an
      illegal memory access, and one bad row must not take down the whole batch.

  --worker
      Verify exactly one row and print its result as json on stdout.

A row without a candidate is a *baseline probe*: it only checks that the
original torch program runs and records its runtime. That is step 0 of the
pipeline, used to drop rows whose reference program is itself broken.

The driver takes a single *bundle* json holding the source for the whole batch
and materializes the .py files itself. That keeps the whole batch to one file
transfer plus one exec: writing each row's files individually would cost two
round-trips per row and undo the point of batching. The files must be real .py
on disk because Triton's @jit needs inspect.getsource.

Bundle:         {"rows": [{"uid":..., "ref_code":..., "cand_code": <str|null>}]}
Result line:    {"uid":..., "stage":..., "ok":..., "baseline_ms":...,
                 "candidate_ms":..., "speedup":..., "correct":..., "error":...}

``stage`` localizes the failure so the report can bucket causes:
    baseline | load_candidate | init_candidate | run_candidate | correctness | timing | ok
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


# ── single-row verification (worker) ─────────────────────────────────────────


def _load_ref(ref_src: str):
    ns: dict = {}
    exec(ref_src, ns)
    if "Model" not in ns:
        raise RuntimeError("reference program defines no Model")
    get_inputs = ns.get("get_inputs")
    if get_inputs is None:
        raise RuntimeError("reference program defines no get_inputs")
    return ns["Model"], get_inputs, ns.get("get_init_inputs", lambda: [])


def _load_modelnew_from_path(path: Path):
    """Import ModelNew from a real .py file.

    Triton's @jit needs inspect.getsource, which fails for exec()-only strings.
    """
    import importlib.util
    import uuid

    path = Path(path)
    mod_name = f"seedgen_cand_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ModelNew"):
        raise RuntimeError("candidate module defines no ModelNew")
    return mod.ModelNew


def _copy_compatible_state(src_model, dst_model) -> list[str]:
    src = src_model.state_dict()
    dst = dst_model.state_dict()
    payload = {}
    copied = []
    for k, v in dst.items():
        if k in src and tuple(src[k].shape) == tuple(v.shape):
            payload[k] = src[k]
            copied.append(k)
    if payload:
        dst_model.load_state_dict(payload, strict=False)
    return copied


def _time_model(model, inputs, warmup: int, repeats: int) -> float:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(*inputs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _allclose(ref_y, cand_y, atol: float, rtol: float) -> bool:
    import torch

    if isinstance(ref_y, torch.Tensor):
        if not isinstance(cand_y, torch.Tensor):
            return False
        if tuple(ref_y.shape) != tuple(cand_y.shape):
            return False
        return bool(torch.allclose(cand_y, ref_y, atol=atol, rtol=rtol))
    if isinstance(ref_y, (tuple, list)):
        if not isinstance(cand_y, (tuple, list)) or len(ref_y) != len(cand_y):
            return False
        return all(_allclose(a, b, atol, rtol) for a, b in zip(ref_y, cand_y))
    return ref_y == cand_y


def verify_row(
    ref_src: str,
    cand_src: str | None,
    *,
    cand_path: Path | None = None,
    warmup: int = 3,
    repeats: int = 10,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict:
    """Verify one row. Returns a result dict (never raises)."""
    import torch

    out: dict = {
        "stage": "baseline",
        "ok": False,
        "baseline_ms": None,
        "candidate_ms": None,
        "speedup": None,
        "correct": False,
        "error": "",
    }
    if not torch.cuda.is_available():
        out["error"] = "cuda_unavailable"
        return out

    device = torch.device("cuda:0")

    # ── step 0: the reference torch program must itself run ──────────────────
    try:
        Model, get_inputs, get_init = _load_ref(ref_src)
        init_args = list(get_init())
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = Model(*init_args).to(device).eval()
        inputs = [x.to(device) if hasattr(x, "to") else x for x in get_inputs()]
        with torch.no_grad():
            ref_y = model(*inputs)
            if isinstance(ref_y, torch.Tensor):
                ref_y = ref_y.detach().clone()
            elif isinstance(ref_y, (tuple, list)):
                ref_y = type(ref_y)(
                    x.detach().clone() if hasattr(x, "detach") else x for x in ref_y
                )
        out["baseline_ms"] = _time_model(model, inputs, warmup, repeats)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out

    if cand_src is None and cand_path is None:
        # Baseline probe only.
        out["stage"] = "ok"
        out["ok"] = True
        out["correct"] = True
        return out

    # ── candidate: import ───────────────────────────────────────────────────
    out["stage"] = "load_candidate"
    try:
        if cand_path is not None and Path(cand_path).exists():
            ModelNew = _load_modelnew_from_path(Path(cand_path))
        else:
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="seedgen_")) / "candidate.py"
            tmp.write_text(cand_src or "", encoding="utf-8")
            ModelNew = _load_modelnew_from_path(tmp)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out

    # ── candidate: construct with the same init args ────────────────────────
    out["stage"] = "init_candidate"
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        cand = ModelNew(*init_args).to(device).eval()
        out["copied_params"] = _copy_compatible_state(model, cand)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out

    # ── candidate: run (this is where compile / IMA failures surface) ───────
    out["stage"] = "run_candidate"
    try:
        with torch.no_grad():
            cand_y = cand(*inputs)
            torch.cuda.synchronize()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out

    # ── candidate: numerics ────────────────────────────────────────────────
    out["stage"] = "correctness"
    try:
        ok = _allclose(ref_y, cand_y, atol, rtol)
    except Exception as e:
        out["error"] = f"allclose_failed: {type(e).__name__}: {e}"
        return out
    out["correct"] = bool(ok)
    if not ok:
        out["error"] = "correctness_failed: outputs differ beyond atol/rtol"
        return out

    # ── candidate: timing (speedup is recorded but never gates acceptance) ──
    out["stage"] = "timing"
    try:
        cand_ms = _time_model(cand, inputs, warmup, repeats)
        out["candidate_ms"] = cand_ms
        base_ms = float(out["baseline_ms"] or 0.0)
        out["speedup"] = (base_ms / cand_ms) if cand_ms > 0 else None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["stage"] = "ok"
    out["ok"] = True
    return out


# ── batch driver ─────────────────────────────────────────────────────────────


def _run_worker(
    self_path: Path,
    row: dict,
    *,
    warmup: int,
    repeats: int,
    timeout_s: int,
) -> dict:
    """Run one row in a subprocess so a hard crash cannot kill the batch."""
    cmd = [
        sys.executable,
        str(self_path),
        "--worker",
        "--ref",
        str(row["ref_path"]),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    if row.get("cand_path"):
        cmd += ["--candidate", str(row["cand_path"])]
    try:
        # Successive CUDA candidates for the same task reuse the same load_inline
        # extension name (the model writes "bmm_cuda" every time), and torch keys its
        # build cache on that name alone. A repair round that genuinely changed the
        # kernel then reloads the PREVIOUS round's compiled .so -- torch sees the
        # name is "already built" and skips recompiling -- and fails with
        # "undefined symbol" because the new C++ declares a function the old binary
        # never had. A fresh TORCH_EXTENSIONS_DIR per subprocess handles torch's
        # default cache location, but some candidates pass an explicit relative
        # build_directory (e.g. "./build") to load_inline themselves -- that bypasses
        # the env var entirely and resolves against the subprocess cwd, which is the
        # same directory every round. Running each attempt in its own fresh cwd closes
        # that path too, so an explicit relative build_directory also lands somewhere
        # new each time.
        with tempfile.TemporaryDirectory(prefix="torch_ext_") as ext_dir:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=ext_dir,
                env={**os.environ, "PYTHONWARNINGS": "ignore", "TORCH_EXTENSIONS_DIR": ext_dir},
            )
    except subprocess.TimeoutExpired:
        return {
            "stage": "timeout",
            "ok": False,
            "correct": False,
            "error": f"worker_timeout after {timeout_s}s",
        }
    # The worker prints exactly one json object on its last non-empty stdout line.
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "stage": "worker_crash",
        "ok": False,
        "correct": False,
        "error": (
            f"worker exited {proc.returncode} with no result; "
            f"stderr={(proc.stderr or '')[-800:]}"
        ),
    }


def _materialize(bundle_path: Path, workdir: Path) -> list[dict]:
    """Write each row's ref/cand source to real .py files; return driver rows."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for rec in bundle.get("rows", []):
        uid = str(rec["uid"])
        row_dir = workdir / uid
        row_dir.mkdir(parents=True, exist_ok=True)
        ref_path = row_dir / "ref.py"
        ref_path.write_text(rec["ref_code"], encoding="utf-8")
        cand_path = None
        if rec.get("cand_code"):
            cand_path = row_dir / "cand.py"
            cand_path.write_text(rec["cand_code"], encoding="utf-8")
        rows.append(
            {
                "uid": uid,
                "ref_path": str(ref_path),
                "cand_path": str(cand_path) if cand_path else None,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="verify a single row")
    ap.add_argument("--ref", help="reference program path (worker mode)")
    ap.add_argument("--candidate", default=None, help="candidate path (worker mode)")
    ap.add_argument("--bundle", help="batch source bundle json (driver mode)")
    ap.add_argument("--manifest", help="jsonl manifest of rows (driver mode)")
    ap.add_argument("--workdir", help="where to materialize bundle sources")
    ap.add_argument("--out", help="jsonl results path (driver mode)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--timeout-s", type=int, default=300, help="per-row timeout")
    args = ap.parse_args()

    if args.worker:
        # Catch everything: a worker that dies without printing a result is
        # reported as "worker_crash", and that label should mean a genuine hard
        # crash (segfault / CUDA abort), not an ordinary Python exception.
        try:
            ref_src = Path(args.ref).read_text(encoding="utf-8")
            cand_path = Path(args.candidate) if args.candidate else None
            res = verify_row(
                ref_src,
                None,
                cand_path=cand_path,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except BaseException as e:  # noqa: BLE001 - must never escape
            res = {
                "stage": "worker_error",
                "ok": False,
                "correct": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
            }
        print(json.dumps(res))
        return

    if not args.out or not (args.bundle or args.manifest):
        ap.error("driver mode needs --out and one of --bundle / --manifest")

    self_path = Path(__file__).resolve()
    if args.bundle:
        workdir = Path(args.workdir or Path(args.bundle).parent / "src")
        workdir.mkdir(parents=True, exist_ok=True)
        rows = _materialize(Path(args.bundle), workdir)
    else:
        rows = [
            json.loads(line)
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append + flush per row: if the pod dies mid-batch we keep what we measured.
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            res = _run_worker(
                self_path,
                row,
                warmup=args.warmup,
                repeats=args.repeats,
                timeout_s=args.timeout_s,
            )
            res["uid"] = row["uid"]
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(rows)}] {row['uid']} stage={res.get('stage')} "
                f"ok={res.get('ok')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
