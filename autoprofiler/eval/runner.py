from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable

from autoprofiler.config import REPO_ROOT
from autoprofiler.core.benchmark import Benchmark
from autoprofiler.core.component import get, load_builtin
from autoprofiler.core.env import Env, build_env as _build_env_orig
build_env = _build_env_orig  # module-level name so tests can monkeypatch it
from autoprofiler.core.types import Task, Trajectory
from autoprofiler.eval.config import EvalConfig
from autoprofiler.eval.report import write_report


def build_method(run_name: str, spec: dict):
    """One method, as this experiment configures it.

    ``spec`` comes straight from the experiment file: ``method`` names the class and
    everything else is a setting, so a run can override a knob (``n_samples``) or a
    whole component (``profiler: {type: tree, ...}``) without touching the class.
    """
    load_builtin()
    spec = dict(spec)
    cls = get("method", spec.pop("method"))
    method = cls(**spec)
    method.run_name = run_name   # two runs of one method stay distinguishable
    return method


def _score(task: Task, traj: Trajectory, env: Env) -> dict:
    """Re-time the method's chosen kernel with the strict protocol; this is the reported number."""
    if traj.best is None:
        return {"correct": False, "speedup": None, "baseline_ms": None, "candidate_ms": None,
                "error": traj.error or "no correct candidate"}
    r = env.evaluator.evaluate(task, traj.best, tag=f"{task.id}_final", final=True)
    return {"correct": r.ok, "speedup": r.speedup if r.ok else None, "baseline_ms": r.baseline_ms,
            "candidate_ms": r.candidate_ms, "error": "" if r.ok else (r.error or r.stage)[:300]}


def _tee(out_dir: Path, log) -> tuple[Callable[[str], None], Path]:
    """Log to the console and to a file, with timestamps in the file.

    A run takes hours, so the console scroll is gone by the time anything is worth
    asking about; the file is what you read afterwards.
    """
    path = out_dir / "run.log"
    handle = path.open("a", encoding="utf-8")

    def emit(msg: str = "") -> None:
        log(msg)
        # A line ending in the ellipsis is a "this is running now" ping: worth seeing
        # live so a 90-second LLM call does not look like a hang, worthless afterwards
        # because the line that follows carries the outcome and the duration.
        if not msg.endswith("…"):
            handle.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            handle.flush()

    return emit, path


def run_eval(cfg: EvalConfig, *, resume: bool = True, log=print) -> Path:
    """Run every method in ``cfg.methods`` on the same tasks, LLM, backend and budget.

    Results are written per (method, task) as they finish, so an interrupted run
    resumes where it stopped. Returns the run directory.
    """
    tasks = Benchmark.load(cfg.benchmark).tasks(groups=cfg.groups, ids=cfg.ids, limit=cfg.limit)
    out = Path(cfg.output)
    out = out if out.is_absolute() else REPO_ROOT / out
    out = out / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "config": cfg.model_dump(), "tasks": [{"id": t.id, "group": t.group} for t in tasks],
    }, indent=2), encoding="utf-8")

    say, log_path = _tee(out, log)
    specs = cfg.method_specs()
    env = build_env(llm=cfg.llm, backend=cfg.backend, work_dir=out / "_work", rounds=cfg.rounds,
                    fast=cfg.fast, final=cfg.final, log=say)
    say(f"[eval] {cfg.name}: {len(tasks)} tasks x {cfg.rounds} rounds on {env.llm.config.model}")
    say(f"[eval] -> {out}")

    started = time.time()
    for mname, spec in specs.items():
        method = build_method(mname, spec)
        mdir = out / mname
        (mdir / "tasks").mkdir(parents=True, exist_ok=True)
        (mdir / "code").mkdir(exist_ok=True)
        done_before = sum(1 for t in tasks if (mdir / "tasks" / f"{t.id}.json").exists())
        say(f"[eval] {mname}: {type(method.seeder).__name__} -> "
            f"{type(method.profiler).__name__} -> {type(method.optimizer).__name__}"
            + (f" ({done_before}/{len(tasks)} already done)" if resume and done_before else ""))
        for i, task in enumerate(tasks, 1):
            rec = mdir / "tasks" / f"{task.id}.json"
            if resume and rec.exists():
                continue
            say(f"[{mname}] {i}/{len(tasks)} {task.id}")
            env = env.for_task(out / "_work" / mname / task.id)
            t0 = time.time()
            try:
                traj = method.run(task, env)
            except Exception as e:  # a method crash is that method's failure on this task
                say(f" CRASHED {type(e).__name__}: {e}")
                traj = Trajectory(task_id=task.id, method=mname,
                                  error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
            score = _score(task, traj, env)
            row = {"task_id": task.id, "group": task.group, "method": mname, **score,
                   "wall_s": round(time.time() - t0, 1), "trajectory": traj.to_dict()}
            rec.write_text(json.dumps(row, indent=2), encoding="utf-8")
            if traj.best is not None:
                (mdir / "code" / f"{task.id}.py").write_text(traj.best.code, encoding="utf-8")
            u = traj.to_dict().get("usage", {})
            sp = f"{score['speedup']:.3f}x" if score["speedup"] else score["error"][:50]
            say(f"  = {sp}  [{row['wall_s']:.0f}s, {u.get('llm', {}).get('calls', 0)} llm, "
                f"{u.get('llm', {}).get('total_tokens', 0) // 1000}k tok]")

    summary = write_report(out)
    say(f"[eval] done in {(time.time() - started) / 60:.1f} min")
    for m, r in summary["methods"].items():
        a = r["all"]
        say(f"[eval] {m:22} correct {a['correct']:.0%}  fast_1 {a['fast_1']:.0%}  "
            f"geomean_all {a['geomean_all']:.3f}")
    say(f"[eval] table -> {out}/results.md")
    return out
