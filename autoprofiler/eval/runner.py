from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

from autoprofiler.components.seeders.llm_port import LLMPortSeeder
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


def _make_seed(seeder, task: Task, env: Env):
    """The Triton port of ``task``: read from data/seeds/, generated and stored there if missing."""
    if env.cache_dir is not None:   # a hand-built env keeps no cache, and stays that way
        env = replace(env, cache_dir=REPO_ROOT / "data")
    return seeder.seed(task, env)


def _score(task: Task, traj: Trajectory, env: Env, vs_seed: bool = False) -> dict:
    """Re-time the method's chosen kernel with the strict protocol; this is the reported number.

    ``vs_seed`` also times the starting kernel the same way and records the ratio, which is
    what the report uses in place of the speedup over PyTorch.
    """
    if traj.best is None:
        return {"correct": False, "speedup": None, "baseline_ms": None, "candidate_ms": None,
                "seed_ms": None, "speedup_vs_seed": None,
                "error": traj.error or "no correct candidate"}
    r = env.evaluator.evaluate(task, traj.best, tag=f"{task.id}_final", final=True)
    out = {"correct": r.ok, "speedup": r.speedup if r.ok else None, "baseline_ms": r.baseline_ms,
           "candidate_ms": r.candidate_ms, "error": "" if r.ok else (r.error or r.stage)[:300]}
    if vs_seed:
        out["seed_ms"] = out["speedup_vs_seed"] = None
        seed = traj.steps[0].candidate if traj.steps and traj.steps[0].round == 0 else None
        if seed is not None and r.ok and r.candidate_ms:
            if seed is traj.best:
                seed_ms = r.candidate_ms
            else:  # same strict protocol as the final kernel, so the two times are comparable
                sr = env.evaluator.evaluate(task, seed, tag=f"{task.id}_seedfinal", final=True)
                seed_ms = sr.candidate_ms if sr.ok else None
            if seed_ms:
                out["seed_ms"] = seed_ms
                out["speedup_vs_seed"] = seed_ms / r.candidate_ms
    return out


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

    seeder = None
    if cfg.use_seed:
        others = [m for m, s in specs.items() if s["method"] != "autoprofiler"]
        if others:
            raise ValueError(f"use_seed starts from a Triton seed, which only autoprofiler runs "
                             f"from; remove {others} or set use_seed: false")
        seeder = LLMPortSeeder(language="triton", attempts=4)
        say("[eval] use_seed: start from the Triton seed (data/seeds), scored against the seed")

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
                traj = None
                kw = {}
                if seeder is not None:
                    start = _make_seed(seeder, task, env)
                    if start is None or start.result is None or not start.result.ok:
                        r = start.result if start else None
                        traj = Trajectory(task_id=task.id, method=mname, error="no working seed: " + (
                            f"{r.stage}: {r.error[:200]}" if r else "none produced"))
                    kw["start"] = start
                if traj is None:
                    traj = method.run(task, env, **kw)
            except Exception as e:  # a method crash is that method's failure on this task
                say(f" CRASHED {type(e).__name__}: {e}")
                traj = Trajectory(task_id=task.id, method=mname,
                                  error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
            score = _score(task, traj, env, vs_seed=cfg.use_seed)
            row = {"task_id": task.id, "group": task.group, "method": mname, **score,
                   "wall_s": round(time.time() - t0, 1), "trajectory": traj.to_dict()}
            rec.write_text(json.dumps(row, indent=2), encoding="utf-8")
            if traj.best is not None:
                (mdir / "code" / f"{task.id}.py").write_text(traj.best.code, encoding="utf-8")
            u = traj.to_dict().get("usage", {})
            shown = score["speedup_vs_seed"] if cfg.use_seed else score["speedup"]
            sp = (f"{shown:.3f}x" + (" vs seed" if cfg.use_seed else "")) if shown else score["error"][:50]
            say(f"  = {sp}  [{row['wall_s']:.0f}s, {u.get('llm', {}).get('calls', 0)} llm, "
                f"{u.get('llm', {}).get('total_tokens', 0) // 1000}k tok]")

    summary = write_report(out)
    say(f"[eval] done in {(time.time() - started) / 60:.1f} min")
    for m, r in summary["methods"].items():
        a = r["all"]
        base = summary["baseline"]   # what the speedups below are measured against
        say(f"[eval] {m:22} correct {a['correct']:.0%}  faster than {base} {a['fast_1']:.0%}  "
            f"geomean_all vs {base} {a['geomean_all']:.3f}")
    say(f"[eval] table -> {out}/results.md")
    return out
