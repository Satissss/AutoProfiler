"""Aggregate per-task results into the main table.

Every method is scored over the SAME task list (from run.json). A task a method has no
result for, or failed, counts as a failure, never as a skipped row, so methods cannot
look better by dropping hard tasks.

Metrics (per method, per group and overall):
  correct      fraction of tasks whose chosen kernel is correct
  fast_1       fraction correct AND faster than the baseline (speedup > 1)
  geomean      geometric-mean speedup over correct tasks only
  geomean_all  geometric-mean speedup over all tasks, a failed task counting as 1.0
               (i.e. falling back to the baseline); this is the headline number
The baseline is PyTorch, or with ``use_seed`` the seed kernel each task started from
(the table then calls the first metric fast_seed).
95% bootstrap CIs are given for fast_1 and geomean_all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _geomean(xs) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else None


def _ci(values: np.ndarray, stat, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = [stat(values[i]) for i in idx]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _summarize(rows: list[dict], key: str = "speedup") -> dict:
    """``key`` picks the speedup being scored: over PyTorch, or over the seed (``speedup_vs_seed``)."""
    n = len(rows)
    sp = np.array([r[key] if r.get("correct") and r.get(key) else 0.0 for r in rows])
    correct = sp > 0
    fast1 = sp > 1.0
    eff = np.where(correct, sp, 1.0)  # failure falls back to the baseline
    log_eff = np.log(eff)
    usage = [r["trajectory"].get("usage", {}) for r in rows if "trajectory" in r]  # finished tasks only
    tok = [u.get("llm", {}).get("total_tokens", 0) or u.get("total_tokens", 0) for u in usage]
    return {
        "n": n,
        "correct": float(correct.mean()) if n else None,
        "fast_1": float(fast1.mean()) if n else None,
        "fast_1_ci": _ci(fast1.astype(float), np.mean),
        "geomean": _geomean(sp[correct]),
        "geomean_all": float(np.exp(log_eff.mean())) if n else None,
        "geomean_all_ci": ([math.exp(x) for x in c] if (c := _ci(log_eff, np.mean)) else None),
        "mean_llm_tokens": float(np.mean(tok)) if tok else None,
        "mean_llm_calls": float(np.mean([u.get("llm", {}).get("calls", 0) or u.get("llm_calls", 0) for u in usage])) if usage else None,
        "mean_gpu_seconds": float(np.mean([u.get("gpu_seconds", 0) for u in usage])) if usage else None,
    }


def build_summary(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    tasks = meta["tasks"]
    # With use_seed the scores are over each task's seed, not over PyTorch.
    key, baseline = ("speedup_vs_seed", "seed") if meta["config"].get("use_seed") else ("speedup", "PyTorch")
    summary: dict = {"config": meta["config"], "baseline": baseline, "methods": {}}
    for m in meta["config"]["methods"]:
        by_task = {}
        for t in tasks:
            f = run_dir / m / "tasks" / f"{t['id']}.json"
            by_task[t["id"]] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else \
                {"task_id": t["id"], "group": t["group"], "correct": False, "speedup": None}
        groups = sorted({t["group"] for t in tasks})
        res = {"all": _summarize(list(by_task.values()), key)}
        for g in groups:
            res[g] = _summarize([r for r in by_task.values() if r["group"] == g], key)
        res["missing"] = sum(1 for t in tasks if not (run_dir / m / "tasks" / f"{t['id']}.json").exists())
        summary["methods"][m] = res
    return summary


def _fmt(x, pct=False, ci=None) -> str:
    if x is None:
        return "-"
    s = f"{100 * x:.1f}%" if pct else f"{x:.3f}"
    if ci:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if pct else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return s


def to_markdown(summary: dict) -> str:
    groups = [g for g in next(iter(summary["methods"].values())) if g not in ("all", "missing")]
    base = summary.get("baseline", "PyTorch")
    tag = "seed" if base == "seed" else "1"   # fast_1 / fast_seed
    lines = [f"# {summary['config']['name']}", "",
             f"LLM `{summary['config']['llm']}`, {summary['config']['rounds']} rounds, "
             f"benchmark `{summary['config']['benchmark']}`. All speedups are over the {base}"
             f"{' kernel each task started from' if base == 'seed' else ' reference'}. "
             f"Failed/missing tasks count as 1.0x in geomean_all.", ""]
    for scope in ["all"] + groups:
        lines += [f"## {scope}", "",
                  f"| method | n | correct | fast_{tag} [95% CI] | geomean_all [95% CI] | geomean (correct) | tokens/task | gpu s/task |",
                  "|---|---|---|---|---|---|---|---|"]
        for m, r in summary["methods"].items():
            s = r[scope]
            lines.append(f"| {m} | {s['n']} | {_fmt(s['correct'], True)} | {_fmt(s['fast_1'], True, s['fast_1_ci'])} | "
                         f"{_fmt(s['geomean_all'], ci=s['geomean_all_ci'])} | {_fmt(s['geomean'])} | "
                         f"{s['mean_llm_tokens']:.0f} | {s['mean_gpu_seconds']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path) -> dict:
    summary = build_summary(Path(run_dir))
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(run_dir) / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary
