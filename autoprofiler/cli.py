"""``autoprofiler`` command line: train a tree, evaluate methods, build a report."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _train(a: argparse.Namespace) -> None:
    from autoprofiler.train.build import build_state
    from autoprofiler.train.loop import train

    state = build_state(a.config, a.backend, limit=a.limit, resume_run_id=a.resume, data_path=a.data)
    summary = train(state)
    out = a.out or state.train_cfg.checkpoint.tree_out
    final = state.run_dir() / "tree.json"
    if out and final.exists():
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final, dest)
        print(f"[train] tree -> {dest}")
    print(f"[train] run dir: {state.run_dir()}")
    return summary


def _eval(a: argparse.Namespace) -> None:
    from autoprofiler.eval import load_eval_config, run_eval

    cfg = load_eval_config(a.config)
    if a.methods:
        cfg.methods = a.methods
    if a.limit is not None:
        cfg.limit = a.limit
    if a.rounds is not None:
        cfg.rounds = a.rounds
    out = run_eval(cfg, resume=not a.fresh)
    print(f"[eval] results: {out}/results.md")


def _report(a: argparse.Namespace) -> None:
    from autoprofiler.eval.report import write_report

    write_report(Path(a.run_dir))
    print((Path(a.run_dir) / "results.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="autoprofiler", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a profiling tree on a Triton-seed corpus")
    t.add_argument("config", nargs="?", default="configs/train/default.yaml")
    t.add_argument("--backend", default="configs/backends/local.yaml", help="GPU backend config")
    t.add_argument("--data", help="corpus dir with seed_index.jsonl (overrides the config)")
    t.add_argument("--limit", type=int, help="use only the first N training programs")
    t.add_argument("--out", help="where to save the finished tree (default: checkpoint.tree_out)")
    t.add_argument("--resume", help="resume a run by run id")
    t.set_defaults(fn=_train)

    e = sub.add_parser("eval", help="run methods on a benchmark and write the results table")
    e.add_argument("config", help="eval config, e.g. configs/eval/main_table.yaml")
    e.add_argument("--methods", nargs="+", help="override the method list")
    e.add_argument("--limit", type=int, help="only the first N tasks")
    e.add_argument("--rounds", type=int, help="override the round budget")
    e.add_argument("--fresh", action="store_true", help="ignore existing per-task results")
    e.set_defaults(fn=_eval)

    r = sub.add_parser("report", help="(re)build the results table of a finished run")
    r.add_argument("run_dir")
    r.set_defaults(fn=_report)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        # Misconfiguration and missing credentials are the common case here; a stack trace
        # buries the one line that says what to fix. AUTOPROFILER_TRACEBACK=1 brings it back.
        if os.environ.get("AUTOPROFILER_TRACEBACK"):
            raise
        raise SystemExit(f"autoprofiler {a.cmd}: {type(e).__name__}: {e}") from e


if __name__ == "__main__":
    main()
