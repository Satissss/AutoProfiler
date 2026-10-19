"""CLI for the torch -> Triton seed pre-experiment.

    python -m autoprofiler.corpus --limit 50            # small pilot
    python -m autoprofiler.corpus                       # full corpus
    python -m autoprofiler.corpus --report-only         # rebuild artifacts
    python -m autoprofiler.corpus --dry-run --limit 5   # no GPU, no LLM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoprofiler.corpus.pipeline import SeedgenConfig, SeedPipeline


def _build_llm_client(
    profile: str,
    *,
    model: str = "",
    base_url: str = "",
    api_style: str = "",
    max_tokens: int = 0,
    llm_retries: int | None = None,
):
    """LLM client from a configs/llm/<profile>.yaml profile, with optional overrides."""
    from autoprofiler.llm.client import LLMClient
    from autoprofiler.llm.config import load_llm_profile

    cfg = load_llm_profile(profile)
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url.rstrip("/")
    if api_style:
        cfg.api_style = api_style
    if max_tokens:
        cfg.max_tokens = max_tokens
    if llm_retries is not None:
        cfg.request_retries = llm_retries
    return LLMClient(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autoprofiler.corpus")
    ap.add_argument("--source", default="", help="parquet corpus (default data/source/data.parquet)")
    ap.add_argument("--out-dir", default="./output/seed_triton")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (0 = all)")
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="with --limit, choose a reproducible random subset instead of the first rows",
    )
    ap.add_argument("--wave-size", type=int, default=24, help="rows per remote batch")
    ap.add_argument("--llm-workers", type=int, default=8, help="concurrent LLM calls")
    ap.add_argument("--max-rounds", type=int, default=3, help="1 port + N-1 repairs")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--row-timeout-s", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--llm", default="gpt", help="LLM profile in configs/llm/ (the porting model)")
    ap.add_argument("--model", default="", help="override the profile's model")
    ap.add_argument("--base-url", default="", help="override the LLM base URL")
    ap.add_argument(
        "--api-style",
        choices=("openai", "anthropic"),
        default="",
        help="request/response protocol (default from config, normally openai)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="override response token budget (0 = use the profile)",
    )
    ap.add_argument(
        "--llm-retries",
        type=int,
        default=None,
        help="retries per API call after the first attempt",
    )
    ap.add_argument("--no-resume", action="store_true", help="ignore previous attempts.jsonl")
    ap.add_argument("--exec-config", default="", help="backend config (default: configs/backends/local.yaml)")
    ap.add_argument(
        "--baseline-cache",
        default="",
        help="shared torch-baseline JSONL (useful for fair multi-model comparisons)",
    )
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="measure/cache original torch programs only; no LLM porting",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the frozen seed set + report from attempts.jsonl; no LLM, no GPU",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="load the corpus and print a preview; no LLM, no GPU",
    )
    args = ap.parse_args(argv)

    cfg = SeedgenConfig(
        source=args.source,
        out_dir=args.out_dir,
        limit=args.limit or None,
        sample_seed=args.sample_seed,
        wave_size=args.wave_size,
        llm_workers=args.llm_workers,
        max_rounds=args.max_rounds,
        warmup=args.warmup,
        repeats=args.repeats,
        row_timeout_s=args.row_timeout_s,
        temperature=args.temperature,
        resume=not args.no_resume,
        exec_config=args.exec_config,
        baseline_cache=args.baseline_cache,
    )

    if args.dry_run:
        from autoprofiler.corpus.dataset import load_seed_rows
        from autoprofiler.codegen.port_prompts import build_user_prompt

        rows, rejected = load_seed_rows(
            cfg.source or None, limit=cfg.limit, sample_seed=cfg.sample_seed
        )
        print(f"[dry-run] usable rows: {len(rows)}, malformed: {len(rejected)}")
        for r in rows[:3]:
            print(f"  {r.uid} row={r.row_index} ops={r.ops[:60]!r} chars={len(r.code)}")
        if rows:
            prompt = build_user_prompt(rows[0].code)
            print(f"[dry-run] first user prompt: {len(prompt)} chars")
        return 0

    client = _build_llm_client(
        args.llm,
        model=args.model,
        base_url=args.base_url,
        api_style=args.api_style,
        max_tokens=args.max_tokens,
        llm_retries=args.llm_retries,
    )

    if args.report_only:
        # Verifier is never touched on this path; pass a stub so nothing can
        # accidentally reach the pod while "just rebuilding the report".
        pipeline = SeedPipeline(cfg, client=client, verifier=None)  # type: ignore[arg-type]
        report = pipeline.finalize()
        print(json.dumps({k: v for k, v in report.items() if k != "frozen"}, indent=2))
        return 0

    from autoprofiler.backends import build_backend
    from autoprofiler.backends.verify import RemoteVerifier
    from autoprofiler.config import load_exec_config

    exec_cfg = load_exec_config(cfg.exec_config or "configs/backends/local.yaml")
    backend = build_backend(exec_cfg)
    verifier = RemoteVerifier(
        backend,
        exec_cfg,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        row_timeout_s=cfg.row_timeout_s,
    )
    print(
        f"[seedgen] model={client.config.model} backend={exec_cfg.backend} "
        f"out={cfg.out_dir}"
    )
    pipeline = SeedPipeline(cfg, client=client, verifier=verifier)
    if args.baseline_only:
        from autoprofiler.corpus.dataset import load_seed_rows

        rows, rejected = load_seed_rows(
            cfg.source or None, limit=cfg.limit, sample_seed=cfg.sample_seed
        )
        baselines = pipeline.probe_baselines(rows)
        ok = sum(1 for rec in baselines.values() if rec.get("ok"))
        print(
            json.dumps(
                {
                    "rows_selected": len(rows) + len(rejected),
                    "rows_usable": len(rows),
                    "rows_baseline_ok": ok,
                    "baseline_cache": str(pipeline.baselines_path),
                },
                indent=2,
            )
        )
        pipeline.attempts.close()
        return 0 if ok else 1
    report = pipeline.run()
    return 0 if report["rows_accepted"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
