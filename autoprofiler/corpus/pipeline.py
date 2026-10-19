"""Orchestration for the torch -> Triton seed pre-experiment.

Shape of the run: rows are processed in *waves*. Within a wave every row's LLM
turn happens concurrently (network-bound, so threads are enough), then the whole
wave is verified in ONE remote batch, then the rows that failed go into another
round carrying the real machine error as feedback. Rounds continue until the row
is accepted or the repair budget is spent.

Two properties this buys us, both load-bearing:
  - GPU round-trips scale with waves x rounds, not rows x rounds.
  - Repair is grounded in what the machine actually reported, not in a guess.

Baseline probing is a separate first pass with its own cache: whether the ORIGINAL
torch program runs is a property of the row, not of any port, so it is measured
once and reused across resumes.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoprofiler.llm.client import LLMClient
from autoprofiler.corpus.dataset import SeedRow, load_seed_rows
from autoprofiler.codegen.port import PortRejected, propose_port
from autoprofiler.codegen.port_prompts import PROMPT_VERSION
from autoprofiler.backends.verify import RemoteVerifier, VerifyRequest, VerifyResult
from autoprofiler.corpus.store import (
    Attempt,
    AttemptLog,
    build_report,
    freeze_seed_set,
)

Logger = Callable[..., None]


def _default_log(*args: Any) -> None:
    print(*args, flush=True)


@dataclass
class SeedgenConfig:
    source: str = ""
    out_dir: str = "./output/seed_triton"
    limit: int | None = None
    sample_seed: int | None = None
    wave_size: int = 24
    llm_workers: int = 8
    max_rounds: int = 3
    warmup: int = 3
    repeats: int = 10
    row_timeout_s: int = 300
    temperature: float = 0.2
    # Each repair round nudges temperature up: a deterministic retry of a failed
    # deterministic attempt tends to reproduce the same mistake.
    repair_temperature_step: float = 0.15
    resume: bool = True
    exec_config: str = ""
    baseline_cache: str = ""

    def temperature_for(self, round_idx: int) -> float:
        return min(1.0, self.temperature + self.repair_temperature_step * round_idx)


@dataclass
class _Pending:
    """A row still working its way through the rounds."""

    row: SeedRow
    round: int = 0
    code: str = ""
    error: str = ""
    stage: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class SeedPipeline:
    def __init__(
        self,
        cfg: SeedgenConfig,
        *,
        client: LLMClient,
        verifier: RemoteVerifier,
        log: Logger = _default_log,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.verifier = verifier
        self.log = log
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_path = self.out_dir / "attempts.jsonl"
        self.baselines_path = (
            Path(cfg.baseline_cache)
            if cfg.baseline_cache
            else self.out_dir / "baselines.jsonl"
        )
        self.baselines_path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts = AttemptLog(self.attempts_path)
        self.tag_prefix = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", self.client.config.model
        )[:48]

    # ── step 0: baselines ────────────────────────────────────────────────────

    def _load_baseline_cache(self) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        if not self.cfg.resume or not self.baselines_path.exists():
            return cache
        for line in self.baselines_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("uid"):
                cache[str(rec["uid"])] = rec
        return cache

    def probe_baselines(self, rows: list[SeedRow]) -> dict[str, dict]:
        """Measure the original torch programs; cached across resumes."""
        cache = self._load_baseline_cache()
        todo = [r for r in rows if r.uid not in cache]
        if cache:
            self.log(f"[seedgen] baseline cache hit for {len(cache)} rows")
        if not todo:
            return cache

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with self.baselines_path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), self.cfg.wave_size):
                chunk = todo[i : i + self.cfg.wave_size]
                tag = (
                    f"baseline_{self.tag_prefix}_{stamp}_"
                    f"{i // self.cfg.wave_size:04d}"
                )
                results = self.verifier.probe_baselines(
                    [(r.uid, r.code) for r in chunk], tag=tag
                )
                n_ok = 0
                for r in chunk:
                    vr = results.get(r.uid)
                    rec = {
                        "uid": r.uid,
                        "row_index": r.row_index,
                        "ok": bool(vr and vr.ok),
                        "baseline_ms": vr.baseline_ms if vr else None,
                        "stage": vr.stage if vr else "missing",
                        "error": (vr.error if vr else "no result")[:1000],
                    }
                    cache[r.uid] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += int(rec["ok"])
                fh.flush()
                self.log(
                    f"[seedgen] baseline wave {i // self.cfg.wave_size + 1}: "
                    f"{n_ok}/{len(chunk)} torch programs run"
                )
        return cache

    # ── LLM turn ─────────────────────────────────────────────────────────────

    def _propose_one(self, item: _Pending) -> _Pending:
        temp = self.cfg.temperature_for(item.round)
        try:
            cand = propose_port(
                item.row.code,
                client=self.client,
                temperature=temp,
                prev_code=item.code or None,
                error=item.error or None,
                stage=item.stage,
            )
        except PortRejected as e:
            # Locally invalid: no GPU spent, but it still consumed a round and
            # its reason becomes the feedback for the next one.
            item.code = ""
            item.error = e.reason
            item.stage = e.stage
            item.meta = {"local_reject": True}
            return item
        except Exception as e:
            item.code = ""
            item.error = f"llm_call_failed: {type(e).__name__}: {e}"
            item.stage = "llm"
            item.meta = {"llm_error": True}
            return item
        item.code = cand.code
        item.error = ""
        item.stage = ""
        item.meta = cand.meta
        return item

    def _propose_wave(self, items: list[_Pending]) -> list[_Pending]:
        if len(items) == 1:
            return [self._propose_one(items[0])]
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.llm_workers)) as pool:
            return list(pool.map(self._propose_one, items))

    # ── one wave through all rounds ──────────────────────────────────────────

    def _record(
        self,
        item: _Pending,
        *,
        ok: bool,
        stage: str,
        error: str,
        vr: VerifyResult | None = None,
    ) -> None:
        self.attempts.append(
            Attempt(
                uid=item.row.uid,
                row_index=item.row.row_index,
                round=item.round,
                ok=ok,
                stage=stage,
                error=error[:2000],
                baseline_ms=vr.baseline_ms if vr else None,
                candidate_ms=vr.candidate_ms if vr else None,
                speedup=vr.speedup if vr else None,
                code=item.code if ok else "",
                ref_code=item.row.code if ok else "",
                ops=item.row.ops,
                prompt_version=PROMPT_VERSION,
                model=self.client.config.model,
                meta={
                    **item.meta,
                    # Keep failed source out of the ledger's hot path but retain
                    # enough to debug: the tail is where the kernel usually is.
                    **({} if ok else {"failed_code_tail": item.code[-1500:]}),
                },
            )
        )

    def run_wave(self, rows: list[SeedRow], *, tag: str) -> dict[str, bool]:
        """Drive one wave through up to max_rounds. Returns uid -> accepted."""
        pending = [_Pending(row=r) for r in rows]
        accepted: dict[str, bool] = {r.uid: False for r in rows}

        for rnd in range(self.cfg.max_rounds):
            if not pending:
                break
            for item in pending:
                item.round = rnd
            pending = self._propose_wave(pending)

            # Locally rejected rows never reach the GPU.
            to_verify = [it for it in pending if it.code]
            for it in pending:
                if not it.code:
                    self._record(it, ok=False, stage=it.stage or "local_check", error=it.error)

            results: dict[str, VerifyResult] = {}
            if to_verify:
                results = self.verifier.verify_batch(
                    [
                        VerifyRequest(
                            uid=it.row.uid, ref_code=it.row.code, cand_code=it.code
                        )
                        for it in to_verify
                    ],
                    tag=f"{tag}_r{rnd}",
                )

            still: list[_Pending] = []
            n_ok = 0
            for it in pending:
                if not it.code:
                    still.append(it)
                    continue
                vr = results.get(it.row.uid)
                if vr is None:
                    it.error = "no verification result"
                    it.stage = "transport"
                    self._record(it, ok=False, stage=it.stage, error=it.error)
                    still.append(it)
                    continue
                self._record(it, ok=vr.ok, stage=vr.stage, error=vr.error, vr=vr)
                if vr.ok:
                    accepted[it.row.uid] = True
                    n_ok += 1
                else:
                    it.error = vr.error or f"failed at stage {vr.stage}"
                    it.stage = vr.stage
                    still.append(it)

            self.log(
                f"[seedgen] {tag} round {rnd}: accepted {n_ok}, "
                f"retry {len(still)} (local rejects "
                f"{sum(1 for it in pending if not it.code)})"
            )
            pending = still

        return accepted

    # ── driver ───────────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        rows, rejected = load_seed_rows(
            self.cfg.source or None,
            limit=self.cfg.limit,
            sample_seed=self.cfg.sample_seed,
        )
        self.log(
            f"[seedgen] corpus: {len(rows)} usable rows, "
            f"{len(rejected)} rejected as malformed source"
        )

        if self.cfg.resume:
            done, exhausted = AttemptLog.resume_state(
                self.attempts_path, max_rounds=self.cfg.max_rounds
            )
            skip = done | exhausted
            if skip:
                rows = [r for r in rows if r.uid not in skip]
                self.log(
                    f"[seedgen] resume: skipping {len(done)} accepted + "
                    f"{len(exhausted)} budget-exhausted rows; {len(rows)} left"
                )

        baselines = self.probe_baselines(rows) if rows else {}
        runnable = [r for r in rows if (baselines.get(r.uid) or {}).get("ok")]
        n_bad_baseline = len(rows) - len(runnable)
        if n_bad_baseline:
            self.log(
                f"[seedgen] dropping {n_bad_baseline} rows whose original torch "
                f"program does not run"
            )

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        total_accepted = 0
        for i in range(0, len(runnable), self.cfg.wave_size):
            chunk = runnable[i : i + self.cfg.wave_size]
            wave = i // self.cfg.wave_size + 1
            tag = f"port_{self.tag_prefix}_{stamp}_w{wave:04d}"
            self.log(
                f"[seedgen] wave {wave}/"
                f"{(len(runnable) + self.cfg.wave_size - 1) // self.cfg.wave_size}"
                f" ({len(chunk)} rows)"
            )
            accepted = self.run_wave(chunk, tag=tag)
            total_accepted += sum(1 for v in accepted.values() if v)
            self.log(
                f"[seedgen] cumulative accepted this run: {total_accepted}"
            )

        return self.finalize(
            rejected=rejected,
            baselines=baselines,
            n_bad_baseline=n_bad_baseline,
        )

    def finalize(
        self,
        *,
        rejected: list[dict] | None = None,
        baselines: dict[str, dict] | None = None,
        n_bad_baseline: int = 0,
    ) -> dict[str, Any]:
        frozen = freeze_seed_set(self.attempts_path, self.out_dir)
        all_rows, all_rejected = load_seed_rows(
            self.cfg.source or None,
            limit=self.cfg.limit,
            sample_seed=self.cfg.sample_seed,
        )
        report = build_report(
            self.attempts_path,
            total_rows=len(all_rows) + len(all_rejected),
            rejected=rejected if rejected is not None else all_rejected,
            extra={
                "rows_torch_baseline_failed": n_bad_baseline
                or sum(
                    1 for r in (baselines or {}).values() if not r.get("ok")
                ),
                "frozen": frozen,
                "config": {
                    "model": self.client.config.model,
                    "sample_seed": self.cfg.sample_seed,
                    "prompt_version": PROMPT_VERSION,
                    "max_rounds": self.cfg.max_rounds,
                    "wave_size": self.cfg.wave_size,
                    "temperature": self.cfg.temperature,
                    "warmup": self.cfg.warmup,
                    "repeats": self.cfg.repeats,
                },
            },
        )
        report_path = self.out_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.attempts.close()
        self.log(
            f"[seedgen] done: {report['rows_accepted']} seeds "
            f"(yield {report['yield_rate']:.1%}) -> {frozen['index_path']}"
        )
        self.log(f"[seedgen] report -> {report_path}")
        return report
