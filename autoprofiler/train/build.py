"""Construction of a training run (LLM clients, kernel store, tree, corpus)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoprofiler.backends import SingleGPUQueue, build_backend
from autoprofiler.backends.ncu import MicrobatchNCUProfiler
from autoprofiler.backends.verify import MicrobatchRemoteVerifier, RemoteVerifier
from autoprofiler.config import load_exec_config
from autoprofiler.llm.client import LLMClient
from autoprofiler.train.config import TrainConfig, load_train_config
from autoprofiler.tree.catalog import BOUND_KEYS
from autoprofiler.tree.store import KernelStore
from autoprofiler.tree.structure import MetricTree, seed_tree
from autoprofiler.train.checkpoint import _resolve_checkpoint_dir
from autoprofiler.train.corpus import load_datasets, load_uid_manifest, split_holdout
from autoprofiler.train.log import _log, configure_logging
from autoprofiler.train.loop import _close_runtime
from autoprofiler.train.recovery import _recover_leaf_updates, _repair_persisted_failure_states
from autoprofiler.train.state import LeafUpdateCycle, TrainState, _restore_history_wins


def _build_llm(stage) -> LLMClient:
    """One client per training stage, from its own LLM settings.

    ``stage`` carries where to call (a ``profile`` in configs/llm/, or an explicit
    ``model`` + ``base_url`` + ``api_key_env``) plus how to sample it, so REWRITE/SPLIT
    reasoning and the leaf optimizer can run on entirely different models.
    """
    from autoprofiler.llm.config import resolve_llm

    cfg = resolve_llm(stage)
    cfg.request_retries = max(0, int(stage.attempts) - 1)
    return LLMClient(cfg)


def _open_training_store(
    base: Path,
    run_id: str,
    *,
    checkpoint_manifest: dict[str, Any] | None,
    isolated_subset: bool,
) -> KernelStore:
    """Open a store whose corpus scope matches the loaded training samples."""
    if checkpoint_manifest is not None:
        return KernelStore.fork_checkpoint(
            checkpoint_manifest,
            base / "_run" / run_id / "kernels",
        )
    if isolated_subset:
        return KernelStore(base / "_run" / run_id / "kernels").load()
    return KernelStore(base / "kernels").load()


def build_state(
    config_path: str | None = None,
    exec_config_path: str | None = None,
    *,
    limit: int | None = None,
    uid_manifest: str | None = None,
    resume_run_id: str | None = None,
    resume_checkpoint: str | None = None,
    data_path: str | None = None,
) -> TrainState:
    import datetime as _dt

    train_cfg = load_train_config(config_path) if config_path else TrainConfig()
    if data_path:
        train_cfg.data.datasets = [{"id": "triton_seed", "path": data_path}]
    cfg = load_exec_config(Path(exec_config_path) if exec_config_path else None)
    backend = build_backend(cfg)
    llm = _build_llm(train_cfg.propose)
    optimize_llm = _build_llm(train_cfg.optimize)

    uid_man = uid_manifest if uid_manifest is not None else (train_cfg.data.uid_manifest or "")
    lim = limit if limit is not None else train_cfg.data.limit
    uid_filter = load_uid_manifest(uid_man) if uid_man else None
    if not train_cfg.data.datasets:
        raise RuntimeError("no training data: set data.datasets (or pass --data) to a triton_seed corpus dir")
    samples = load_datasets(cfg, train_cfg.data.datasets, limit=lim, uids=uid_filter)
    source_desc: Any = [d.get("path") for d in train_cfg.data.datasets]
    if uid_filter is not None:
        source_desc = f"uid_manifest:{uid_man}"
    if not samples:
        raise RuntimeError(f"no samples loaded (source={source_desc})")

    train_set, holdout = split_holdout(samples, train_cfg)
    if resume_checkpoint and not resume_run_id:
        raise RuntimeError("--checkpoint requires --resume RUN_ID")
    # Historical resume is a fork. It must never append into or overwrite the
    # source run whose future history the user may still need.
    run_id = (
        _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if resume_checkpoint
        else resume_run_id or _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )

    base = Path(train_cfg.checkpoint.dir)
    tree = seed_tree(
        train_cfg.seed_metrics,
        sm_hi=train_cfg.bucketize.sm_hi,
        mem_hi=train_cfg.bucketize.mem_hi,
    )
    history: list[dict[str, Any]] = []
    restored_leaf_updates: dict[str, LeafUpdateCycle] = {}
    restored_epoch = 0
    restored_next_epoch = 0
    restored_bound_index = 0
    restored_value = {bk: 0.0 for bk in BOUND_KEYS}
    checkpoint_store_manifest: dict[str, Any] | None = None
    resumed_checkpoint_id = ""
    if resume_run_id:
        source_run_dir = base / "_run" / resume_run_id
        if resume_checkpoint:
            checkpoint_dir = _resolve_checkpoint_dir(
                source_run_dir, resume_checkpoint
            )
            run_dir = checkpoint_dir
            checkpoint_meta_path = checkpoint_dir / "checkpoint.json"
            if not checkpoint_meta_path.exists():
                raise RuntimeError(
                    f"resume checkpoint incomplete: missing {checkpoint_meta_path}"
                )
            checkpoint_meta = json.loads(
                checkpoint_meta_path.read_text(encoding="utf-8")
            )
            checkpoint_store_manifest = dict(
                checkpoint_meta.get("store") or {}
            )
            resumed_checkpoint_id = checkpoint_dir.name
        else:
            run_dir = source_run_dir
        tree_path = run_dir / "tree.json"
        if not tree_path.exists():
            raise RuntimeError(f"resume checkpoint incomplete: missing {tree_path}")
        tree = MetricTree.load(tree_path)
        meta_path = run_dir / "progress.json"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            history = list(raw.get("history") or [])
            restored_epoch = int(raw.get("epoch", 0))
            restored_next_epoch = int(
                raw.get("next_epoch", restored_epoch + 1)
            )
            restored_bound_index = int(raw.get("bound_index", 0))
            saved_value = dict(raw.get("value") or {})
            restored_value = {
                bk: float(saved_value.get(bk, 0.0)) for bk in BOUND_KEYS
            }
            restored_leaf_updates = {
                str(leaf_id): LeafUpdateCycle(
                    proposal_ids=list(rec.get("proposal_ids") or []),
                    operator=str(rec.get("operator") or "rewrite"),
                    base_leaf_version=int(rec.get("base_leaf_version", 0)),
                )
                for leaf_id, rec in (raw.get("leaf_updates") or {}).items()
            }
    tree.assert_invariants()

    # Historical checkpoint resumes always fork. So does any run on a SUBSET of the
    # corpus (--limit or a uid manifest): the shared store is keyed by the full pool,
    # and routing a subset against it would retire every kernel whose origin this run
    # did not load, silently emptying the shared pool for later full runs.
    store = _open_training_store(
        base,
        run_id,
        checkpoint_manifest=checkpoint_store_manifest,
        isolated_subset=uid_filter is not None or lim is not None,
    )

    state = TrainState(
        cfg=cfg, train_cfg=train_cfg, backend=backend, llm=llm,
        optimize_llm=optimize_llm, tree=tree, store=store,
        train_samples=train_set, holdout_samples=holdout,
        by_uid={s.uid: s for s in samples},
        value=restored_value,
        epoch=restored_epoch, next_epoch=restored_next_epoch,
        bound_index=restored_bound_index,
        run_id=run_id, history=history,
        leaf_updates=restored_leaf_updates,
    )
    _restore_history_wins(state)
    _repair_persisted_failure_states(state)
    _recover_leaf_updates(state)
    try:
        state.gpu_queue = SingleGPUQueue(name=f"npt-gpu-{run_id}")
        state.ncu_batcher = MicrobatchNCUProfiler(
            backend,
            cfg,
            gpu_queue=state.gpu_queue,
            window_s=float(train_cfg.schedule.ncu_batch_wait_ms) / 1000.0,
            max_batch_size=int(train_cfg.schedule.ncu_batch_size),
        )
        remote_verifier = RemoteVerifier(
            backend, cfg,
            warmup=train_cfg.training.warmup,
            repeats=train_cfg.training.repeats,
        )
        state.verifier = MicrobatchRemoteVerifier(
            remote_verifier,
            window_s=float(train_cfg.schedule.verify_batch_wait_ms) / 1000.0,
            max_batch_size=int(train_cfg.schedule.verify_batch_size),
            gpu_queue=state.gpu_queue,
            journal_path=state.run_dir() / "gpu_verify_jobs.jsonl",
        )
        state.run_dir().mkdir(parents=True, exist_ok=True)
        configure_logging(state.run_dir(), train_cfg)
    except Exception:
        _close_runtime(state)
        raise
    _log(
        f"[init] run_id={run_id} resume={bool(resume_run_id)} "
        f"checkpoint={resumed_checkpoint_id or 'latest'} "
        f"n_train={len(train_set)} n_holdout={len(holdout)} source={source_desc}\n"
        f"[init] propose_model={llm.config.model} "
        f"optimize_model={optimize_llm.config.model}\n"
        f"[init] artifacts -> {state.run_dir()}\n"
        f"[init] kernel store -> {store.root} ({store.stats()})"
    )
    return state
