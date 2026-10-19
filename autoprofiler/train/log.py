"""Training log: console plus an optional train.log file."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from autoprofiler.train.config import TrainConfig


_LOG_FH = None


def configure_logging(out_dir: Path, train_cfg: TrainConfig) -> None:
    global _LOG_FH
    if not train_cfg.logging.to_file:
        return
    log_dir = Path(train_cfg.logging.dir) if train_cfg.logging.dir else out_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except Exception:
            pass
    _LOG_FH = open(log_dir / "train.log", "a", encoding="utf-8")
    _log(f"[log] run log -> {log_dir / 'train.log'}")


def _log(*args, **kwargs):
    import datetime as _dt

    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    if _LOG_FH is not None:
        try:
            stamp = _dt.datetime.now().strftime("%H:%M:%S")
            _LOG_FH.write(f"{stamp} {' '.join(str(a) for a in args)}\n")
            _LOG_FH.flush()
        except Exception:
            pass


@contextmanager
def _nonblocking_executor(max_workers: int):
    """Do not make a finished update wait for obsolete in-flight requests."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
