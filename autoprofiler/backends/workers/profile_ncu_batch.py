"""Run multiple isolated profile_ncu jobs inside one remote exec."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from profile_ncu import _failure_scope, _run_group  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-script", required=True)
    ap.add_argument("--timeout-s", type=int, default=900)
    args = ap.parse_args()

    rows = json.loads(Path(args.bundle).read_text(encoding="utf-8")).get("rows") or []
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for index, row in enumerate(rows):
            uid = str(row.get("uid") or index)
            row_dir = work / f"row_{index}"
            row_dir.mkdir(parents=True, exist_ok=True)
            ref = row_dir / "reference.py"
            candidate = row_dir / "candidate.py"
            result = row_dir / "profile.json"
            ref.write_text(
                str(row.get("reference_code") or ""), encoding="utf-8"
            )
            candidate.write_text(
                str(row.get("candidate_code") or ""), encoding="utf-8"
            )
            cmd = [
                sys.executable,
                args.profile_script,
                "--ref", str(ref),
                "--candidate", str(candidate),
                "--metrics", ",".join(row.get("metrics") or []),
                "--out", str(result),
            ]
            try:
                # _run_group: on timeout OR exit, SIGKILL the child's whole
                # process group so an inner ncu/runner.py can never orphan and
                # keep the batch worker from advancing to the next row.
                proc = _run_group(
                    cmd, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=args.timeout_s,
                )
                if proc.returncode == 0 and result.exists():
                    payload = json.loads(result.read_text(encoding="utf-8"))
                else:
                    stderr = proc.stderr[-2000:]
                    payload = {
                        "ok": False, "metrics": {}, "kernel_count": 0,
                        "metadata": {
                            "error": "remote_failed",
                            "stdout": proc.stdout[-2000:],
                            "stderr": stderr,
                            "ncu_returncode": proc.returncode,
                            "failure_scope": _failure_scope(
                                proc.returncode, stderr
                            ),
                        },
                    }
            except subprocess.TimeoutExpired as exc:
                payload = {
                    "ok": False, "metrics": {}, "kernel_count": 0,
                    "metadata": {
                        "error": "batch_worker",
                        "failure_scope": "sample",
                        "stderr": f"TimeoutExpired: {exc}",
                    },
                }
            except Exception as exc:
                payload = {
                    "ok": False, "metrics": {}, "kernel_count": 0,
                    "metadata": {
                        "error": "batch_worker",
                        "failure_scope": "sample",
                        "stderr": f"{type(exc).__name__}: {exc}",
                    },
                }
            fh.write(json.dumps({"uid": uid, "payload": payload}) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
