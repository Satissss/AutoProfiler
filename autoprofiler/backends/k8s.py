from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from autoprofiler.backends.base import Backend, ExecResult
from autoprofiler.config import REPO_ROOT, ExecConfig


class K8sExecBackend(Backend):
    """Remote execution via ``kubectl exec`` into a GPU pod (optional plugin)."""

    def __init__(self, exec_cfg: ExecConfig, repo_root: Path = REPO_ROOT):
        self.cfg = exec_cfg.k8s
        self.root = exec_cfg.remote_root
        self.repo_root = repo_root
        self.kubeconfig = self.cfg.kubeconfig_path(repo_root)
        if self.kubeconfig is not None and not self.kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {self.kubeconfig}")
        self._shipped: dict[str, str] = {}

    @property
    def python_bin(self) -> str:
        return self.cfg.python

    def script(self, name: str) -> str:
        if name not in self._shipped:
            self._shipped[name] = self._ship_script(name)
        return self._shipped[name]

    def _kubectl_base(self) -> list[str]:
        cmd = ["kubectl"]
        if self.kubeconfig is not None:
            cmd += ["--kubeconfig", str(self.kubeconfig)]
        return cmd + ["-n", self.cfg.namespace]

    def get_pod_name(self) -> str:
        cmd = self._kubectl_base() + [
            "get",
            "pod",
            "-l",
            f"app={self.cfg.deployment}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        try:
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.STDOUT, timeout=60
            )
        except subprocess.TimeoutExpired as e:
            # No timeout here used to hang the whole run forever when the pod was
            # unschedulable or the apiserver was unreachable. Fail fast instead.
            raise RuntimeError(
                f"kubectl get pod timed out after 60s for app={self.cfg.deployment} "
                f"(apiserver unreachable or pod not scheduled)"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"kubectl get pod failed for app={self.cfg.deployment}: "
                f"{(e.output or '').strip()[:500]}"
            ) from e
        name = out.strip()
        if not name:
            raise RuntimeError(f"No running pod for app={self.cfg.deployment}")
        return name

    def run(
        self,
        remote_command: str,
        *,
        use_env: bool = True,
        timeout_s: int = 600,
        retries: int = 3,
    ) -> ExecResult:
        if use_env and self.cfg.conda_env:
            wrapped = (
                f"source {shlex.quote(self.cfg.conda_sh)} && "
                f"conda activate {shlex.quote(self.cfg.conda_env)} && "
                f"{remote_command}"
            )
        else:
            wrapped = remote_command

        # Put the interpreter's own bin directory on PATH. Tools that live beside it
        # are otherwise invisible: torch shells out to `ninja` by name to build a
        # load_inline extension, so without this every CUDA kernel fails to compile
        # with "Ninja is required to load C++ extensions" while ninja sits unused
        # next to the python we are running.
        bin_dir = str(Path(self.cfg.python).parent)
        if bin_dir not in ("", "."):
            wrapped = f"export PATH={shlex.quote(bin_dir)}:$PATH && {wrapped}"

        # Pin this exec to one physical GPU when the caller's process is bound to a
        # device (parallel eval, one task per GPU). GPU work runs in the pod via
        # kubectl exec, so the pin must be exported INSIDE the remote shell, not in
        # this local process. Empty/unset => pod default (all GPUs), i.e. unchanged
        # serial behaviour.
        # Prefer a thread-local binding (in-process parallel harness), else the
        # process env (subprocess-based eval worker). Empty/unset => pod default.
        from autoprofiler.backends.gpu import thread_gpu

        _tg = thread_gpu()
        _gpu = str(_tg) if _tg is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
        if _gpu not in (None, ""):
            wrapped = f"export CUDA_VISIBLE_DEVICES={shlex.quote(_gpu)} && {wrapped}"

        cmd = self._kubectl_base() + [
            "exec",
            f"deploy/{self.cfg.deployment}",
            "--",
            "bash",
            "-lc",
            wrapped,
        ]
        last: ExecResult | None = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    encoding="utf-8",
                    errors="replace",
                )
                last = ExecResult(
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    elapsed_s=time.time() - t0,
                )
                # retry only transport-ish failures
                transport_fail = (
                    proc.returncode != 0
                    and ("i/o timeout" in (proc.stderr or "") or "error sending request" in (proc.stderr or ""))
                )
                if not transport_fail:
                    return last
            except subprocess.TimeoutExpired as e:
                last = ExecResult(
                    returncode=124,
                    stdout=(e.stdout or b"").decode("utf-8", errors="replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or ""),
                    stderr=f"local_timeout: {e}",
                    elapsed_s=time.time() - t0,
                )
            time.sleep(2 + attempt * 2)
        assert last is not None
        return last

    def put_bytes(self, remote_path: str, data: bytes, *, retries: int = 4) -> None:
        """Write bytes to the pod via kubectl exec stdin (reliable on Windows)."""
        parent = str(Path(remote_path).parent).replace("\\", "/")
        last_err = ""
        for attempt in range(retries):
            try:
                self.run(f"mkdir -p {shlex.quote(parent)}", use_env=False, timeout_s=60)
                pod = self.get_pod_name()
                cmd = self._kubectl_base() + [
                    "exec",
                    "-i",
                    pod,
                    "--",
                    "bash",
                    "-lc",
                    f"cat > {shlex.quote(remote_path)}",
                ]
                proc = subprocess.run(
                    cmd,
                    input=data,
                    capture_output=True,
                    timeout=300,
                )
                if proc.returncode == 0:
                    return
                last_err = proc.stderr.decode("utf-8", errors="replace") or proc.stdout.decode(
                    "utf-8", errors="replace"
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
                last_err = str(e)
            time.sleep(2 + attempt * 2)
        raise RuntimeError(f"put_bytes failed after {retries} retries: {last_err}")

    def get_text(self, remote_path: str) -> str:
        res = self.run(f"cat {shlex.quote(remote_path)}", use_env=False, timeout_s=120)
        if not res.ok:
            raise RuntimeError(f"get_text failed: {res.stderr or res.stdout}")
        return res.stdout

    def mkdir(self, remote_path: str) -> None:
        self.run(f"mkdir -p {shlex.quote(remote_path)}", use_env=False, timeout_s=60)
