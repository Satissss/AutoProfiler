"""Execution-backend configuration (where GPU work runs).

``backend: local`` runs everything on this machine's GPU; ``backend: k8s`` ships
the same worker scripts to a pod via ``kubectl exec``. Everything downstream only
sees the :class:`ExecConfig` properties ``remote_root`` and ``python``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
# Worker scripts (stdlib + torch only) that run next to the GPU.
REMOTE_DIR = PACKAGE_DIR / "backends" / "workers"
# Path of the worker directory relative to a backend's working root.
REMOTE_REL = "autoprofiler/backends/workers"


class LocalBackendConfig(BaseModel):
    python: str = ""  # empty -> the interpreter running this process
    workdir: str = "output/_jobs"  # job files + worker scripts are staged here

    def python_path(self) -> str:
        return self.python or sys.executable

    def workdir_path(self, repo_root: Path = REPO_ROOT) -> Path:
        p = Path(self.workdir).expanduser()
        return p if p.is_absolute() else (repo_root / p).resolve()


class K8sBackendConfig(BaseModel):
    kubeconfig: str = ""  # path to a kubeconfig; empty -> kubectl's default
    namespace: str = "default"
    deployment: str = ""  # deployment (and pod label ``app=``) holding the GPU
    conda_env: str = ""  # conda env activated before running workers (optional)
    conda_sh: str = "/root/miniconda3/etc/profile.d/conda.sh"
    remote_workdir: str = "/root/autoprofiler"
    python: str = "python"

    def kubeconfig_path(self, repo_root: Path = REPO_ROOT) -> Path | None:
        if not self.kubeconfig:
            return None
        p = Path(self.kubeconfig).expanduser()
        return p if p.is_absolute() else repo_root / p


class ExecConfig(BaseModel):
    backend: Literal["local", "k8s"] = "local"
    local: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    k8s: K8sBackendConfig = Field(default_factory=K8sBackendConfig)

    @property
    def remote_root(self) -> str:
        """Working root, as seen by the machine that runs the workers."""
        if self.backend == "k8s":
            return self.k8s.remote_workdir.rstrip("/")
        return str(self.local.workdir_path()).rstrip("/")

    @property
    def python(self) -> str:
        return self.k8s.python if self.backend == "k8s" else self.local.python_path()


def load_exec_config(path: Path | str | None = None) -> ExecConfig:
    """Load a backend config yaml; a missing path yields the local default."""
    data: dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ExecConfig.model_validate(data)
