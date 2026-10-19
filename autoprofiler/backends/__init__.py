from autoprofiler.backends.base import Backend, ExecResult
from autoprofiler.backends.gpu import GPUQueueMetrics, SingleGPUQueue, gpu_flock, gpu_lock_path
from autoprofiler.config import ExecConfig


def build_backend(cfg: ExecConfig) -> Backend:
    """Instantiate the backend named in the config (``local`` or ``k8s``)."""
    if cfg.backend == "k8s":
        from autoprofiler.backends.k8s import K8sExecBackend

        return K8sExecBackend(cfg)
    from autoprofiler.backends.local import LocalBackend

    return LocalBackend(cfg)


__all__ = [
    "Backend",
    "ExecResult",
    "GPUQueueMetrics",
    "SingleGPUQueue",
    "build_backend",
    "gpu_flock",
    "gpu_lock_path",
]
