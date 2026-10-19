"""Config-driven IterativeMethod: seeder/profiler/optimizer resolved by type name at instantiation."""
from __future__ import annotations

from autoprofiler.core.component import get
from autoprofiler.core.interfaces import Optimizer, Profiler, Seeder
from autoprofiler.core.iterative import IterativeMethod
from autoprofiler.core.interfaces import Method


def _build(kind: str, spec) -> object:
    if isinstance(spec, str):
        return get(kind, spec)()
    spec = dict(spec)
    name = spec.pop("type")
    return get(kind, name)(**spec)


class ConfigurableIterative(IterativeMethod):
    """Instantiate by specifying seeder/profiler/optimizer in the config yaml.

    This is the ``method: iterative`` entry that every baseline config uses.
    """

    name = "iterative"

    class Params(IterativeMethod.Params):
        seeder: dict | str = "torch_baseline"
        profiler: dict | str = "none"
        optimizer: dict | str = "judge_coder"

    # from_config lets the test/shim call IterativeMethod.from_config({...})
    @classmethod
    def from_config(cls, cfg: dict) -> "ConfigurableIterative":
        params = {k: v for k, v in cfg.items() if k not in ("method", "name")}
        return cls(**params)

    def make_seeder(self) -> Seeder:
        return _build("seeder", self.params.seeder)

    def make_profiler(self) -> Profiler:
        return _build("profiler", self.params.profiler)

    def make_optimizer(self) -> Optimizer:
        return _build("optimizer", self.params.optimizer)
