"""The one base class every pluggable part (seeder, profiler, optimizer, method) derives from.

A component

* has a ``kind`` (set once by its interface class: ``"seeder"``, ``"profiler"``, ...),
* has a ``name`` (set by the concrete class; this is what configs refer to),
* declares its settings as a nested pydantic ``Params`` model, so a misspelt or
  mistyped option fails when the config is loaded, with the offending key named,
* registers itself when the class is defined; there is no separate registration call.

    class MyProfiler(Profiler):
        name = "my_profiler"

        class Params(Profiler.Params):
            window: int = 3

        def profile(self, task, cand, env): ...   # settings are at self.params.window
"""
from __future__ import annotations

import importlib
from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

_REGISTRY: dict[str, dict[str, type["Component"]]] = {}
_BUILTIN_PACKAGES = ("autoprofiler.components", "autoprofiler.methods")
_builtin_loaded = False


class Component(ABC):
    kind: ClassVar[str] = ""
    name: ClassVar[str] = ""
    # Set by the eval runner when a config runs one method under a different label
    # (an ablation arm, a component swap); defaults to ``name``.
    run_name: str = ""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    def __init__(self, **params: Any) -> None:
        self.params = self.Params(**params)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("name")  # only classes that name themselves are registered
        if not name:
            return
        if not cls.kind:
            raise TypeError(f"{cls.__name__} sets name={name!r} but derives from no interface with a kind")
        bucket = _REGISTRY.setdefault(cls.kind, {})
        if name in bucket and bucket[name].__module__ != cls.__module__:
            raise ValueError(f"{cls.kind} {name!r} is already registered by {bucket[name].__module__}")
        bucket[name] = cls

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params.model_dump()})"


def load_builtin() -> None:
    """Import the packages that define the shipped components (idempotent)."""
    global _builtin_loaded
    if _builtin_loaded:
        return
    _builtin_loaded = True
    for pkg in _BUILTIN_PACKAGES:
        importlib.import_module(pkg)


def get(kind: str, name: str) -> type[Component]:
    load_builtin()
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(bucket)}")
    return bucket[name]


def available(kind: str) -> list[str]:
    load_builtin()
    return sorted(_REGISTRY.get(kind, {}))


def build(kind: str, spec: str | dict) -> Component:
    """Instantiate a component from a config entry.

    ``spec`` is either a bare name (``"tree"``) or a mapping whose ``type`` key names
    the component and whose other keys are its settings
    (``{"type": "tree", "tree": "output/trees/mine.json"}``).
    """
    if isinstance(spec, str):
        return get(kind, spec)()
    params = dict(spec)
    name = params.pop("type", None)
    if not name:
        raise ValueError(f"{kind} spec needs a 'type' key, got {spec!r}")
    return get(kind, name)(**params)
