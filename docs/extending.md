# Extending

## A new profiler (or seeder / optimizer)

Subclass the interface, set `name`, and declare a `Params` model:

```python
# autoprofiler/components/profilers/my_profiler.py
from autoprofiler.core.interfaces import Profiler
from autoprofiler.core.env import Env
from autoprofiler.core.types import Candidate, Profile, Task


class MyProfiler(Profiler):
    """doc"""

    name = "my_profiler"  # registered automatically when the class is defined

    class Params(Profiler.Params):
        window: int = 3  # every key is validated against this schema

    def profile(self, task: Task, cand: Candidate, env: Env) -> Profile:
        # env.backend / env.exec_cfg to run GPU work; env.llm is the shared LLM.
        return Profile(metrics={...}, text="what the optimizer will read", info={...})
```

Register it by importing in the package `__init__`:

```python
# autoprofiler/components/profilers/__init__.py
from autoprofiler.components.profilers.my_profiler import MyProfiler  # noqa: F401
```

Then reference it in a method class:

```python
class MyMethod(IterativeMethod):
    name = "my_method"
    def make_profiler(self): return MyProfiler(window=5)
    ...
```

Settings (like `window`) are passed to the constructor; pydantic validates and rejects unknown keys.

## A new method

Subclass `IterativeMethod` and implement `make_seeder` / `make_profiler` / `make_optimizer`.
The loop (seed → N rounds of profile/propose/evaluate) is inherited:

```python
class MyMethod(IterativeMethod):
    name = "my_method"

    class Params(Method.Params):
        gpu: str = "RTX-4090"
        temperature: float = 0.7

    def make_seeder(self) -> Seeder:   return LLMPortSeeder()
    def make_profiler(self) -> Profiler: return NCUFixedProfiler(metric_set="roofline")
    def make_optimizer(self) -> Optimizer: return JudgeCoder(gpu=self.params.gpu,
                                                              temperature=self.params.temperature)
```

Add the method config:

```yaml
# in any experiment config under configs/
methods:
  my_method:
    gpu: A100      # a Params knob; an unknown key is a validation error
```

For a completely different loop, subclass `Method` directly and implement `run(task, env) -> Trajectory`.

## A new benchmark

1. Add task files to `data/benchmarks/<name>/<group>/*.py`. Each must define `Model`,
   `get_inputs()`, and optionally `get_init_inputs()`.
2. Add `configs/benchmarks/<name>.yaml`:

```yaml
name: mybench
root: data/benchmarks/mybench
id_prefix: mb
groups:
  easy: {dir: easy, tag: e}
```

3. Reference it in an eval config (`benchmark: mybench`). No code changes needed.

## A new LLM backbone

Copy `configs/llm/openai_compatible.example.yaml`, fill in `model`, `base_url`,
`api_style` (`openai` or `anthropic`), and `api_key_env`, then use its name in any
`llm:` field in an eval or train config.

## A new GPU backend

Subclass `autoprofiler.backends.base.Backend` (three methods: `run`, `put_bytes`,
`get_text`), add a branch to `build_backend`, and point `backend:` in a config at a
yaml that specifies it.
