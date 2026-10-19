"""End-to-end check of runner -> method -> components -> report with a fake LLM and evaluator.

Needs no GPU and no network, so it runs in CI.
"""
import json
from pathlib import Path

import pytest

from autoprofiler.core.benchmark import Benchmark
from autoprofiler.core.env import Env
from autoprofiler.core.types import EvalResult
from autoprofiler.eval.runner import build_method
from autoprofiler.eval import EvalConfig, run_eval
from autoprofiler.eval import runner as runner_mod
from autoprofiler.llm.usage import Usage


class FakeLLM:
    class config:          # the runner logs which model a run used
        model = "fake-model"

    def __init__(self):
        self.usage = Usage()

    def chat(self, messages, **kw):
        self.usage.add(10, 5)
        if "performance engineer" in messages[0]["content"] or "auditor" in messages[0]["content"]:
            return '```json\n{"bottleneck": "memory", "optimisation method": "vectorise loads", "modification plan": "x"}\n```'
        return "```python\nclass ModelNew:\n    pass\n```"


class FakeEvaluator:
    """Each successive evaluation is a little faster, so 'best' tracking is observable."""

    def __init__(self):
        self.n_evals, self.gpu_seconds = 0, 0.0

    def evaluate(self, task, cand, tag="", final=False):
        self.n_evals += 1
        return EvalResult(ok=True, stage="ok", baseline_ms=2.0, candidate_ms=2.0 / (1 + 0.1 * self.n_evals),
                          speedup=1 + 0.1 * self.n_evals)


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    def build_env(**kw):
        return Env(FakeLLM(), FakeEvaluator(), backend=None, exec_cfg=None, work_dir=tmp_path, rounds=kw["rounds"])

    monkeypatch.setattr(runner_mod, "build_env", build_env)
    monkeypatch.setattr("autoprofiler.backends.ncu.remote_ncu_profile",
                        lambda *a, **k: {"ok": True, "metrics": {"sm__cycles_active.avg": 1.0}, "csv": ""})


def test_kernelbench_has_250_tasks():
    tasks = Benchmark.load("kernelbench").tasks()
    assert len(tasks) == 250 and tasks[0].id == "kb_l1_1" and tasks[-1].id == "kb_l3_50"


@pytest.mark.parametrize("name,parts", [
    ("autoprofiler", ("LLMPortSeeder", "TreeProfiler", "LeafOptimizer")),
    ("cudaforge", ("LLMCudaSeeder", "NCUFixedProfiler", "JudgeCoder")),
    ("kernelmem", ("LLMCudaSeeder", "NCUFixedProfiler", "MemoryJudgeCoder")),
    ("kernelagent", ("LLMTritonSeeder", "NCUFixedProfiler", "JudgeCoder")),
    ("autokernel", ("LLMTritonSeeder", "NCUFixedProfiler", "EditLoop")),
])
def test_a_method_name_is_enough_to_build_it(name, parts):
    """`methods: {cudaforge: {}}` in an experiment file builds the published method."""
    m = build_method(name, {"method": name})
    assert (type(m.seeder).__name__, type(m.profiler).__name__,
            type(m.optimizer).__name__) == parts


def test_run_eval_two_methods(fake_env, tmp_path):
    cfg = EvalConfig(name="t", groups=["level1"], limit=2, methods=["cudaforge", "kernelmem"],
                     rounds=2, output=str(tmp_path / "res"))
    out = run_eval(cfg, log=lambda m: None)
    summary = json.loads((out / "summary.json").read_text())
    for m in ("cudaforge", "kernelmem"):
        s = summary["methods"][m]["all"]
        assert s["n"] == 2 and s["correct"] == 1.0 and s["fast_1"] == 1.0
        rec = json.loads((out / m / "tasks" / "kb_l1_1.json").read_text())
        # seed + 2 rounds, one candidate each, all counted against the same LLM
        assert len(rec["trajectory"]["rounds"]) == 3 and rec["trajectory"]["usage"]["llm"]["calls"] > 0
        assert (out / m / "code" / "kb_l1_1.py").exists()
    assert (out / "results.md").read_text().startswith("# t")


def test_resume_skips_finished_tasks(fake_env, tmp_path):
    cfg = EvalConfig(name="r", groups=["level1"], limit=1, methods=["cudaforge"], rounds=1,
                     output=str(tmp_path / "res"))
    out = run_eval(cfg, log=lambda m: None)
    first = (out / "cudaforge" / "tasks" / "kb_l1_1.json").read_text()
    run_eval(cfg, log=lambda m: None)  # fake evaluator would give different numbers if it re-ran
    assert (out / "cudaforge" / "tasks" / "kb_l1_1.json").read_text() == first


def test_empty_round_does_not_end_the_task(tmp_path):
    from autoprofiler.core import Task
    from autoprofiler.core.interfaces import Optimizer, Profiler
    from autoprofiler.core.component import _REGISTRY
    from autoprofiler.core.types import Profile

    def register(kind, name):
        def deco(cls): _REGISTRY.setdefault(kind, {})[name] = cls; return cls
        return deco


    @register("profiler", "t_empty")
    class P(Profiler):
        def profile(self, task, cand, env):
            return Profile()

    @register("optimizer", "t_empty")
    class O(Optimizer):
        calls = 0

        def propose(self, task, history, profile, env):
            O.calls += 1
            return []  # e.g. the LLM produced nothing usable

    env = Env(FakeLLM(), FakeEvaluator(), backend=None, exec_cfg=None, work_dir=tmp_path, rounds=3)
    from autoprofiler.methods.iterative import ConfigurableIterative
    m = ConfigurableIterative(seeder="torch_baseline", profiler="t_empty", optimizer="t_empty")
    traj = m.run(Task("t", "b", "g", "class Model: pass"), env)
    assert O.calls == 3 and len(traj.steps) == 1  # all rounds attempted; only the seed was evaluated


def test_shipped_experiments_are_self_contained():
    """Each experiment file alone says what to run, on what, with which model and budget."""
    from autoprofiler.eval.config import load_eval_config

    for path in sorted(Path("configs/eval").glob("*.yaml")):
        cfg = load_eval_config(path)
        assert cfg.methods, f"{name}: no methods"
        assert cfg.rounds >= 0 and cfg.benchmark and cfg.llm and cfg.backend
        for run_name, spec in cfg.method_specs().items():
            assert build_method(run_name, spec).run_name == run_name


def test_an_experiment_overrides_a_method_knob():
    """A knob named in the experiment file wins over the method's default."""
    default = build_method("autoprofiler", {"method": "autoprofiler"})
    tuned = build_method("autoprofiler", {"method": "autoprofiler", "n_samples": 7})
    assert default.optimizer.params.n_samples == 2
    assert tuned.optimizer.params.n_samples == 7


def test_base_runs_one_method_twice_under_two_names():
    """An ablation arm: same method, one component swapped, two rows in the table."""
    plain = build_method("cudaforge", {"method": "cudaforge"})
    swapped = build_method("cudaforge_with_tree", {
        "method": "cudaforge",
        "profiler": {"type": "tree", "path": "output/trees/autoprofiler.json"}})
    assert plain.name == swapped.name == "cudaforge"          # one method
    assert plain.run_name != swapped.run_name                  # two rows
    assert type(plain.profiler).__name__ == "NCUFixedProfiler"
    assert type(swapped.profiler).__name__ == "TreeProfiler"   # the swap took


def test_start_kernel_replaces_the_seeder(tmp_path):
    """Handed a kernel, a method optimizes it instead of generating its own first draft."""
    from autoprofiler.core.types import Candidate, Task

    class RefusingSeeder:
        """Fails loudly if the method ever falls back to seeding."""
        def seed(self, task, env):
            raise AssertionError("seeder ran even though a start kernel was given")

    m = build_method("cudaforge", {"method": "cudaforge", "profiler": "none"})
    m.seeder = RefusingSeeder()
    env = Env(FakeLLM(), FakeEvaluator(), backend=None, exec_cfg=None, work_dir=tmp_path, rounds=1)
    mine = Candidate("# my own kernel", "cuda", origin="caller")

    traj = m.run(Task("t", "b", "g", "class Model: pass"), env, start=mine)
    assert traj.steps[0].candidate is mine        # round 0 is the kernel we passed in
    assert len(traj.steps) > 1                    # and it went on to optimize it


def test_a_hand_built_env_caches_nothing(tmp_path):
    """Only a real run caches. Tests and demos build an Env directly, and must not.

    They did: 245 `class ModelNew: pass` drafts from a fake LLM, each stamped correct
    by a fake evaluator, sat in output/_cache/seeds until real runs loaded them and
    failed scoring with "ModelNew.forward() takes 2 positional arguments".
    """
    from autoprofiler.core.types import Task
    from autoprofiler.components.seeders.base import LLMSeeder

    env = Env(FakeLLM(), FakeEvaluator(), backend=None, exec_cfg=None, work_dir=tmp_path, rounds=0)
    assert env.cache_dir is None
    seeder = LLMSeeder.__subclasses__()[0](attempts=1)
    assert seeder._cache_path(Task("t", "b", "g", "class Model: pass"), env) is None


def test_a_real_run_caches_under_its_cache_dir(tmp_path):
    from autoprofiler.core.types import Task
    from autoprofiler.components.seeders.base import LLMSeeder
    from dataclasses import replace

    env = replace(Env(FakeLLM(), FakeEvaluator(), backend=None, exec_cfg=None,
                      work_dir=tmp_path, rounds=0), cache_dir=tmp_path / "cache")
    seeder = LLMSeeder.__subclasses__()[0](attempts=1)
    path = seeder._cache_path(Task("t", "b", "g", "class Model: pass"), env)
    assert path is not None and tmp_path in path.parents
