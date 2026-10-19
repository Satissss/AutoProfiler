"""AutoProfiler: Triton seed + learned profiling tree + leaf optimizer.

TreeProfiler (registered as "tree") and LeafOptimizer (registered as "leaf")
are also available as standalone components for experiments.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from autoprofiler.components.seeders.llm_port import LLMPortSeeder
from autoprofiler.config import REPO_ROOT
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Method, Optimizer, Profiler, Seeder
from autoprofiler.core.iterative import IterativeMethod
from autoprofiler.core.types import Candidate, Profile, Step, Task


class TreeProfiler(Profiler):
    """AutoProfiler's learned profiler: walks root→leaf, measuring only the NCU
    counters each node asks for, and returns the union of metrics as a Profile."""

    name = "tree"

    class Params(Profiler.Params):
        path: str = "output/trees/autoprofiler.json"   # the trained tree
        train_config: str = "configs/train/default.yaml"  # for the NCU retry budget

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._tree = None
        self._train_cfg = None

    def _load(self):
        if self._tree is None:
            from autoprofiler.train.config import load_train_config
            from autoprofiler.tree.structure import MetricTree

            tp = Path(self.params.path)
            tp = tp if tp.is_absolute() else REPO_ROOT / tp
            if not tp.exists():
                raise FileNotFoundError(f"trained tree not found: {tp}")
            self._tree = MetricTree.load(str(tp))
            cp = Path(self.params.train_config)
            cp = cp if cp.is_absolute() or cp.exists() else REPO_ROOT / cp
            if not cp.exists():
                raise FileNotFoundError(f"training config not found: {cp}")
            self._train_cfg = load_train_config(str(cp))
        return self._tree, self._train_cfg

    def profile(self, task: Task, cand: Candidate, env: Env) -> Profile:
        from autoprofiler.tree.route import collect_route, make_profile_fn
        from autoprofiler.tree.store import KernelStore
        from autoprofiler.tree.leaf_prompts import format_measurements

        tree, train_cfg = self._load()
        store = KernelStore(env.work_dir / "tree_store" / task.id)
        state = store.add_seed(task.id, cand.code)
        profile_fn = make_profile_fn(
            env.backend, env.exec_cfg,
            max_retries=train_cfg.schedule.ncu_max_retries,
        )
        route = collect_route(tree, store, state, profile_fn, task.ref_code, log=env.log)
        return Profile(
            metrics=dict(route.path_metrics),
            text=format_measurements(route.path_metrics),
            info={"leaf_id": route.leaf.node_id, "bound": route.leaf.bound,
                  "node_path": list(route.node_path), "profile_ms": route.profile_ms},
        )


class LeafOptimizer(Optimizer):
    """Leaf-level Triton optimizer: one LLM call per sample, up to max_repair repairs."""

    name = "leaf"

    class Params(Optimizer.Params):
        strong: bool = True
        n_samples: int = 1
        max_repair: int = 2
        temperature: float = 0.4
        max_tokens: int = 16384

    def parent(self, history: list[Step]) -> Step | None:
        from autoprofiler.core.types import best_step
        return best_step(history)

    def propose(self, task: Task, history: list[Step], profile: Profile, env: Env) -> list[Candidate]:
        from autoprofiler.tree.leaf_prompts import (
            OPTIMIZE_SYSTEM, OPTIMIZE_SYSTEM_V2,
            build_optimize_user, build_optimize_repair,
        )
        from autoprofiler.tree.leaf import solver_seed
        from autoprofiler.tree.store import kernel_id_for
        from autoprofiler.codegen.port import PortRejected, check_candidate, extract_code
        from autoprofiler.codegen.port_prompts import repair_hint
        from autoprofiler.core.types import Candidate, best_step

        parent = best_step(history)
        if parent is None:
            return []
        system = OPTIMIZE_SYSTEM_V2 if self.params.strong else OPTIMIZE_SYSTEM
        out: list[Candidate] = []
        for si in range(self.params.n_samples):
            salt = f"#s{si}" if self.params.n_samples > 1 else ""
            seed = solver_seed(kernel_id_for(parent.candidate.code), salt)
            code = parent.candidate.code
            prev_code = prev_err = prev_stage = ""
            cand = None
            for rnd in range(self.params.max_repair + 1):
                user = (build_optimize_repair(prev_code or code, profile.metrics, prev_err, prev_stage,
                                              hint=repair_hint(prev_err, prev_stage))
                        if prev_err else build_optimize_user(code, profile.metrics,
                                                             parent.result.candidate_ms, seed=seed))
                tag = f"  leaf {si + 1}/{self.params.n_samples}"
                if rnd:
                    tag += f" repair {rnd}"
                try:
                    with env.step(tag) as done:
                        raw = env.llm.chat(
                            [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
                        )
                        _, new_code = extract_code(raw)
                        check_candidate(new_code)
                        done(f"{len(new_code)} chars")
                except PortRejected as e:
                    new_code = locals().get("new_code", "")
                    prev_code, prev_err, prev_stage = new_code, e.reason, e.stage
                    continue
                cand = Candidate(new_code, parent.candidate.language, "leaf", {"sample": si, "repair": rnd})
                with env.step(f"{tag} verify") as done:
                    cand.result = env.evaluator.evaluate(task, cand, tag=f"{task.id}_leaf{si}_{rnd}")
                    r = cand.result
                    done(f"{r.speedup:.2f}x" if r.ok and r.speedup else f"{r.stage}: {r.error[:60]}")
                if cand.result.ok:
                    break
                prev_code, prev_err, prev_stage = new_code, cand.result.error or cand.result.stage, cand.result.stage
            if cand is not None:
                out.append(cand)
        return out


class AutoProfiler(IterativeMethod):
    """The full AutoProfiler method used in evaluations."""

    name = "autoprofiler"

    class Params(IterativeMethod.Params):
        tree: str = "output/trees/autoprofiler.json"  # the trained tree to route with
        n_samples: int = 2                            # best-of-N at the leaf

    def make_seeder(self) -> Seeder:
        return LLMPortSeeder(language="triton", attempts=4)

    def make_profiler(self) -> Profiler:
        return TreeProfiler(path=self.params.tree)

    def make_optimizer(self) -> Optimizer:
        return LeafOptimizer(strong=True, n_samples=self.params.n_samples, max_repair=2)
