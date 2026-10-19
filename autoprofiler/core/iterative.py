from __future__ import annotations

from abc import abstractmethod

from autoprofiler.metering import metering, charge_llm, charge_gpu
from autoprofiler.core.component import build
from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Method, Optimizer, Profiler, Seeder
from autoprofiler.core.types import Candidate, Profile, Step, Task, Trajectory


def _snap(meter, env=None) -> dict:
    """Snapshot merging the metering context and (if available) the LLM client's own counter.

    The metering context tracks calls made through the real LLMClient (which calls
    charge_llm). Tests use a FakeLLM that updates llm.usage directly without going
    through charge_llm; merging both sources makes the snapshot correct in both cases.
    """
    s = meter.snapshot()
    llm_u = getattr(getattr(env, "llm", None), "usage", None) if env is not None else None
    if llm_u is not None:
        lu = llm_u.snapshot() if hasattr(llm_u, "snapshot") else {}
        return {
            "llm": {
                "calls": max(s["llm_calls"], lu.get("llm_calls", lu.get("calls", 0))),
                "total_tokens": max(s["total_tokens"], lu.get("total_tokens", 0)),
                "prompt_tokens": max(s["prompt_tokens"], lu.get("prompt_tokens", 0)),
                "completion_tokens": max(s["completion_tokens"], lu.get("completion_tokens", 0)),
            },
            "gpu_seconds": s["gpu_seconds"],
            "n_evals": s["n_evals"],
        }
    return {
        "llm": {"calls": s["llm_calls"], "total_tokens": s["total_tokens"],
                "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"]},
        "gpu_seconds": s["gpu_seconds"],
        "n_evals": s["n_evals"],
    }







class IterativeMethod(Method):
    """seed -> repeat(profile the parent, propose, evaluate) up to ``env.rounds`` times.

    A concrete method names its three parts by implementing ``make_seeder`` /
    ``make_profiler`` / ``make_optimizer``; that is all that distinguishes CudaForge from
    AutoProfiler here. ``n`` candidates proposed in one round count as one round
    (best-of-N); their LLM calls and GPU time are still charged to that round.

    Any of the three can be replaced from the config, which is how a component is swapped
    for an experiment -- giving CudaForge AutoProfiler's tree, say, without new code.
    The shipped method configs spell the defaults out, so what a method is made of is
    readable without opening the class.
    """

    class Params(Method.Params):
        # None = the method's own default (``make_*`` below). A name or a
        # ``{type: ..., **settings}`` mapping replaces it.
        seeder: str | dict | None = None
        profiler: str | dict | None = None
        optimizer: str | dict | None = None

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.seeder: Seeder = self._part("seeder", self.make_seeder)
        self.profiler: Profiler = self._part("profiler", self.make_profiler)
        self.optimizer: Optimizer = self._part("optimizer", self.make_optimizer)

    def _part(self, kind: str, default):
        spec = getattr(self.params, kind, None)
        return build(kind, spec) if spec else default()

    @abstractmethod
    def make_seeder(self) -> Seeder: ...

    @abstractmethod
    def make_profiler(self) -> Profiler: ...

    @abstractmethod
    def make_optimizer(self) -> Optimizer: ...

    def run(self, task: Task, env: Env, start: Candidate | None = None) -> Trajectory:
        """Optimize ``task``, from ``start`` if given, else from this method's own seed."""
        traj = Trajectory(task_id=task.id, method=getattr(self, "run_name", self.name))
        with metering() as total:
            with metering() as spent:
                seed = start if start is not None else self.seeder.seed(task, env)
                if seed is None:
                    traj.error = "seeder produced no candidate"
                    env.log("  seed: none produced, skipping task")
                else:
                    if seed.result is None:
                        with env.step("  seed verify") as done:
                            seed.result = env.evaluator.evaluate(task, seed, tag=f"{task.id}_r0")
                            r = seed.result
                            done(f"{r.speedup:.2f}x" if r.ok and r.speedup else f"{r.stage}: {r.error[:60]}")
                    traj.record(Step(0, seed, seed.result, usage=_snap(spent, env)))
            if seed is not None:
                for rnd in range(1, env.rounds + 1):
                    self._round(task, env, traj, rnd)
        traj.usage = _snap(total, env)
        return traj

    def _round(self, task: Task, env: Env, traj: Trajectory, rnd: int) -> None:
        with metering() as spent:
            parent = self.optimizer.parent(traj.steps)
            profile = self.profiler.profile(task, parent.candidate, env) if parent else Profile()
            proposals = self.optimizer.propose(task, traj.steps, profile, env)
            if not proposals:
                env.log("  optimizer proposed nothing")  # the round is still spent
            for i, cand in enumerate(proposals):
                if cand.result:
                    continue  # the optimizer already measured it and logged the verdict
                with env.step(f"  verify {i + 1}/{len(proposals)}") as done:
                    cand.result = env.evaluator.evaluate(task, cand, tag=f"{task.id}_r{rnd}_{i}")
                    r = cand.result
                    done(f"{r.speedup:.2f}x" if r.ok and r.speedup else f"{r.stage}: {r.error[:60]}")
            for cand in proposals:
                traj.record(Step(rnd, cand, cand.result, profile=profile))
        if proposals:
            traj.steps[-1].usage = _snap(spent, env)  # the round's cost, attached once
        best = traj.best_result.speedup if traj.best_result else None
        env.log(f"  round {rnd}/{env.rounds}: best {best:.2f}x" if best
                else f"  round {rnd}/{env.rounds}: nothing correct yet")
