"""Component-level checks that need no GPU: the learned tree router and the leaf optimizer."""
from pathlib import Path
from unittest import mock

from autoprofiler.core import Candidate, EvalResult, Profile, Step, Task
from autoprofiler.core.env import Env
from autoprofiler.core.component import get as _get, load_builtin as _lb

def build(kind, name_or_spec):
    _lb()
    if isinstance(name_or_spec, str): return _get(kind, name_or_spec)()
    spec = dict(name_or_spec); name = spec.pop('type', spec.pop('name', ''))
    return _get(kind, name)(**spec)

from autoprofiler.llm.usage import Usage

TASK = Task("kb_l1_1", "kernelbench", "level1", "class Model: pass")
TRITON = '''import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * 2.0, mask=m)

class ModelNew(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        _k[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
        return y
'''


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.usage, self.calls = list(replies), Usage(), 0

    def chat(self, messages, **kw):
        self.calls += 1
        self.usage.add(1, 1)
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


class FakeEvaluator:
    n_evals, gpu_seconds = 0, 0.0

    def evaluate(self, task, cand, tag="", final=False):
        self.n_evals += 1
        return EvalResult(ok=True, stage="ok", baseline_ms=2.0, candidate_ms=1.0, speedup=2.0)


def _history():
    seed = Candidate("seed", "triton")
    return [Step(0, seed, EvalResult(ok=True, stage="ok", baseline_ms=2.0, candidate_ms=2.0, speedup=1.0))]


def test_tree_profiler_routes_with_released_tree(tmp_path):
    asked = []

    def fake_make(backend, cfg, *, max_retries=2, **kw):
        def fn(kernel, metrics, ref):
            asked.append(list(metrics))
            return {m: 70.0 for m in metrics}, 1.0
        return fn

    prof = build("profiler", {"type": "tree", "path": "output/trees/autoprofiler.json"})
    env = Env(None, None, None, None, tmp_path)
    with mock.patch("autoprofiler.tree.route.make_profile_fn", fake_make):
        p = prof.profile(TASK, Candidate("x", "triton"), env)
    assert p.info["node_path"][0] == "root" and p.info["leaf_id"] == p.info["node_path"][-1]
    assert len(asked) == len(p.info["node_path"]) and p.metrics and p.text


def test_leaf_optimizer_returns_evaluated_candidates(tmp_path):
    llm = ScriptedLLM([f"reasoning\n```python\n{TRITON}\n```"])
    env = Env(llm, FakeEvaluator(), None, None, tmp_path)
    opt = build("optimizer", {"type": "leaf", "n_samples": 2})
    out = opt.propose(TASK, _history(), Profile(metrics={"sm__throughput": 70.0}), env)
    assert len(out) == 2 and all(c.result is not None and c.result.ok for c in out)
    assert llm.calls == 2  # one draft per sample, no repair needed


def test_leaf_optimizer_survives_unusable_llm_output(tmp_path):
    llm = ScriptedLLM(["I cannot do that."])
    env = Env(llm, FakeEvaluator(), None, None, tmp_path)
    opt = build("optimizer", {"type": "leaf", "max_repair": 2})
    assert opt.propose(TASK, _history(), Profile(), env) == []
    assert llm.calls == 3  # initial draft + 2 repair turns, then give up


def test_forbidden_triton_symbols_are_really_absent():
    """A symbol on the blacklist must not exist in the Triton we target.

    Listing a real symbol is worse than listing none: the model writes correct code,
    check_candidate rejects it unread, and the repair budget is spent re-deriving it.
    `tl.abs` sat on this list while being a perfectly good Triton 3.x builtin.
    """
    from autoprofiler.codegen.port import FORBIDDEN_TL

    # Triton 3.x surface, as reported by `triton.language` on the GPU host.
    real = {"abs", "maximum", "minimum", "where", "exp", "log", "sqrt", "sigmoid",
            "erf", "cos", "sin", "floor", "ceil", "fma", "clamp", "rsqrt", "softmax"}
    wrongly_banned = [s for s in FORBIDDEN_TL
                      if s.startswith("tl.") and s.count(".") == 1 and s[3:] in real]
    assert not wrongly_banned, f"these exist in Triton but are blacklisted: {wrongly_banned}"


def test_client_drops_a_rejected_parameter_and_retries():
    """A model that rejects one sampling parameter must not crash every call with it.

    DashScope's kimi-k3 answers 400 InvalidParameter for any non-default temperature.
    Hard-failing on that would mean nobody can point autoprofiler at such a model.
    """
    import json
    import urllib.error
    from unittest import mock

    from autoprofiler.llm.client import LLMClient
    from autoprofiler.llm.config import LLMConfig

    client = LLMClient(LLMConfig(api_key="k", model="kimi-k3", temperature=0.2))
    reject = urllib.error.HTTPError(
        "url", 400, "Bad Request", {},
        mock.Mock(read=lambda: json.dumps(
            {"error": {"message": "Parameter 'temperature'=0.2 is not supported for kimi-k3 model."}}
        ).encode()),
    )
    ok_body = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1:
            raise reject
        return mock.Mock(__enter__=lambda s: mock.Mock(read=lambda: ok_body), __exit__=lambda *a: None)

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        text = client.chat([{"role": "user", "content": "hi"}], retries=1)

    assert text == "ok"
    assert "temperature" in calls[0] and "temperature" not in calls[1]


def test_default_timeout_tolerates_a_slow_reasoning_model():
    """180s was the default; a reasoning model's genuine thinking time looks identical
    to a hung connection at that budget, and 8 retries turned one slow call into a
    30-minute one. The shipped profiles already needed 600s -- make it the default."""
    from autoprofiler.llm.config import LLMConfig

    assert LLMConfig(api_key="k").timeout_s >= 600


def test_two_evaluators_never_produce_the_same_remote_job_name():
    """Two Evaluators evaluating the same task must not collide on a remote path.

    Each Evaluator's uid counter starts at 0, so without a process-unique prefix,
    two methods run as separate processes against the same pod (the natural way to
    run several methods side by side) would generate the identical uid on their Nth
    call each -- the same remote job directory, one process's write landing mid-read
    of another's.
    """
    from autoprofiler.core.evaluator import Evaluator
    from autoprofiler.core.types import Candidate, Task

    e1, e2 = Evaluator(backend=None, cfg=None), Evaluator(backend=None, cfg=None)
    task = Task("kb_l1_1", "kernelbench", "level1", "class Model: pass")
    cand = Candidate("class ModelNew: pass")
    seen = []
    e1.verify_batch = lambda requests, tag="", final=False: seen.append(requests[0].uid) or {requests[0].uid: _OkResult()}
    e2.verify_batch = e1.verify_batch
    e1.evaluate(task, cand)
    e2.evaluate(task, cand)
    assert seen[0] != seen[1]


class _OkResult:
    ok = True
    stage = "ok"
    error = ""
    baseline_ms = 2.0
    candidate_ms = 1.0
    speedup = 2.0
    correct = True


def test_cuda_workers_give_every_attempt_its_own_torch_extension_cache():
    """A stale load_inline .so must never survive into the next attempt.

    torch keys its build cache on the extension name alone, and the model reuses
    the same name (e.g. "bmm_cuda") across repair rounds. A round whose code
    genuinely changed then reloaded the PREVIOUS round's compiled binary and failed
    with "undefined symbol", because torch saw the name as already built and never
    recompiled. Both places that exec a CUDA candidate must point
    TORCH_EXTENSIONS_DIR at a fresh directory for that one attempt.
    """
    import re

    for path in ("autoprofiler/backends/workers/seed_verify.py",
                 "autoprofiler/backends/workers/profile_ncu.py"):
        src = open(path).read()
        assert "TORCH_EXTENSIONS_DIR" in src, f"{path}: no isolated extension cache"
        # it must be set from something unique to this call, not a fixed path
        assert re.search(r'TemporaryDirectory|tempfile\.mkdtemp|td_path', src), \
            f"{path}: TORCH_EXTENSIONS_DIR is not tied to a fresh directory"
