from __future__ import annotations

from autoprofiler.codegen.port import PortRejected
from autoprofiler.codegen.text import extract_python_block
from autoprofiler.components.gpu_specs import gpu_block
from autoprofiler.components.seeders.base import LLMSeeder
from autoprofiler.core.env import Env
from autoprofiler.core.types import Task

SYSTEM = (
    "You are a senior CUDA-kernel optimisation specialist. Generate a high-quality, compilable, "
    "runnable Python script that builds and launches hand-written CUDA kernels. "
    "Output the complete code inside one ```python block."
)

OUTPUT_RULES = """\
OUTPUT RULES (STRICT)
1. Inside the code block, in this order: imports (`torch`, `torch.nn`, `load_inline`); `source` (CUDA
   kernel + host wrapper as a triple-quoted string); `cpp_src` (prototypes of everything exposed); one
   `load_inline` call per kernel group; `class ModelNew(nn.Module)`.
2. No testing code, no `if __name__ == "__main__"`, no prose outside the block.
3. Use `c10::cuda::getCurrentCUDAStream()` (include <c10/cuda/CUDAStream.h>)."""

FIRST = """\
# Target GPU
{gpu}

Task: write hand-written CUDA kernels that replace the PyTorch operator(s) inside the original
`class Model` below. You may fuse operators if that is faster. Leave parts you do not replace unchanged.
Name the new class `ModelNew`; it must take the same constructor arguments and produce the same
outputs as `Model`.

{rules}

Original architecture:
```python
{arch}
```
"""

REPAIR = """\
# Target GPU
{gpu}

The CUDA implementation below failed at stage `{stage}`. Fix it so it compiles, runs and matches the
reference; keep the public API.

{rules}

Original architecture:
```python
{arch}
```

Failing implementation:
```python
{code}
```

Error log:
{error}
"""


class LLMCudaSeeder(LLMSeeder):
    """Ask the LLM for a CUDA (``load_inline``) implementation."""

    name = "llm_cuda"
    language = "cuda"
    prompt_version = "cuda_v1"

    class Params(LLMSeeder.Params):
        gpu: str = "RTX-4090"
        temperature: float = 1.0

    def draft(self, task: Task, env: Env, *, prev_code: str, error: str, stage: str) -> str:
        gpu, arch = gpu_block(self.params.gpu), task.ref_code.strip()
        user = (REPAIR.format(gpu=gpu, rules=OUTPUT_RULES, arch=arch, code=prev_code, stage=stage, error=error[-4000:])
                if prev_code and error else FIRST.format(gpu=gpu, rules=OUTPUT_RULES, arch=arch))
        raw = env.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                           temperature=self.params.temperature, max_tokens=self.params.max_tokens)
        try:
            return extract_python_block(raw)
        except (ValueError, SyntaxError) as e:
            raise PortRejected(f"unusable reply: {e}") from e
