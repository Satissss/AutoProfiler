from __future__ import annotations

from autoprofiler.codegen.port import propose_port
from autoprofiler.codegen.port_prompts import PROMPT_VERSION
from autoprofiler.components.seeders.base import LLMSeeder
from autoprofiler.core.env import Env
from autoprofiler.core.types import Task


class LLMTritonSeeder(LLMSeeder):
    """Port the PyTorch reference to Triton (the seed AutoProfiler's tree is trained on)."""

    name = "llm_triton"
    language = "triton"
    prompt_version = PROMPT_VERSION

    def draft(self, task: Task, env: Env, *, prev_code: str, error: str, stage: str) -> str:
        pc = propose_port(task.ref_code, client=env.llm, temperature=self.params.temperature,
                          prev_code=prev_code or None, error=error or None, stage=stage)
        return pc.code
