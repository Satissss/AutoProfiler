"""llm_port: port the PyTorch reference to the requested language (default: Triton).

Registered as "llm_port" so method configs can write ``seeder: {type: llm_port, language: triton}``.
"""
from __future__ import annotations

from autoprofiler.components.seeders.base import LLMSeeder
from autoprofiler.codegen.port import propose_port
from autoprofiler.codegen.port_prompts import PROMPT_VERSION
from autoprofiler.core.env import Env
from autoprofiler.core.types import Task


class LLMPortSeeder(LLMSeeder):
    """Port the PyTorch reference to Triton (or another language in the future).

    Registered as both ``llm_port`` (the config name) and distinguishable from
    ``llm_triton`` by the ``language`` param so it can be extended later.
    """

    name = "llm_port"
    language = "triton"
    prompt_version = PROMPT_VERSION

    class Params(LLMSeeder.Params):
        language: str = "triton"  # only triton is currently supported
        max_repair: int = 3

    def __init__(self, **params) -> None:
        params.setdefault("attempts", params.pop("max_repair", 3) + 1)
        super().__init__(**params)

    def draft(self, task: Task, env: Env, *, prev_code: str, error: str, stage: str) -> str:
        pc = propose_port(task.ref_code, client=env.llm, temperature=self.params.temperature,
                          prev_code=prev_code or None, error=error or None, stage=stage)
        return pc.code
