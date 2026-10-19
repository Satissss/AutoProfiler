from __future__ import annotations

from autoprofiler.core.env import Env
from autoprofiler.core.interfaces import Seeder
from autoprofiler.core.types import Candidate, Task


class TorchBaselineSeeder(Seeder):
    """Start from the original PyTorch program itself (``ModelNew = Model``)."""

    name = "torch_baseline"

    def seed(self, task: Task, env: Env) -> Candidate:
        return Candidate(code=task.ref_code.rstrip() + "\n\nModelNew = Model\n",
                         language="torch", origin=self.name)
