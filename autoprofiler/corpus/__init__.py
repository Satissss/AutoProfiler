"""Torch -> Triton seed generation (one-off pre-experiment).

Turns the raw torch corpus in ``data/source/data.parquet`` into a frozen,
correctness-verified Triton seed set for training the profiling tree. KernelBench stays a pure
test set and is never touched here.
"""

from autoprofiler.corpus.dataset import SeedRow, load_seed_rows
from autoprofiler.corpus.pipeline import SeedgenConfig, SeedPipeline

__all__ = ["SeedRow", "load_seed_rows", "SeedgenConfig", "SeedPipeline"]
