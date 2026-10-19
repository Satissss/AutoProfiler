# data/

```
benchmarks/kernelbench/level{1,2,3}/   KernelBench problems (250). Test set only.
train/example/seed_index.jsonl          100 training programs, in git: enough to smoke-test training.
train/full/seed_index.jsonl             all 999 training programs. NOT in git, download it (see top-level README).
source/data.parquet                     raw PyTorch programs the corpus was built from. NOT in git; only needed to rebuild it.
```

## Training corpus format

One JSON object per line of `seed_index.jsonl`:

| field | meaning |
|---|---|
| `uid` | unique id, `syn_<row>_<hash>` (synthetic programs, disjoint from KernelBench) |
| `orig_code` | the original PyTorch program (`Model`, `get_inputs`, `get_init_inputs`) |
| `code` | a verified Triton port defining `ModelNew` |
| `seed_baseline_ms`, `seed_triton_ms`, `seed_speedup_vs_torch` | measured runtimes of the PyTorch program and the port |
| `model`, `prompt_version`, `rounds_used` | provenance of the port |

## Adding a benchmark

See `docs/extending.md`: drop problem files under `data/benchmarks/<name>/` and add `configs/benchmarks/<name>.yaml`.
