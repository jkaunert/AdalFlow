# optimize_anything benchmark (GEPA-style scaffold)

This benchmark provides a reproducible scaffold for `adalflow.optim.optimize_anything` across
three artifact categories:

1. text/prompt artifact
2. code artifact
3. config/SVG-like artifact

Each run reports:
- quality score (maximize)
- token_cost (minimize)
- latency_ms (minimize)

## Run

```bash
python benchmarks/optimize_anything/gepa_parity_benchmark.py
```

The script prints JSON containing baseline (seed) and optimized metrics for each case.