"""GEPA-style benchmark scaffold for optimize_anything.

Tracks:
1) prompt/text artifact
2) code artifact
3) config/svg-like artifact

Metrics:
- quality score (maximize)
- token_cost (minimize)
- latency_ms (minimize)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

# Ensure benchmark resolves local source package from this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adalflow"))

from adalflow import EngineConfig, GEPAConfig, optimize_anything, log


def _run_case(name: str, seed_candidate: str, target_keyword: str) -> Dict[str, float]:
    def evaluator(candidate: str):
        quality = 1.0 if target_keyword in candidate else 0.2
        quality += min(candidate.count(target_keyword), 3) * 0.1
        quality -= 0.01 * len(candidate.split())
        log(f"target={target_keyword}, contains={target_keyword in candidate}")
        return {"score": quality}

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        objective=f"Increase quality for {name} while reducing verbosity.",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=30, random_seed=42),
            population_size=10,
            elite_size=3,
            mutation_rate=0.8,
            crossover_rate=0.2,
        ),
    )

    baseline = result.history[0]
    return {
        "baseline_score": baseline.score,
        "best_score": result.best_score,
        "baseline_token_cost": baseline.token_cost,
        "best_token_cost": min(item.token_cost for item in result.pareto_frontier),
        "baseline_latency_ms": baseline.latency_ms,
        "best_latency_ms": min(item.latency_ms for item in result.pareto_frontier),
    }


def run_benchmark() -> Dict[str, Dict[str, float]]:
    return {
        "text_artifact": _run_case(
            "text_artifact",
            "You are an assistant. Provide detailed answer.",
            "concise",
        ),
        "code_artifact": _run_case(
            "code_artifact",
            "def add(a,b): return a+b",
            "type hints",
        ),
        "config_svg_artifact": _run_case(
            "config_svg_artifact",
            "<svg><circle r='5'/></svg>",
            "viewBox",
        ),
    }


if __name__ == "__main__":
    import json

    results = run_benchmark()
    print(json.dumps(results, indent=2))