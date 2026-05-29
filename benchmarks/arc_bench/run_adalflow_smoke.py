"""Run a 5-topic ARC-Bench-style smoke benchmark for AdalFlow research demos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.arc_bench.metrics import build_smoke_metrics
from use_cases.multi_agent_dag.auto_research_claw_demo import AutoResearchPipeline
from use_cases.multi_agent_dag.open_evolve_research_demo import OpenEvolveResearchLoop


PipelineMode = Literal["simple", "evolution"]


def load_topics(path: Path) -> List[Dict[str, str]]:
    topics: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            topics.append(json.loads(line))
    return topics


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_pipeline(
    mode: PipelineMode,
    max_iterations: int,
    *,
    use_llm: bool,
    provider: str,
    model_kwargs: Dict,
):
    if mode == "simple":
        return "adalflow-auto-research-demo", AutoResearchPipeline(hitl_mode="auto")
    if mode == "evolution":
        system = (
            "adalflow-open-evolve-research-demo-llm"
            if use_llm
            else "adalflow-open-evolve-research-demo"
        )
        return system, OpenEvolveResearchLoop(
            max_iterations=max_iterations,
            use_llm=use_llm,
            provider=provider,
            model_kwargs=model_kwargs,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def summarize(metrics: Iterable[Dict], *, system: str, mode: PipelineMode, use_llm: bool) -> Dict:
    rows = list(metrics)
    count = len(rows)
    if count == 0:
        return {"system": system, "mode": mode, "use_llm": use_llm, "topics": 0}

    summary = {
        "system": system,
        "mode": mode,
        "use_llm": use_llm,
        "topics": count,
        "completed": sum(1 for row in rows if row["completed"]),
        "completion_rate": sum(1 for row in rows if row["completed"]) / count,
        "mean_stage_completion_rate": sum(row["stage_completion_rate"] for row in rows) / count,
        "mean_retry_count": sum(row["retry_count"] for row in rows) / count,
        "mean_wall_time_sec": sum(row["wall_time_sec"] for row in rows) / count,
        "experiment_success_rate": sum(1 for row in rows if row["experiment_success"]) / count,
        "paper_generated_rate": sum(1 for row in rows if row["paper_generated"]) / count,
        "mean_artifact_completeness": sum(row["artifact_completeness"] for row in rows) / count,
    }

    if any(row.get("best_score") is not None for row in rows):
        summary.update(
            {
                "mean_best_score": sum(row.get("best_score") or 0 for row in rows) / count,
                "mean_initial_score": sum(row.get("initial_score") or 0 for row in rows) / count,
                "mean_score_improvement": sum(row.get("score_improvement") or 0 for row in rows) / count,
                "mean_num_candidates": sum(row.get("num_candidates") or 0 for row in rows) / count,
                "mean_convergence_iteration": sum(
                    row.get("convergence_iteration") or 0 for row in rows
                )
                / count,
                "mean_archive_size": sum(row.get("archive_size") or 0 for row in rows) / count,
            }
        )

    return summary


def run(
    topics_path: Path,
    output_dir: Path,
    *,
    mode: PipelineMode = "evolution",
    max_iterations: int = 5,
    use_llm: bool = False,
    provider: str = "openai",
    model_kwargs: Dict | None = None,
) -> Dict:
    topics = load_topics(topics_path)
    model_kwargs = model_kwargs or {"model": "gpt-4o-mini", "temperature": 0.3}
    system, pipeline = build_pipeline(
        mode,
        max_iterations,
        use_llm=use_llm,
        provider=provider,
        model_kwargs=model_kwargs,
    )
    all_metrics: List[Dict] = []

    for topic_record in topics:
        topic_dir = output_dir / topic_record["topic_id"]
        start = time.perf_counter()
        result = None
        error = None

        try:
            result = pipeline(topic_record["topic"])
        except Exception as exc:  # pragma: no cover - benchmark safety path
            error = f"{type(exc).__name__}: {exc}"

        wall_time_sec = time.perf_counter() - start
        metrics = build_smoke_metrics(
            system=system,
            topic_record=topic_record,
            result=result,
            wall_time_sec=wall_time_sec,
            error=error,
        ).to_dict()

        write_json(topic_dir / "result.json", result or {})
        write_json(topic_dir / "metrics.json", metrics)
        all_metrics.append(metrics)

    summary = summarize(all_metrics, system=system, mode=mode, use_llm=use_llm)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=Path("benchmarks/arc_bench/topics_smoke.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/arc_bench/adalflow_evolution_smoke"))
    parser.add_argument("--mode", choices=["simple", "evolution"], default="evolution")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--provider", choices=["openai", "groq", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    summary = run(
        args.topics,
        args.output,
        mode=args.mode,
        max_iterations=args.max_iterations,
        use_llm=args.use_llm,
        provider=args.provider,
        model_kwargs={"model": args.model, "temperature": args.temperature},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
