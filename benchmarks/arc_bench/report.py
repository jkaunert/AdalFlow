"""Generate a Markdown report for ARC-Bench smoke benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(input_dir: Path) -> List[Dict]:
    return [load_json(path) for path in sorted(input_dir.glob("*/metrics.json"))]


def _format_optional(value, fmt: str = ".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:{fmt}}"
    return str(value)


def render_report(input_dir: Path) -> str:
    summary = load_json(input_dir / "summary.json")
    rows = collect_metrics(input_dir)
    has_evolution_metrics = any(row.get("best_score") is not None for row in rows)

    lines = [
        "# AdalFlow ARC-Bench Smoke Benchmark Report",
        "",
        "> This is a 5-topic smoke benchmark for the AdalFlow autonomous research demo only. "
        "It is not a verified comparison against AutoResearchClaw until both systems are run "
        "under the same model, hardware, timeout, budget, and scoring rubric.",
        "",
        "## Summary",
        "",
        f"- System: `{summary.get('system', 'unknown')}`",
        f"- Mode: `{summary.get('mode', 'unknown')}`",
        f"- Topics: {summary.get('topics', 0)}",
        f"- Completed: {summary.get('completed', 0)}",
        f"- Completion rate: {summary.get('completion_rate', 0):.2%}",
        f"- Mean stage completion rate: {summary.get('mean_stage_completion_rate', 0):.2%}",
        f"- Mean retry count: {summary.get('mean_retry_count', 0):.2f}",
        f"- Mean wall time: {summary.get('mean_wall_time_sec', 0):.4f}s",
        f"- Experiment success rate: {summary.get('experiment_success_rate', 0):.2%}",
        f"- Paper generated rate: {summary.get('paper_generated_rate', 0):.2%}",
        f"- Mean artifact completeness: {summary.get('mean_artifact_completeness', 0):.2%}",
    ]

    if has_evolution_metrics:
        lines.extend(
            [
                f"- Mean initial score: {summary.get('mean_initial_score', 0):.4f}",
                f"- Mean best score: {summary.get('mean_best_score', 0):.4f}",
                f"- Mean score improvement: {summary.get('mean_score_improvement', 0):.4f}",
                f"- Mean candidates explored: {summary.get('mean_num_candidates', 0):.2f}",
                f"- Mean convergence iteration: {summary.get('mean_convergence_iteration', 0):.2f}",
                f"- Mean archive size: {summary.get('mean_archive_size', 0):.2f}",
            ]
        )

    lines.extend(["", "## Per-topic Results", ""])

    if has_evolution_metrics:
        lines.extend(
            [
                "| Topic ID | Domain | Completed | Best Score | Improvement | Candidates | Retries | Converged At | Wall Time (s) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {topic_id} | {domain} | {completed} | {best_score} | {improvement} | "
                "{candidates} | {retry_count} | {converged} | {wall_time_sec:.4f} |".format(
                    topic_id=row["topic_id"],
                    domain=row["domain"],
                    completed=row["completed"],
                    best_score=_format_optional(row.get("best_score")),
                    improvement=_format_optional(row.get("score_improvement")),
                    candidates=_format_optional(row.get("num_candidates"), ".0f"),
                    retry_count=row["retry_count"],
                    converged=_format_optional(row.get("convergence_iteration"), ".0f"),
                    wall_time_sec=row["wall_time_sec"],
                )
            )
    else:
        lines.extend(
            [
                "| Topic ID | Domain | Completed | Stage Completion | Retries | Wall Time (s) | Experiment | Paper |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {topic_id} | {domain} | {completed} | {stage_completion_rate:.2%} | "
                "{retry_count} | {wall_time_sec:.4f} | {experiment_success} | {paper_generated} |".format(**row)
            )

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Add a second runner for AutoResearchClaw and execute both systems over the same topic file "
            "to produce a real baseline-vs-candidate comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/arc_bench/adalflow_evolution_smoke"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/arc_bench/adalflow_evolution_smoke_report.md"))
    args = parser.parse_args()

    report = render_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
