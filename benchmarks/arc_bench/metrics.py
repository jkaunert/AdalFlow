"""Shared metrics for ARC-Bench smoke benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class SmokeMetrics:
    """Normalized per-topic benchmark metrics.

    These metrics intentionally measure only the local smoke benchmark behavior.
    They are not comparable to AutoResearchClaw's published ARC-Bench results
    until both systems are run under the same environment and scoring rubric.
    """

    system: str
    topic_id: str
    domain: str
    topic: str
    completed: bool
    stages_total: int
    stages_completed: int
    stage_completion_rate: float
    retry_count: int
    wall_time_sec: float
    experiment_success: bool
    paper_generated: bool
    artifact_completeness: float
    best_score: Optional[float] = None
    initial_score: Optional[float] = None
    score_improvement: Optional[float] = None
    num_candidates: Optional[int] = None
    convergence_iteration: Optional[int] = None
    archive_size: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_smoke_metrics(
    *,
    system: str,
    topic_record: Dict[str, str],
    result: Optional[Dict[str, Any]],
    wall_time_sec: float,
    error: Optional[str] = None,
) -> SmokeMetrics:
    """Convert a pipeline result into normalized smoke benchmark metrics."""

    completed = error is None and bool(result)
    experiment_metrics = result.get("experiment_metrics", {}) if result else {}
    output_status = result.get("output_status", "") if result else ""

    experiment_success = bool(experiment_metrics)
    paper_generated = output_status.startswith("Successfully")

    is_evolution_loop = "best_score" in experiment_metrics
    if is_evolution_loop:
        stages_total = 6
        stages_completed = sum(
            [
                completed,
                bool(result and result.get("hypothesis")),
                bool(result and result.get("evolution_history")),
                experiment_success,
                bool(experiment_metrics.get("num_candidates", 0)),
                paper_generated,
            ]
        )
        retry_count = int(experiment_metrics.get("retry_count", 0))
        artifacts = [
            bool(result and result.get("hypothesis")),
            bool(result and result.get("best_candidate")),
            bool(result and result.get("evolution_history")),
            experiment_success,
            paper_generated,
        ]
    else:
        # The simple demo models four coarse stage groups:
        # literature, hypothesis/HITL, code execution/repair, paper status.
        stages_total = 4
        stages_completed = sum(
            [
                completed,
                bool(result and result.get("hypothesis")),
                experiment_success,
                paper_generated,
            ]
        )
        # The simple demo intentionally triggers one failed sandbox attempt and repairs it.
        retry_count = 1 if experiment_success else 0
        artifacts = [
            bool(result and result.get("hypothesis")),
            experiment_success,
            paper_generated,
        ]

    artifact_completeness = sum(artifacts) / len(artifacts)

    return SmokeMetrics(
        system=system,
        topic_id=topic_record["topic_id"],
        domain=topic_record["domain"],
        topic=topic_record["topic"],
        completed=completed and stages_completed == stages_total,
        stages_total=stages_total,
        stages_completed=stages_completed,
        stage_completion_rate=stages_completed / stages_total,
        retry_count=retry_count,
        wall_time_sec=wall_time_sec,
        experiment_success=experiment_success,
        paper_generated=paper_generated,
        artifact_completeness=artifact_completeness,
        best_score=experiment_metrics.get("best_score"),
        initial_score=experiment_metrics.get("initial_score"),
        score_improvement=experiment_metrics.get("score_improvement"),
        num_candidates=experiment_metrics.get("num_candidates"),
        convergence_iteration=experiment_metrics.get("convergence_iteration"),
        archive_size=experiment_metrics.get("archive_size"),
        error=error,
    )
