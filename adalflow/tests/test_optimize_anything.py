from adalflow.optim.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    optimize_anything,
    log,
)


def test_optimize_anything_respects_max_metric_calls():
    calls = {"count": 0}

    def evaluator(candidate: str) -> float:
        calls["count"] += 1
        log(f"len={len(candidate)}")
        return float(candidate.count("good"))

    result = optimize_anything(
        seed_candidate="good",
        evaluator=evaluator,
        objective="increase 'good' count",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=5, random_seed=1)),
    )

    assert result.metric_calls == 5
    assert calls["count"] == 5
    assert len(result.history) == 5


def test_optimize_anything_returns_seed_when_no_improvement():
    def evaluator(candidate: str) -> float:
        log("constant score")
        return 0.5

    seed = "artifact"
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective="no-op",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=4, random_seed=2)),
    )

    assert result.best_candidate == seed
    assert result.best_score == 0.5


def test_optimize_anything_improves_on_toy_objective():
    def evaluator(candidate: str):
        # shorter candidate is better, but still uses score max convention
        score = 1.0 / (1 + len(candidate))
        log(f"candidate={candidate[:20]}")
        return {"score": score}

    result = optimize_anything(
        seed_candidate="this is a very long seed candidate",
        evaluator=evaluator,
        objective="minimize length",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=12, random_seed=7),
            population_size=6,
            elite_size=2,
            mutation_rate=0.9,
            crossover_rate=0.0,
        ),
    )

    assert result.best_score >= result.history[0].score
    assert result.pareto_frontier


def test_optimize_anything_is_exported_from_top_level():
    import adalflow as adal

    assert hasattr(adal, "optimize_anything")
    assert hasattr(adal, "GEPAConfig")
    assert hasattr(adal, "EngineConfig")