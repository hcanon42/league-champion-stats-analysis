"""Tests for the statistics engine and game labelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from league_stats.analysis.statistics import StatisticsEngine


def _synthetic_matches(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic match table where fewer deaths mean more wins.

    Args:
        n: Number of games.
        seed: RNG seed for reproducibility.

    Returns:
        A dataframe shaped like ``MatchRecord.to_row`` output (subset).
    """
    rng = np.random.default_rng(seed)
    deaths = rng.integers(0, 9, n)
    win = (deaths + rng.normal(0, 1.5, n) < 4).astype(int)
    gd15 = rng.normal(0, 1200, n).round()
    return pd.DataFrame(
        {
            "match_id": [f"M{i}" for i in range(n)],
            "win": win,
            "deaths": deaths,
            "deaths_pre20": np.minimum(deaths, 5),
            "deaths_pre14": np.minimum(deaths, 3),
            "control_wards": rng.integers(0, 6, n),
            "first_item_min": rng.normal(9.5, 1.2, n).round(2),
            "cs10": rng.normal(72, 8, n).round(),
            "gd10": rng.normal(0, 500, n).round(),
            "gd15": gd15,
            "xpd10": rng.normal(0, 400, n).round(),
            "csd10": rng.normal(0, 10, n).round(),
            "dpm": rng.normal(700, 120, n).round(),
            "vspm": rng.normal(1.3, 0.3, n).round(2),
            "avg_unspent_gold": rng.normal(800, 250, n).round(),
            "roams_pre15": rng.integers(0, 3, n),
            "lane_priority": rng.uniform(0.2, 0.9, n).round(2),
            "solo_deaths": rng.integers(0, 4, n),
            "kill_participation": rng.uniform(0.3, 0.8, n).round(2),
            "damage_share": rng.uniform(0.18, 0.35, n).round(3),
            "duration_min": rng.normal(30, 4, n).round(1),
        }
    )


@pytest.fixture()
def engine(tmp_path: Path) -> StatisticsEngine:
    """A statistics engine over the synthetic table."""
    return StatisticsEngine(_synthetic_matches(), tmp_path)


def test_correlation_matrix_contains_win(engine: StatisticsEngine) -> None:
    """The correlation matrix is square and includes the win column."""
    corr = engine.correlation_matrix()
    assert "win" in corr.columns
    assert corr.shape[0] == corr.shape[1]


def test_win_correlations_detect_deaths(engine: StatisticsEngine) -> None:
    """Deaths correlate negatively with winning in the synthetic data."""
    correlations = {c.feature: c.correlation for c in engine.win_correlations()}
    assert correlations["deaths_pre20"] < -0.3


def test_split_test_significance(engine: StatisticsEngine) -> None:
    """The win-rate split on early deaths is large and significant."""
    result = engine.winrate_split_test("deaths_pre20", 3)
    assert result is not None
    assert result["winrate_low"] > result["winrate_high"]
    assert result["p_value"] < 0.05


def test_random_forest_trains(engine: StatisticsEngine, tmp_path: Path) -> None:
    """The predictor trains, persists and ranks deaths as informative."""
    model = engine.train_win_predictor()
    assert model.trained is True
    assert (tmp_path / "win_predictor.joblib").exists()
    top_features = model.feature_importance["feature"].head(3).tolist()
    assert "deaths_pre14" in top_features


def test_clustering_produces_labels(engine: StatisticsEngine) -> None:
    """Every game receives a cluster and an archetype label."""
    clusters = engine.cluster_games()
    assert len(clusters) == 60
    assert clusters["cluster"].nunique() >= 2
    assert set(clusters["label"]).issubset(
        {
            "Lane stomp win", "Comeback win", "Scaling win", "Clean win",
            "Throw", "One-sided loss", "Close loss",
        }
    )
