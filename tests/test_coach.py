"""Tests for the coach engine's rule evaluation and ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.coach import (
    CoachEngine,
    VISIBLE_RECOMMENDATIONS,
    recommendations_markdown,
)
from analysis.matchups import matchups_dataframe
from analysis.statistics import StatisticsEngine
from tests.test_statistics import _synthetic_matches


@pytest.fixture()
def coach(tmp_path: Path) -> CoachEngine:
    """A coach over synthetic data with a strong early-deaths signal."""
    matches = _synthetic_matches()
    matches["opponent"] = (["Syndra"] * 20 + ["Orianna"] * 20 + ["Akali"] * 20)
    # Make Akali a clear counter and Orianna a clear win.
    matches.loc[matches["opponent"] == "Akali", "win"] = 0
    matches.loc[matches["opponent"] == "Orianna", "win"] = 1
    matches["gd10"] = np.where(matches["win"] == 1, 650, -450)
    matches["kill_participation"] = np.where(matches["win"] == 1, 0.78, 0.32)
    matches["deaths_before_dragon"] = np.where(matches["win"] == 0, 2, 0)
    matches["deaths_before_baron"] = np.where(matches["win"] == 0, 1, 0)
    matches["greed_deaths"] = np.where(matches["win"] == 0, 3, 0)
    matches["shutdown_given"] = np.where(matches["win"] == 0, 400, 50)
    matches["tf_participation"] = np.where(matches["win"] == 1, 0.8, 0.4)
    matches.loc[matches["gd15"] < 0, "gd15"] = 900
    matches.loc[matches["win"] == 0, "gd15"] = 900
    deaths = pd.DataFrame(
        {
            "match_id": ["M0"] * 6,
            "win": [0, 0, 0, 0, 1, 0],
            "minute": [23.0, 25.0, 12.0, 30.0, 18.0, 27.0],
            "side_lane_push": [True, True, False, True, False, True],
            "alone": [True, True, False, True, False, True],
            "team_wards_recent": [0, 1, 2, 0, 1, 0],
            "zone": ["bot", "top", "mid", "bot", "mid", "bot"],
            "shutdown_given": [300, 0, 0, 450, 0, 250],
        }
    )
    objectives = pd.DataFrame(
        {
            "match_id": ["M0"] * 14,
            "win": [0] * 8 + [1] * 6,
            "kind": ["dragon"] * 8 + ["baron"] * 6,
            "present": [False] * 10 + [True] * 4,
            "dead_before": [True] * 5 + [False] * 9,
        }
    )
    stats = StatisticsEngine(matches, tmp_path)
    return CoachEngine(
        matches_df=matches,
        deaths_df=deaths,
        matchups_df=matchups_dataframe(matches),
        objectives_df=objectives,
        stats_engine=stats,
    )


def test_recommendations_generated_and_sorted(coach: CoachEngine) -> None:
    """Recommendations exist and are sorted by descending priority."""
    recommendations = coach.generate()
    assert recommendations
    priorities = [r.priority for r in recommendations]
    assert priorities == sorted(priorities, reverse=True)


def test_matchup_rules_fire(coach: CoachEngine) -> None:
    """Best (Orianna) and worst (Akali) matchups are both reported."""
    titles = " | ".join(r.title for r in coach.generate())
    assert "Orianna" in titles
    assert "Akali" in titles


def test_new_rules_fire(coach: CoachEngine) -> None:
    """New coaching rules surface on the synthetic dataset."""
    titles = " | ".join(r.title for r in coach.generate())
    assert "Your personal win conditions" in titles
    assert "Greed deaths are a recurring pattern" in titles
    assert "You give away too many shutdown bounties" in titles
    assert "You throw leads you build in lane" in titles
    assert "Low teamfight participation is limiting your impact" in titles
    assert "You're dead too often right before objectives" in titles
    assert "Deaths right before epic monsters are throwing objectives" in titles


def test_merged_objective_death_rule_not_dragon_only(coach: CoachEngine) -> None:
    """Pre-objective deaths cover dragon and baron together."""
    rec = next(r for r in coach.generate() if "epic monsters" in r.title)
    assert "dragon or baron" in rec.detail
    assert "pre-dragon" in rec.evidence
    assert "pre-baron" in rec.evidence


def test_markdown_rendering(coach: CoachEngine) -> None:
    """The Markdown export lists every recommendation with evidence."""
    recommendations = coach.generate()
    markdown = recommendations_markdown(recommendations)
    assert markdown.startswith("# Viktor Mid Coaching Recommendations")
    assert "Evidence:" in markdown


def test_visible_recommendation_limit_constant() -> None:
    """The report shows five recommendations before expanding."""
    assert VISIBLE_RECOMMENDATIONS == 5


def test_empty_data_yields_no_recommendations(tmp_path: Path) -> None:
    """A near-empty dataset produces no recommendations."""
    empty = pd.DataFrame({"match_id": ["M0"], "win": [1]})
    stats = StatisticsEngine(empty, tmp_path)
    coach = CoachEngine(empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats)
    assert coach.generate() == []
