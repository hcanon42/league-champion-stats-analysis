"""Tests for the coach engine's rule evaluation and ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from league_stats.analysis.coach.engine import (
    CoachEngine,
    VISIBLE_RECOMMENDATIONS,
    recommendations_markdown,
)
from league_stats.analysis.matchups import matchups_dataframe
from league_stats.analysis.statistics import StatisticsEngine
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
    matches["deaths_before_neutral_objective"] = np.where(matches["win"] == 0, 2, 0)
    matches["fights_disadvantaged"] = np.where(matches["win"] == 0, 3, 0)
    matches["avg_gold_at_death"] = np.where(matches["win"] == 0, 1600, 450)
    matches["avg_unspent_gold_per_fight"] = np.where(matches["win"] == 0, 1800, 650)
    matches["grouped_share"] = np.where(matches["win"] == 0, 0.72, 0.38)
    matches["solo_share"] = np.where(matches["win"] == 1, 0.55, 0.18)
    matches["dist_jungle"] = np.where(matches["win"] == 1, 2200, 6200)
    matches["outnumbered_deaths"] = np.where(matches["win"] == 0, 3, 0)
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
        role="MIDDLE",
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
    assert "Win more when ahead @10" in titles
    assert "Greed deaths are a recurring pattern" in titles
    assert "You give away too many shutdown bounties" in titles
    assert "You throw leads you build in lane" in titles
    assert "Low teamfight participation is limiting your impact" in titles
    assert "You're dead too often right before objectives" in titles
    assert "Deaths right before epic monsters are throwing objectives" in titles


def test_merged_objective_death_rule_not_dragon_only(coach: CoachEngine) -> None:
    """Pre-objective deaths cover dragon, elder, and baron together."""
    rec = next(r for r in coach.generate() if "epic monsters" in r.title)
    assert "dragon, elder, or baron" in rec.detail


def test_new_metric_rules_fire(coach: CoachEngine) -> None:
    """Coach rules for newer death, fight, and positioning metrics surface."""
    titles = " | ".join(r.title for r in coach.generate())
    assert "Dying with unspent gold is costing you games" in titles
    assert "Too much gold unspent when fights start" in titles
    assert "Too many fights started at a numbers disadvantage" in titles
    assert "Over-grouping is hurting your win rate" in titles
    assert "Splitting for farm wins you more games" in titles
    assert "Stay closer to your jungle" in titles
    assert "Outnumbered deaths are throwing fights" in titles


def test_markdown_rendering(coach: CoachEngine) -> None:
    """The Markdown export lists every recommendation with evidence."""
    recommendations = coach.generate()
    markdown = recommendations_markdown(recommendations)
    assert markdown.startswith("# Viktor Mid Coaching Recommendations")
    assert "Evidence:" in markdown


def test_visible_recommendation_limit_constant() -> None:
    """The report shows three recommendations before expanding."""
    assert VISIBLE_RECOMMENDATIONS == 3


def test_empty_data_yields_no_recommendations(tmp_path: Path) -> None:
    """A near-empty dataset produces no recommendations."""
    empty = pd.DataFrame({"match_id": ["M0"], "win": [1]})
    stats = StatisticsEngine(empty, tmp_path)
    coach = CoachEngine(empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats)
    assert coach.generate() == []
