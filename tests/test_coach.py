"""Tests for the coach engine's rule evaluation and ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.coach import CoachEngine, recommendations_markdown
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
    deaths = pd.DataFrame(
        {
            "match_id": ["M0"] * 4,
            "win": [0, 0, 0, 1],
            "minute": [23.0, 25.0, 12.0, 30.0],
            "side_lane_push": [True, True, False, True],
            "alone": [True, True, False, True],
            "team_wards_recent": [0, 1, 2, 0],
            "zone": ["bot", "top", "mid", "bot"],
        }
    )
    stats = StatisticsEngine(matches, tmp_path)
    return CoachEngine(
        matches_df=matches,
        deaths_df=deaths,
        matchups_df=matchups_dataframe(matches),
        objectives_df=pd.DataFrame(),
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


def test_markdown_rendering(coach: CoachEngine) -> None:
    """The Markdown export lists every recommendation with evidence."""
    recommendations = coach.generate()
    markdown = recommendations_markdown(recommendations)
    assert markdown.startswith("# Viktor Mid Coaching Recommendations")
    assert "Evidence:" in markdown


def test_empty_data_yields_no_recommendations(tmp_path: Path) -> None:
    """A near-empty dataset produces no recommendations."""
    empty = pd.DataFrame({"match_id": ["M0"], "win": [1]})
    stats = StatisticsEngine(empty, tmp_path)
    coach = CoachEngine(empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats)
    assert coach.generate() == []
