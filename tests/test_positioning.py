"""Tests for macro positioning metrics and win-rate hints."""

from __future__ import annotations

import pandas as pd
import pytest

from league_stats.analysis.positioning import extract_positioning, positioning_hints, positioning_summary
from league_stats.analysis.timeline import build_context
from league_stats.core.models import MatchRecord
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


@pytest.fixture()
def record() -> MatchRecord:
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    return parser.parse(make_match(), make_timeline(), MY_PUUID)


def test_role_distances_exclude_player_role(record: MatchRecord) -> None:
    """Per-role distances are tracked for allies only."""
    distances = record.timeline.role_distances
    assert "MIDDLE" not in distances
    assert set(distances) == {"TOP", "JUNGLE", "BOTTOM", "UTILITY"}
    assert all(value > 0 for value in distances.values())


def test_extract_positioning_from_context() -> None:
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    shares = extract_positioning(ctx)
    assert shares["avg_teammate_distance"] is not None
    assert shares["role_distances"]["JUNGLE"] == shares["role_distances"]["TOP"]


def test_positioning_summary_includes_role_columns(record: MatchRecord) -> None:
    from league_stats.pipeline.frames import build_analysis_frames

    frames = build_analysis_frames([record])
    summary = positioning_summary(frames.matches_df, record.role)
    assert summary["avg_teammate_distance"] is not None
    assert summary["dist_jungle"] is not None
    assert "dist_middle" not in summary


def test_positioning_hints_need_enough_games() -> None:
    frame = pd.DataFrame({"win": [1, 0, 1, 0], "dist_jungle": [1000, 5000, 900, 4800]})
    assert positioning_hints(frame, "MIDDLE") == []


def test_positioning_hint_when_close_to_jungle_wins_more() -> None:
    rng = pd.Series([1200, 1300, 1100, 1400, 1500, 1200, 1300, 1250, 5000, 5200, 4800, 5100])
    wins = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    frame = pd.DataFrame({"win": wins, "dist_jungle": rng})
    hints = positioning_hints(frame, "MIDDLE")
    assert hints
    assert any("jungle" in hint["text"].lower() for hint in hints)
    assert hints[0]["tone"] == "positive"
