"""Tests for rank-peer comparison and tier benchmarks."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.benchmarks import adjacent_tiers, resolve_benchmark_path, tier_benchmark
from analysis.peer_comparison import (
    build_comparisons,
    peer_recommendations,
    _extract_champion_role_from_match,
)
from models import RankedEntry
from tests.fixtures import MY_PUUID, make_match


def test_tier_benchmark_maps_winrate() -> None:
    """JSON ``winrate`` is exposed as ``win`` for comparisons."""
    emerald = tier_benchmark("EMERALD", "Viktor", "MIDDLE")
    assert emerald["win"] == 0.5
    assert emerald["winrate"] == 0.5


def test_tier_benchmark_returns_gold_defaults() -> None:
    """Unknown tiers fall back to GOLD benchmarks."""
    gold = tier_benchmark("GOLD", "Viktor", "MIDDLE")
    unknown = tier_benchmark("NOT_A_TIER", "Viktor", "MIDDLE")
    assert gold["dpm"] == unknown["dpm"]
    assert gold["cspm"] > 6.0


def test_benchmark_path_prefers_champion_specific() -> None:
    """Champion-specific benchmarks are preferred over role fallback."""
    path = resolve_benchmark_path("Viktor", "MIDDLE")
    assert path.name == "viktor_middle.json"


def test_adjacent_tiers_includes_neighbours() -> None:
    """PLATINUM neighbours are GOLD and EMERALD."""
    neighbours = adjacent_tiers("PLATINUM")
    assert neighbours == {"PLATINUM", "GOLD", "EMERALD"}


def test_infer_platform_from_match_id() -> None:
    """Match id prefixes map to platform routing hosts."""
    from riot_api import RiotApiClient

    assert RiotApiClient.infer_platform_from_match_id("EUW1_12345") == "euw1"
    assert RiotApiClient.infer_platform_from_match_id("EUN1_999") == "eun1"
    assert RiotApiClient.infer_platform_from_match_id("UNKNOWN_1") is None


def test_extract_champion_role_excludes_player() -> None:
    """The tracked player is not included in peer rows."""
    match = make_match()
    rows = _extract_champion_role_from_match(match, MY_PUUID, "Viktor", "MIDDLE")
    assert rows == []


def test_extract_champion_role_finds_enemy() -> None:
    """A matching opponent is extracted with combat stats."""
    match = make_match()
    match["info"]["participants"][5]["championName"] = "Viktor"
    match["info"]["participants"][5]["teamPosition"] = "MIDDLE"
    rows = _extract_champion_role_from_match(match, MY_PUUID, "Viktor", "MIDDLE")
    assert len(rows) == 1
    assert rows[0]["puuid"] != MY_PUUID
    assert rows[0]["dpm"] > 0


def test_extract_champion_role_filters_lane() -> None:
    """Only the configured lane is included."""
    match = make_match()
    match["info"]["participants"][5]["championName"] = "Viktor"
    match["info"]["participants"][5]["teamPosition"] = "TOP"
    rows = _extract_champion_role_from_match(match, MY_PUUID, "Viktor", "MIDDLE")
    assert rows == []


def test_comparison_summary_handles_none_delta_pct() -> None:
    """Summary lines work when peer average is zero (no % gap)."""
    from analysis.peer_comparison import _comparison_summary_line
    from models import MetricComparison

    comp = MetricComparison(
        metric="gd10",
        label="Gold diff @10",
        yours=120.0,
        peer_avg=0.0,
        delta=120.0,
        delta_pct=None,
        direction="higher",
        verdict="above",
    )
    line = _comparison_summary_line(comp)
    assert "120.0" in line
    assert "%" not in line


def test_build_comparisons_verdicts() -> None:
    """Higher-is-better metrics classify gaps correctly."""
    user = {"kda": 3.0, "deaths": 4.0, "dpm": 800.0, "win": 0.6}
    peer = {"kda": 2.4, "deaths": 5.4, "dpm": 640.0, "win": 0.5}
    comparisons = build_comparisons(user, peer)
    by_key = {c.metric: c for c in comparisons}
    assert by_key["kda"].verdict == "above"
    assert by_key["deaths"].verdict == "above"  # fewer deaths is better
    assert by_key["dpm"].verdict == "above"


def test_peer_recommendations_flag_weaknesses() -> None:
    """Large negative gaps produce rank-peer coaching tips."""
    user = {"deaths": 7.0, "cspm": 5.5, "vspm": 0.7, "dpm": 500.0}
    peer = {"deaths": 5.0, "cspm": 7.0, "vspm": 1.2, "dpm": 680.0}
    comparisons = build_comparisons(user, peer)
    recs = peer_recommendations(
        comparisons, "Gold II", peer_games=20, build_label="Viktor mid"
    )
    titles = " | ".join(r.title for r in recs)
    assert "die more" in titles.lower() or "farming" in titles.lower()


def test_comparisons_dataframe_from_result() -> None:
    """Comparison export is one row per metric."""
    from analysis.peer_comparison import comparisons_dataframe
    from models import MetricComparison, PeerComparisonResult

    result = PeerComparisonResult(
        rank_label="Gold II",
        tier="GOLD",
        source="test",
        peer_games=0,
        peer_players=0,
        comparisons=[
            MetricComparison(
                metric="kda",
                label="KDA",
                yours=3.0,
                peer_avg=2.4,
                delta=0.6,
                delta_pct=25.0,
                direction="higher",
                verdict="above",
            )
        ],
    )
    frame = comparisons_dataframe(result)
    assert len(frame) == 1
    assert frame.iloc[0]["metric"] == "kda"
