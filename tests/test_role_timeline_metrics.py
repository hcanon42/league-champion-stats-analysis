"""Tests for jungle and support timeline metrics."""

from __future__ import annotations

from league_stats.analysis.jungle import extract_jungle_metrics
from league_stats.analysis.support import extract_support_metrics
from league_stats.analysis.timeline import build_context, extract_roams
from tests.fixtures import make_player_match, make_timeline


def test_extract_jungle_metrics_counts_laning_ganks() -> None:
    match = make_player_match("JG-1", champion="LeeSin", position="JUNGLE", puuid="jungle-puuid")
    timeline = make_timeline()
    ctx = build_context(match, timeline, "jungle-puuid")
    metrics = extract_jungle_metrics(ctx)
    assert "early_ganks" in metrics
    assert metrics["early_ganks"] >= 0


def test_extract_support_metrics_tracks_roam_conversions() -> None:
    match = make_player_match("SUP-1", champion="Thresh", position="UTILITY", puuid="sup-puuid")
    timeline = make_timeline()
    ctx = build_context(match, timeline, "sup-puuid")
    roams = extract_roams(ctx)
    metrics = extract_support_metrics(ctx, roams)
    assert "roam_conversions" in metrics
    assert "kp15" in metrics
    assert "vspm10" in metrics
