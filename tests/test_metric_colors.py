"""Tests for gradient metric color helpers."""

from __future__ import annotations

from league_stats.presentation.metric_colors import (
    color_winrate,
    interpolate_metric_color,
    score_deaths_per_game,
    score_lane_diff,
    score_winrate,
)


def test_interpolate_metric_color_endpoints() -> None:
    assert interpolate_metric_color(-1.0) == "#e05563"
    assert interpolate_metric_color(1.0) == "#3fb68b"


def test_interpolate_metric_color_midpoint_is_neutral() -> None:
    assert interpolate_metric_color(0.0) == "#c4aa6a"


def test_score_winrate_is_centered_on_fifty_percent() -> None:
    assert score_winrate(50.0) == 0.0
    assert score_winrate(70.0) == 1.0
    assert score_winrate(30.0) == -1.0


def test_score_lane_diff_scales_signed_gold() -> None:
    assert score_lane_diff(300.0) == 1.0
    assert score_lane_diff(-300.0) == -1.0


def test_score_deaths_prefers_fewer_deaths() -> None:
    assert score_deaths_per_game(3.0) == 1.0
    assert score_deaths_per_game(6.0) == -1.0


def test_color_winrate_graduates_near_fifty() -> None:
    assert color_winrate(0.5) == interpolate_metric_color(0.0)
    assert color_winrate(0.53) != color_winrate(0.47)
