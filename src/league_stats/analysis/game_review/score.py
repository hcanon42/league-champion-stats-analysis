"""Personal-baseline game score for a single match."""

from __future__ import annotations

from typing import Any

from league_stats.core.models import GameScoreBreakdown
from league_stats.core.role_metrics import role_profile
from league_stats.presentation.metric_colors import (
    score_deaths_per_game,
    score_form_delta,
    score_lane_diff,
)

# Map ingredient columns onto the fixed game-review score dimensions.
_COLUMN_TO_DIMENSION: dict[str, str] = {
    "gd10": "laning",
    "csd10": "laning",
    "deaths_pre14": "laning",
    "cs10": "laning",
    "early_ganks": "laning",
    "roams_pre15": "laning",
    "kp15": "laning",
    "lane_priority": "laning",
    "deaths": "survival",
    "avg_unspent_gold": "survival",
    "first_item_min": "survival",
    "damage_share": "impact",
    "ccpm": "impact",
    "kill_participation": "impact",
    "tf_participation": "impact",
    "tf_won_share": "impact",
    "gold_share": "impact",
    "damage_taken_share": "impact",
    "hpm": "impact",
    "spm": "impact",
    "vspm": "vision",
    "control_wards": "vision",
    "objectives_present_rate": "objectives",
}

_SCORE_DIMENSIONS = ("laning", "survival", "impact", "vision", "objectives")

_LOWER_IS_BETTER = frozenset(
    {"deaths", "avg_unspent_gold", "deaths_pre14", "first_item_min"}
)


def _score_tier(overall: int) -> str:
    if overall >= 90:
        return "S"
    if overall >= 75:
        return "A"
    if overall >= 60:
        return "B"
    if overall >= 45:
        return "C"
    return "D"


def _to_percent_score(raw: float | None) -> int:
    if raw is None:
        return 50
    return max(0, min(100, round((raw + 1.0) * 50)))


def _metric_direction(column: str) -> str:
    return "lower" if column in _LOWER_IS_BETTER else "higher"


def _component_score(
    column: str,
    game_value: float | None,
    baseline: float | None,
    *,
    game_row: dict[str, Any],
) -> int:
    if game_value is None:
        return 50
    if baseline is None:
        if column == "deaths":
            duration = float(game_row.get("duration_min") or 30.0)
            return _to_percent_score(score_deaths_per_game(float(game_value), duration_min=duration))
        return 50

    direction = _metric_direction(column)
    if column in {"gd10", "cs10", "gd15", "xpd10", "csd10"}:
        return _to_percent_score(score_lane_diff(float(game_value) - float(baseline)))

    improvement = float(game_value) - float(baseline)
    if direction == "lower":
        improvement = float(baseline) - float(game_value)

    if column == "deaths":
        duration = float(game_row.get("duration_min") or 30.0)
        return _to_percent_score(score_deaths_per_game(float(game_value), duration_min=duration))

    return _to_percent_score(score_form_delta(column, improvement))


def compute_game_score(
    game_row: dict[str, Any],
    baseline_means: dict[str, float],
    *,
    role: str,
) -> GameScoreBreakdown:
    """Score one game against personal baseline means."""
    profile = role_profile(role)
    dimension_scores: dict[str, list[int]] = {key: [] for key in _SCORE_DIMENSIONS}

    for spec in profile.score_components:
        for metric in spec.metrics:
            dimension = _COLUMN_TO_DIMENSION.get(metric.column)
            if dimension not in dimension_scores:
                continue
            game_value = game_row.get(metric.column)
            if game_value is None:
                continue
            baseline = baseline_means.get(metric.column)
            dimension_scores[dimension].append(
                _component_score(
                    metric.column, float(game_value), baseline, game_row=game_row
                )
            )

    def dim_avg(key: str) -> int:
        values = dimension_scores[key]
        return round(sum(values) / len(values)) if values else 50

    breakdown = {key: dim_avg(key) for key in _SCORE_DIMENSIONS}
    overall = round(sum(breakdown.values()) / len(breakdown))
    return GameScoreBreakdown(
        overall=overall,
        tier=_score_tier(overall),
        **breakdown,
    )
