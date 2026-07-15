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

_SCORE_NAME_TO_DIMENSION: dict[str, str] = {
    "Laning": "laning",
    "Farming": "laning",
    "Clear @10": "laning",
    "Early ganks": "laning",
    "Survival": "survival",
    "Damage": "impact",
    "CC impact": "impact",
    "Utility": "impact",
    "Impact": "impact",
    "Vision": "vision",
    "Objectives": "objectives",
    "Map control": "objectives",
    "Resets": "economy",
}

_LOWER_IS_BETTER = frozenset({"deaths", "avg_unspent_gold", "deaths_pre14"})


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


def _component_score(column: str, game_value: float | None, baseline: float | None) -> int:
    if game_value is None:
        return 50
    if baseline is None:
        if column == "deaths":
            return _to_percent_score(score_deaths_per_game(float(game_value)))
        return 50

    direction = _metric_direction(column)
    if column in {"gd10", "cs10", "gd15", "xpd10", "csd10"}:
        return _to_percent_score(score_lane_diff(float(game_value) - float(baseline)))

    improvement = float(game_value) - float(baseline)
    if direction == "lower":
        improvement = float(baseline) - float(game_value)

    if column == "deaths":
        return _to_percent_score(score_deaths_per_game(float(game_value)))

    return _to_percent_score(score_form_delta(column, improvement))


def compute_game_score(
    game_row: dict[str, Any],
    baseline_means: dict[str, float],
    *,
    role: str,
) -> GameScoreBreakdown:
    """Score one game against personal baseline means."""
    profile = role_profile(role)
    dimension_scores: dict[str, list[int]] = {
        "laning": [],
        "survival": [],
        "impact": [],
        "vision": [],
        "objectives": [],
        "economy": [],
    }

    for spec in profile.score_components:
        dimension = _SCORE_NAME_TO_DIMENSION.get(spec.name, "impact")
        game_value = game_row.get(spec.column)
        if game_value is None:
            continue
        baseline = baseline_means.get(spec.column)
        dimension_scores[dimension].append(_component_score(spec.column, float(game_value), baseline))

    def dim_avg(key: str) -> int:
        values = dimension_scores[key]
        return round(sum(values) / len(values)) if values else 50

    breakdown = {
        "laning": dim_avg("laning"),
        "survival": dim_avg("survival"),
        "impact": dim_avg("impact"),
        "vision": dim_avg("vision"),
        "objectives": dim_avg("objectives"),
        "economy": dim_avg("economy"),
    }
    overall = round(sum(breakdown.values()) / len(breakdown))
    return GameScoreBreakdown(
        overall=overall,
        tier=_score_tier(overall),
        **breakdown,
    )
