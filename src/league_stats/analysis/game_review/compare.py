"""Per-game comparison rows vs personal baseline."""

from __future__ import annotations

from typing import Any

from league_stats.analysis.progression.metrics import progression_metrics_for_role
from league_stats.core.config import GAME_REVIEW_MAX_COMPARISONS
from league_stats.core.models import GameComparisonRow


def _verdict(delta: float, direction: str) -> str:
    if abs(delta) < 1e-9:
        return "on_par"
    if direction == "higher":
        return "above" if delta > 0 else "below"
    return "below" if delta > 0 else "above"


def _top_rows(rows: list[GameComparisonRow]) -> list[GameComparisonRow]:
    ranked = sorted(rows, key=lambda row: abs(row.delta), reverse=True)
    return ranked[:GAME_REVIEW_MAX_COMPARISONS]


def _comparison_decimals(metric: str) -> int:
    if metric in {
        "win",
        "kill_participation",
        "damage_share",
        "objectives_present_rate",
        "lane_priority",
    } or metric.endswith("_rate"):
        return 2
    if metric in {
        "deaths",
        "deaths_pre14",
        "control_wards",
        "gd10",
        "gd15",
        "cs10",
        "early_ganks",
        "roams_pre15",
        "avg_unspent_gold",
        "solo_deaths",
        "greed_deaths",
    }:
        return 0
    return 1


def _round_comparison(metric: str, value: float) -> float:
    return round(value, _comparison_decimals(metric))


def compare_to_baseline(
    game_row: dict[str, Any],
    baseline_means: dict[str, float],
    *,
    role: str,
) -> list[GameComparisonRow]:
    """Compare one game to personal baseline means."""
    rows: list[GameComparisonRow] = []
    for spec in progression_metrics_for_role(role):
        if spec.source != "matches_df":
            continue
        game_value = game_row.get(spec.metric)
        baseline = baseline_means.get(spec.metric)
        if game_value is None or baseline is None:
            continue
        game_f = float(game_value)
        base_f = float(baseline)
        delta = game_f - base_f
        rows.append(
            GameComparisonRow(
                metric=spec.metric,
                label=spec.label,
                game_value=_round_comparison(spec.metric, game_f),
                benchmark_value=_round_comparison(spec.metric, base_f),
                delta=_round_comparison(spec.metric, delta),
                verdict=_verdict(delta, spec.direction),
            )
        )
    return _top_rows(rows)
