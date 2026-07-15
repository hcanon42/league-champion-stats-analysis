"""Per-game comparison rows vs personal baseline and rank peers."""

from __future__ import annotations

from typing import Any

from league_stats.analysis.progression.metrics import progression_metrics_for_role
from league_stats.core.config import GAME_REVIEW_MAX_COMPARISONS
from league_stats.core.models import GameComparisonRow, PeerComparisonResult


def _verdict(delta: float, direction: str) -> str:
    if abs(delta) < 1e-9:
        return "on_par"
    if direction == "higher":
        return "above" if delta > 0 else "below"
    return "below" if delta > 0 else "above"


def _top_rows(rows: list[GameComparisonRow]) -> list[GameComparisonRow]:
    ranked = sorted(rows, key=lambda row: abs(row.delta), reverse=True)
    return ranked[:GAME_REVIEW_MAX_COMPARISONS]


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
                game_value=round(game_f, 3),
                benchmark_value=round(base_f, 3),
                delta=round(delta, 3),
                verdict=_verdict(delta, spec.direction),
            )
        )
    return _top_rows(rows)


def compare_to_peers(
    game_row: dict[str, Any],
    peer_comparison: PeerComparisonResult | None,
) -> list[GameComparisonRow]:
    """Compare one game to rank-peer averages (display only)."""
    if peer_comparison is None:
        return []
    rows: list[GameComparisonRow] = []
    for comp in peer_comparison.comparisons:
        game_value = game_row.get(comp.metric)
        if game_value is None:
            continue
        game_f = float(game_value)
        peer_f = float(comp.peer_avg)
        delta = game_f - peer_f
        rows.append(
            GameComparisonRow(
                metric=comp.metric,
                label=comp.label,
                game_value=round(game_f, 3),
                benchmark_value=round(peer_f, 3),
                delta=round(delta, 3),
                verdict=_verdict(delta, comp.direction),
            )
        )
    return _top_rows(rows)
