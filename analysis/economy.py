"""Economy analysis: income, gold efficiency and reset habits."""

from __future__ import annotations

from typing import Any

import pandas as pd

from models import MatchRecord


def economy_summary(matches_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate economic performance from the master match table.

    Args:
        matches_df: One row per game (from :meth:`models.MatchRecord.to_row`).

    Returns:
        Income/farm aggregates split by result plus reset-habit metrics.
    """
    if matches_df.empty:
        return {}
    wins = matches_df[matches_df["win"] == 1]
    losses = matches_df[matches_df["win"] == 0]

    def mean_of(frame: pd.DataFrame, column: str, digits: int = 1) -> float | None:
        """Rounded mean of a column, ignoring missing values."""
        series = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        return round(float(series.mean()), digits) if not series.empty else None

    return {
        "avg_gpm": mean_of(matches_df, "gpm"),
        "avg_gpm_wins": mean_of(wins, "gpm"),
        "avg_gpm_losses": mean_of(losses, "gpm"),
        "avg_cspm": mean_of(matches_df, "cspm", 2),
        "avg_gold_share": mean_of(matches_df, "gold_share", 3),
        "avg_dpm": mean_of(matches_df, "dpm"),
        "avg_damage_per_gold": (
            round(float((matches_df["damage"] / matches_df["gold"]).mean()), 3)
            if not matches_df.empty
            else None
        ),
        "avg_unspent_gold_before_recall": mean_of(matches_df, "avg_unspent_gold", 0),
        "avg_recalls_per_game": mean_of(matches_df, "recalls"),
        "avg_time_dead_s": mean_of(matches_df, "time_dead_s", 0),
        "gold_lost_to_death_timers_pct": (
            round(
                float(
                    (matches_df["time_dead_s"] / (matches_df["duration_min"] * 60)).mean() * 100
                ),
                1,
            )
            if not matches_df.empty
            else None
        ),
    }


def reset_quality(records: list[MatchRecord]) -> dict[str, Any]:
    """Analyse recall habits: timing regularity and banked gold.

    Args:
        records: Parsed match records.

    Returns:
        Average first-recall timing and unspent gold distribution stats.
    """
    first_recalls: list[float] = []
    unspent: list[int] = []
    for record in records:
        if record.timeline.recalls:
            first_recalls.append(record.timeline.recalls[0].minute)
        unspent.extend(r.unspent_gold for r in record.timeline.recalls)
    series = pd.Series(unspent, dtype=float)
    return {
        "avg_first_recall_min": (
            round(sum(first_recalls) / len(first_recalls), 2) if first_recalls else None
        ),
        "avg_unspent_gold": round(float(series.mean()), 0) if not series.empty else None,
        "p90_unspent_gold": (
            round(float(series.quantile(0.9)), 0) if not series.empty else None
        ),
        "recalls_analyzed": int(series.size),
    }
