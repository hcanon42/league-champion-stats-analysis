"""Laning phase analysis: checkpoint differentials and early-game outcomes."""

from __future__ import annotations

from typing import Any

import pandas as pd


def laning_summary(matches_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate laning-phase performance from the master match table.

    Args:
        matches_df: One row per game (from :meth:`models.MatchRecord.to_row`).

    Returns:
        Average differentials at checkpoints, lane win rates and early
        deaths, split by game result where informative.
    """
    if matches_df.empty:
        return {}

    def mean_of(column: str, frame: pd.DataFrame | None = None) -> float | None:
        """Rounded mean of a column, ignoring missing values."""
        source = matches_df if frame is None else frame
        if column not in source or source[column].dropna().empty:
            return None
        return round(float(source[column].dropna().mean()), 1)

    wins = matches_df[matches_df["win"] == 1]
    losses = matches_df[matches_df["win"] == 0]
    gd10 = matches_df["gd10"].dropna()
    summary: dict[str, Any] = {
        "avg_gd5": mean_of("gd5"),
        "avg_gd10": mean_of("gd10"),
        "avg_gd15": mean_of("gd15"),
        "avg_gd20": mean_of("gd20"),
        "avg_xpd10": mean_of("xpd10"),
        "avg_csd10": mean_of("csd10"),
        "avg_cs10": mean_of("cs10"),
        "avg_gold10": mean_of("gold10"),
        "avg_gd10_wins": mean_of("gd10", wins),
        "avg_gd10_losses": mean_of("gd10", losses),
        "lane_win_rate": round(float((gd10 > 0).mean()), 3) if not gd10.empty else None,
        "avg_deaths_pre14": mean_of("deaths_pre14"),
        "avg_gank_deaths_laning": mean_of("gank_deaths_laning"),
        "avg_under_own_tower_laning_deaths": mean_of("under_own_tower_laning_deaths"),
        "avg_under_enemy_tower_laning_deaths": mean_of("under_enemy_tower_laning_deaths"),
        "avg_lane_priority": (
            round(float(matches_df["lane_priority"].dropna().mean()), 3)
            if matches_df["lane_priority"].notna().any()
            else None
        ),
        "avg_roams_pre15": mean_of("roams_pre15"),
    }
    # Win rate when winning/losing lane at 10 (gold diff sign).
    if not gd10.empty:
        ahead = matches_df[matches_df["gd10"] > 0]
        behind = matches_df[matches_df["gd10"] < 0]
        summary["winrate_when_ahead_at_10"] = (
            round(float(ahead["win"].mean()), 3) if not ahead.empty else None
        )
        summary["winrate_when_behind_at_10"] = (
            round(float(behind["win"].mean()), 3) if not behind.empty else None
        )
    return summary
