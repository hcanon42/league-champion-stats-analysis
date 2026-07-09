"""Matchup analysis: per-lane-opponent outcomes and recommendations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils import wilson_lower_bound

MIN_GAMES_FOR_VERDICT: int = 3


def matchups_dataframe(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every lane matchup into one row per opposing champion.

    Args:
        matches_df: One row per game (from :meth:`models.MatchRecord.to_row`).

    Returns:
        Per-champion games, winrate, lane differentials and death profile,
        sorted by games played.
    """
    if matches_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for champion, group in matches_df.groupby("opponent"):
        wins = int(group["win"].sum())
        games = int(len(group))
        rows.append(
            {
                "opponent": str(champion),
                "games": games,
                "wins": wins,
                "winrate": round(wins / games, 3),
                "wilson_lb": round(wilson_lower_bound(wins, games), 3),
                "avg_gd10": _mean(group, "gd10"),
                "avg_gd15": _mean(group, "gd15"),
                "avg_xpd10": _mean(group, "xpd10"),
                "avg_csd10": _mean(group, "csd10"),
                "avg_dpm": _mean(group, "dpm"),
                "avg_deaths": _mean(group, "deaths", 2),
                "avg_deaths_pre14": _mean(group, "deaths_pre14", 2),
                "avg_kills": _mean(group, "kills", 2),
            }
        )
    return pd.DataFrame(rows).sort_values("games", ascending=False).reset_index(drop=True)


def _mean(group: pd.DataFrame, column: str, digits: int = 1) -> float | None:
    """Rounded column mean ignoring missing values.

    Args:
        group: Games against one champion.
        column: Column name.
        digits: Rounding digits.

    Returns:
        The mean or ``None`` when no data exists.
    """
    series = group[column].dropna() if column in group else pd.Series(dtype=float)
    return round(float(series.mean()), digits) if not series.empty else None


def matchup_recommendation(row: pd.Series) -> str:
    """Generate a short human recommendation for one matchup row.

    Args:
        row: A row of :func:`matchups_dataframe`.

    Returns:
        A one-sentence coaching hint.
    """
    hints: list[str] = []
    if row.get("avg_gd10") is not None and row["avg_gd10"] < -300:
        hints.append("you lose lane early - play for scaling and avoid pre-6 trades")
    if row.get("avg_deaths_pre14") is not None and row["avg_deaths_pre14"] >= 1.5:
        hints.append("you die too often before 14 min - respect all-in windows")
    if row.get("avg_csd10") is not None and row["avg_csd10"] < -8:
        hints.append("large CS deficit at 10 - focus on safe wave clear")
    if row.get("winrate", 0) >= 0.6 and row.get("games", 0) >= MIN_GAMES_FOR_VERDICT:
        hints.append("strong matchup - look to snowball and roam")
    return "; ".join(hints).capitalize() if hints else "Even matchup; play standard."


def matchup_summary(matchups_df: pd.DataFrame) -> dict[str, Any]:
    """Detect the best and worst matchups with enough games.

    Ranking uses the Wilson lower bound so small samples don't dominate.

    Args:
        matchups_df: Output of :func:`matchups_dataframe`.

    Returns:
        Best/worst matchup names and their stats, or an empty dict.
    """
    if matchups_df.empty:
        return {}
    eligible = matchups_df[matchups_df["games"] >= MIN_GAMES_FOR_VERDICT]
    if eligible.empty:
        return {}
    best = eligible.loc[eligible["wilson_lb"].idxmax()]
    worst = eligible.loc[(eligible["winrate"] + (1 - eligible["wilson_lb"])).idxmin()]
    return {
        "best_matchup": str(best["opponent"]),
        "best_matchup_winrate": float(best["winrate"]),
        "best_matchup_games": int(best["games"]),
        "worst_matchup": str(worst["opponent"]),
        "worst_matchup_winrate": float(worst["winrate"]),
        "worst_matchup_games": int(worst["games"]),
        "worst_matchup_deaths_pre14": (
            float(worst["avg_deaths_pre14"]) if worst["avg_deaths_pre14"] is not None else None
        ),
    }
