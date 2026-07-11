"""Item build analysis: build paths, timings and per-build outcomes."""

from __future__ import annotations

from typing import Any

import pandas as pd

MIN_GAMES_PER_BUILD: int = 3


def items_dataframe(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Per-build-slot outcome table (winrate, DPM, deaths, gold).

    Groups games by first, second and third completed item and by boots.

    Args:
        matches_df: One row per game (from :meth:`models.MatchRecord.to_row`).

    Returns:
        One row per ``(slot, item)`` pair with outcome aggregates.
    """
    rows: list[dict[str, Any]] = []
    if matches_df.empty:
        return pd.DataFrame(rows)
    for slot in ("first_item", "second_item", "third_item", "boots"):
        subset = matches_df.dropna(subset=[slot])
        for item, group in subset.groupby(slot):
            timing_col = f"{slot}_min"
            timings = (
                group[timing_col].dropna() if timing_col in group else pd.Series(dtype=float)
            )
            rows.append(
                {
                    "slot": slot,
                    "item": str(item),
                    "games": int(len(group)),
                    "winrate": round(float(group["win"].mean()), 3),
                    "avg_dpm": round(float(group["dpm"].mean()), 1),
                    "avg_deaths": round(float(group["deaths"].mean()), 2),
                    "avg_gold": round(float(group["gold"].mean()), 0),
                    "avg_timing_min": (
                        round(float(timings.mean()), 2) if not timings.empty else None
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["slot", "games"], ascending=[True, False])


def build_path_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Outcomes per two-item core (first + second completed item).

    Args:
        matches_df: One row per game.

    Returns:
        One row per core with winrate, DPM, deaths and gold aggregates.
    """
    if matches_df.empty:
        return pd.DataFrame()
    subset = matches_df.dropna(subset=["first_item", "second_item"]).copy()
    if subset.empty:
        return pd.DataFrame()
    subset["core"] = subset["first_item"] + " -> " + subset["second_item"]
    grouped = subset.groupby("core").agg(
        first_item=("first_item", "first"),
        second_item=("second_item", "first"),
        games=("win", "size"),
        winrate=("win", "mean"),
        avg_dpm=("dpm", "mean"),
        avg_deaths=("deaths", "mean"),
        avg_gold=("gold", "mean"),
    )
    grouped = grouped.round({"winrate": 3, "avg_dpm": 1, "avg_deaths": 2, "avg_gold": 0})
    return grouped.sort_values("games", ascending=False).reset_index()


def item_summary(items_df: pd.DataFrame) -> dict[str, Any]:
    """Headline item insights: best/worst first item with enough games.

    Args:
        items_df: Output of :func:`items_dataframe`.

    Returns:
        Best and worst first items (min. :data:`MIN_GAMES_PER_BUILD` games).
    """
    if items_df.empty:
        return {}
    firsts = items_df[
        (items_df["slot"] == "first_item") & (items_df["games"] >= MIN_GAMES_PER_BUILD)
    ]
    if firsts.empty:
        return {}
    best = firsts.loc[firsts["winrate"].idxmax()]
    worst = firsts.loc[firsts["winrate"].idxmin()]
    return {
        "best_first_item": str(best["item"]),
        "best_first_item_winrate": float(best["winrate"]),
        "best_first_item_games": int(best["games"]),
        "worst_first_item": str(worst["item"]),
        "worst_first_item_winrate": float(worst["winrate"]),
        "worst_first_item_games": int(worst["games"]),
    }
