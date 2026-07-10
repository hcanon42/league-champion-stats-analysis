"""Rune analysis: outcomes per keystone, secondary tree and shard setup."""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.core.models import MatchRecord

MIN_GAMES_PER_SETUP: int = 3


def runes_dataframe(records: list[MatchRecord]) -> pd.DataFrame:
    """Per-game rune setup with outcomes, one row per game.

    Args:
        records: Parsed match records.

    Returns:
        A dataframe with keystone/trees/shards and result metrics.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "match_id": record.match_id,
                "win": int(record.win),
                "keystone": record.runes.keystone,
                "primary_tree": record.runes.primary_tree,
                "secondary_tree": record.runes.secondary_tree,
                "minor_runes": " | ".join(
                    record.runes.primary_runes + record.runes.secondary_runes
                ),
                "shards": " | ".join(record.runes.shards),
                "dpm": round(record.combat.dpm, 1),
                "damage": record.combat.damage_to_champions,
                "gold": record.economy.gold,
                "deaths": record.combat.deaths,
            }
        )
    return pd.DataFrame(rows)


def rune_setup_stats(runes_df: pd.DataFrame, group_by: str = "keystone") -> pd.DataFrame:
    """Outcome aggregates grouped by a rune dimension.

    Args:
        runes_df: Output of :func:`runes_dataframe`.
        group_by: Column to group by (``keystone``, ``secondary_tree``,
            ``shards`` or ``minor_runes``).

    Returns:
        One row per setup with games, winrate, damage, gold and deaths.
    """
    if runes_df.empty:
        return pd.DataFrame()
    grouped = runes_df.groupby(group_by).agg(
        games=("win", "size"),
        winrate=("win", "mean"),
        avg_dpm=("dpm", "mean"),
        avg_damage=("damage", "mean"),
        avg_gold=("gold", "mean"),
        avg_deaths=("deaths", "mean"),
    )
    grouped = grouped.round(
        {"winrate": 3, "avg_dpm": 1, "avg_damage": 0, "avg_gold": 0, "avg_deaths": 2}
    )
    return grouped.sort_values("games", ascending=False).reset_index()


def rune_summary(runes_df: pd.DataFrame) -> dict[str, Any]:
    """Headline rune insights across keystones and secondary trees.

    Args:
        runes_df: Output of :func:`runes_dataframe`.

    Returns:
        Best keystone/secondary tree with sufficient sample size.
    """
    summary: dict[str, Any] = {}
    for dimension in ("keystone", "secondary_tree"):
        stats = rune_setup_stats(runes_df, dimension)
        if stats.empty:
            continue
        eligible = stats[stats["games"] >= MIN_GAMES_PER_SETUP]
        if eligible.empty:
            continue
        best = eligible.loc[eligible["winrate"].idxmax()]
        summary[f"best_{dimension}"] = str(best[dimension])
        summary[f"best_{dimension}_winrate"] = float(best["winrate"])
        summary[f"best_{dimension}_games"] = int(best["games"])
    return summary
