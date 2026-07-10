"""Vision analysis: control ward lifetimes, vision trends and blind spots."""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.models import MatchRecord


def extract_control_ward_lifetime(ctx: TimelineContext) -> float | None:
    """Average lifetime of the player's control wards, in seconds.

    Placements (``WARD_PLACED`` by the player, type ``CONTROL_WARD``) are
    matched FIFO against enemy ``WARD_KILL`` events of the same type; wards
    never cleared live until game end. Ward positions are not exposed by the
    API, so FIFO matching is a documented approximation.

    Args:
        ctx: Timeline context.

    Returns:
        Mean lifetime in seconds, or ``None`` if no control wards were placed.
    """
    placements = [
        int(e["timestamp"])
        for e in ctx.events_of("WARD_PLACED")
        if int(e.get("creatorId", 0)) == ctx.participant_id
        and e.get("wardType") == "CONTROL_WARD"
    ]
    kills = sorted(
        int(e["timestamp"])
        for e in ctx.events_of("WARD_KILL")
        if int(e.get("killerId", 0)) in ctx.enemy_ids and e.get("wardType") == "CONTROL_WARD"
    )
    if not placements:
        return None
    game_end_ms = ctx.duration_s * 1000
    lifetimes: list[float] = []
    kill_iter = iter(kills)
    next_kill = next(kill_iter, None)
    for placed in sorted(placements):
        while next_kill is not None and next_kill < placed:
            next_kill = next(kill_iter, None)
        if next_kill is not None:
            lifetimes.append((next_kill - placed) / 1000.0)
            next_kill = next(kill_iter, None)
        else:
            lifetimes.append(max(0.0, (game_end_ms - placed) / 1000.0))
    return sum(lifetimes) / len(lifetimes)


def vision_dataframe(records: list[MatchRecord]) -> pd.DataFrame:
    """Per-game vision metrics as a dataframe.

    Args:
        records: Parsed match records.

    Returns:
        One row per game with vision score, wards and objective-setup vision.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        objective_wards = [o.team_wards_before for o in record.objectives]
        rows.append(
            {
                "match_id": record.match_id,
                "win": int(record.win),
                "vision_score": record.vision.vision_score,
                "vspm": round(record.vision.vision_score_per_min, 2),
                "wards_placed": record.vision.wards_placed,
                "wards_killed": record.vision.wards_killed,
                "control_wards": record.vision.control_wards_bought,
                "avg_control_ward_lifetime_s": (
                    round(record.vision.avg_control_ward_lifetime_s, 1)
                    if record.vision.avg_control_ward_lifetime_s is not None
                    else None
                ),
                "avg_wards_before_objectives": (
                    round(sum(objective_wards) / len(objective_wards), 2)
                    if objective_wards
                    else None
                ),
                "avg_wards_around_deaths": (
                    round(
                        sum(d.team_wards_recent for d in record.deaths) / len(record.deaths), 2
                    )
                    if record.deaths
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def vision_summary(vision_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate vision statistics split by game result.

    Args:
        vision_df: Output of :func:`vision_dataframe`.

    Returns:
        Mean vision metrics overall and per win/loss.
    """
    if vision_df.empty:
        return {}
    summary: dict[str, Any] = {
        "avg_vision_score": round(float(vision_df["vision_score"].mean()), 1),
        "avg_vspm": round(float(vision_df["vspm"].mean()), 2),
        "avg_control_wards": round(float(vision_df["control_wards"].mean()), 2),
        "avg_control_ward_lifetime_s": (
            round(float(vision_df["avg_control_ward_lifetime_s"].dropna().mean()), 1)
            if vision_df["avg_control_ward_lifetime_s"].notna().any()
            else None
        ),
    }
    for label, flag in (("wins", 1), ("losses", 0)):
        subset = vision_df[vision_df["win"] == flag]
        summary[f"avg_vspm_{label}"] = (
            round(float(subset["vspm"].mean()), 2) if not subset.empty else None
        )
        summary[f"avg_control_wards_{label}"] = (
            round(float(subset["control_wards"].mean()), 2) if not subset.empty else None
        )
    return summary
