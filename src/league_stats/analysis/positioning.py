"""Macro positioning analysis: grouped vs solo time and side-lane habits.

All metrics are derived from 60-second timeline frames, so they are coarse
by nature and documented as approximations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.models import MatchRecord, Position, Zone
from league_stats.utils import classify_zone, distance, is_side_lane

GROUPED_RADIUS: float = 3_000.0
MACRO_PHASE_START_MIN: int = 14


def extract_positioning(ctx: TimelineContext) -> dict[str, Any]:
    """Compute grouped/solo/side-lane time shares from timeline frames.

    Args:
        ctx: Timeline context.

    Returns:
        Frame-share metrics for the mid/late game (post 14 minutes):
        ``grouped_share``, ``solo_share``, ``side_lane_share`` and
        ``avg_allies_nearby``.
    """
    grouped = 0
    solo = 0
    side_lane = 0
    total = 0
    allies_nearby_counts: list[int] = []
    last_minute = len(ctx.frames) - 1
    for minute in range(MACRO_PHASE_START_MIN, last_minute + 1):
        frame = ctx.frame_at_minute(minute)
        mine = ctx.participant_frame(frame, ctx.participant_id) if frame else None
        if not mine or "position" not in mine:
            continue
        pos = Position(**mine["position"])
        zone = classify_zone(pos)
        if zone == Zone.BASE:
            continue
        total += 1
        allies_near = 0
        for pid in ctx.team_ids:
            if pid == ctx.participant_id:
                continue
            other = ctx.participant_frame(frame, pid) if frame else None
            if other and "position" in other:
                if distance(pos, Position(**other["position"])) <= GROUPED_RADIUS:
                    allies_near += 1
        allies_nearby_counts.append(allies_near)
        if allies_near >= 2:
            grouped += 1
        elif allies_near == 0:
            solo += 1
        if is_side_lane(zone):
            side_lane += 1
    if total == 0:
        return {
            "grouped_share": None,
            "solo_share": None,
            "side_lane_share": None,
            "avg_allies_nearby": None,
        }
    return {
        "grouped_share": round(grouped / total, 3),
        "solo_share": round(solo / total, 3),
        "side_lane_share": round(side_lane / total, 3),
        "avg_allies_nearby": round(sum(allies_nearby_counts) / total, 2),
    }


def macro_summary(records: list[MatchRecord], matches_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate macro habits across all games.

    Args:
        records: Parsed match records.
        matches_df: The master per-game table.

    Returns:
        Reset timings before objectives, side-lane death rates and
        recall/roam habits.
    """
    if not records:
        return {}
    resets_before_dragon: list[float] = []
    resets_before_baron: list[float] = []
    for record in records:
        recall_minutes = [r.minute for r in record.timeline.recalls]
        for obj in record.objectives:
            gaps = [obj.minute - m for m in recall_minutes if 0 < obj.minute - m <= 3.0]
            if not gaps:
                continue
            if obj.kind.value in ("dragon", "elder"):
                resets_before_dragon.append(min(gaps))
            elif obj.kind.value == "baron":
                resets_before_baron.append(min(gaps))

    side_lane_deaths = matches_df["side_lane_deaths"] if "side_lane_deaths" in matches_df else pd.Series(dtype=float)
    return {
        "avg_recall_min_gap": (
            round(
                float(
                    pd.Series(
                        [
                            record.duration_min / max(1, len(record.timeline.recalls))
                            for record in records
                        ]
                    ).mean()
                ),
                1,
            )
        ),
        "reset_before_dragon_rate": (
            round(len(resets_before_dragon) / max(1, sum(
                1 for r in records for o in r.objectives if o.kind.value in ("dragon", "elder")
            )), 3)
        ),
        "reset_before_baron_rate": (
            round(len(resets_before_baron) / max(1, sum(
                1 for r in records for o in r.objectives if o.kind.value == "baron"
            )), 3)
        ),
        "avg_side_lane_deaths": (
            round(float(side_lane_deaths.mean()), 2) if not side_lane_deaths.empty else None
        ),
    }


def positioning_dataframe(records: list[MatchRecord], per_game: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine per-game positioning shares with results.

    Args:
        records: Parsed match records (order-aligned with ``per_game``).
        per_game: Output of :func:`extract_positioning` per record.

    Returns:
        One row per game with positioning shares and the result.
    """
    rows: list[dict[str, Any]] = []
    for record, shares in zip(records, per_game):
        rows.append({"match_id": record.match_id, "win": int(record.win), **shares})
    return pd.DataFrame(rows)
