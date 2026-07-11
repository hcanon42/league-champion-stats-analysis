"""Macro positioning analysis: grouped vs solo time, ally distances and win hints.

All metrics are derived from 60-second timeline frames, so they are coarse
by nature and documented as approximations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.champions import role_display
from league_stats.core.models import MatchRecord, Position, Zone
from league_stats.utils import classify_zone, distance, is_side_lane

GROUPED_RADIUS: float = 3_000.0
MACRO_PHASE_START_MIN: int = 14
ALLY_ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_COLUMNS: dict[str, str] = {
    "TOP": "dist_top",
    "JUNGLE": "dist_jungle",
    "MIDDLE": "dist_middle",
    "BOTTOM": "dist_bottom",
    "UTILITY": "dist_support",
}
MIN_HINT_GAMES: int = 8
MIN_HINT_GROUP: int = 3
MIN_WR_DELTA: float = 0.12


def extract_positioning(ctx: TimelineContext) -> dict[str, Any]:
    """Compute grouped/solo/side-lane shares and ally distances from timeline frames.

    Args:
        ctx: Timeline context.

    Returns:
        Frame-share metrics for the mid/late game (post 14 minutes) plus
        average distance to each ally role (excluding the tracked player).
    """
    grouped = 0
    solo = 0
    side_lane = 0
    total = 0
    allies_nearby_counts: list[int] = []
    teammate_distances: list[float] = []
    my_role = ctx.id_to_role.get(ctx.participant_id, "")
    role_distances: dict[str, list[float]] = {
        role: [] for role in ALLY_ROLES if role != my_role
    }
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
        frame_distances: list[float] = []
        for pid in ctx.team_ids:
            if pid == ctx.participant_id:
                continue
            other = ctx.participant_frame(frame, pid) if frame else None
            if other and "position" in other:
                other_pos = Position(**other["position"])
                dist = distance(pos, other_pos)
                frame_distances.append(dist)
                role = ctx.id_to_role.get(pid, "")
                if role in role_distances:
                    role_distances[role].append(dist)
                if dist <= GROUPED_RADIUS:
                    allies_near += 1
        if frame_distances:
            teammate_distances.append(sum(frame_distances) / len(frame_distances))
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
            "avg_teammate_distance": None,
            "role_distances": {},
        }
    return {
        "grouped_share": round(grouped / total, 3),
        "solo_share": round(solo / total, 3),
        "side_lane_share": round(side_lane / total, 3),
        "avg_allies_nearby": round(sum(allies_nearby_counts) / total, 2),
        "avg_teammate_distance": round(sum(teammate_distances) / len(teammate_distances), 0)
        if teammate_distances
        else None,
        "role_distances": {
            role: round(sum(values) / len(values), 0)
            for role, values in role_distances.items()
            if values
        },
    }


def _mean_of(matches_df: pd.DataFrame, column: str) -> float | None:
    if column not in matches_df or matches_df[column].dropna().empty:
        return None
    return round(float(matches_df[column].dropna().mean()), 1)


def _median_split_winrates(
    matches_df: pd.DataFrame,
    column: str,
) -> tuple[float | None, float | None, int, int] | None:
    """Return win rates below/above the column median with group sizes."""
    if column not in matches_df.columns:
        return None
    values = pd.to_numeric(matches_df[column], errors="coerce")
    wins = pd.to_numeric(matches_df["win"], errors="coerce")
    mask = values.notna() & wins.notna()
    if mask.sum() < MIN_HINT_GAMES or values[mask].nunique() < 2:
        return None
    median = float(values[mask].median())
    low_mask = mask & (values <= median)
    high_mask = mask & (values > median)
    if low_mask.sum() < MIN_HINT_GROUP or high_mask.sum() < MIN_HINT_GROUP:
        return None
    return (
        round(float(wins[low_mask].mean()), 3),
        round(float(wins[high_mask].mean()), 3),
        int(low_mask.sum()),
        int(high_mask.sum()),
    )


def _hint_from_distance(
    matches_df: pd.DataFrame,
    column: str,
    role_label: str,
) -> dict[str, str] | None:
    split = _median_split_winrates(matches_df, column)
    if split is None:
        return None
    wr_close, wr_far, _, _ = split
    delta = wr_close - wr_far
    if abs(delta) < MIN_WR_DELTA:
        return None
    if delta >= MIN_WR_DELTA:
        return {
            "tone": "positive",
            "text": (
                f"You tend to win more when you stay closer to your {role_label} "
                f"({wr_close:.0%} WR vs {wr_far:.0%} when farther)."
            ),
        }
    return {
        "tone": "negative",
        "text": (
            f"You win more when you play farther from your {role_label} "
            f"({wr_far:.0%} WR vs {wr_close:.0%} when grouped on that lane)."
        ),
    }


def _hint_from_share(
    matches_df: pd.DataFrame,
    column: str,
    *,
    high_label: str,
    low_label: str,
    high_is_good: bool | None = None,
) -> dict[str, str] | None:
    split = _median_split_winrates(matches_df, column)
    if split is None:
        return None
    wr_low, wr_high, _, _ = split
    delta = wr_high - wr_low
    if abs(delta) < MIN_WR_DELTA:
        return None
    if high_is_good is None:
        high_is_good = delta >= MIN_WR_DELTA
    if high_is_good and delta >= MIN_WR_DELTA:
        return {
            "tone": "positive",
            "text": (
                f"You win more when {high_label} ({wr_high:.0%} WR vs {wr_low:.0%} otherwise)."
            ),
        }
    if not high_is_good and delta <= -MIN_WR_DELTA:
        return {
            "tone": "negative",
            "text": (
                f"You tend to lose more when {high_label} ({wr_high:.0%} WR vs {wr_low:.0%} otherwise). "
                f"{low_label}"
            ),
        }
    return None


def positioning_hints(matches_df: pd.DataFrame, player_role: str) -> list[dict[str, str]]:
    """Build short win-rate insight callouts for positioning habits."""
    if matches_df.empty or len(matches_df) < MIN_HINT_GAMES:
        return []

    candidates: list[tuple[float, dict[str, str]]] = []

    for role, column in ROLE_COLUMNS.items():
        if role == player_role or column not in matches_df.columns:
            continue
        hint = _hint_from_distance(matches_df, column, role_display(role))
        if hint is not None:
            split = _median_split_winrates(matches_df, column)
            delta = abs(split[0] - split[1]) if split else 0.0
            candidates.append((delta, hint))

    overall_split = _median_split_winrates(matches_df, "avg_teammate_distance")
    if overall_split is not None:
        wr_close, wr_far, _, _ = overall_split
        delta = wr_close - wr_far
        if delta >= MIN_WR_DELTA:
            candidates.append((
                abs(delta),
                {
                    "tone": "positive",
                    "text": (
                        f"You tend to win more when you stay closer to your team overall "
                        f"({wr_close:.0%} WR vs {wr_far:.0%} when more spread out)."
                    ),
                },
            ))
        elif delta <= -MIN_WR_DELTA:
            candidates.append((
                abs(delta),
                {
                    "tone": "negative",
                    "text": (
                        f"You tend to lose more when you stick with teammates all the time "
                        f"({wr_close:.0%} WR when grouped vs {wr_far:.0%} when spread out). "
                        "Don't forget side waves and jungle camps between fights."
                    ),
                },
            ))

    grouped = _hint_from_share(
        matches_df,
        "grouped_share",
        high_label="you stay grouped with teammates most of the mid/late game",
        low_label="Make room for side-wave and jungle camp income between fights.",
        high_is_good=False,
    )
    if grouped is not None:
        split = _median_split_winrates(matches_df, "grouped_share")
        candidates.append((abs(split[0] - split[1]) if split else 0.0, grouped))

    solo = _hint_from_share(
        matches_df,
        "solo_share",
        high_label="you spend more time alone on the map",
        low_label="Keep catching side waves, but track enemy collapse timers.",
        high_is_good=True,
    )
    if solo is not None:
        split = _median_split_winrates(matches_df, "solo_share")
        candidates.append((abs(split[0] - split[1]) if split else 0.0, solo))

    side_lane = _hint_from_share(
        matches_df,
        "side_lane_share",
        high_label="you spend more time in side lanes",
        low_label="Keep pressuring side waves when the team doesn't need you grouped.",
        high_is_good=True,
    )
    if side_lane is not None:
        split = _median_split_winrates(matches_df, "side_lane_share")
        candidates.append((abs(split[0] - split[1]) if split else 0.0, side_lane))

    candidates.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    hints: list[dict[str, str]] = []
    for _, hint in candidates:
        if hint["text"] in seen:
            continue
        seen.add(hint["text"])
        hints.append(hint)
        if len(hints) >= 3:
            break
    return hints


def positioning_summary(matches_df: pd.DataFrame, player_role: str) -> dict[str, Any]:
    """Aggregate macro positioning metrics for the dashboard section."""
    if matches_df.empty:
        return {"hints": []}

    summary: dict[str, Any] = {
        "avg_grouped_share": _mean_of(matches_df, "grouped_share"),
        "avg_solo_share": _mean_of(matches_df, "solo_share"),
        "avg_side_lane_share": _mean_of(matches_df, "side_lane_share"),
        "avg_allies_nearby": _mean_of(matches_df, "avg_allies_nearby"),
        "avg_teammate_distance": _mean_of(matches_df, "avg_teammate_distance"),
        "hints": positioning_hints(matches_df, player_role),
    }
    for role, column in ROLE_COLUMNS.items():
        if role != player_role:
            summary[column] = _mean_of(matches_df, column)
    return summary


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
        row = {"match_id": record.match_id, "win": int(record.win), **shares}
        for role, column in ROLE_COLUMNS.items():
            row[column] = shares.get("role_distances", {}).get(role)
        rows.append(row)
    return pd.DataFrame(rows)
