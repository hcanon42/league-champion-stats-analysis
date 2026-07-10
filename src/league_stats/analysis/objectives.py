"""Objective (dragon/baron/herald/grubs/elder) setup analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.models import MatchRecord, ObjectiveKind, ObjectiveRecord
from league_stats.utils import BARON_PIT, DRAGON_PIT, distance, ms_to_min

PRESENCE_RADIUS: float = 4_500.0
EARLY_RADIUS: float = 5_000.0
EARLY_LOOKBACK_MS: int = 60_000
DEAD_BEFORE_WINDOW_MS: int = 45_000
VISION_WINDOW_MS: int = 120_000


def _kind_of(event: dict[str, Any]) -> ObjectiveKind | None:
    """Map an ``ELITE_MONSTER_KILL`` event to an :class:`ObjectiveKind`.

    Args:
        event: The raw timeline event.

    Returns:
        The objective kind, or ``None`` for untracked monsters.
    """
    monster = str(event.get("monsterType", ""))
    subtype = str(event.get("monsterSubType", ""))
    if monster == "DRAGON":
        return ObjectiveKind.ELDER if subtype == "ELDER_DRAGON" else ObjectiveKind.DRAGON
    if monster == "BARON_NASHOR":
        return ObjectiveKind.BARON
    if monster == "RIFTHERALD":
        return ObjectiveKind.HERALD
    if monster == "HORDE":
        return ObjectiveKind.GRUBS
    return None


def extract_objectives(ctx: TimelineContext) -> list[ObjectiveRecord]:
    """Contextualise every epic monster take in the game.

    Args:
        ctx: Timeline context.

    Returns:
        One :class:`~models.ObjectiveRecord` per epic monster kill.
    """
    my_team_id = 100 if ctx.blue_side else 200
    my_death_ts = [
        int(e["timestamp"])
        for e in ctx.events_of("CHAMPION_KILL")
        if int(e.get("victimId", 0)) == ctx.participant_id
    ]
    wards = ctx.events_of("WARD_PLACED")

    records: list[ObjectiveRecord] = []
    for event in ctx.events_of("ELITE_MONSTER_KILL"):
        kind = _kind_of(event)
        if kind is None:
            continue
        ts = int(event["timestamp"])
        pit = DRAGON_PIT if kind in (ObjectiveKind.DRAGON, ObjectiveKind.ELDER) else BARON_PIT

        pos_now = ctx.position_at_ms(ctx.participant_id, ts)
        pos_before = ctx.position_at_ms(ctx.participant_id, max(0, ts - EARLY_LOOKBACK_MS))
        present = pos_now is not None and distance(pos_now, pit) <= PRESENCE_RADIUS
        early = pos_before is not None and distance(pos_before, pit) <= EARLY_RADIUS
        team_wards = [
            w
            for w in wards
            if int(w.get("creatorId", 0)) in ctx.team_ids
            and 0 <= ts - int(w["timestamp"]) <= VISION_WINDOW_MS
        ]
        records.append(
            ObjectiveRecord(
                minute=ms_to_min(ts),
                kind=kind,
                taken_by_team=int(event.get("killerTeamId", 0)) == my_team_id,
                present=present,
                arrived_early=early,
                arrived_late=present and not early,
                dead_before=any(0 <= ts - t <= DEAD_BEFORE_WINDOW_MS for t in my_death_ts),
                team_wards_before=len(team_wards),
                control_wards_before=sum(
                    1 for w in team_wards if w.get("wardType") == "CONTROL_WARD"
                ),
            )
        )
    return records


def objectives_dataframe(records: list[MatchRecord]) -> pd.DataFrame:
    """Flatten every objective event into a dataframe.

    Args:
        records: Parsed match records.

    Returns:
        One row per epic monster take with the player's setup context.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        for obj in record.objectives:
            rows.append(
                {
                    "match_id": record.match_id,
                    "win": int(record.win),
                    "kind": obj.kind.value,
                    "minute": round(obj.minute, 2),
                    "taken_by_team": obj.taken_by_team,
                    "present": obj.present,
                    "arrived_early": obj.arrived_early,
                    "arrived_late": obj.arrived_late,
                    "dead_before": obj.dead_before,
                    "team_wards_before": obj.team_wards_before,
                    "control_wards_before": obj.control_wards_before,
                }
            )
    return pd.DataFrame(rows)


def objective_summary(obj_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-objective-kind setup statistics.

    Args:
        obj_df: Output of :func:`objectives_dataframe`.

    Returns:
        Presence/death/vision rates per objective kind plus overall rates.
    """
    if obj_df.empty:
        return {"total_objectives": 0, "by_kind": {}}
    by_kind: dict[str, Any] = {}
    for kind, group in obj_df.groupby("kind"):
        by_kind[str(kind)] = {
            "count": int(len(group)),
            "taken_rate": round(float(group["taken_by_team"].mean()), 3),
            "presence_rate": round(float(group["present"].mean()), 3),
            "early_rate": round(float(group["arrived_early"].mean()), 3),
            "dead_before_rate": round(float(group["dead_before"].mean()), 3),
            "avg_wards_before": round(float(group["team_wards_before"].mean()), 2),
            "avg_control_wards_before": round(float(group["control_wards_before"].mean()), 2),
        }
    return {
        "total_objectives": int(len(obj_df)),
        "overall_presence_rate": round(float(obj_df["present"].mean()), 3),
        "overall_dead_before_rate": round(float(obj_df["dead_before"].mean()), 3),
        "by_kind": by_kind,
    }
