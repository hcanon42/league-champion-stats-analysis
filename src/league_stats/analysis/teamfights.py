"""Automatic teamfight detection and per-fight involvement metrics.

A teamfight is a spatio-temporal cluster of champion kills: consecutive
kills join the same fight when they happen within ``FIGHT_GAP_MS`` of the
previous kill and within ``FIGHT_RADIUS`` of the running centroid, and the
cluster contains at least ``MIN_KILLS`` kills.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.models import MatchRecord, Position, TeamfightRecord
from league_stats.utils import distance, ms_to_min, push_progress

FIGHT_GAP_MS: int = 25_000
FIGHT_RADIUS: float = 4_000.0
MIN_KILLS: int = 3
PROXIMITY_PARTICIPATION: float = 3_000.0


def _cluster_kills(kills: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group kill events into spatio-temporal clusters.

    Args:
        kills: ``CHAMPION_KILL`` events sorted by timestamp.

    Returns:
        Clusters (lists of kill events) with at least :data:`MIN_KILLS` kills.
    """
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for kill in kills:
        if not current:
            current = [kill]
            continue
        centroid = _centroid(current)
        pos = Position(**kill.get("position", {"x": 0, "y": 0}))
        close_in_time = int(kill["timestamp"]) - int(current[-1]["timestamp"]) <= FIGHT_GAP_MS
        close_in_space = distance(pos, centroid) <= FIGHT_RADIUS
        if close_in_time and close_in_space:
            current.append(kill)
        else:
            clusters.append(current)
            current = [kill]
    if current:
        clusters.append(current)
    return [c for c in clusters if len(c) >= MIN_KILLS]


def _centroid(kills: list[dict[str, Any]]) -> Position:
    """Mean position of a kill cluster.

    Args:
        kills: Kill events with ``position`` fields.

    Returns:
        The centroid position.
    """
    xs = [float(k.get("position", {}).get("x", 0)) for k in kills]
    ys = [float(k.get("position", {}).get("y", 0)) for k in kills]
    return Position(x=sum(xs) / len(xs), y=sum(ys) / len(ys))


def _damage_from_player(kill: dict[str, Any], pid: int) -> int:
    """Damage the player contributed to a victim, from the kill event.

    Args:
        kill: A ``CHAMPION_KILL`` event.
        pid: The player's participant id.

    Returns:
        Summed physical + magic + true damage dealt by the player.
    """
    total = 0
    for entry in kill.get("victimDamageReceived", []) or []:
        if int(entry.get("participantId", 0)) == pid:
            total += (
                int(entry.get("physicalDamage", 0))
                + int(entry.get("magicDamage", 0))
                + int(entry.get("trueDamage", 0))
            )
    return total


def detect_teamfights(ctx: TimelineContext) -> list[TeamfightRecord]:
    """Detect teamfights and summarise the player's involvement in each.

    Args:
        ctx: Timeline context.

    Returns:
        One :class:`~models.TeamfightRecord` per detected fight.
    """
    kills = ctx.events_of("CHAMPION_KILL")
    fights: list[TeamfightRecord] = []
    for cluster in _cluster_kills(kills):
        start_ts = int(cluster[0]["timestamp"])
        end_ts = int(cluster[-1]["timestamp"])
        centroid = _centroid(cluster)

        my_kills = 0
        my_assists = 0
        died = False
        death_ts: int | None = None
        damage_dealt = 0
        damage_taken: int | None = None
        ally_kills = 0
        enemy_kills = 0
        involved = False
        for kill in cluster:
            killer = int(kill.get("killerId", 0))
            victim = int(kill.get("victimId", 0))
            assists = [int(a) for a in kill.get("assistingParticipantIds", []) or []]
            if killer in ctx.team_ids:
                ally_kills += 1
            elif killer in ctx.enemy_ids:
                enemy_kills += 1
            if killer == ctx.participant_id:
                my_kills += 1
            if ctx.participant_id in assists:
                my_assists += 1
            if victim == ctx.participant_id:
                died = True
                death_ts = int(kill["timestamp"])
                damage_taken = sum(
                    int(e.get("physicalDamage", 0))
                    + int(e.get("magicDamage", 0))
                    + int(e.get("trueDamage", 0))
                    for e in kill.get("victimDamageReceived", []) or []
                )
            damage_dealt += _damage_from_player(kill, ctx.participant_id)
            if ctx.participant_id in (killer, victim) or ctx.participant_id in assists:
                involved = True

        if not involved:
            my_pos = ctx.position_at_ms(ctx.participant_id, (start_ts + end_ts) // 2)
            involved = my_pos is not None and distance(my_pos, centroid) <= PROXIMITY_PARTICIPATION

        window_s = max(1.0, (end_ts - start_ts) / 1000.0)
        time_alive = ((death_ts - start_ts) / 1000.0) if died and death_ts else window_s

        front_to_back: float | None = None
        my_pos = ctx.position_at_ms(ctx.participant_id, start_ts)
        if my_pos is not None:
            ally_progress = [
                push_progress(pos, ctx.blue_side)
                for pid in ctx.team_ids
                if pid != ctx.participant_id
                and (pos := ctx.position_at_ms(pid, start_ts)) is not None
            ]
            if ally_progress:
                front_to_back = push_progress(my_pos, ctx.blue_side) - (
                    sum(ally_progress) / len(ally_progress)
                )

        fights.append(
            TeamfightRecord(
                start_minute=ms_to_min(start_ts),
                end_minute=ms_to_min(end_ts),
                participated=involved,
                kills=my_kills,
                assists=my_assists,
                died=died,
                damage_dealt=damage_dealt,
                damage_taken=damage_taken,
                time_alive_s=time_alive,
                centroid=centroid,
                front_to_back=front_to_back,
                enemies_hit_by_ult=None,  # not derivable from the public API
                ally_kills=ally_kills,
                enemy_kills=enemy_kills,
                won=(ally_kills > enemy_kills) if ally_kills != enemy_kills else None,
            )
        )
    return fights


def teamfights_dataframe(records: list[MatchRecord]) -> pd.DataFrame:
    """Flatten every detected teamfight into a dataframe.

    Args:
        records: Parsed match records.

    Returns:
        One row per teamfight with the player's involvement metrics.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        for fight in record.teamfights:
            rows.append(
                {
                    "match_id": record.match_id,
                    "win": int(record.win),
                    "start_minute": round(fight.start_minute, 2),
                    "participated": fight.participated,
                    "kills": fight.kills,
                    "assists": fight.assists,
                    "died": fight.died,
                    "damage_dealt": fight.damage_dealt,
                    "damage_taken": fight.damage_taken,
                    "time_alive_s": round(fight.time_alive_s, 1),
                    "front_to_back": (
                        round(fight.front_to_back, 1) if fight.front_to_back is not None else None
                    ),
                    "ally_kills": fight.ally_kills,
                    "enemy_kills": fight.enemy_kills,
                    "fight_won": fight.won,
                }
            )
    return pd.DataFrame(rows)


def teamfight_summary(tf_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate headline teamfight statistics.

    Args:
        tf_df: Output of :func:`teamfights_dataframe`.

    Returns:
        Participation, win-share and positioning aggregates.
    """
    if tf_df.empty:
        return {"total_fights": 0}
    joined = tf_df[tf_df["participated"]]
    decided = joined[joined["fight_won"].notna()]
    return {
        "total_fights": int(len(tf_df)),
        "participation_rate": round(float(tf_df["participated"].mean()), 3),
        "fight_win_rate": (
            round(float(decided["fight_won"].astype(float).mean()), 3) if not decided.empty else None
        ),
        "avg_damage_per_fight": round(float(joined["damage_dealt"].mean()), 0) if not joined.empty else 0,
        "death_rate_in_fights": round(float(joined["died"].mean()), 3) if not joined.empty else 0,
        "avg_front_to_back": (
            round(float(joined["front_to_back"].dropna().mean()), 1)
            if not joined.empty and joined["front_to_back"].notna().any()
            else None
        ),
    }
