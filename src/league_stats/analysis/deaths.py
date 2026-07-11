"""Death analysis: full contextualisation of every death plus aggregates.

Extraction runs on the raw timeline (called by the parser); aggregation runs
on parsed :class:`~models.MatchRecord` collections (called by the pipeline).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.timeline import TimelineContext, avg_teammate_distance_at_ms, current_gold_at_ms, headcount_near
from league_stats.core.models import DeathEvent, MatchRecord, Position, RecallEvent, Zone
from league_stats.utils import (
    LANING_PHASE_END_MIN,
    classify_zone,
    is_side_lane,
    ms_to_min,
    near_enemy_lane_tower,
    near_major_objective,
    near_own_lane_tower,
    push_progress,
)

NEARBY_RADIUS: float = 2_200.0
GREED_PUSH_THRESHOLD: float = 2_600.0
BOUNTY_THRESHOLD_GOLD: int = 300
RECENT_WARDS_WINDOW_MS: int = 60_000
AFTER_TOWER_WINDOW_MS: int = 45_000
AFTER_OBJECTIVE_WINDOW_MS: int = 60_000
BEFORE_OBJECTIVE_WINDOW_MS: int = 60_000
AFTER_RECALL_WINDOW_MS: int = 45_000
DEAD_BEFORE_WINDOW_MS: int = 45_000
SIDE_LANE_MIN: float = 14.0
LANE_GANK_ZONES: frozenset[Zone] = frozenset({Zone.TOP_LANE, Zone.MID_LANE, Zone.BOT_LANE})
ZHONYA_ITEM_IDS: frozenset[int] = frozenset({3157, 2420})


def _zhonya_in_inventory_at(ctx: TimelineContext, timestamp_ms: int) -> bool:
    """Return whether Zhonya's or Stopwatch is in inventory at a timestamp.

    Walks purchase, destroy, sell and undo events. Stopwatch consumption
    emits ``ITEM_DESTROYED``; Zhonya's active does not remove the item.
    """
    owned = 0
    for event in ctx.events:
        if int(event.get("participantId", 0)) != ctx.participant_id:
            continue
        if int(event["timestamp"]) > timestamp_ms:
            break
        event_type = event.get("type")
        if event_type == "ITEM_PURCHASED":
            item_id = int(event.get("itemId", 0))
            if item_id in ZHONYA_ITEM_IDS:
                owned += 1
        elif event_type in {"ITEM_DESTROYED", "ITEM_SOLD"}:
            item_id = int(event.get("itemId", 0))
            if item_id in ZHONYA_ITEM_IDS:
                owned = max(0, owned - 1)
        elif event_type == "ITEM_UNDO":
            undone = int(event.get("beforeId", 0))
            if undone in ZHONYA_ITEM_IDS:
                owned = max(0, owned - 1)
    return owned > 0


def _is_gank_death(
    event: dict[str, Any],
    ctx: TimelineContext,
    *,
    minute: float,
    zone: Zone,
) -> bool:
    """Whether a laning-phase lane death involved an extra enemy roaming in.

    Counts only deaths before :data:`~utils.LANING_PHASE_END_MIN` in a lane
    zone where the killer or an assist is not the lane opponent (e.g. enemy
    jungler or roamer joined the fight).
    """
    if minute >= LANING_PHASE_END_MIN or zone not in LANE_GANK_ZONES:
        return False
    killer_id = int(event.get("killerId", 0))
    assists = {int(a) for a in (event.get("assistingParticipantIds") or [])}
    enemy_champs = {killer_id, *assists}
    enemy_champs = {pid for pid in enemy_champs if pid in ctx.enemy_ids}
    if not enemy_champs:
        return False
    if ctx.opponent_id is None:
        return len(enemy_champs) >= 2
    return any(pid != ctx.opponent_id for pid in enemy_champs)


def _headcount_near(ctx: TimelineContext, pos: Position, timestamp_ms: int) -> tuple[int, int]:
    """Count allies (excluding the player) and enemies near a position."""
    return headcount_near(ctx, pos, timestamp_ms, radius=NEARBY_RADIUS)


def extract_deaths(
    ctx: TimelineContext,
    recalls: list[RecallEvent],
    ult_learned_min: float | None,
) -> list[DeathEvent]:
    """Build a fully contextualised :class:`~models.DeathEvent` per death.

    Args:
        ctx: Timeline context.
        recalls: Inferred recalls (for death-after-recall detection).
        ult_learned_min: Minute R was first skilled, or ``None``.

    Returns:
        Death events in chronological order.
    """
    kills = ctx.events_of("CHAMPION_KILL")
    my_team_id = 100 if ctx.blue_side else 200
    enemy_team_id = 200 if ctx.blue_side else 100
    tower_kills_ts = [
        int(e["timestamp"])
        for e in ctx.events_of("BUILDING_KILL")
        if int(e.get("teamId", 0)) == enemy_team_id  # teamId = building owner -> our team took it
    ]
    monsters = ctx.events_of("ELITE_MONSTER_KILL")
    team_monster_ts = [
        int(e["timestamp"]) for e in monsters if int(e.get("killerTeamId", 0)) == my_team_id
    ]
    dragon_ts = [int(e["timestamp"]) for e in monsters if e.get("monsterType") == "DRAGON"]
    baron_ts = [int(e["timestamp"]) for e in monsters if e.get("monsterType") == "BARON_NASHOR"]
    # Dragon timestamps include elder (monsterType DRAGON, subType ELDER_DRAGON).
    neutral_objective_ts = dragon_ts + baron_ts
    ward_events = ctx.events_of("WARD_PLACED")

    deaths: list[DeathEvent] = []
    for event in kills:
        if int(event.get("victimId", 0)) != ctx.participant_id:
            continue
        ts = int(event["timestamp"])
        minute = ms_to_min(ts)
        pos = Position(**event.get("position", {"x": 0, "y": 0}))
        zone = classify_zone(pos)
        allies, enemies = _headcount_near(ctx, pos, ts)
        alone = allies == 0
        zhonya = _zhonya_in_inventory_at(ctx, ts)
        laning = minute < LANING_PHASE_END_MIN
        under_own_tower_laning = laning and near_own_lane_tower(pos, ctx.blue_side)
        under_enemy_tower_laning = laning and near_enemy_lane_tower(pos, ctx.blue_side)
        wards_recent = sum(
            1
            for w in ward_events
            if int(w.get("creatorId", 0)) in ctx.team_ids
            and 0 <= ts - int(w["timestamp"]) <= RECENT_WARDS_WINDOW_MS
        )
        deaths.append(
            DeathEvent(
                minute=minute,
                position=pos,
                zone=zone,
                near_objective=near_major_objective(pos),
                shutdown_given=int(event.get("shutdownBounty", 0)),
                bounty_held=int(event.get("bounty", 0)) > BOUNTY_THRESHOLD_GOLD,
                flash_available=None,  # summoner cooldowns are not exposed by the API
                ult_available=(ult_learned_min is not None and minute >= ult_learned_min),
                zhonya_available=zhonya,
                alone=alone,
                outnumbered=enemies > allies + 1,
                team_wards_recent=wards_recent,
                enemy_seen=None,  # fog-of-war state is not exposed by the API
                after_greed=alone and push_progress(pos, ctx.blue_side) > GREED_PUSH_THRESHOLD,
                after_tower=any(0 <= ts - t <= AFTER_TOWER_WINDOW_MS for t in tower_kills_ts),
                after_objective=any(
                    0 <= ts - t <= AFTER_OBJECTIVE_WINDOW_MS for t in team_monster_ts
                ),
                side_lane_push=is_side_lane(zone) and minute > SIDE_LANE_MIN,
                before_dragon=any(0 <= t - ts <= BEFORE_OBJECTIVE_WINDOW_MS for t in dragon_ts),
                before_baron=any(0 <= t - ts <= BEFORE_OBJECTIVE_WINDOW_MS for t in baron_ts),
                before_neutral_objective=any(
                    0 <= t - ts <= BEFORE_OBJECTIVE_WINDOW_MS for t in neutral_objective_ts
                ),
                after_recall=any(
                    0 <= ts - int(r.minute * 60_000) <= AFTER_RECALL_WINDOW_MS for r in recalls
                ),
                to_gank=_is_gank_death(event, ctx, minute=minute, zone=zone),
                under_own_tower_laning=under_own_tower_laning,
                under_enemy_tower_laning=under_enemy_tower_laning,
                killer_champion=ctx.id_to_champion.get(int(event.get("killerId", 0))),
                current_gold=current_gold_at_ms(ctx, ts),
                avg_teammate_distance=avg_teammate_distance_at_ms(ctx, ts),
            )
        )
    return deaths


def deaths_dataframe(records: list[MatchRecord]) -> pd.DataFrame:
    """Flatten every death of every game into a dataframe.

    Args:
        records: Parsed match records.

    Returns:
        One row per death, including match context (win, opponent, patch).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        for death in record.deaths:
            rows.append(
                {
                    "match_id": record.match_id,
                    "win": int(record.win),
                    "opponent": record.lane_opponent or "Unknown",
                    "patch": record.patch,
                    "minute": round(death.minute, 2),
                    "x": death.position.x,
                    "y": death.position.y,
                    "zone": death.zone.value,
                    "near_objective": death.near_objective,
                    "shutdown_given": death.shutdown_given,
                    "bounty_held": death.bounty_held,
                    "ult_available": death.ult_available,
                    "zhonya_available": death.zhonya_available,
                    "alone": death.alone,
                    "outnumbered": death.outnumbered,
                    "team_wards_recent": death.team_wards_recent,
                    "after_greed": death.after_greed,
                    "after_tower": death.after_tower,
                    "after_objective": death.after_objective,
                    "side_lane_push": death.side_lane_push,
                    "before_dragon": death.before_dragon,
                    "before_baron": death.before_baron,
                    "before_neutral_objective": death.before_neutral_objective,
                    "after_recall": death.after_recall,
                    "to_gank": death.to_gank,
                    "under_own_tower_laning": death.under_own_tower_laning,
                    "under_enemy_tower_laning": death.under_enemy_tower_laning,
                    "killer": death.killer_champion or "Unknown",
                    "current_gold": death.current_gold,
                    "avg_teammate_distance": (
                        round(death.avg_teammate_distance, 0)
                        if death.avg_teammate_distance is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def death_summary(deaths_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate headline death statistics.

    Args:
        deaths_df: Output of :func:`deaths_dataframe`.

    Returns:
        Headline metrics (rates of solo/greed/objective-adjacent deaths...).
    """
    if deaths_df.empty:
        return {"total_deaths": 0}
    total = len(deaths_df)
    return {
        "total_deaths": total,
        "solo_death_rate": round(float(deaths_df["alone"].mean()), 3),
        "greed_death_rate": round(float(deaths_df["after_greed"].mean()), 3),
        "gank_death_rate": round(float(deaths_df["to_gank"].mean()), 3),
        "under_own_tower_laning_death_rate": round(
            float(deaths_df["under_own_tower_laning"].mean()), 3
        ),
        "under_enemy_tower_laning_death_rate": round(
            float(deaths_df["under_enemy_tower_laning"].mean()), 3
        ),
        "side_lane_death_rate": round(float(deaths_df["side_lane_push"].mean()), 3),
        "death_before_neutral_objective_rate": round(
            float(deaths_df["before_neutral_objective"].mean()), 3
        ),
        "shutdowns_given": int((deaths_df["shutdown_given"] > 0).sum()),
        "avg_death_minute": round(float(deaths_df["minute"].mean()), 1),
        "avg_gold_at_death": (
            round(float(deaths_df["current_gold"].dropna().mean()), 0)
            if "current_gold" in deaths_df and deaths_df["current_gold"].notna().any()
            else None
        ),
        "avg_teammate_distance_at_death": (
            round(float(deaths_df["avg_teammate_distance"].dropna().mean()), 0)
            if "avg_teammate_distance" in deaths_df and deaths_df["avg_teammate_distance"].notna().any()
            else None
        ),
        "outnumbered_death_rate": (
            round(float(deaths_df["outnumbered"].mean()), 3)
            if "outnumbered" in deaths_df.columns
            else None
        ),
        "most_common_zone": str(deaths_df["zone"].mode().iat[0]),
        "deaths_by_zone": deaths_df["zone"].value_counts().to_dict(),
        "most_common_killer": str(deaths_df["killer"].mode().iat[0]),
    }


def blind_spot_zones(deaths_df: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
    """Find the zones with the most solo deaths and poor recent team vision.

    A documented proxy for "blind spots": ward positions are not exposed by
    the API, so low recent team ward activity around solo deaths is used.

    Args:
        deaths_df: Output of :func:`deaths_dataframe`.
        top_n: Number of zones to return.

    Returns:
        Zones ranked by count of low-vision solo deaths.
    """
    if deaths_df.empty:
        return []
    risky = deaths_df[(deaths_df["alone"]) & (deaths_df["team_wards_recent"] <= 1)]
    counts = risky.groupby("zone").size().sort_values(ascending=False).head(top_n)
    return [{"zone": zone, "deaths": int(count)} for zone, count in counts.items()]
