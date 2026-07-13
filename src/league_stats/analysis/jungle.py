"""Jungle-specific timeline metrics: ganks, KP @15, invade pressure."""

from __future__ import annotations

from typing import Any

from league_stats.analysis.timeline import TimelineContext
from league_stats.utils import ms_to_min

LANER_ROLES: frozenset[str] = frozenset({"TOP", "MIDDLE", "BOTTOM", "UTILITY"})
EARLY_GAME_MS: int = 15 * 60 * 1000
KP_CUTOFF_MS: int = 15 * 60 * 1000


def _role_of(ctx: TimelineContext, participant_id: int) -> str:
    return str(ctx.id_to_role.get(participant_id, ""))


def _is_laner_victim(ctx: TimelineContext, victim_id: int) -> bool:
    role = _role_of(ctx, victim_id)
    return role in LANER_ROLES


def extract_jungle_metrics(ctx: TimelineContext) -> dict[str, Any]:
    """Derive jungle map-impact metrics from timeline kill events."""
    early_ganks = 0
    gank_assists = 0
    player_ka_pre15 = 0
    team_kills_pre15 = 0

    for event in ctx.events_of("CHAMPION_KILL"):
        ts = int(event["timestamp"])
        killer_id = int(event.get("killerId", 0))
        victim_id = int(event.get("victimId", 0))
        assists = {int(a) for a in (event.get("assistingParticipantIds") or [])}
        my_involved = ctx.participant_id == killer_id or ctx.participant_id in assists

        if ts <= KP_CUTOFF_MS and killer_id in ctx.team_ids:
            team_kills_pre15 += 1
            if my_involved:
                player_ka_pre15 += 1

        if ts > EARLY_GAME_MS:
            continue
        if not _is_laner_victim(ctx, victim_id):
            continue
        if killer_id == ctx.participant_id:
            early_ganks += 1
        elif ctx.participant_id in assists:
            gank_assists += 1

    kp15 = player_ka_pre15 / team_kills_pre15 if team_kills_pre15 else None
    return {
        "early_ganks": early_ganks,
        "gank_assists": gank_assists,
        "kp15": round(kp15, 3) if kp15 is not None else None,
    }


def jungle_summary(matches_df) -> dict[str, Any]:
    """Aggregate jungle metrics from the master match table."""
    if matches_df.empty:
        return {}

    def mean_of(column: str) -> float | None:
        if column not in matches_df or matches_df[column].dropna().empty:
            return None
        return round(float(matches_df[column].dropna().mean()), 2)

    return {
        "avg_early_ganks": mean_of("early_ganks"),
        "avg_gank_assists": mean_of("gank_assists"),
        "avg_kp15": mean_of("kp15"),
    }
