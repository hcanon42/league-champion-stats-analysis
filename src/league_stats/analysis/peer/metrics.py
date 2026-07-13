"""Shared end-of-game metric extraction for peer comparison."""

from __future__ import annotations

from typing import Any, Final

from league_stats.core.config import REMAKE_MAX_DURATION_S, RANKED_SOLO_QUEUE_ID
from league_stats.utils import safe_div

BENCHMARK_METRIC_KEYS: Final[tuple[str, ...]] = (
    "win",
    "kda",
    "dpm",
    "ccpm",
    "cspm",
    "deaths",
    "vspm",
    "control_wards",
    "kill_participation",
    "damage_share",
    "assists",
    "healing",
    "shielding",
    "objectives_present_rate",
    "roams_pre15",
    "early_ganks",
)


def participant_position(participant: dict[str, Any]) -> str:
    """Return the normalised lane for a participant."""
    return str(participant.get("teamPosition") or participant.get("individualPosition") or "")


def match_duration_minutes(match: dict[str, Any]) -> float | None:
    """Return game length in minutes when the match is a valid solo queue game."""
    info = match.get("info", {})
    if int(info.get("queueId", 0)) != RANKED_SOLO_QUEUE_ID:
        return None
    duration_s = int(info.get("gameDuration", 0))
    if duration_s > 100_000:
        duration_s //= 1000
    if duration_s <= REMAKE_MAX_DURATION_S:
        return None
    return duration_s / 60.0


def participant_row(participant: dict[str, Any], duration_min: float) -> dict[str, Any]:
    """Extract comparable scalars from a match participant block."""
    minutes = max(1.0, duration_min)
    kills = int(participant.get("kills", 0))
    deaths = int(participant.get("deaths", 0))
    assists = int(participant.get("assists", 0))
    damage = int(participant.get("totalDamageDealtToChampions", 0))
    gold = int(participant.get("goldEarned", 0))
    cs = int(participant.get("totalMinionsKilled", 0)) + int(
        participant.get("neutralMinionsKilled", 0)
    )
    challenges = participant.get("challenges", {}) or {}
    cc_score = int(participant.get("timeCCingOthers", 0))
    healing = int(participant.get("totalHealsOnTeammates", 0)) + int(participant.get("totalHeal", 0))
    shielding = int(participant.get("totalDamageShieldedOnTeammates", 0))
    return {
        "puuid": str(participant.get("puuid", "")),
        "win": int(bool(participant.get("win"))),
        "kda": (kills + assists) / max(1, deaths),
        "dpm": damage / minutes,
        "ccpm": cc_score / minutes,
        "cspm": cs / minutes,
        "deaths": float(deaths),
        "vspm": int(participant.get("visionScore", 0)) / minutes,
        "control_wards": float(int(participant.get("visionWardsBoughtInGame", 0))),
        "kill_participation": float(challenges.get("killParticipation", 0.0)),
        "damage_share": safe_div(damage, damage),
        "assists": float(assists),
        "healing": float(healing),
        "shielding": float(shielding),
        "gold": gold,
        "damage": damage,
    }


def extract_champion_role_rows(
    match: dict[str, Any],
    *,
    exclude_puuid: str,
    champion: str,
    role: str,
) -> list[dict[str, Any]]:
    """Pull performances on the configured champion + lane from a raw match."""
    duration_min = match_duration_minutes(match)
    if duration_min is None:
        return []

    participants: list[dict[str, Any]] = match.get("info", {}).get("participants", [])
    team_damage = team_damage_totals(participants)
    rows: list[dict[str, Any]] = []
    for participant in participants:
        if str(participant.get("puuid", "")) == exclude_puuid:
            continue
        if str(participant.get("championName", "")) != champion:
            continue
        if participant_position(participant) != role:
            continue
        row = participant_row(participant, duration_min)
        team_id = int(participant.get("teamId", 0))
        row["damage_share"] = safe_div(row["damage"], team_damage.get(team_id, row["damage"]))
        rows.append(row)
    return rows


def team_damage_totals(participants: list[dict[str, Any]]) -> dict[int, int]:
    """Sum champion damage dealt per team id."""
    totals: dict[int, int] = {}
    for participant in participants:
        team_id = int(participant.get("teamId", 0))
        totals[team_id] = totals.get(team_id, 0) + int(
            participant.get("totalDamageDealtToChampions", 0)
        )
    return totals
