"""Support-specific timeline metrics: roam conversion, KP @15, early vision."""

from __future__ import annotations

from typing import Any

from league_stats.analysis.timeline import TimelineContext
from league_stats.core.models import RoamEvent

ROAM_CONVERSION_WINDOW_MS: int = 120_000
KP_CUTOFF_MS: int = 15 * 60 * 1000


def _player_involved_in_kill(ctx: TimelineContext, event: dict[str, Any]) -> bool:
    killer_id = int(event.get("killerId", 0))
    assists = {int(a) for a in (event.get("assistingParticipantIds") or [])}
    return ctx.participant_id == killer_id or ctx.participant_id in assists


def extract_support_metrics(ctx: TimelineContext, roams: list[RoamEvent]) -> dict[str, Any]:
    """Derive support setup metrics from roams and early kills."""
    kills = ctx.events_of("CHAMPION_KILL")
    player_ka_pre15 = 0
    team_kills_pre15 = 0
    wards_pre10 = 0

    for event in kills:
        ts = int(event["timestamp"])
        killer_id = int(event.get("killerId", 0))
        if ts <= KP_CUTOFF_MS and killer_id in ctx.team_ids:
            team_kills_pre15 += 1
            if _player_involved_in_kill(ctx, event):
                player_ka_pre15 += 1

    for event in ctx.events_of("WARD_PLACED"):
        if int(event.get("creatorId", 0)) != ctx.participant_id:
            continue
        if int(event["timestamp"]) <= 10 * 60 * 1000:
            wards_pre10 += 1

    roam_conversions = 0
    for roam in roams:
        start_ms = int(roam.start_minute * 60_000)
        end_ms = int(roam.end_minute * 60_000) + ROAM_CONVERSION_WINDOW_MS
        for event in kills:
            ts = int(event["timestamp"])
            if start_ms <= ts <= end_ms and _player_involved_in_kill(ctx, event):
                roam_conversions += 1
                break

    kp15 = player_ka_pre15 / team_kills_pre15 if team_kills_pre15 else None
    vspm10 = round(wards_pre10 / 10.0, 2) if wards_pre10 else 0.0
    return {
        "roam_conversions": roam_conversions,
        "kp15": round(kp15, 3) if kp15 is not None else None,
        "vspm10": vspm10,
    }


def utility_summary(matches_df) -> dict[str, Any]:
    """Aggregate support/utility metrics from the master match table."""
    if matches_df.empty:
        return {}

    def mean_of(column: str) -> float | None:
        if column not in matches_df or matches_df[column].dropna().empty:
            return None
        return round(float(matches_df[column].dropna().mean()), 2)

    minutes = matches_df["duration_min"].clip(lower=1.0)
    hpm = None
    if "healing" in matches_df.columns:
        hpm = round(float((matches_df["healing"] / minutes).mean()), 1)
    spm = None
    if "shielding" in matches_df.columns:
        spm = round(float((matches_df["shielding"] / minutes).mean()), 1)

    return {
        "avg_assists": mean_of("assists"),
        "avg_hpm": hpm,
        "avg_spm": spm,
        "avg_roam_conversions": mean_of("roam_conversions"),
        "avg_kp15": mean_of("kp15"),
        "avg_vspm10": mean_of("vspm10"),
    }
