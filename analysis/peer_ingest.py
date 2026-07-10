"""Extract and persist peer game rows from raw match documents."""

from __future__ import annotations

import time
from typing import Any

from analysis.peer_metrics import (
    BENCHMARK_METRIC_KEYS,
    match_duration_minutes,
    participant_position,
    participant_row,
    team_damage_totals,
)
from cache import MatchStore
from utils import get_logger, safe_div


def extract_peer_rows(
    match: dict[str, Any],
    *,
    match_id: str,
    platform: str,
) -> list[dict[str, Any]]:
    """Extract one peer row per solo queue participant in a match."""
    duration_min = match_duration_minutes(match)
    if duration_min is None:
        return []

    info = match.get("info", {})
    queue_id = int(info.get("queueId", 0))
    participants: list[dict[str, Any]] = info.get("participants", [])
    team_damage = team_damage_totals(participants)
    ingested_at = time.time()
    rows: list[dict[str, Any]] = []

    for participant in participants:
        puuid = str(participant.get("puuid", ""))
        champion = str(participant.get("championName", ""))
        role = participant_position(participant)
        if not puuid or not champion or not role:
            continue

        row = participant_row(participant, duration_min)
        team_id = int(participant.get("teamId", 0))
        row["damage_share"] = safe_div(row["damage"], team_damage.get(team_id, row["damage"]))
        metrics = {key: row[key] for key in BENCHMARK_METRIC_KEYS if key in row}
        rows.append(
            {
                "match_id": match_id,
                "puuid": puuid,
                "champion": champion,
                "role": role,
                "tier": "",
                "rank": "",
                "platform": platform,
                "queue_id": queue_id,
                "metrics": metrics,
                "ingested_at": ingested_at,
                "rank_verified": 0,
            }
        )
    return rows


def ingest_match(
    store: MatchStore,
    match_id: str,
    match: dict[str, Any],
    platform: str,
) -> int:
    """Persist peer rows extracted from one match document."""
    inserted = 0
    for row in extract_peer_rows(match, match_id=match_id, platform=platform):
        if store.upsert_peer_game(row):
            inserted += 1
    if inserted:
        get_logger("peer_ingest").debug(
            "Ingested %d peer row(s) from %s", inserted, match_id
        )
    return inserted


def backfill_all_matches(store: MatchStore, platform: str) -> int:
    """Scan every stored match and ingest peer rows."""
    total = 0
    for match_id in store.iter_all_match_ids():
        match = store.load_match(match_id)
        if match is None:
            continue
        total += ingest_match(store, match_id, match, platform)
    get_logger("peer_ingest").info("Backfilled %d peer row(s) from stored matches", total)
    return total
