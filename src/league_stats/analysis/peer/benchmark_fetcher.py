"""Build rank-peer benchmarks by sampling league players via the Riot API."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
from tqdm import tqdm

from league_stats.analysis.peer.ingest import ingest_match
from league_stats.analysis.peer.metrics import (
    BENCHMARK_METRIC_KEYS,
    match_duration_minutes,
    participant_position,
    participant_row,
    team_damage_totals,
)
from league_stats.analysis.peer.rank_scope import RankScope, build_widened_scope, league_lookup_pairs, rank_matches
from league_stats.infra.cache import MatchStore
from league_stats.core.config import RANKED_SOLO_QUEUE_ID
from league_stats.core.models import RankedEntry
from league_stats.infra.riot_api import RiotApiClient, RiotApiError
from league_stats.utils import get_logger, safe_div

MIN_BENCHMARK_GAMES: Final[int] = 12
TARGET_PEER_GAMES: Final[int] = 100
MAX_PLAYERS_TO_SCAN: Final[int] = 150
MATCH_IDS_PER_PLAYER: Final[int] = 100
MAX_MATCH_DOWNLOADS: Final[int] = 400
LEAGUE_PAGES: Final[int] = 3
MAX_LEAGUE_CANDIDATES: Final[int] = 500


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Peer baseline built from Riot API sampling."""

    metrics: dict[str, float]
    games_sampled: int
    players_sampled: int
    from_cache: bool
    platform: str


def extract_champion_role_for_puuid(
    match: dict[str, Any],
    puuid: str,
    champion: str,
    role: str,
) -> dict[str, Any] | None:
    """Return end-of-game stats when a player used the configured champion + lane."""
    duration_min = match_duration_minutes(match)
    if duration_min is None:
        return None

    participants: list[dict[str, Any]] = match.get("info", {}).get("participants", [])
    team_damage = team_damage_totals(participants)
    for participant in participants:
        if str(participant.get("puuid", "")) != puuid:
            continue
        if str(participant.get("championName", "")) != champion:
            return None
        if participant_position(participant) != role:
            return None
        row = participant_row(participant, duration_min)
        team_id = int(participant.get("teamId", 0))
        row["damage_share"] = safe_div(row["damage"], team_damage.get(team_id, row["damage"]))
        return row
    return None


def _match_has_build(match: dict[str, Any], champion: str, role: str) -> bool:
    """Whether any participant played the configured champion + lane."""
    if match_duration_minutes(match) is None:
        return False
    for participant in match.get("info", {}).get("participants", []):
        if str(participant.get("championName", "")) != champion:
            continue
        if participant_position(participant) == role:
            return True
    return False


def _participant_puuids(match: dict[str, Any]) -> list[str]:
    """Return every participant PUUID in a match."""
    return [
        str(participant.get("puuid", ""))
        for participant in match.get("info", {}).get("participants", [])
        if participant.get("puuid")
    ]


def _average_metrics(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute column means, skipping missing values."""
    result: dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not series.empty:
            result[column] = float(series.mean())
    return result


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average comparable metrics across sampled games."""
    frame = pd.DataFrame(rows)
    metrics = _average_metrics(frame, list(BENCHMARK_METRIC_KEYS))
    if "win" in metrics:
        metrics["winrate"] = metrics["win"]
    return metrics


def _gather_seed_puuids(client: RiotApiClient, scope: RankScope, exclude_puuid: str | None) -> list[str]:
    """Collect league entry PUUIDs across the configured rank scope."""
    seen: set[str] = set()
    puuids: list[str] = []
    for tier, division in league_lookup_pairs(scope):
        try:
            entries = client.fetch_league_entries_pages(tier, division, max_pages=LEAGUE_PAGES)
        except RiotApiError:
            continue
        for entry in entries:
            puuid = str(entry.get("puuid", ""))
            if not puuid or puuid in seen or puuid == (exclude_puuid or ""):
                continue
            seen.add(puuid)
            puuids.append(puuid)
            if len(puuids) >= MAX_LEAGUE_CANDIDATES:
                return puuids
    random.shuffle(puuids)
    return puuids[:MAX_PLAYERS_TO_SCAN]


def _load_or_fetch_match(
    client: RiotApiClient,
    store: MatchStore,
    match_id: str,
    owner_puuid: str,
) -> dict[str, Any] | None:
    """Return a match document from the store or Riot API."""
    cached = store.load_match(match_id)
    if cached is not None:
        return cached
    try:
        match = client.fetch_match(match_id)
    except RiotApiError as exc:
        get_logger("benchmark_fetcher").debug("Skipping match %s: %s", match_id, exc)
        return None
    store.save_match(match_id, owner_puuid, match)
    ingest_match(store, match_id, match, client.platform)
    return match


def _collect_sample_rows(
    client: RiotApiClient,
    store: MatchStore,
    *,
    champion: str,
    role: str,
    ranked: RankedEntry,
    seed_puuids: list[str],
    exclude_puuid: str | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Snowball-scan recent games for matching champion + lane performances."""
    log = get_logger("benchmark_fetcher")
    scope = build_widened_scope(ranked)
    rows: list[dict[str, Any]] = []
    players_used: set[str] = set()
    downloads = 0
    seen_puuids: set[str] = set()
    seen_matches: set[str] = set()
    queue: deque[str] = deque(seed_puuids)

    progress = tqdm(total=MAX_MATCH_DOWNLOADS, desc="Sampling rank peers", unit="match")
    try:
        while queue and downloads < MAX_MATCH_DOWNLOADS and len(rows) < TARGET_PEER_GAMES:
            puuid = queue.popleft()
            if puuid in seen_puuids:
                continue
            seen_puuids.add(puuid)

            try:
                match_ids = client.fetch_match_ids(
                    puuid, MATCH_IDS_PER_PLAYER, queue_id=RANKED_SOLO_QUEUE_ID
                )
            except RiotApiError as exc:
                log.debug("Skipping %s...: %s", puuid[:12], exc)
                continue

            for match_id in match_ids:
                if len(rows) >= TARGET_PEER_GAMES or downloads >= MAX_MATCH_DOWNLOADS:
                    break
                if match_id in seen_matches:
                    continue
                seen_matches.add(match_id)

                match = _load_or_fetch_match(client, store, match_id, puuid)
                if match is None:
                    continue
                downloads += 1
                progress.update(1)

                if _match_has_build(match, champion, role):
                    for other_puuid in _participant_puuids(match):
                        if other_puuid and other_puuid not in seen_puuids:
                            queue.append(other_puuid)

                row = extract_champion_role_for_puuid(match, puuid, champion, role)
                if row is None:
                    continue
                if exclude_puuid and row.get("puuid") == exclude_puuid:
                    continue

                peer_rank = client.fetch_solo_rank(puuid)
                if peer_rank is None:
                    continue
                store.set_puuid_rank(puuid, peer_rank.tier, peer_rank.rank)
                if not rank_matches(peer_rank.tier, peer_rank.rank, scope):
                    continue

                row["match_id"] = match_id
                rows.append(row)
                players_used.add(puuid)
    finally:
        progress.close()

    return rows, len(rows), len(players_used)


def fetch_benchmark_from_api(
    client: RiotApiClient,
    store: MatchStore,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None = None,
) -> BenchmarkSnapshot | None:
    """Sample rank-scoped players and aggregate their champion + lane stats."""
    log = get_logger("benchmark_fetcher")
    scope = build_widened_scope(ranked)
    seed_puuids = _gather_seed_puuids(client, scope, exclude_puuid)
    if not seed_puuids:
        log.warning(
            "No league entries returned for %s on %s",
            ranked.label,
            client.platform,
        )
        return None

    rows, games, players = _collect_sample_rows(
        client,
        store,
        champion=champion,
        role=role,
        ranked=ranked,
        seed_puuids=seed_puuids,
        exclude_puuid=exclude_puuid,
    )
    if games < MIN_BENCHMARK_GAMES:
        log.warning(
            "Only found %d %s %s games near %s (need %d).",
            games,
            champion,
            role,
            ranked.label,
            MIN_BENCHMARK_GAMES,
        )
        return None

    metrics = _aggregate_rows(rows)
    log.info(
        "Built benchmark from %d games across %d players (%s %s)",
        games,
        players,
        champion,
        role,
    )
    return BenchmarkSnapshot(
        metrics=metrics,
        games_sampled=games,
        players_sampled=players,
        from_cache=False,
        platform=client.platform,
    )
