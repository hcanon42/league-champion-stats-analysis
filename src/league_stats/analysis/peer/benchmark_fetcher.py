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
    extract_champion_role_rows,
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

MIN_BENCHMARK_GAMES: Final[int] = 50
TARGET_PEER_GAMES: Final[int] = 50
MAX_PLAYERS_TO_SCAN: Final[int] = 150
MATCH_IDS_PER_PLAYER: Final[int] = 30
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


def _gather_seeds(
    client: RiotApiClient,
    scope: RankScope,
    exclude_puuid: str | None,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Collect league entry PUUIDs and their known ranks across the configured rank scope.

    Returns a (puuids, rank_cache) pair where rank_cache maps PUUID to (tier, rank)
    using the data already present in the league-v4 response, avoiding extra API calls.
    """
    seen: set[str] = set()
    puuids: list[str] = []
    rank_cache: dict[str, tuple[str, str]] = {}
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
            entry_tier = (str(entry.get("tier", "")) or tier).upper()
            entry_rank = (str(entry.get("rank", "")) or division).upper()
            rank_cache[puuid] = (entry_tier, entry_rank)
            if len(puuids) >= MAX_LEAGUE_CANDIDATES:
                selected = puuids[:MAX_PLAYERS_TO_SCAN]
                return selected, {p: rank_cache[p] for p in selected}
    random.shuffle(puuids)
    selected = puuids[:MAX_PLAYERS_TO_SCAN]
    return selected, {p: rank_cache[p] for p in selected if p in rank_cache}


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


def _resolve_rank(
    puuid: str,
    rank_cache: dict[str, tuple[str, str]],
    client: RiotApiClient,
    store: MatchStore,
) -> tuple[str, str] | None:
    """Return (tier, rank) for a PUUID using cache before falling back to the API.

    Updates the in-memory cache and the peer store for any newly-fetched ranks.
    Returns None when the player is unranked or the lookup fails.
    """
    if puuid in rank_cache:
        tier, rank = rank_cache[puuid]
        return None if tier == "UNRANKED" else (tier, rank)

    peer_rank = client.fetch_solo_rank(puuid)
    if peer_rank is None:
        rank_cache[puuid] = ("UNRANKED", "")
        store.set_puuid_rank(puuid, "UNRANKED", "")
        return None

    rank_cache[puuid] = (peer_rank.tier, peer_rank.rank)
    store.set_puuid_rank(puuid, peer_rank.tier, peer_rank.rank)
    return peer_rank.tier, peer_rank.rank


def _collect_sample_rows(
    client: RiotApiClient,
    store: MatchStore,
    *,
    champion: str,
    role: str,
    ranked: RankedEntry,
    seed_puuids: list[str],
    seed_ranks: dict[str, tuple[str, str]],
    exclude_puuid: str | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Snowball-scan recent games for matching champion + lane performances.

    Extracts rows from ALL participants who played the target build in each downloaded
    match (not just the currently-scanned player), and uses the rank data already
    present in league entry seeds to avoid redundant fetch_solo_rank calls.
    Exits as soon as TARGET_PEER_GAMES rows are collected.
    """
    log = get_logger("benchmark_fetcher")
    scope = build_widened_scope(ranked)
    rows: list[dict[str, Any]] = []
    players_used: set[str] = set()
    downloads = 0
    seen_for_snowball: set[str] = set()
    seen_matches: set[str] = set()
    rank_cache: dict[str, tuple[str, str]] = dict(seed_ranks)
    queue: deque[str] = deque(seed_puuids)

    progress = tqdm(total=MAX_MATCH_DOWNLOADS, desc="Sampling rank peers", unit="match")
    try:
        while queue and downloads < MAX_MATCH_DOWNLOADS and len(rows) < TARGET_PEER_GAMES:
            puuid = queue.popleft()
            if puuid in seen_for_snowball:
                continue
            seen_for_snowball.add(puuid)

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

                if not _match_has_build(match, champion, role):
                    continue

                # Snowball: enqueue all participants for future scanning
                for other_puuid in _participant_puuids(match):
                    if other_puuid and other_puuid not in seen_for_snowball:
                        queue.append(other_puuid)

                # Extract ALL players who played the target champion+lane in this match,
                # not just the currently-scanned player. Since each ranked match has at
                # most one player per champion, this extracts the one Azir (or whoever)
                # from every match that contains them, regardless of who we're scanning.
                match_rows = extract_champion_role_rows(
                    match,
                    exclude_puuid=exclude_puuid or "",
                    champion=champion,
                    role=role,
                )
                for row in match_rows:
                    p_puuid = str(row.get("puuid", ""))
                    if not p_puuid:
                        continue
                    resolved = _resolve_rank(p_puuid, rank_cache, client, store)
                    if resolved is None:
                        continue
                    tier, rank_str = resolved
                    if not rank_matches(tier, rank_str, scope):
                        continue

                    row["match_id"] = match_id
                    rows.append(row)
                    players_used.add(p_puuid)

                    if len(rows) >= TARGET_PEER_GAMES:
                        break
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
    seed_puuids, seed_ranks = _gather_seeds(client, scope, exclude_puuid)
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
        seed_ranks=seed_ranks,
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
