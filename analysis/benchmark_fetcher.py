"""Build rank-peer benchmarks by sampling league players via the Riot API.

Riot does not expose champion-by-rank population averages directly. This module
approximates them by:

1. Listing players in the requester's solo queue league (same tier + division),
2. Scanning their recent ranked games for the configured champion + lane,
3. Averaging end-of-game stats across the collected sample.

Results are cached on disk for :data:`BENCHMARK_CACHE_TTL_S` so later runs do
not repeat the full scan.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
from tqdm import tqdm

from analysis.benchmarks import BENCHMARKS_DIR
from champions import champion_slug
from config import REMAKE_MAX_DURATION_S, RANKED_SOLO_QUEUE_ID
from models import RankedEntry
from riot_api import RiotApiClient, RiotApiError
from utils import get_logger, safe_div

MIN_BENCHMARK_GAMES: Final[int] = 12
MAX_PLAYERS_TO_SCAN: Final[int] = 25
MATCH_IDS_PER_PLAYER: Final[int] = 20
MAX_MATCH_DOWNLOADS: Final[int] = 80
BENCHMARK_CACHE_TTL_S: Final[float] = 7 * 24 * 3600

BENCHMARK_METRIC_KEYS: Final[tuple[str, ...]] = (
    "win",
    "kda",
    "dpm",
    "cspm",
    "deaths",
    "vspm",
    "control_wards",
    "kill_participation",
    "damage_share",
)


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Peer baseline built from Riot API sampling or a recent cache."""

    metrics: dict[str, float]
    games_sampled: int
    players_sampled: int
    from_cache: bool
    platform: str


def benchmark_cache_path(
    champion: str, role: str, platform: str, tier: str, rank: str
) -> Path:
    """Return the on-disk cache path for a champion + lane + rank sample."""
    slug = champion_slug(champion, role)
    rank_part = rank.lower() if rank else "masterplus"
    return BENCHMARKS_DIR / f"{slug}__{platform}__{tier.lower()}__{rank_part}.json"


def _participant_row(participant: dict[str, Any], duration_min: float) -> dict[str, Any]:
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
    return {
        "puuid": str(participant.get("puuid", "")),
        "win": int(bool(participant.get("win"))),
        "kda": (kills + assists) / max(1, deaths),
        "dpm": damage / minutes,
        "cspm": cs / minutes,
        "deaths": float(deaths),
        "vspm": int(participant.get("visionScore", 0)) / minutes,
        "control_wards": float(int(participant.get("visionWardsBoughtInGame", 0))),
        "kill_participation": float(challenges.get("killParticipation", 0.0)),
        "damage_share": safe_div(damage, damage),
        "gold": gold,
        "damage": damage,
    }


def extract_champion_role_for_puuid(
    match: dict[str, Any],
    puuid: str,
    champion: str,
    role: str,
) -> dict[str, Any] | None:
    """Return end-of-game stats when a player used the configured champion + lane."""
    info = match.get("info", {})
    if int(info.get("queueId", 0)) != RANKED_SOLO_QUEUE_ID:
        return None
    duration_s = int(info.get("gameDuration", 0))
    if duration_s > 100_000:
        duration_s //= 1000
    if duration_s <= REMAKE_MAX_DURATION_S:
        return None
    duration_min = duration_s / 60.0
    participants: list[dict[str, Any]] = info.get("participants", [])
    team_damage: dict[int, int] = {}
    for participant in participants:
        team_id = int(participant.get("teamId", 0))
        team_damage[team_id] = team_damage.get(team_id, 0) + int(
            participant.get("totalDamageDealtToChampions", 0)
        )
    for participant in participants:
        if str(participant.get("puuid", "")) != puuid:
            continue
        if str(participant.get("championName", "")) != champion:
            return None
        if str(participant.get("teamPosition", "")) != role:
            return None
        row = _participant_row(participant, duration_min)
        team_id = int(participant.get("teamId", 0))
        row["damage_share"] = safe_div(row["damage"], team_damage.get(team_id, row["damage"]))
        return row
    return None


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


def _load_cache(path: Path) -> BenchmarkSnapshot | None:
    """Load a cached benchmark when metadata matches and TTL has not expired."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    meta = payload.get("_meta", {})
    fetched_at = float(meta.get("fetched_at", 0))
    if time.time() - fetched_at > BENCHMARK_CACHE_TTL_S:
        return None

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None

    return BenchmarkSnapshot(
        metrics={key: float(value) for key, value in metrics.items()},
        games_sampled=int(meta.get("games_sampled", 0)),
        players_sampled=int(meta.get("players_sampled", 0)),
        from_cache=True,
        platform=str(meta.get("platform", "")),
    )


def _save_cache(
    path: Path,
    *,
    champion: str,
    role: str,
    platform: str,
    tier: str,
    rank: str,
    metrics: dict[str, float],
    games_sampled: int,
    players_sampled: int,
) -> None:
    """Persist a freshly sampled benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "champion": champion,
            "role": role,
            "platform": platform,
            "tier": tier.upper(),
            "rank": rank.upper() if rank else "",
            "fetched_at": time.time(),
            "games_sampled": games_sampled,
            "players_sampled": players_sampled,
        },
        "metrics": {key: round(value, 4) for key, value in metrics.items()},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _collect_sample_rows(
    client: RiotApiClient,
    *,
    champion: str,
    role: str,
    puuids: list[str],
    exclude_puuid: str | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Scan recent games for matching champion + lane performances."""
    log = get_logger("benchmark_fetcher")
    rows: list[dict[str, Any]] = []
    players_used: set[str] = set()
    downloads = 0

    for puuid in tqdm(puuids, desc="Sampling rank peers", unit="player"):
        if len(rows) >= MIN_BENCHMARK_GAMES and downloads >= MIN_BENCHMARK_GAMES:
            break
        if downloads >= MAX_MATCH_DOWNLOADS:
            break

        try:
            match_ids = client.fetch_match_ids(puuid, MATCH_IDS_PER_PLAYER)
        except RiotApiError as exc:
            log.debug("Skipping %s...: %s", puuid[:12], exc)
            continue

        for match_id in match_ids:
            if len(rows) >= MIN_BENCHMARK_GAMES * 3:
                break
            if downloads >= MAX_MATCH_DOWNLOADS:
                break

            try:
                match = client.fetch_match(match_id)
            except RiotApiError as exc:
                log.debug("Skipping match %s: %s", match_id, exc)
                continue
            downloads += 1

            row = extract_champion_role_for_puuid(match, puuid, champion, role)
            if row is None:
                continue
            if exclude_puuid and row.get("puuid") == exclude_puuid:
                continue
            row["match_id"] = match_id
            rows.append(row)
            players_used.add(puuid)

    return rows, len(rows), len(players_used)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average comparable metrics across sampled games."""
    frame = pd.DataFrame(rows)
    metrics = _average_metrics(frame, list(BENCHMARK_METRIC_KEYS))
    if "win" in metrics:
        metrics["winrate"] = metrics["win"]
    return metrics


def fetch_benchmark_from_api(
    client: RiotApiClient,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None = None,
) -> BenchmarkSnapshot | None:
    """Sample same-rank players and aggregate their champion + lane stats."""
    log = get_logger("benchmark_fetcher")
    entries = client.fetch_league_entries(ranked.tier, ranked.rank)
    puuids = [
        str(entry["puuid"])
        for entry in entries
        if entry.get("puuid") and str(entry["puuid"]) != (exclude_puuid or "")
    ]
    if not puuids:
        log.warning(
            "No league entries returned for %s %s on %s",
            ranked.tier,
            ranked.rank or "Master+",
            client.platform,
        )
        return None

    random.shuffle(puuids)
    puuids = puuids[:MAX_PLAYERS_TO_SCAN]

    rows, games, players = _collect_sample_rows(
        client,
        champion=champion,
        role=role,
        puuids=puuids,
        exclude_puuid=exclude_puuid,
    )
    if games < MIN_BENCHMARK_GAMES:
        log.warning(
            "Only found %d %s %s games at %s (need %d). "
            "Try again later or pick a more popular champion.",
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


def ensure_tier_benchmark(
    client: RiotApiClient,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None = None,
) -> BenchmarkSnapshot | None:
    """Return a peer baseline, using cache when fresh else sampling via Riot API."""
    cache_path = benchmark_cache_path(
        champion, role, client.platform, ranked.tier, ranked.rank
    )
    cached = _load_cache(cache_path)
    if cached is not None:
        get_logger("benchmark_fetcher").info("Using cached benchmark: %s", cache_path.name)
        return cached

    snapshot = fetch_benchmark_from_api(
        client,
        ranked,
        champion,
        role,
        exclude_puuid=exclude_puuid,
    )
    if snapshot is None:
        return None

    _save_cache(
        cache_path,
        champion=champion,
        role=role,
        platform=client.platform,
        tier=ranked.tier,
        rank=ranked.rank,
        metrics=snapshot.metrics,
        games_sampled=snapshot.games_sampled,
        players_sampled=snapshot.players_sampled,
    )
    return snapshot
