"""Load peer games from the local match store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from league_stats.analysis.peer.ingest import ingest_match
from league_stats.analysis.peer.metrics import BENCHMARK_METRIC_KEYS
from league_stats.analysis.peer.rank_scope import RankScope, rank_matches
from league_stats.infra.cache import MatchStore
from league_stats.core.models import RankedEntry
from league_stats.infra.riot_api import RiotApiClient
from league_stats.utils import get_logger

MAX_RANK_LOOKUPS: int = 100


@dataclass(frozen=True)
class PeerSample:
    """Peer games collected for one champion + lane baseline."""

    rows: list[dict[str, Any]]
    games: int
    players: int
    source: str


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten stored peer rows into a metrics dataframe."""
    flat: list[dict[str, Any]] = []
    for row in rows:
        entry = {"puuid": row["puuid"], "match_id": row["match_id"]}
        entry.update(row["metrics"])
        flat.append(entry)
    return pd.DataFrame(flat)


def aggregate_peer_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average end-of-game metrics across peer rows."""
    if not rows:
        return {}
    frame = _rows_to_frame(rows)
    metrics: dict[str, float] = {}
    for key in BENCHMARK_METRIC_KEYS:
        if key not in frame.columns:
            continue
        series = pd.to_numeric(frame[key], errors="coerce").dropna()
        if not series.empty:
            metrics[key] = float(series.mean())
    if "win" in metrics:
        metrics["winrate"] = metrics["win"]
    return metrics


def _backfill_ranks(store: MatchStore, client: RiotApiClient | None) -> None:
    """Resolve unknown peer ranks via league-v4."""
    if client is None:
        return
    puuids = store.iter_unverified_puuids(MAX_RANK_LOOKUPS)
    for puuid in puuids:
        ranked = client.fetch_solo_rank(puuid)
        if ranked is None:
            store.set_puuid_rank(puuid, "UNRANKED", "")
            continue
        store.set_puuid_rank(puuid, ranked.tier, ranked.rank)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    scope: RankScope,
    exclude_puuid: str,
) -> list[dict[str, Any]]:
    """Keep peer rows inside the rank scope."""
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row["puuid"] == exclude_puuid:
            continue
        if not row.get("rank_verified"):
            continue
        if row.get("tier") == "UNRANKED":
            continue
        if rank_matches(str(row.get("tier", "")), str(row.get("rank", "")), scope):
            filtered.append(row)
    return filtered


def collect_peer_games_from_store(
    store: MatchStore,
    *,
    champion: str,
    role: str,
    platform: str,
    scope: RankScope,
    exclude_puuid: str,
    client: RiotApiClient | None = None,
) -> PeerSample:
    """Load peer games for a champion + lane from the persistent store."""
    log = get_logger("peer_cache")

    if store.count_peer_games(champion=champion, role=role, platform=platform) == 0:
        for match_id in store.iter_all_match_ids():
            match = store.load_match(match_id)
            if match is None:
                continue
            ingest_match(store, match_id, match, platform)

    _backfill_ranks(store, client)
    rows = store.load_peer_games(champion=champion, role=role, platform=platform)
    filtered = _filter_rows(rows, scope=scope, exclude_puuid=exclude_puuid)
    players = len({row["puuid"] for row in filtered})
    log.debug(
        "Loaded %d peer game(s) for %s %s (%d players) from store",
        len(filtered),
        champion,
        role,
        players,
    )
    return PeerSample(
        rows=filtered,
        games=len(filtered),
        players=players,
        source="cached peer store",
    )


def collect_user_history_peers(
    store: MatchStore,
    exclude_puuid: str,
    champion: str,
    role: str,
) -> pd.DataFrame:
    """Scan the tracked player's matches for same champion + lane opponents."""
    from league_stats.analysis.peer.metrics import extract_champion_role_rows

    rows: list[dict[str, Any]] = []
    for match_id in store.iter_match_ids(exclude_puuid):
        match = store.load_match(match_id)
        if not match:
            continue
        for row in extract_champion_role_rows(
            match, exclude_puuid=exclude_puuid, champion=champion, role=role
        ):
            row["match_id"] = match_id
            rows.append(row)
    return pd.DataFrame(rows)
