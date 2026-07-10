"""Match download and record loading."""

from __future__ import annotations

from tqdm import tqdm

from league_stats.core.models import MatchRecord
from league_stats.ingest.parser import BaseMatchFilter, ItemCatalog, MatchParser
from league_stats.pipeline.services import PlayerContext, Services
from league_stats.utils import get_logger


def fetch_matches(services: Services) -> list[PlayerContext]:
    """Resolve every tracked player and download their match histories."""
    config = services.config
    contexts: list[PlayerContext] = []
    for player in config.players:
        puuid = services.client.resolve_puuid(player.riot_id, player.tagline)
        match_ids = services.client.fetch_ranked_match_ids(puuid, config.match_count)
        services.client.download_matches(puuid, match_ids)
        contexts.append(
            PlayerContext(riot_id=player.riot_id, tagline=player.tagline, puuid=puuid)
        )
    return contexts


def resolve_player_contexts(services: Services) -> list[PlayerContext]:
    """Resolve PUUIDs for every configured player without downloading."""
    return [
        PlayerContext(
            riot_id=player.riot_id,
            tagline=player.tagline,
            puuid=services.client.resolve_puuid(player.riot_id, player.tagline),
        )
        for player in services.config.players
    ]


def load_all_records(services: Services, puuids: str | list[str]) -> list[MatchRecord]:
    """Parse stored ranked queue games for one or more players."""
    if isinstance(puuids, str):
        puuid_list = [puuids]
    else:
        puuid_list = list(puuids)
    log = get_logger("pipeline")
    catalog = ItemCatalog(services.client.fetch_item_catalog())
    match_filter = BaseMatchFilter(services.config)
    parser = MatchParser(catalog)
    records: list[MatchRecord] = []
    for puuid in puuid_list:
        match_ids = list(services.store.iter_match_ids(puuid))
        for match_id in tqdm(match_ids, desc="Parsing matches", unit="match"):
            match = services.store.load_match(match_id)
            timeline = services.store.load_timeline(match_id)
            if not match or not timeline:
                continue
            if not match_filter.accept(match, puuid):
                continue
            try:
                records.append(parser.parse(match, timeline, puuid))
            except Exception as exc:
                log.warning("Failed to parse %s: %s", match_id, exc)
    records.sort(key=lambda r: r.game_creation_ms, reverse=True)
    log.info("Parsed %d qualifying ranked queue games", len(records))
    return records


def group_records(records: list[MatchRecord], champion: str, role: str) -> list[MatchRecord]:
    """Filter parsed records to one champion + lane build."""
    return [r for r in records if r.champion == champion and r.role == role]
