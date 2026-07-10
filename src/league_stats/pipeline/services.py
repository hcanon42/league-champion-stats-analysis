"""Composition root and wired application services."""

from __future__ import annotations

from dataclasses import dataclass

import typer

from league_stats.core.champions import parse_riot_id
from league_stats.core.config import AppConfig, PlayerIdentity, load_config
from league_stats.infra.cache import HttpCache, MatchStore
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.infra.riot_api import RiotApiClient
from league_stats.utils import setup_logging


@dataclass
class Services:
    """Wired application services (composition root for DI)."""

    config: AppConfig
    http_cache: HttpCache
    store: MatchStore
    client: RiotApiClient
    assets: DDragonAssets


@dataclass
class PlayerContext:
    """Resolved player identity and PUUID."""

    riot_id: str
    tagline: str
    puuid: str

    @property
    def label(self) -> str:
        return f"{self.riot_id}#{self.tagline}"


def parse_players_cli(
    player_flags: list[str],
    riot_id: str | None,
    tagline: str | None,
) -> list[PlayerIdentity] | None:
    """Resolve CLI player identities from ``--player`` or ``--riot-id``/``--tagline``."""
    if player_flags:
        players: list[PlayerIdentity] = []
        for value in player_flags:
            try:
                name, tag = parse_riot_id(value)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            players.append(PlayerIdentity(riot_id=name, tagline=tag))
        return players
    if riot_id and tagline:
        return [PlayerIdentity(riot_id=riot_id, tagline=tagline)]
    return None


def build_services(
    riot_id: str | None,
    tagline: str | None,
    region: str | None,
    platform: str | None,
    api_key: str | None,
    count: int | None,
    min_games: int | None,
    verbose: bool,
    *,
    players: list[PlayerIdentity] | None = None,
) -> Services:
    """Load configuration and construct every service."""
    setup_logging(verbose)
    config = load_config(
        riot_id=riot_id,
        tagline=tagline,
        region=region,
        platform=platform,
        api_key=api_key,
        match_count=count,
        min_games=min_games,
        verbose=verbose,
        players=players,
    )
    config.ensure_directories()
    http_cache = HttpCache(config.http_cache_dir)
    store = MatchStore(config.db_path)
    client = RiotApiClient(config, http_cache, store)
    assets = DDragonAssets(config)
    return Services(
        config=config,
        http_cache=http_cache,
        store=store,
        client=client,
        assets=assets,
    )
