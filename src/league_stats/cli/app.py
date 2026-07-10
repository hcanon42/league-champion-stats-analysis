"""Champion coaching analyzer CLI."""

from __future__ import annotations

import typer

from league_stats.core.champions import normalize_role
from league_stats.core.config import load_config
from league_stats.infra.cache import HttpCache, MatchStore
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.pipeline.fetch import fetch_matches, resolve_player_contexts
from league_stats.pipeline.orchestrator import run_all_builds
from league_stats.pipeline.services import (
    PlayerContext,
    Services,
    build_services,
    parse_players_cli,
)
from league_stats.presentation.report import discover_reports, refresh_all_player_hubs, refresh_report_indexes
from league_stats.utils import get_logger, setup_logging

# Re-export pipeline symbols used by tests and external callers.
from league_stats.analysis.peer import build_peer_comparison  # noqa: F401
from league_stats.pipeline.bundles import (  # noqa: F401
    build_window_bundle as _build_window_bundle,
    default_game_window_key as _default_game_window_key,
    default_queue_filter_key as _default_queue_filter_key,
    filter_records_by_queue as _filter_records_by_queue,
    game_window_options as _game_window_options,
    queue_filter_options as _queue_filter_options,
    slice_records as _slice_records,
)
from league_stats.pipeline.fetch import group_records as _group_records  # noqa: F401
from league_stats.pipeline.orchestrator import run_analysis  # noqa: F401
from league_stats.pipeline.services import parse_players_cli as _parse_players_cli  # noqa: F401

app = typer.Typer(
    help="Ranked queue coaching analyzer for any champion + lane (Riot Match-V5 API).",
    no_args_is_help=True,
)


@app.command()
def analyze(
    player: list[str] = typer.Option(
        [],
        "--player",
        help='Riot ID as "Name#Tag". Repeat to pool multiple players into one report.',
    ),
    riot_id: str = typer.Option(None, "--riot-id", help="Riot ID game name (e.g. 'Faker')."),
    tagline: str = typer.Option(None, "--tagline", help="Riot ID tagline without '#'."),
    region: str = typer.Option(
        "europe",
        "--region",
        help="Routing region (europe/americas/asia/sea) or platform (euw1, na1...).",
    ),
    platform: str = typer.Option(
        None,
        "--platform",
        help="Platform for league-v4 rank lookup (euw1, eun1, na1...).",
    ),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY", help="Riot API key."),
    count: int = typer.Option(None, "--count", help="Max matches to scan (default 500)."),
    min_games: int = typer.Option(None, "--min-games", help="Min ranked games per build (default 20)."),
    champion: str | None = typer.Option(None, "--champion", help="Analyse only this champion."),
    role: str | None = typer.Option(None, "--role", "--lane", help="Analyse only this lane/role."),
    skip_peer: bool = typer.Option(False, "--skip-peer", help="Skip rank-peer comparison."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Run the full pipeline: download, analyse all eligible builds and generate reports."""
    players = parse_players_cli(player, riot_id, tagline)
    services = build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose, players=players
    )
    if champion or role:
        updates: dict[str, str | None] = {}
        if champion:
            updates["filter_champion"] = champion
        if role:
            updates["filter_role"] = normalize_role(role)
        services.config = services.config.model_copy(update=updates)
    try:
        contexts = fetch_matches(services)
        run_all_builds(services, contexts, fetch=False, skip_peer=skip_peer)
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def fetch(
    player: list[str] = typer.Option([], "--player", help='Riot ID as "Name#Tag". Repeat for multiple.'),
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option("europe", "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    count: int = typer.Option(None, "--count"),
    min_games: int = typer.Option(None, "--min-games"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download matches into the local store without analysing them."""
    players = parse_players_cli(player, riot_id, tagline)
    services = build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose, players=players
    )
    try:
        fetch_matches(services)
        get_logger().info("Store now holds %d complete matches.", services.store.count())
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def report(
    player: list[str] = typer.Option([], "--player", help='Riot ID as "Name#Tag". Repeat for multiple.'),
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option("europe", "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    min_games: int = typer.Option(None, "--min-games"),
    champion: str | None = typer.Option(None, "--champion", help="Analyse only this champion."),
    role: str | None = typer.Option(None, "--role", "--lane", help="Analyse only this lane/role."),
    skip_peer: bool = typer.Option(False, "--skip-peer", help="Skip rank-peer comparison."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild all eligible build reports from already-downloaded matches."""
    players = parse_players_cli(player, riot_id, tagline)
    services = build_services(
        riot_id, tagline, region, platform, api_key, None, min_games, verbose, players=players
    )
    if champion or role:
        updates: dict[str, str | None] = {}
        if champion:
            updates["filter_champion"] = champion
        if role:
            updates["filter_role"] = normalize_role(role)
        services.config = services.config.model_copy(update=updates)
    try:
        contexts = resolve_player_contexts(services)
        run_all_builds(services, contexts, fetch=False, skip_peer=skip_peer)
    finally:
        services.store.close()
        services.http_cache.close()


@app.command("ingest-peers")
def ingest_peers(
    region: str = typer.Option("europe", "--region"),
    platform: str = typer.Option(None, "--platform"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Backfill peer game rows from every match already stored locally."""
    from league_stats.analysis.peer.ingest import backfill_all_matches
    from league_stats.core.config import load_paths_config

    setup_logging(verbose)
    config = load_paths_config(region=region, platform=platform)
    config.ensure_directories()
    store = MatchStore(config.db_path)
    try:
        inserted = backfill_all_matches(store, config.routing_platform)
        get_logger().info("Peer store now holds rows for %d ingested performances.", inserted)
    finally:
        store.close()


@app.command("clear-cache")
def clear_cache(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Clear the HTTP response cache (downloaded matches are kept)."""
    from league_stats.core.config import load_paths_config

    setup_logging(verbose)
    config = load_paths_config()
    cache = HttpCache(config.http_cache_dir)
    cache.clear()
    cache.close()
    get_logger().info("HTTP cache cleared.")


@app.command("download-assets")
def download_assets(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    force: bool = typer.Option(False, "--force", help="Re-download icons even when cached."),
) -> None:
    """Download champion and keystone icons from Data Dragon for report UI."""
    from league_stats.core.config import load_paths_config

    setup_logging(verbose)
    config = load_paths_config()
    config.ensure_directories()
    assets = DDragonAssets(config)
    version = assets.ensure_downloaded(force=force)
    get_logger().info("Assets ready in %s (patch %s).", assets.assets_root, version)


@app.command()
def reports(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Rebuild the report index from saved reports (no analysis)."""
    from league_stats.core.config import load_paths_config

    setup_logging(verbose)
    config = load_paths_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    assets = DDragonAssets(config)
    assets.ensure_downloaded()
    index_path = refresh_report_indexes(
        config.output_dir,
        config.template_dir,
        assets=assets,
    )[0]
    refresh_all_player_hubs(config.output_dir, config.template_dir, assets=assets)
    count = len(discover_reports(config.output_dir))
    get_logger().info("Index refreshed with %d report(s). Open %s", count, index_path)


if __name__ == "__main__":
    app()
