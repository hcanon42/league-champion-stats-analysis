"""Tests for multi-player pooling and CLI player parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from league_stats.infra.cache import MatchStore
from league_stats.core.config import AppConfig, PlayerIdentity, load_config
from league_stats.cli.app import PlayerContext, Services, _group_records, _parse_players_cli, run_all_builds
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.ingest.parser import ItemCatalog, MatchParser, discover_build_pools
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_player_match, make_timeline
from tests.test_build_pools import _config, _seed_store

ALT_PUUID = "alt-puuid-22222222-2222-2222-2222-222222222222"


def test_parse_players_cli_accepts_repeatable_flag() -> None:
    """--player flags are parsed into identities."""
    players = _parse_players_cli(["Alice#EUW", "Bob#NA1"], None, None)
    assert players == [
        PlayerIdentity(riot_id="Alice", tagline="EUW"),
        PlayerIdentity(riot_id="Bob", tagline="NA1"),
    ]


def test_parse_players_cli_prefers_player_over_riot_id() -> None:
    """Explicit --player wins over --riot-id/--tagline."""
    players = _parse_players_cli(["Solo#EUW"], "Other", "EUW")
    assert players == [PlayerIdentity(riot_id="Solo", tagline="EUW")]


def test_discover_build_pools_pools_multiple_players(tmp_path: Path) -> None:
    """Champion+lane counts combine games from every tracked player."""
    config = _config(tmp_path)
    store = MatchStore(config.db_path)
    _seed_store(store, MY_PUUID, viktor=10, ahri=0)
    for index in range(15):
        match_id = f"EUW1_alt_v{index}"
        store.save_match(
            match_id,
            ALT_PUUID,
            make_player_match(
                match_id, champion="Viktor", position="MIDDLE", puuid=ALT_PUUID
            ),
        )
        store.save_timeline(match_id, make_timeline())
    try:
        pools = discover_build_pools(store, [MY_PUUID, ALT_PUUID], config, min_games=20)
        assert len(pools) == 1
        assert pools[0].champion == "Viktor"
        assert pools[0].games == 25
    finally:
        store.close()


def test_group_records_pools_same_build_across_players() -> None:
    """Grouped records include every player's games for one build."""
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    mine = parser.parse(
        make_player_match("EUW1_1", champion="Viktor", position="MIDDLE"),
        make_timeline(),
        MY_PUUID,
    )
    alt_match = make_player_match(
        "EUW1_2", champion="Viktor", position="MIDDLE", puuid=ALT_PUUID
    )
    theirs = parser.parse(alt_match, make_timeline(), ALT_PUUID)
    grouped = _group_records([mine, theirs], "Viktor", "MIDDLE")
    assert len(grouped) == 2


def test_multi_player_config_uses_group_slug(tmp_path: Path) -> None:
    """Multiple players share one report directory slug."""
    config = load_config(
        api_key="RGAPI-test",
        riot_id="Alice",
        tagline="EUW",
        players=[
            PlayerIdentity(riot_id="Alice", tagline="EUW"),
            PlayerIdentity(riot_id="Bob", tagline="NA1"),
        ],
        output_dir=tmp_path / "output",
    )
    assert config.players_label == "Alice#EUW, Bob#NA1"
    assert config.reports_group_slug == "alice_euw__bob_na1"


def test_run_all_builds_pools_multi_player_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch analysis pools qualifying games from multiple players."""
    from league_stats.infra.cache import HttpCache
    from league_stats.core.models import RankedEntry
    from league_stats.infra.riot_api import RiotApiClient

    config = load_config(
        api_key="RGAPI-test",
        riot_id="Alice",
        tagline="EUW",
        players=[
            PlayerIdentity(riot_id="Alice", tagline="EUW"),
            PlayerIdentity(riot_id="Bob", tagline="NA1"),
        ],
        min_games=20,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        template_dir=Path(__file__).resolve().parent.parent / "src/league_stats/presentation/templates",
    )
    config.ensure_directories()
    store = MatchStore(config.db_path)
    http_cache = HttpCache(config.http_cache_dir)
    client = RiotApiClient(config, http_cache, store)
    _seed_store(store, MY_PUUID, viktor=10, ahri=0)
    for index in range(15):
        match_id = f"EUW1_alt_v{index}"
        store.save_match(
            match_id,
            ALT_PUUID,
            make_player_match(
                match_id, champion="Viktor", position="MIDDLE", puuid=ALT_PUUID
            ),
        )
        store.save_timeline(match_id, make_timeline())

    monkeypatch.setattr(
        client,
        "fetch_solo_rank",
        lambda puuid: RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75),
    )
    monkeypatch.setattr(client, "fetch_item_catalog", lambda: FAKE_ITEMS)

    services = Services(
        config=config,
        http_cache=http_cache,
        store=store,
        client=client,
        assets=DDragonAssets(config),
    )
    contexts = [
        PlayerContext(riot_id="Alice", tagline="EUW", puuid=MY_PUUID),
        PlayerContext(riot_id="Bob", tagline="NA1", puuid=ALT_PUUID),
    ]
    try:
        hub_path = run_all_builds(services, contexts, fetch=False, skip_peer=True)
    finally:
        store.close()
        http_cache.close()

    report_html = (config.player_reports_dir / "viktor_middle" / "report.html").read_text(
        encoding="utf-8"
    )
    assert hub_path.exists()
    assert "Alice#EUW, Bob#NA1" in report_html
    assert "25 games" in report_html or ">25<" in report_html
