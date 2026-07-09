"""Tests for build pool discovery and batch grouping."""

from __future__ import annotations

from pathlib import Path

import pytest

from cache import MatchStore
from config import AppConfig
from main import _group_records, run_all_builds
from parser import ItemCatalog, MatchParser, discover_build_pools
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_player_match, make_timeline


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        riot_id="Test",
        tagline="EUW",
        region="europe",
        api_key="RGAPI-test",
        min_games=20,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        template_dir=Path(__file__).resolve().parent.parent / "templates",
    )


def _seed_store(store: MatchStore, puuid: str, *, viktor: int, ahri: int) -> None:
    for index in range(viktor):
        match_id = f"EUW1_v{index}"
        store.save_match(match_id, puuid, make_player_match(match_id, champion="Viktor", position="MIDDLE"))
        store.save_timeline(match_id, make_timeline())
    for index in range(ahri):
        match_id = f"EUW1_a{index}"
        store.save_match(match_id, puuid, make_player_match(match_id, champion="Ahri", position="MIDDLE"))
        store.save_timeline(match_id, make_timeline())


def test_discover_build_pools_respects_min_games(tmp_path: Path) -> None:
    """Only champion+lane pairs with enough games are returned."""
    config = _config(tmp_path)
    store = MatchStore(config.db_path)
    _seed_store(store, MY_PUUID, viktor=25, ahri=10)
    try:
        pools = discover_build_pools(store, MY_PUUID, config, min_games=20)
        assert len(pools) == 1
        assert pools[0].champion == "Viktor"
        assert pools[0].role == "MIDDLE"
        assert pools[0].games == 25
    finally:
        store.close()


def test_discover_build_pools_treats_lanes_separately(tmp_path: Path) -> None:
    """Same champion on different lanes counts as separate builds."""
    config = _config(tmp_path)
    store = MatchStore(config.db_path)
    for index in range(20):
        match_id = f"EUW1_t{index}"
        store.save_match(
            match_id,
            MY_PUUID,
            make_player_match(match_id, champion="Akali", position="TOP"),
        )
        store.save_timeline(match_id, make_timeline())
    for index in range(20):
        match_id = f"EUW1_m{index}"
        store.save_match(
            match_id,
            MY_PUUID,
            make_player_match(match_id, champion="Akali", position="MIDDLE"),
        )
        store.save_timeline(match_id, make_timeline())
    try:
        pools = discover_build_pools(store, MY_PUUID, config, min_games=20)
        assert len(pools) == 2
        labels = {pool.build_label for pool in pools}
        assert labels == {"Akali top", "Akali mid"}
    finally:
        store.close()


def test_group_records_filters_by_champion_and_lane() -> None:
    """Grouped records match one build only."""
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    viktor = parser.parse(
        make_player_match("EUW1_1", champion="Viktor", position="MIDDLE"),
        make_timeline(),
        MY_PUUID,
    )
    ahri = parser.parse(
        make_player_match("EUW1_2", champion="Ahri", position="MIDDLE"),
        make_timeline(),
        MY_PUUID,
    )
    grouped = _group_records([viktor, ahri], "Viktor", "MIDDLE")
    assert len(grouped) == 1
    assert grouped[0].champion == "Viktor"


def test_run_all_builds_generates_player_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch analysis writes every eligible report and a player hub."""
    from cache import HttpCache
    from main import Services
    from models import RankedEntry
    from riot_api import RiotApiClient

    config = _config(tmp_path)
    config.ensure_directories()
    store = MatchStore(config.db_path)
    http_cache = HttpCache(config.http_cache_dir)
    client = RiotApiClient(config, http_cache, store)
    _seed_store(store, MY_PUUID, viktor=20, ahri=20)

    monkeypatch.setattr(
        "main.build_peer_comparison",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        client,
        "fetch_solo_rank",
        lambda puuid: RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75),
    )
    monkeypatch.setattr(
        client,
        "fetch_item_catalog",
        lambda: FAKE_ITEMS,
    )

    services = Services(config=config, http_cache=http_cache, store=store, client=client)
    try:
        hub_path = run_all_builds(services, MY_PUUID, fetch=False)
    finally:
        store.close()
        http_cache.close()

    assert hub_path.exists()
    hub_html = hub_path.read_text(encoding="utf-8")
    assert "Viktor mid" in hub_html
    assert "Ahri mid" in hub_html
    assert (config.player_reports_dir / "manifest.json").exists()
    assert (config.player_reports_dir / "viktor_middle" / "report.html").exists()
    assert (config.player_reports_dir / "ahri_middle" / "report.html").exists()

    report_html = (config.player_reports_dir / "viktor_middle" / "report.html").read_text(
        encoding="utf-8"
    )
    assert 'id="build-switcher"' in report_html
    assert "../ahri_middle/report.html" in report_html
