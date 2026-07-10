"""Tests for dynamic benchmark fetching from the Riot API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from analysis.benchmark_fetcher import (
    extract_champion_role_for_puuid,
    fetch_benchmark_from_api,
)
from cache import MatchStore
from models import RankedEntry
from tests.fixtures import make_match


def _league_entry(puuid: str) -> dict[str, str]:
    return {"puuid": puuid, "tier": "GOLD", "rank": "II"}


def _match_for(puuid: str, champion: str = "Zac", role: str = "JUNGLE") -> dict:
    match = make_match()
    participant = match["info"]["participants"][1]
    participant["puuid"] = puuid
    participant["championName"] = champion
    participant["teamPosition"] = role
    return match


def test_extract_champion_role_for_puuid_finds_player() -> None:
    """A matching participant row is returned for the requested player."""
    row = extract_champion_role_for_puuid(
        _match_for("peer-1"), "peer-1", "Zac", "JUNGLE"
    )
    assert row is not None
    assert row["puuid"] == "peer-1"
    assert row["dpm"] > 0


def test_extract_champion_role_for_puuid_filters_lane() -> None:
    """Wrong lane returns None."""
    row = extract_champion_role_for_puuid(
        _match_for("peer-1", role="TOP"), "peer-1", "Zac", "JUNGLE"
    )
    assert row is None


def test_fetch_benchmark_from_api_aggregates_league_sample(tmp_path, monkeypatch) -> None:
    """League entries are scanned until enough champion games are found."""
    monkeypatch.setattr("analysis.benchmark_fetcher.MIN_BENCHMARK_GAMES", 3)
    monkeypatch.setattr("analysis.benchmark_fetcher.TARGET_PEER_GAMES", 3)
    monkeypatch.setattr("analysis.benchmark_fetcher.MAX_MATCH_DOWNLOADS", 10)

    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_league_entries_pages.return_value = [
        _league_entry(f"peer-{index}") for index in range(5)
    ]

    def match_ids(puuid: str, count: int, queue_id: int | None = None) -> list[str]:
        return [f"EUW1_{puuid}"]

    client.fetch_match_ids.side_effect = match_ids
    client.fetch_match.side_effect = lambda match_id: _match_for(match_id.removeprefix("EUW1_"))
    client.fetch_solo_rank.return_value = RankedEntry(
        tier="GOLD", rank="II", league_points=45, wins=10, losses=10
    )

    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=10, losses=10)
    snapshot = fetch_benchmark_from_api(client, store, ranked, "Zac", "JUNGLE")
    assert snapshot is not None
    assert snapshot.games_sampled >= 3
    assert snapshot.metrics["kda"] > 0
    assert store.count_peer_games(champion="Zac", role="JUNGLE", platform="euw1") >= 3


def test_fetch_benchmark_persists_downloaded_matches(tmp_path, monkeypatch) -> None:
    """Downloaded peer matches are stored for later runs."""
    monkeypatch.setattr("analysis.benchmark_fetcher.MIN_BENCHMARK_GAMES", 1)
    monkeypatch.setattr("analysis.benchmark_fetcher.TARGET_PEER_GAMES", 1)

    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_league_entries_pages.return_value = [_league_entry("peer-1")]
    client.fetch_match_ids.return_value = ["EUW1_peer-1"]
    client.fetch_match.return_value = _match_for("peer-1")
    client.fetch_solo_rank.return_value = RankedEntry(
        tier="GOLD", rank="II", league_points=45, wins=10, losses=10
    )

    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=10, losses=10)
    snapshot = fetch_benchmark_from_api(client, store, ranked, "Zac", "JUNGLE")
    assert snapshot is not None
    assert store.load_match("EUW1_peer-1") is not None
