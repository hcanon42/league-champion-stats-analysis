"""Tests for dynamic benchmark fetching from the Riot API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from analysis.benchmark_fetcher import (
    ensure_tier_benchmark,
    extract_champion_role_for_puuid,
    fetch_benchmark_from_api,
)
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


def test_fetch_benchmark_from_api_aggregates_league_sample(monkeypatch) -> None:
    """League entries are scanned until enough champion games are found."""
    monkeypatch.setattr("analysis.benchmark_fetcher.MIN_BENCHMARK_GAMES", 3)

    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_league_entries.return_value = [
        _league_entry(f"peer-{index}") for index in range(5)
    ]

    def match_ids(puuid: str, count: int) -> list[str]:
        return [f"EUW1_{puuid}"]

    client.fetch_match_ids.side_effect = match_ids
    client.fetch_match.side_effect = lambda match_id: _match_for(match_id.removeprefix("EUW1_"))

    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=10, losses=10)
    snapshot = fetch_benchmark_from_api(client, ranked, "Zac", "JUNGLE")
    assert snapshot is not None
    assert snapshot.games_sampled >= 3
    assert snapshot.metrics["kda"] > 0


def test_ensure_tier_benchmark_uses_cache(tmp_path, monkeypatch) -> None:
    """Fresh cache files are reused without calling the league endpoints."""
    import analysis.benchmark_fetcher as fetcher

    monkeypatch.setattr(fetcher, "BENCHMARKS_DIR", tmp_path)
    cache_path = tmp_path / "zac_jungle__euw1__gold__ii.json"
    cache_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "fetched_at": fetcher.time.time(),
                    "games_sampled": 20,
                    "players_sampled": 8,
                    "platform": "euw1",
                },
                "metrics": {"win": 0.52, "kda": 3.1, "dpm": 420.0},
            }
        ),
        encoding="utf-8",
    )

    client = MagicMock()
    client.configure_mock(platform="euw1")
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=10, losses=10)
    snapshot = ensure_tier_benchmark(client, ranked, "Zac", "JUNGLE")
    assert snapshot is not None
    assert snapshot.from_cache is True
    assert snapshot.metrics["kda"] == 3.1
    client.fetch_league_entries.assert_not_called()


def test_ensure_tier_benchmark_refetches_when_cache_expired(tmp_path, monkeypatch) -> None:
    """Expired cache files trigger a new API sample attempt."""
    import analysis.benchmark_fetcher as fetcher

    monkeypatch.setattr(fetcher, "BENCHMARK_CACHE_TTL_S", 60)
    monkeypatch.setattr(fetcher, "MIN_BENCHMARK_GAMES", 1)
    monkeypatch.setattr(fetcher, "BENCHMARKS_DIR", tmp_path)
    cache_path = tmp_path / "zac_jungle__euw1__gold__ii.json"
    cache_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "fetched_at": fetcher.time.time() - 120,
                    "games_sampled": 20,
                    "players_sampled": 8,
                    "platform": "euw1",
                },
                "metrics": {"win": 0.52},
            }
        ),
        encoding="utf-8",
    )

    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_league_entries.return_value = [_league_entry("peer-1")]
    client.fetch_match_ids.return_value = ["EUW1_peer-1"]
    client.fetch_match.return_value = _match_for("peer-1")

    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=10, losses=10)
    snapshot = ensure_tier_benchmark(client, ranked, "Zac", "JUNGLE")
    assert snapshot is not None
    assert snapshot.from_cache is False
    client.fetch_league_entries.assert_called_once()
