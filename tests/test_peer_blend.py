"""Tests for peer comparison improvements: multi-participant extraction, rank cache, file cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from league_stats.analysis.peer.benchmark_cache import read_live_cache, write_live_cache
from league_stats.analysis.peer.benchmark_fetcher import (
    BenchmarkSnapshot,
    fetch_benchmark_from_api,
)
from league_stats.analysis.peer.baseline import resolve_peer_baseline
from league_stats.analysis.peer.rank_scope import (
    build_exact_scope,
    build_wider_scope,
    build_widened_scope,
    league_lookup_pairs,
    rank_matches,
)
from league_stats.infra.cache import MatchStore
from league_stats.core.models import RankedEntry
from tests.fixtures import make_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _league_entry(puuid: str, tier: str = "GOLD", rank: str = "II") -> dict:
    return {"puuid": puuid, "tier": tier, "rank": rank}


def _match_with_build(seed_puuid: str, build_puuid: str, champion: str = "Zac", role: str = "JUNGLE") -> dict:
    """Return a match where seed_puuid is at MIDDLE and build_puuid plays champion+role."""
    match = make_match()
    match["info"]["participants"][0]["puuid"] = seed_puuid
    match["info"]["participants"][0]["championName"] = "Viktor"
    match["info"]["participants"][0]["teamPosition"] = "MIDDLE"
    match["info"]["participants"][1]["puuid"] = build_puuid
    match["info"]["participants"][1]["championName"] = champion
    match["info"]["participants"][1]["teamPosition"] = role
    return match


@pytest.fixture
def ranked_gold() -> RankedEntry:
    return RankedEntry(tier="GOLD", rank="II", league_points=50, wins=10, losses=10)


@pytest.fixture
def ranked_emerald() -> RankedEntry:
    return RankedEntry(tier="EMERALD", rank="II", league_points=50, wins=10, losses=10)


# ---------------------------------------------------------------------------
# Multi-participant extraction
# ---------------------------------------------------------------------------


def test_non_scanned_player_build_is_captured(tmp_path, monkeypatch, ranked_gold: RankedEntry) -> None:
    """A player who played the target build in a seeded player's match is captured
    even though the seeded player didn't play that build themselves."""
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.MIN_BENCHMARK_GAMES", 1)
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.TARGET_PEER_GAMES", 1)
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.MAX_MATCH_DOWNLOADS", 5)

    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")

    # Seed: "seed-player" at GOLD II (plays Viktor mid, NOT Zac jungle)
    client.fetch_league_entries_pages.return_value = [_league_entry("seed-player")]

    # Their match contains "zac-player" at Zac JUNGLE
    match = _match_with_build("seed-player", "zac-player")
    client.fetch_match_ids.return_value = ["EUW1_match1"]
    client.fetch_match.return_value = match
    # "zac-player" is not a seed, so rank must come from API
    client.fetch_solo_rank.return_value = RankedEntry(
        tier="GOLD", rank="II", league_points=50, wins=5, losses=5
    )

    snapshot = fetch_benchmark_from_api(client, store, ranked_gold, "Zac", "JUNGLE")

    assert snapshot is not None
    assert snapshot.games_sampled >= 1
    # fetch_solo_rank called for zac-player (not a seed), not for seed-player
    called_puuids = [c.args[0] for c in client.fetch_solo_rank.call_args_list]
    assert "zac-player" in called_puuids
    assert "seed-player" not in called_puuids


# ---------------------------------------------------------------------------
# Rank cache: seed ranks avoid redundant API calls
# ---------------------------------------------------------------------------


def test_seed_rank_skips_fetch_solo_rank(tmp_path, monkeypatch, ranked_gold: RankedEntry) -> None:
    """A seed player's rank from the league entry avoids a fetch_solo_rank call."""
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.MIN_BENCHMARK_GAMES", 1)
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.TARGET_PEER_GAMES", 1)
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.MAX_MATCH_DOWNLOADS", 5)
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_fetcher.MATCH_IDS_PER_PLAYER", 1)

    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")

    client.fetch_league_entries_pages.return_value = [_league_entry("seed-player")]

    # Seed player themselves play Zac JUNGLE in their match
    match = make_match()
    match["info"]["participants"][1]["puuid"] = "seed-player"
    match["info"]["participants"][1]["championName"] = "Zac"
    match["info"]["participants"][1]["teamPosition"] = "JUNGLE"

    client.fetch_match_ids.return_value = ["EUW1_seed"]
    client.fetch_match.return_value = match

    snapshot = fetch_benchmark_from_api(client, store, ranked_gold, "Zac", "JUNGLE")

    assert snapshot is not None
    # fetch_solo_rank must NOT be called for "seed-player" (rank known from league entry)
    called_puuids = [c.args[0] for c in client.fetch_solo_rank.call_args_list]
    assert "seed-player" not in called_puuids


# ---------------------------------------------------------------------------
# File cache (7-day TTL)
# ---------------------------------------------------------------------------


def test_write_and_read_live_cache(tmp_path, monkeypatch) -> None:
    """A written cache entry is read back correctly within the TTL."""
    monkeypatch.setattr(
        "league_stats.analysis.peer.benchmark_cache._LIVE_CACHE_DIR", tmp_path / "cache"
    )
    snapshot = BenchmarkSnapshot(
        metrics={"win": 0.52, "kda": 3.1, "dpm": 640.0, "cspm": 7.4,
                 "deaths": 4.5, "vspm": 0.95, "control_wards": 1.5,
                 "kill_participation": 0.56, "damage_share": 0.24},
        games_sampled=50,
        players_sampled=20,
        from_cache=False,
        platform="euw1",
    )
    write_live_cache("euw1", "GOLD", "Zac", "JUNGLE", snapshot)

    cached = read_live_cache("euw1", "GOLD", "Zac", "JUNGLE")
    assert cached is not None
    assert cached.from_cache is True
    assert cached.games_sampled == 50
    assert cached.players_sampled == 20
    assert abs(cached.metrics["kda"] - 3.1) < 0.01


def test_live_cache_expired_returns_none(tmp_path, monkeypatch) -> None:
    """A cache entry older than 7 days is ignored."""
    import time as _time

    monkeypatch.setattr(
        "league_stats.analysis.peer.benchmark_cache._LIVE_CACHE_DIR", tmp_path / "cache"
    )
    snapshot = BenchmarkSnapshot(
        metrics={"win": 0.5},
        games_sampled=20,
        players_sampled=10,
        from_cache=False,
        platform="euw1",
    )
    write_live_cache("euw1", "GOLD", "Zac", "JUNGLE", snapshot)

    # Advance clock by 8 days so the entry appears stale
    eight_days = 8 * 24 * 3600
    now = _time.time() + eight_days
    monkeypatch.setattr("league_stats.analysis.peer.benchmark_cache.time.time", lambda: now)

    cached = read_live_cache("euw1", "GOLD", "Zac", "JUNGLE")
    assert cached is None


def test_live_cache_hit_skips_live_api(tmp_path, monkeypatch, ranked_gold: RankedEntry) -> None:
    """When a fresh file cache entry exists, no live API sampling is performed."""
    monkeypatch.setattr(
        "league_stats.analysis.peer.benchmark_cache._LIVE_CACHE_DIR", tmp_path / "cache"
    )

    # Pre-populate the cache
    snapshot = BenchmarkSnapshot(
        metrics={"win": 0.5, "kda": 3.0, "dpm": 620.0, "cspm": 7.2,
                 "deaths": 4.8, "vspm": 0.9, "control_wards": 1.4,
                 "kill_participation": 0.55, "damage_share": 0.24},
        games_sampled=50,
        players_sampled=18,
        from_cache=False,
        platform="euw1",
    )
    write_live_cache("euw1", "GOLD", "Zac", "JUNGLE", snapshot)

    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")

    baseline = resolve_peer_baseline(
        client, store, ranked_gold, "Zac", "JUNGLE", exclude_puuid="puuid-me"
    )

    assert baseline is not None
    assert baseline.games == 50
    assert "Cached" in baseline.source
    # No live API calls should have been made
    client.fetch_league_entries_pages.assert_not_called()
    client.fetch_match.assert_not_called()


# ---------------------------------------------------------------------------
# Wider scope fallback (±2 tiers)
# ---------------------------------------------------------------------------


def test_wider_scope_includes_two_adjacent_tiers(ranked_emerald: RankedEntry) -> None:
    """build_wider_scope covers tiers two steps away from the player's rank."""
    scope = build_wider_scope(ranked_emerald)
    allowed = scope.allowed_tiers
    # EMERALD is adjacent to PLATINUM and DIAMOND (±1)
    assert "EMERALD" in allowed
    assert "PLATINUM" in allowed
    assert "DIAMOND" in allowed
    # GOLD is ±2 from EMERALD (EMERALD → PLATINUM → GOLD)
    assert "GOLD" in allowed


def test_rank_matches_accepts_wider_tiers(ranked_emerald: RankedEntry) -> None:
    """rank_matches returns True for players two tiers away when using wider scope."""
    scope = build_wider_scope(ranked_emerald)
    assert rank_matches("GOLD", "I", scope)


# ---------------------------------------------------------------------------
# League lookup ordering
# ---------------------------------------------------------------------------


def test_league_lookup_pairs_prioritises_player_division(ranked_gold: RankedEntry) -> None:
    """The player's exact tier+division is the first pair returned."""
    scope = build_widened_scope(ranked_gold)
    pairs = league_lookup_pairs(scope)
    assert len(pairs) > 0
    first_tier, first_div = pairs[0]
    assert first_tier == "GOLD"
    assert first_div == "II"


# ---------------------------------------------------------------------------
# Store threshold: lowered thresholds resolve at level 0/1
# ---------------------------------------------------------------------------


def test_store_threshold_requires_fifty_games(
    tmp_path, monkeypatch, ranked_gold: RankedEntry
) -> None:
    """A store with fewer than 50 games does not satisfy the exact-rank baseline."""
    import league_stats.analysis.peer.baseline as peer_baseline

    store = MatchStore(tmp_path / "matches.sqlite")
    for index in range(30):
        match = make_match()
        match["info"]["participants"][1]["puuid"] = f"peer-{index}"
        from league_stats.analysis.peer.ingest import ingest_match
        ingest_match(store, f"EUW1_{index}", match, "euw1")
        store.set_puuid_rank(f"peer-{index}", "GOLD", "II")

    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_league_entries_pages.return_value = []
    client.fetch_solo_rank.return_value = ranked_gold

    baseline = resolve_peer_baseline(
        client, store, ranked_gold, "LeeSin", "JUNGLE", exclude_puuid="puuid-me"
    )

    # 30 games is below MIN_EXACT_GAMES=50; live API also empty → static fallback
    assert baseline is not None
    assert baseline.fallback_level in {4, 5}
    assert baseline.confidence == "low"


def test_widened_store_resolves_at_fifty_games(
    tmp_path, monkeypatch, ranked_gold: RankedEntry
) -> None:
    """Games from an adjacent tier resolve at level 1 once MIN_WIDENED_GAMES is met."""
    import league_stats.analysis.peer.baseline as peer_baseline

    store = MatchStore(tmp_path / "matches.sqlite")
    for index in range(50):
        match = make_match()
        match["info"]["participants"][1]["puuid"] = f"peer-{index}"
        from league_stats.analysis.peer.ingest import ingest_match
        ingest_match(store, f"EUW1_{index}", match, "euw1")
        store.set_puuid_rank(f"peer-{index}", "PLATINUM", "I")

    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_solo_rank.return_value = ranked_gold

    baseline = resolve_peer_baseline(
        client, store, ranked_gold, "LeeSin", "JUNGLE", exclude_puuid="puuid-me"
    )

    assert baseline is not None
    assert baseline.fallback_level == 1
    assert baseline.games >= 50
    client.fetch_league_entries_pages.assert_not_called()
