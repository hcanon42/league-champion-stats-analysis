"""Tests for the persistent peer game store."""

from __future__ import annotations

from league_stats.infra.cache import MatchStore
from league_stats.analysis.peer.ingest import ingest_match
from tests.fixtures import make_match


def test_upsert_peer_game_is_idempotent(tmp_path) -> None:
    """Ingesting the same match twice only stores one row per participant."""
    store = MatchStore(tmp_path / "matches.sqlite")
    match = make_match()
    first = ingest_match(store, "EUW1_1", match, "euw1")
    second = ingest_match(store, "EUW1_1", match, "euw1")
    assert first == 10
    assert second == 0
    assert store.count_peer_games(champion="Viktor", role="MIDDLE", platform="euw1") == 1


def test_set_puuid_rank_backfills_rows(tmp_path) -> None:
    """Rank metadata is applied to every row for one player."""
    store = MatchStore(tmp_path / "matches.sqlite")
    match = make_match()
    ingest_match(store, "EUW1_1", match, "euw1")
    updated = store.set_puuid_rank("puuid-2", "EMERALD", "II")
    assert updated >= 1
    rows = store.load_peer_games(champion="LeeSin", role="JUNGLE", platform="euw1")
    assert any(row["tier"] == "EMERALD" and row["rank"] == "II" for row in rows)


def test_iter_all_match_ids(tmp_path) -> None:
    """All stored matches are visible regardless of player ownership."""
    store = MatchStore(tmp_path / "matches.sqlite")
    store.save_match("EUW1_a", "puuid-a", make_match())
    store.save_timeline("EUW1_a", {"info": {"frames": []}})
    store.save_match("EUW1_b", "puuid-b", make_match())
    store.save_timeline("EUW1_b", {"info": {"frames": []}})
    assert sorted(store.iter_all_match_ids()) == ["EUW1_a", "EUW1_b"]
