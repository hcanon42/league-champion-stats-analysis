"""Tests for the SQLite match store."""

from __future__ import annotations

from pathlib import Path

from cache import MatchStore


def test_store_roundtrip(tmp_path: Path) -> None:
    """Matches and timelines persist and has_match requires both."""
    store = MatchStore(tmp_path / "m.sqlite")
    assert not store.has_match("EUW1_1")
    store.save_match("EUW1_1", {"metadata": {"matchId": "EUW1_1"}, "info": {"gameCreation": 100}})
    store.register_ownership("EUW1_1", "puuid-x")
    assert not store.has_match("EUW1_1")  # timeline still missing
    store.save_timeline("EUW1_1", {"info": {"frames": []}})
    assert store.has_match("EUW1_1")
    assert store.load_match("EUW1_1") == {
        "metadata": {"matchId": "EUW1_1"},
        "info": {"gameCreation": 100},
    }
    assert list(store.iter_match_ids("puuid-x")) == ["EUW1_1"]
    assert store.count() == 1
    store.close()


def test_iter_match_ids_orders_by_recency_and_respects_limit(tmp_path: Path) -> None:
    """iter_match_ids returns most-recent-first, capped by limit."""
    store = MatchStore(tmp_path / "m.sqlite")
    for match_id, creation in [("EUW1_1", 100), ("EUW1_2", 300), ("EUW1_3", 200)]:
        store.save_match(match_id, {"info": {"gameCreation": creation}})
        store.save_timeline(match_id, {"info": {"frames": []}})
        store.register_ownership(match_id, "puuid-x")
    assert list(store.iter_match_ids("puuid-x")) == ["EUW1_2", "EUW1_3", "EUW1_1"]
    assert list(store.iter_match_ids("puuid-x", limit=2)) == ["EUW1_2", "EUW1_3"]
    store.close()


def test_match_ownership_shared_across_players(tmp_path: Path) -> None:
    """A match seen by two players keeps both ownership rows (duo queue)."""
    store = MatchStore(tmp_path / "m.sqlite")
    store.save_match("EUW1_1", {"info": {"gameCreation": 100}})
    store.save_timeline("EUW1_1", {"info": {"frames": []}})
    store.register_ownership("EUW1_1", "puuid-a")
    store.register_ownership("EUW1_1", "puuid-b")
    assert list(store.iter_match_ids("puuid-a")) == ["EUW1_1"]
    assert list(store.iter_match_ids("puuid-b")) == ["EUW1_1"]
    store.close()
