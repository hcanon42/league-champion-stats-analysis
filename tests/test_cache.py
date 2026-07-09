"""Tests for the SQLite match store."""

from __future__ import annotations

from pathlib import Path

from cache import MatchStore


def test_store_roundtrip(tmp_path: Path) -> None:
    """Matches and timelines persist and has_match requires both."""
    store = MatchStore(tmp_path / "m.sqlite")
    assert not store.has_match("EUW1_1")
    store.save_match("EUW1_1", "puuid-x", {"metadata": {"matchId": "EUW1_1"}})
    assert not store.has_match("EUW1_1")  # timeline still missing
    store.save_timeline("EUW1_1", {"info": {"frames": []}})
    assert store.has_match("EUW1_1")
    assert store.load_match("EUW1_1") == {"metadata": {"matchId": "EUW1_1"}}
    assert list(store.iter_match_ids("puuid-x")) == ["EUW1_1"]
    assert store.count() == 1
    store.close()
