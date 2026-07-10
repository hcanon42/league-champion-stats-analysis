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


def test_duo_queue_keeps_both_player_ownership(tmp_path: Path) -> None:
    """A shared match is indexed for every tracked player."""
    store = MatchStore(tmp_path / "m.sqlite")
    payload = {"metadata": {"matchId": "EUW1_duo"}, "info": {"gameCreation": 1}}
    store.save_match("EUW1_duo", "puuid-a", payload)
    store.save_timeline("EUW1_duo", {"info": {"frames": []}})
    store.save_match("EUW1_duo", "puuid-b", payload)
    assert list(store.iter_match_ids("puuid-a")) == ["EUW1_duo"]
    assert list(store.iter_match_ids("puuid-b")) == ["EUW1_duo"]
    assert store.count() == 1
    store.close()


def test_claim_ownership_links_cached_matches(tmp_path: Path) -> None:
    """A player can index matches downloaded for another account."""
    store = MatchStore(tmp_path / "m.sqlite")
    payload = {"metadata": {"matchId": "EUW1_shared"}, "info": {"gameCreation": 1}}
    store.save_match("EUW1_shared", "puuid-peer", payload)
    store.save_timeline("EUW1_shared", {"info": {"frames": []}})
    assert list(store.iter_match_ids("puuid-me")) == []
    assert store.claim_ownership("puuid-me", ["EUW1_shared", "EUW1_missing"]) == 1
    assert list(store.iter_match_ids("puuid-me")) == ["EUW1_shared"]
    assert store.claim_ownership("puuid-me", ["EUW1_shared"]) == 0
    store.close()


def test_legacy_schema_migrates_puuid_to_match_players(tmp_path: Path) -> None:
    """Existing databases with matches.puuid are migrated on open."""
    import sqlite3

    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE matches (
            match_id TEXT PRIMARY KEY,
            puuid    TEXT NOT NULL,
            payload  TEXT NOT NULL
        );
        CREATE TABLE timelines (
            match_id TEXT PRIMARY KEY,
            payload  TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO matches VALUES (?, ?, ?)",
        ("EUW1_old", "puuid-legacy", '{"metadata": {"matchId": "EUW1_old"}}'),
    )
    conn.execute(
        "INSERT INTO timelines VALUES (?, ?)",
        ("EUW1_old", '{"info": {"frames": []}}'),
    )
    conn.commit()
    conn.close()

    migrated = MatchStore(db_path)
    assert list(migrated.iter_match_ids("puuid-legacy")) == ["EUW1_old"]
    cols = {row[1] for row in migrated._conn.execute("PRAGMA table_info(matches)")}
    assert "puuid" not in cols
    migrated.close()
