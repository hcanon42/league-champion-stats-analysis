"""Persistence layer: HTTP response cache and SQLite match store.

Two complementary stores are used:

* :class:`HttpCache` — a :mod:`diskcache` wrapper that memoises raw API
  responses (account lookups, match-id pages, static data) with TTLs.
* :class:`MatchStore` — a :mod:`sqlite3` database holding raw match and
  timeline JSON documents forever, guaranteeing a match is never downloaded
  twice. Player ownership is tracked in a separate ``match_players`` join
  table so duo-queue games are indexed for every tracked player.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

import diskcache

from utils import get_logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    payload  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timelines (
    match_id TEXT PRIMARY KEY,
    payload  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS match_players (
    match_id TEXT NOT NULL,
    puuid    TEXT NOT NULL,
    PRIMARY KEY (match_id, puuid)
);
CREATE INDEX IF NOT EXISTS idx_match_players_puuid ON match_players (puuid);
"""


class HttpCache:
    """TTL-based cache for raw HTTP responses backed by :mod:`diskcache`."""

    def __init__(self, directory: Path) -> None:
        """Open (or create) the cache.

        Args:
            directory: Directory where the cache files live.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(directory))

    def get(self, key: str) -> Any | None:
        """Fetch a cached value.

        Args:
            key: Cache key (typically the full request URL).

        Returns:
            The cached JSON-decoded payload, or ``None`` on a miss.
        """
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        """Store a value.

        Args:
            key: Cache key.
            value: JSON-serialisable payload.
            ttl_s: Optional time-to-live in seconds (``None`` = forever).
        """
        self._cache.set(key, value, expire=ttl_s)

    def clear(self) -> None:
        """Drop every cached entry."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying cache handle."""
        self._cache.close()


class MatchStore:
    """Permanent SQLite store of raw match and timeline documents."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the store and apply the schema.

        Args:
            db_path: Path of the SQLite database file.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._log = get_logger("cache")
        self._migrate_legacy_schema()

    def __enter__(self) -> "MatchStore":
        """Enter a context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store on context exit."""
        self.close()

    def _migrate_legacy_schema(self) -> None:
        """Move ownership from ``matches.puuid`` into ``match_players``."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(matches)")}
        if "puuid" not in cols:
            return
        self._log.info("Migrating match ownership to match_players table")
        self._conn.execute(
            "INSERT OR IGNORE INTO match_players (match_id, puuid) "
            "SELECT match_id, puuid FROM matches"
        )
        self._conn.executescript(
            """
            CREATE TABLE matches_new (
                match_id TEXT PRIMARY KEY,
                payload  TEXT NOT NULL
            );
            INSERT INTO matches_new (match_id, payload)
            SELECT match_id, payload FROM matches;
            DROP TABLE matches;
            ALTER TABLE matches_new RENAME TO matches;
            """
        )
        self._conn.commit()

    def has_match(self, match_id: str) -> bool:
        """Whether both match and timeline documents are stored.

        Args:
            match_id: Riot match id (e.g. ``EUW1_1234``).

        Returns:
            ``True`` when the match never needs to be downloaded again.
        """
        row = self._conn.execute(
            "SELECT 1 FROM matches m JOIN timelines t USING (match_id) WHERE match_id = ?",
            (match_id,),
        ).fetchone()
        return row is not None

    def save_match(self, match_id: str, puuid: str, match: dict[str, Any]) -> None:
        """Persist a raw match document and record player ownership.

        Args:
            match_id: Riot match id.
            puuid: PUUID of the tracked player (indexed via ``match_players``).
            match: Raw match-v5 JSON document.
        """
        payload = json.dumps(match)
        if self._conn.execute(
            "SELECT 1 FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone():
            self._conn.execute(
                "UPDATE matches SET payload = ? WHERE match_id = ?",
                (payload, match_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO matches (match_id, payload) VALUES (?, ?)",
                (match_id, payload),
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO match_players (match_id, puuid) VALUES (?, ?)",
            (match_id, puuid),
        )
        self._conn.commit()

    def save_timeline(self, match_id: str, timeline: dict[str, Any]) -> None:
        """Persist a raw timeline document.

        Args:
            match_id: Riot match id.
            timeline: Raw match-v5 timeline JSON document.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO timelines (match_id, payload) VALUES (?, ?)",
            (match_id, json.dumps(timeline)),
        )
        self._conn.commit()

    def load_match(self, match_id: str) -> dict[str, Any] | None:
        """Load a stored match document.

        Args:
            match_id: Riot match id.

        Returns:
            The raw match JSON, or ``None`` if absent.
        """
        row = self._conn.execute(
            "SELECT payload FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def load_timeline(self, match_id: str) -> dict[str, Any] | None:
        """Load a stored timeline document.

        Args:
            match_id: Riot match id.

        Returns:
            The raw timeline JSON, or ``None`` if absent.
        """
        row = self._conn.execute(
            "SELECT payload FROM timelines WHERE match_id = ?", (match_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def claim_ownership(self, puuid: str, match_ids: list[str]) -> int:
        """Index already-stored matches for a player without re-downloading.

        When a match was fetched for another account (e.g. rank peers), the
        payload may already exist while this player's ownership row is missing.

        Args:
            puuid: The player's PUUID.
            match_ids: Match ids to claim when present locally.

        Returns:
            Number of ownership rows inserted.
        """
        claimed = 0
        for match_id in match_ids:
            if not self.has_match(match_id):
                continue
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO match_players (match_id, puuid) VALUES (?, ?)",
                (match_id, puuid),
            )
            claimed += cursor.rowcount
        if claimed:
            self._conn.commit()
        return claimed

    def iter_match_ids(self, puuid: str) -> Iterator[str]:
        """Iterate over every stored match id owned by a player.

        Args:
            puuid: The player's PUUID.

        Yields:
            Match ids for that player.
        """
        cursor = self._conn.execute(
            "SELECT match_id FROM match_players WHERE puuid = ?", (puuid,)
        )
        for (match_id,) in cursor:
            yield match_id

    def count(self) -> int:
        """Number of fully stored matches (match + timeline).

        Returns:
            The count of matches with both documents present.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM matches m JOIN timelines t USING (match_id)"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
