"""Persistence layer: HTTP response cache and SQLite match store.

Two complementary stores are used:

* :class:`HttpCache` — a :mod:`diskcache` wrapper that memoises raw API
  responses (account lookups, match-id pages, static data) with TTLs.
* :class:`MatchStore` — a :mod:`sqlite3` database holding raw match and
  timeline JSON documents forever, guaranteeing a match is never downloaded
  twice.
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
    puuid    TEXT NOT NULL,
    payload  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timelines (
    match_id TEXT PRIMARY KEY,
    payload  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_puuid ON matches (puuid);
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
        """Persist a raw match document.

        Args:
            match_id: Riot match id.
            puuid: PUUID of the tracked player (for indexing).
            match: Raw match-v5 JSON document.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO matches (match_id, puuid, payload) VALUES (?, ?, ?)",
            (match_id, puuid, json.dumps(match)),
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

    def iter_match_ids(self, puuid: str) -> Iterator[str]:
        """Iterate over every stored match id for a player.

        Args:
            puuid: The player's PUUID.

        Yields:
            Match ids, most recent insertion order not guaranteed.
        """
        cursor = self._conn.execute("SELECT match_id FROM matches WHERE puuid = ?", (puuid,))
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
