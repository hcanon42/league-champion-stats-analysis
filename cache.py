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
    payload  TEXT NOT NULL,
    game_creation_ms INTEGER
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
        self._migrate()

    def _migrate(self) -> None:
        """Migrate a pre-existing database to the current schema.

        Handles two legacy shapes: a ``matches.puuid`` ownership column
        (moved into ``match_players``) and a missing ``game_creation_ms``
        column (added and backfilled from the stored payload).
        """
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(matches)")}
        if "game_creation_ms" not in columns:
            self._conn.execute("ALTER TABLE matches ADD COLUMN game_creation_ms INTEGER")
        if "puuid" in columns:
            self._conn.execute(
                "INSERT OR IGNORE INTO match_players (match_id, puuid) "
                "SELECT match_id, puuid FROM matches"
            )
            self._conn.execute("ALTER TABLE matches DROP COLUMN puuid")
        rows = self._conn.execute(
            "SELECT match_id, payload FROM matches WHERE game_creation_ms IS NULL"
        ).fetchall()
        for match_id, payload in rows:
            creation_ms = json.loads(payload).get("info", {}).get("gameCreation")
            if creation_ms is not None:
                self._conn.execute(
                    "UPDATE matches SET game_creation_ms = ? WHERE match_id = ?",
                    (creation_ms, match_id),
                )
        self._conn.commit()

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

    def save_match(self, match_id: str, match: dict[str, Any]) -> None:
        """Persist a raw match document.

        Args:
            match_id: Riot match id.
            match: Raw match-v5 JSON document.
        """
        creation_ms = match.get("info", {}).get("gameCreation")
        self._conn.execute(
            "INSERT OR REPLACE INTO matches (match_id, payload, game_creation_ms) VALUES (?, ?, ?)",
            (match_id, json.dumps(match), creation_ms),
        )
        self._conn.commit()

    def register_ownership(self, match_id: str, puuid: str) -> None:
        """Record that ``puuid``'s history includes ``match_id``.

        Idempotent and independent of whether the match payload itself was
        just downloaded or was already stored from another player.

        Args:
            match_id: Riot match id.
            puuid: PUUID whose history includes this match.
        """
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

    def iter_match_ids(self, puuid: str, limit: int | None = None) -> Iterator[str]:
        """Iterate over a player's stored match ids, most recent game first.

        Args:
            puuid: The player's PUUID.
            limit: Maximum number of match ids to yield (most recent by
                actual game date), or ``None`` for every stored match.

        Yields:
            Match ids ordered by ``game_creation_ms`` descending.
        """
        query = (
            "SELECT match_id FROM match_players JOIN matches USING (match_id) "
            "WHERE puuid = ? ORDER BY game_creation_ms DESC"
        )
        params: tuple[Any, ...] = (puuid,)
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        cursor = self._conn.execute(query, params)
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
