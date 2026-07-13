"""Persistent file cache for live-sampled peer benchmarks (7-day TTL).

Cache files are stored under ``data/benchmarks/live/`` and keyed by
``{platform}_{tier}_{champion_slug}_{role_lower}.json``.  A cached entry is
considered fresh when its ``fetched_at`` timestamp is less than ``CACHE_TTL_S``
seconds ago and the sample contains at least ``MIN_BENCHMARK_GAMES`` games.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Final

from league_stats.analysis.peer.benchmark_fetcher import BenchmarkSnapshot, MIN_BENCHMARK_GAMES
from league_stats.core.champions import champion_slug

CACHE_TTL_S: Final[float] = 7 * 24 * 3600  # 7 days

_LIVE_CACHE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "live"
)


def _cache_path(platform: str, tier: str, champion: str, role: str) -> Path:
    slug = champion_slug(champion, role)
    key = f"{platform.lower()}_{tier.upper()}_{slug}"
    return _LIVE_CACHE_DIR / f"{key}.json"


def read_live_cache(
    platform: str,
    tier: str,
    champion: str,
    role: str,
) -> BenchmarkSnapshot | None:
    """Return a cached benchmark snapshot if it is fresh and large enough."""
    path = _cache_path(platform, tier, champion, role)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    fetched_at = float(data.get("fetched_at", 0))
    if time.time() - fetched_at > CACHE_TTL_S:
        return None

    games = int(data.get("games", 0))
    if games < MIN_BENCHMARK_GAMES:
        return None

    return BenchmarkSnapshot(
        metrics=dict(data.get("metrics", {})),
        games_sampled=games,
        players_sampled=int(data.get("players", 0)),
        from_cache=True,
        platform=platform,
    )


def write_live_cache(
    platform: str,
    tier: str,
    champion: str,
    role: str,
    snapshot: BenchmarkSnapshot,
) -> None:
    """Persist a live-sampled benchmark snapshot to the file cache."""
    _LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(platform, tier, champion, role)
    data = {
        "metrics": snapshot.metrics,
        "games": snapshot.games_sampled,
        "players": snapshot.players_sampled,
        "fetched_at": time.time(),
        "tier": tier.upper(),
        "platform": platform.lower(),
    }
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass
