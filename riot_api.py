"""Riot API client: account-v1, match-v5 and Data Dragon static data.

Features:

* sliding-window rate limiting tuned for development keys,
* automatic retry with ``Retry-After`` support on 429 and backoff on 5xx,
* transparent response caching via :class:`cache.HttpCache`,
* permanent match storage via :class:`cache.MatchStore` so a match is never
  fetched twice,
* progress bars for bulk downloads.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Final

import requests
from tqdm import tqdm

from cache import HttpCache, MatchStore
from config import AppConfig, PLATFORM_TO_REGION, RANKED_FLEX_QUEUE_ID, RANKED_SOLO_QUEUE_ID
from models import RankedEntry
from utils import get_logger

MATCH_ID_PAGE_SIZE: Final[int] = 100
MATCH_IDS_TTL_S: Final[float] = 15 * 60
STATIC_TTL_S: Final[float] = 24 * 3600
DDRAGON_BASE: Final[str] = "https://ddragon.leagueoflegends.com"


class RiotApiError(RuntimeError):
    """Raised when the Riot API returns a non-retryable error."""


class RateLimiter:
    """Sliding-window rate limiter for two simultaneous windows."""

    def __init__(self, per_second: int, per_two_minutes: int) -> None:
        """Create the limiter.

        Args:
            per_second: Maximum requests per 1-second window.
            per_two_minutes: Maximum requests per 120-second window.
        """
        self._limits: list[tuple[float, int, deque[float]]] = [
            (1.0, per_second, deque()),
            (120.0, per_two_minutes, deque()),
        ]

    def acquire(self) -> None:
        """Block until a request slot is available, then consume it."""
        while True:
            now = time.monotonic()
            wait = 0.0
            for window, limit, stamps in self._limits:
                while stamps and now - stamps[0] > window:
                    stamps.popleft()
                if len(stamps) >= limit:
                    wait = max(wait, window - (now - stamps[0]) + 0.01)
            if wait <= 0:
                for _, _, stamps in self._limits:
                    stamps.append(now)
                return
            time.sleep(wait)


class RiotApiClient:
    """Thin, cached, rate-limited HTTP client for the Riot API."""

    def __init__(
        self,
        config: AppConfig,
        http_cache: HttpCache,
        store: MatchStore,
        session: requests.Session | None = None,
    ) -> None:
        """Wire the client with its collaborators (dependency injection).

        Args:
            config: Application configuration.
            http_cache: TTL cache for raw responses.
            store: Permanent match/timeline store.
            session: Optional pre-configured :class:`requests.Session`.
        """
        self._config = config
        self._cache = http_cache
        self._store = store
        self._session = session or requests.Session()
        self._limiter = RateLimiter(config.requests_per_second, config.requests_per_two_minutes)
        self._log = get_logger("riot_api")
        self._regional_base = f"https://{config.region}.api.riotgames.com"
        self._platform = config.routing_platform

    @staticmethod
    def infer_platform_from_match_id(match_id: str) -> str | None:
        """Infer platform routing from a match id prefix (e.g. ``EUW1_123`` -> ``euw1``).

        Args:
            match_id: Riot match id.

        Returns:
            Platform code or ``None`` when the prefix is unrecognised.
        """
        prefix = match_id.split("_", 1)[0].lower()
        return prefix if prefix in PLATFORM_TO_REGION else None

    def set_platform(self, platform: str) -> None:
        """Override the platform routing host (league-v4 / summoner-v4).

        Args:
            platform: Platform code such as ``euw1`` or ``na1``.
        """
        key = platform.strip().lower()
        if key not in PLATFORM_TO_REGION:
            raise ValueError(f"Unknown platform {platform!r}")
        self._platform = key
        self._log.info("Using platform routing host: %s", key)

    @property
    def platform(self) -> str:
        """Platform routing host (e.g. ``euw1``)."""
        return self._platform

    @property
    def platform_base(self) -> str:
        """Base URL for platform-routed endpoints (league-v4)."""
        return f"https://{self._platform}.api.riotgames.com"

    @property
    def _base(self) -> str:
        """Base URL for regional endpoints (account-v1, match-v5)."""
        return self._regional_base

    # ------------------------------------------------------------------ HTTP

    def _get(self, url: str, params: dict[str, Any] | None = None, ttl_s: float | None = None,
             use_cache: bool = True, authenticated: bool = True) -> Any:
        """Perform a GET with caching, rate limiting and retries.

        Args:
            url: Absolute request URL.
            params: Optional query parameters.
            ttl_s: Cache TTL; ``None`` caches forever.
            use_cache: Whether to consult/populate the HTTP cache.
            authenticated: Whether to attach the Riot API key header.

        Returns:
            The JSON-decoded response body.

        Raises:
            RiotApiError: On non-retryable HTTP errors or retry exhaustion.
        """
        cache_key = url if not params else f"{url}?{sorted(params.items())!r}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        headers = {"X-Riot-Token": self._config.api_key} if authenticated else {}
        last_error: str = "unknown"
        for attempt in range(self._config.max_retries + 1):
            if authenticated:
                self._limiter.acquire()
            try:
                response = self._session.get(
                    url, params=params, headers=headers, timeout=self._config.request_timeout_s
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._log.warning("Request failed (%s), attempt %d", exc, attempt + 1)
                time.sleep(min(2**attempt, 30))
                continue

            if response.status_code == 200:
                payload = response.json()
                if use_cache:
                    self._cache.set(cache_key, payload, ttl_s)
                return payload
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                self._log.warning("Rate limited; sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                time.sleep(min(2**attempt, 30))
                continue
            raise RiotApiError(f"GET {url} failed: HTTP {response.status_code} {response.text[:200]}")
        raise RiotApiError(f"GET {url} failed after retries ({last_error})")

    # --------------------------------------------------------------- Account

    def resolve_puuid(self, riot_id: str, tagline: str) -> str:
        """Resolve a Riot ID + tagline to a PUUID via account-v1.

        Args:
            riot_id: The game name part of the Riot ID.
            tagline: The tagline part (without ``#``).

        Returns:
            The player's PUUID.
        """
        url = f"{self._base}/riot/account/v1/accounts/by-riot-id/{riot_id}/{tagline}"
        payload = self._get(url, ttl_s=STATIC_TTL_S)
        puuid = str(payload["puuid"])
        self._log.info("Resolved %s#%s -> %s...", riot_id, tagline, puuid[:12])
        return puuid

    def fetch_solo_rank(self, puuid: str) -> RankedEntry | None:
        """Fetch the player's ranked solo queue entry via league-v4.

        League-v4 is **platform-routed** (``euw1.api.riotgames.com``), not
        regional. Set ``platform`` in config or pass ``--platform euw1``.

        Args:
            puuid: The player's PUUID.

        Returns:
            The solo queue :class:`~models.RankedEntry`, or ``None`` if unranked.
        """
        url = f"{self.platform_base}/lol/league/v4/entries/by-puuid/{puuid}"
        try:
            entries = self._get(url, ttl_s=15 * 60)
        except RiotApiError as exc:
            message = str(exc)
            if "403" in message:
                self._log.error(
                    "League-v4 returned 403 on platform %s. Pass the correct "
                    "--platform for your server (e.g. euw1, eun1, na1). %s",
                    self._platform,
                    exc,
                )
            elif "404" in message:
                self._log.info("No ranked data for %s (404)", puuid[:12])
                return None
            else:
                self._log.warning("Could not fetch rank for %s: %s", puuid[:12], exc)
            return None
        if not entries:
            self._log.info("No ranked solo queue entry found for %s", puuid[:12])
            return None
        for entry in entries:
            if entry.get("queueType") != "RANKED_SOLO_5x5":
                continue
            ranked = RankedEntry(
                tier=str(entry["tier"]),
                rank=str(entry.get("rank", "")),
                league_points=int(entry.get("leaguePoints", 0)),
                wins=int(entry.get("wins", 0)),
                losses=int(entry.get("losses", 0)),
            )
            self._log.info("Rank: %s (%dW-%dL)", ranked.label, ranked.wins, ranked.losses)
            return ranked
        self._log.info("No ranked solo queue entry found for %s", puuid[:12])
        return None

    def fetch_tiers_for_puuids(self, puuids: set[str]) -> dict[str, str]:
        """Resolve solo queue tiers for many PUUIDs (cached per player).

        Args:
            puuids: PUUIDs to look up.

        Returns:
            Mapping of PUUID to tier string (e.g. ``"GOLD"``).
        """
        tiers: dict[str, str] = {}
        for puuid in puuids:
            ranked = self.fetch_solo_rank(puuid)
            if ranked is not None:
                tiers[puuid] = ranked.tier
        return tiers

    def fetch_league_entries(self, tier: str, rank: str = "") -> list[dict[str, Any]]:
        """Fetch solo queue league entries for a tier (and division when applicable).

        For ``MASTER``, ``GRANDMASTER`` and ``CHALLENGER`` the corresponding
        league list endpoint is used. Lower tiers use the paginated entries
        endpoint for the given division (``I``–``IV``).

        Args:
            tier: Riot tier string (e.g. ``"PLATINUM"``).
            rank: Division within the tier (``"I"``–``"IV"``); ignored for Master+.

        Returns:
            Raw league entry dicts, each including a ``puuid`` when available.
        """
        tier_key = tier.upper()
        queue = "RANKED_SOLO_5x5"
        if tier_key == "CHALLENGER":
            url = f"{self.platform_base}/lol/league/v4/challengerleagues/by-queue/{queue}"
            payload = self._get(url, ttl_s=15 * 60)
            return list(payload.get("entries", []))
        if tier_key == "GRANDMASTER":
            url = f"{self.platform_base}/lol/league/v4/grandmasterleagues/by-queue/{queue}"
            payload = self._get(url, ttl_s=15 * 60)
            return list(payload.get("entries", []))
        if tier_key == "MASTER":
            url = f"{self.platform_base}/lol/league/v4/masterleagues/by-queue/{queue}"
            payload = self._get(url, ttl_s=15 * 60)
            return list(payload.get("entries", []))

        division = (rank or "I").upper()
        url = f"{self.platform_base}/lol/league/v4/entries/{queue}/{tier_key}/{division}"
        payload = self._get(url, params={"page": 1}, ttl_s=15 * 60)
        return list(payload)

    def fetch_match(self, match_id: str) -> dict[str, Any]:
        """Fetch a single match-v5 document (cached permanently).

        Args:
            match_id: Riot match id.

        Returns:
            Raw match JSON.
        """
        url = f"{self._base}/lol/match/v5/matches/{match_id}"
        return self._get(url, ttl_s=None, use_cache=True)

    # --------------------------------------------------------------- Matches

    def fetch_match_ids(
        self, puuid: str, count: int, *, queue_id: int | None = None
    ) -> list[str]:
        """Fetch up to ``count`` ranked match ids for one queue, paging by 100.

        Args:
            puuid: The player's PUUID.
            count: Maximum number of match ids to fetch.
            queue_id: Riot queue id (defaults to ``config.queue_id``).

        Returns:
            Match ids ordered most-recent first (fewer if history is shorter).
        """
        queue = queue_id if queue_id is not None else self._config.queue_id
        url = f"{self._base}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        ids: list[str] = []
        start = 0
        while len(ids) < count:
            page_size = min(MATCH_ID_PAGE_SIZE, count - len(ids))
            page = self._get(
                url,
                params={"queue": queue, "start": start, "count": page_size},
                ttl_s=MATCH_IDS_TTL_S,
            )
            if not page:
                break
            ids.extend(str(m) for m in page)
            if len(page) < page_size:
                break
            start += len(page)
        self._log.info("Found %d ranked queue %d match ids", len(ids), queue)
        return ids

    def fetch_ranked_match_ids(self, puuid: str, count: int) -> list[str]:
        """Fetch up to ``count`` solo and flex ranked match ids, merged and deduped.

        Args:
            puuid: The player's PUUID.
            count: Maximum match ids to fetch per queue.

        Returns:
            Match ids from both ranked queues, most-recent solo first then flex.
        """
        solo_ids = self.fetch_match_ids(puuid, count, queue_id=RANKED_SOLO_QUEUE_ID)
        flex_ids = self.fetch_match_ids(puuid, count, queue_id=RANKED_FLEX_QUEUE_ID)
        seen: set[str] = set()
        merged: list[str] = []
        for match_id in solo_ids + flex_ids:
            if match_id in seen:
                continue
            seen.add(match_id)
            merged.append(match_id)
        self._log.info(
            "Found %d ranked match ids total (%d solo, %d flex)",
            len(merged),
            len(solo_ids),
            len(flex_ids),
        )
        return merged

    def download_matches(self, puuid: str, match_ids: list[str]) -> None:
        """Download matches + timelines into the store, skipping stored ones.

        Args:
            puuid: The player's PUUID (recorded alongside each match).
            match_ids: Match ids to ensure are stored locally.
        """
        cached = [mid for mid in match_ids if self._store.has_match(mid)]
        pending = [mid for mid in match_ids if mid not in cached]
        if cached:
            claimed = self._store.claim_ownership(puuid, cached)
            self._log.info(
                "Indexed %d cached matches for player (%d already linked)",
                claimed,
                len(cached) - claimed,
            )
        self._log.info("%d matches already cached, %d to download", len(cached), len(pending))
        for match_id in tqdm(pending, desc="Downloading matches", unit="match"):
            match_url = f"{self._base}/lol/match/v5/matches/{match_id}"
            timeline_url = f"{match_url}/timeline"
            try:
                match = self._get(match_url, use_cache=False)
                timeline = self._get(timeline_url, use_cache=False)
            except RiotApiError as exc:
                self._log.error("Skipping %s: %s", match_id, exc)
                continue
            self._store.save_match(match_id, puuid, match)
            self._store.save_timeline(match_id, timeline)

    # ----------------------------------------------------------- Static data

    def fetch_item_catalog(self) -> dict[int, dict[str, Any]]:
        """Download the Data Dragon item catalogue for the latest patch.

        Returns:
            Mapping of item id to its raw Data Dragon definition.
        """
        versions = self._get(f"{DDRAGON_BASE}/api/versions.json", ttl_s=STATIC_TTL_S,
                             authenticated=False)
        latest = str(versions[0])
        items = self._get(
            f"{DDRAGON_BASE}/cdn/{latest}/data/en_US/item.json",
            ttl_s=STATIC_TTL_S,
            authenticated=False,
        )
        return {int(item_id): data for item_id, data in items["data"].items()}

    def fetch_champion_catalog(self) -> dict[str, str]:
        """Download Data Dragon champions and build a name lookup table.

        Returns:
            Mapping of normalised keys to official Riot champion ids.
        """
        from champions import build_champion_catalog

        versions = self._get(
            f"{DDRAGON_BASE}/api/versions.json", ttl_s=STATIC_TTL_S, authenticated=False
        )
        latest = str(versions[0])
        payload = self._get(
            f"{DDRAGON_BASE}/cdn/{latest}/data/en_US/champion.json",
            ttl_s=STATIC_TTL_S,
            authenticated=False,
        )
        return build_champion_catalog(payload["data"])

    def resolve_champion_name(self, user_input: str) -> str:
        """Resolve user input to the official Riot champion id.

        Args:
            user_input: Raw champion string from CLI or config.

        Returns:
            Official champion id used in match-v5 payloads.
        """
        from champions import resolve_champion_name

        return resolve_champion_name(user_input, self.fetch_champion_catalog())
