"""Resolve peer baselines from store, live sampling, and static fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from league_stats.analysis.peer.benchmark_cache import read_live_cache, write_live_cache
from league_stats.analysis.peer.benchmark_fetcher import BenchmarkSnapshot, fetch_benchmark_from_api
from league_stats.analysis.peer.benchmarks import try_role_benchmark, try_static_benchmark
from league_stats.analysis.peer.cache import PeerSample, aggregate_peer_metrics, collect_peer_games_from_store
from league_stats.analysis.peer.rank_scope import RankScope, build_exact_scope, build_wider_scope, build_widened_scope
from league_stats.infra.cache import MatchStore
from league_stats.core.champions import build_label
from league_stats.core.models import RankedEntry
from league_stats.infra.riot_api import RiotApiClient, RiotApiError
from league_stats.utils import get_logger

TARGET_PEER_GAMES: Final[int] = 50
MIN_EXACT_GAMES: Final[int] = 50
MIN_WIDENED_GAMES: Final[int] = 50
MIN_LIVE_GAMES: Final[int] = 50
# Exact-rank store confidence is "high" once we have this many games.
HIGH_CONFIDENCE_GAMES: Final[int] = 100


@dataclass(frozen=True)
class PeerBaseline:
    """Resolved peer baseline for rank comparison."""

    metrics: dict[str, float]
    games: int
    players: int
    source: str
    confidence: str
    fallback_level: int


def _baseline_from_sample(sample: PeerSample, *, level: int, confidence: str) -> PeerBaseline:
    """Build a baseline object from collected peer rows."""
    metrics = aggregate_peer_metrics(sample.rows)
    if "win" not in metrics and "winrate" in metrics:
        metrics = {**metrics, "win": float(metrics["winrate"])}
    return PeerBaseline(
        metrics=metrics,
        games=sample.games,
        players=sample.players,
        source=sample.source,
        confidence=confidence,
        fallback_level=level,
    )


def _try_store_baseline(
    store: MatchStore,
    client: RiotApiClient,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    scope: RankScope,
    exclude_puuid: str,
    min_games: int,
    level: int,
    confidence: str,
    source_label: str,
    high_confidence_threshold: int = 0,
) -> PeerBaseline | None:
    """Return a store-backed baseline when enough games exist.

    When ``high_confidence_threshold`` is set, the confidence is upgraded to
    ``"high"`` once the sample reaches that size (graduated confidence).
    """
    sample = collect_peer_games_from_store(
        store,
        champion=champion,
        role=role,
        platform=client.platform,
        scope=scope,
        exclude_puuid=exclude_puuid,
        client=client,
    )
    if sample.games < min_games:
        return None
    effective_confidence = confidence
    if high_confidence_threshold and sample.games >= high_confidence_threshold:
        effective_confidence = "high"
    sample = PeerSample(
        rows=sample.rows,
        games=sample.games,
        players=sample.players,
        source=source_label,
    )
    return _baseline_from_sample(sample, level=level, confidence=effective_confidence)


def _baseline_from_snapshot(
    snapshot: BenchmarkSnapshot, champion: str, role: str, *, level: int
) -> PeerBaseline:
    """Wrap a BenchmarkSnapshot in a PeerBaseline."""
    metrics = snapshot.metrics
    if "win" not in metrics and "winrate" in metrics:
        metrics = {**metrics, "win": float(metrics["winrate"])}
    return PeerBaseline(
        metrics=metrics,
        games=snapshot.games_sampled,
        players=snapshot.players_sampled,
        source=(
            f"{'Cached' if snapshot.from_cache else 'Live API'} sample: "
            f"{snapshot.games_sampled} ranked solo games "
            f"from {snapshot.players_sampled} players on {build_label(champion, role)}."
        ),
        confidence="medium",
        fallback_level=level,
    )


def _try_live_baseline(
    client: RiotApiClient,
    store: MatchStore,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None,
) -> PeerBaseline | None:
    """Return a peer baseline from the file cache or live snowball sampling.

    The file cache is checked first (7-day TTL) to make re-runs near-instant.
    After a successful live sample the result is written to the cache so the
    next run skips the API entirely.
    """
    log = get_logger("peer_baseline")

    cached = read_live_cache(client.platform, ranked.tier, champion, role)
    if cached is not None:
        log.info(
            "File cache hit for %s %s (platform=%s, tier=%s): %d games",
            champion,
            role,
            client.platform,
            ranked.tier,
            cached.games_sampled,
        )
        return _baseline_from_snapshot(cached, champion, role, level=2)

    snapshot = fetch_benchmark_from_api(
        client,
        store,
        ranked,
        champion,
        role,
        exclude_puuid=exclude_puuid,
    )
    if snapshot is None:
        return None

    write_live_cache(client.platform, ranked.tier, champion, role, snapshot)

    sample = collect_peer_games_from_store(
        store,
        champion=champion,
        role=role,
        platform=client.platform,
        scope=build_widened_scope(ranked),
        exclude_puuid=exclude_puuid or "",
        client=client,
    )
    if sample.games < MIN_LIVE_GAMES:
        return _baseline_from_snapshot(snapshot, champion, role, level=2)

    return _baseline_from_sample(
        PeerSample(
            rows=sample.rows,
            games=sample.games,
            players=sample.players,
            source=(
                f"Peer store + live sample: {sample.games} ranked solo games "
                f"from {sample.players} players on {build_label(champion, role)}."
            ),
        ),
        level=2,
        confidence="medium",
    )


def resolve_peer_baseline(
    client: RiotApiClient,
    store: MatchStore,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None = None,
) -> PeerBaseline | None:
    """Resolve the best available peer baseline using the fallback ladder.

    Levels:
    0 — Peer store, exact rank, ≥50 games (high confidence at ≥100)
    1 — Peer store, ±1 widened rank, ≥50 games
    2 — File cache or live API snowball (7-day TTL), ≥50 games
    3 — Peer store, ±2 wider rank, ≥50 games (post-live-attempt)
    4 — Static champion JSON
    5 — Static role JSON
    """
    import time

    log = get_logger("peer_baseline")
    label = build_label(champion, role)
    exclude = exclude_puuid or ""
    t0 = time.monotonic()

    baseline = _try_store_baseline(
        store,
        client,
        ranked,
        champion,
        role,
        scope=build_exact_scope(ranked),
        exclude_puuid=exclude,
        min_games=MIN_EXACT_GAMES,
        level=0,
        confidence="medium",
        high_confidence_threshold=HIGH_CONFIDENCE_GAMES,
        source_label=(
            f"Peer store: {label} at {ranked.label} "
            f"({TARGET_PEER_GAMES}+ game target)."
        ),
    )
    if baseline is not None:
        log.info(
            "Resolved peer baseline: level=0, games=%d, source=store, took=%.1fs",
            baseline.games,
            time.monotonic() - t0,
        )
        return replace(
            baseline,
            source=(
                f"Peer store: {baseline.games} {label} games at {ranked.label} "
                f"from {baseline.players} players."
            ),
        )

    baseline = _try_store_baseline(
        store,
        client,
        ranked,
        champion,
        role,
        scope=build_widened_scope(ranked),
        exclude_puuid=exclude,
        min_games=MIN_WIDENED_GAMES,
        level=1,
        confidence="medium",
        source_label=f"Peer store (widened rank): {label}.",
    )
    if baseline is not None:
        log.info(
            "Resolved peer baseline: level=1, games=%d, source=store (widened), took=%.1fs",
            baseline.games,
            time.monotonic() - t0,
        )
        return replace(
            baseline,
            source=(
                f"Peer store (widened rank): {baseline.games} {label} games "
                f"from {baseline.players} players near {ranked.label}."
            ),
        )

    try:
        baseline = _try_live_baseline(
            client,
            store,
            ranked,
            champion,
            role,
            exclude_puuid=exclude_puuid,
        )
    except RiotApiError as exc:
        log.warning("Live peer sampling failed: %s", exc)
        baseline = None
    if baseline is not None:
        log.info(
            "Resolved peer baseline: level=2, games=%d, source=live/cache, took=%.1fs",
            baseline.games,
            time.monotonic() - t0,
        )
        return baseline

    # After the live attempt the store may have been populated; try ±2 tiers
    # before falling back to static benchmarks (still requires 50 games).
    baseline = _try_store_baseline(
        store,
        client,
        ranked,
        champion,
        role,
        scope=build_wider_scope(ranked),
        exclude_puuid=exclude,
        min_games=MIN_WIDENED_GAMES,
        level=3,
        confidence="medium",
        source_label=f"Peer store (wider rank ±2 tiers): {label}.",
    )
    if baseline is not None:
        log.info(
            "Resolved peer baseline: level=3, games=%d, source=store (wider ±2), took=%.1fs",
            baseline.games,
            time.monotonic() - t0,
        )
        return replace(
            baseline,
            source=(
                f"Peer store (±2 tier range): {baseline.games} {label} games "
                f"from {baseline.players} players near {ranked.label}."
            ),
        )

    static = try_static_benchmark(ranked.tier, champion, role)
    if static is not None:
        log.info(
            "Resolved peer baseline: level=4, source=static champion JSON, took=%.1fs",
            time.monotonic() - t0,
        )
        return PeerBaseline(
            metrics=static,
            games=0,
            players=0,
            source=f"Static champion benchmark for {label} at {ranked.label}.",
            confidence="low",
            fallback_level=4,
        )

    role_static = try_role_benchmark(ranked.tier, role)
    if role_static is not None:
        log.info(
            "Resolved peer baseline: level=5, source=static role JSON, took=%.1fs",
            time.monotonic() - t0,
        )
        return PeerBaseline(
            metrics=role_static,
            games=0,
            players=0,
            source=f"Static role benchmark for {label} lane at {ranked.label}.",
            confidence="low",
            fallback_level=5,
        )

    log.warning(
        "No peer baseline available for %s at %s (took %.1fs)",
        label,
        ranked.label,
        time.monotonic() - t0,
    )
    return None
