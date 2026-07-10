"""Resolve peer baselines from store, live sampling, and static fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from league_stats.analysis.peer.benchmark_fetcher import fetch_benchmark_from_api
from league_stats.analysis.peer.benchmarks import try_role_benchmark, try_static_benchmark
from league_stats.analysis.peer.cache import PeerSample, aggregate_peer_metrics, collect_peer_games_from_store
from league_stats.analysis.peer.rank_scope import RankScope, build_exact_scope, build_widened_scope
from league_stats.infra.cache import MatchStore
from league_stats.core.champions import build_label
from league_stats.core.models import RankedEntry
from league_stats.infra.riot_api import RiotApiClient, RiotApiError
from league_stats.utils import get_logger

TARGET_PEER_GAMES: Final[int] = 100
MIN_EXACT_GAMES: Final[int] = 100
MIN_WIDENED_GAMES: Final[int] = 50
MIN_LIVE_GAMES: Final[int] = 12


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
) -> PeerBaseline | None:
    """Return a store-backed baseline when enough games exist."""
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
    sample = PeerSample(
        rows=sample.rows,
        games=sample.games,
        players=sample.players,
        source=source_label,
    )
    return _baseline_from_sample(sample, level=level, confidence=confidence)


def _try_live_baseline(
    client: RiotApiClient,
    store: MatchStore,
    ranked: RankedEntry,
    champion: str,
    role: str,
    *,
    exclude_puuid: str | None,
) -> PeerBaseline | None:
    """Top up the peer store via live snowball sampling."""
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
        metrics = snapshot.metrics
        if "win" not in metrics and "winrate" in metrics:
            metrics = {**metrics, "win": float(metrics["winrate"])}
        return PeerBaseline(
            metrics=metrics,
            games=snapshot.games_sampled,
            players=snapshot.players_sampled,
            source=(
                f"Live API sample: {snapshot.games_sampled} ranked solo games "
                f"from {snapshot.players_sampled} players on {build_label(champion, role)}."
            ),
            confidence="medium",
            fallback_level=2,
        )

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
    """Resolve the best available peer baseline using the fallback ladder."""
    log = get_logger("peer_baseline")
    label = build_label(champion, role)
    exclude = exclude_puuid or ""

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
        confidence="high",
        source_label=(
            f"Peer store: {label} at {ranked.label} "
            f"({TARGET_PEER_GAMES}+ game target)."
        ),
    )
    if baseline is not None:
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
        return baseline

    static = try_static_benchmark(ranked.tier, champion, role)
    if static is not None:
        log.info("Using static champion benchmark for %s", label)
        return PeerBaseline(
            metrics=static,
            games=0,
            players=0,
            source=f"Static champion benchmark for {label} at {ranked.label}.",
            confidence="low",
            fallback_level=3,
        )

    role_static = try_role_benchmark(ranked.tier, role)
    if role_static is not None:
        log.info("Using static role benchmark for %s", label)
        return PeerBaseline(
            metrics=role_static,
            games=0,
            players=0,
            source=f"Static role benchmark for {label} lane at {ranked.label}.",
            confidence="low",
            fallback_level=4,
        )

    log.warning("No peer baseline available for %s at %s", label, ranked.label)
    return None
