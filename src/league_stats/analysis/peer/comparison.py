"""Rank-peer comparison: your stats vs same-rank peers on the same champion + lane.

Baselines are resolved from the persistent peer store, live snowball sampling,
and static JSON fallbacks via :func:`analysis.peer_baseline.resolve_peer_baseline`.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import pandas as pd

from league_stats.analysis.peer.baseline import resolve_peer_baseline
from league_stats.analysis.peer.cache import collect_user_history_peers
from league_stats.analysis.peer.metrics import extract_champion_role_rows
from league_stats.infra.cache import MatchStore
from league_stats.core.champions import build_label
from league_stats.core.models import MatchRecord, MetricComparison, PeerComparisonResult, RankedEntry, Recommendation
from league_stats.infra.riot_api import RiotApiClient
from league_stats.utils import get_logger, safe_div

MIN_PEER_GAMES: Final[int] = 12
PEER_LOOKUP_CAP: Final[int] = 80

# (column, display label, whether a higher value is better)
COMPARE_METRICS: Final[tuple[tuple[str, str, Literal["higher", "lower"]], ...]] = (
    ("win", "Win rate", "higher"),
    ("kda", "KDA", "higher"),
    ("dpm", "DPM", "higher"),
    ("cspm", "CS/min", "higher"),
    ("deaths", "Deaths/game", "lower"),
    ("vspm", "Vision/min", "higher"),
    ("control_wards", "Control wards", "higher"),
    ("cs10", "CS @10", "higher"),
    ("gd10", "Gold diff @10", "higher"),
    ("kill_participation", "Kill participation", "higher"),
    ("damage_share", "Damage share", "higher"),
    ("deaths_pre14", "Deaths pre-14", "lower"),
)

# Minimum relative gap (%) to flag a weakness/strength
GAP_THRESHOLD_PCT: Final[float] = 10.0


def _extract_champion_role_from_match(
    match: dict[str, Any],
    exclude_puuid: str,
    champion: str,
    role: str,
) -> list[dict[str, Any]]:
    """Pull performances on the configured champion + lane from a raw match."""
    return extract_champion_role_rows(
        match, exclude_puuid=exclude_puuid, champion=champion, role=role
    )


def _average_metrics(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute column means, skipping missing values.

    Args:
        frame: Input table.
        columns: Columns to average.

    Returns:
        Mapping of column -> mean.
    """
    result: dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not series.empty:
            result[column] = float(series.mean())
    return result


def _user_averages(matches_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate the player's metrics from the master match table.

    Args:
        matches_df: One row per analysed game.

    Returns:
        Mean values for every comparable metric.
    """
    columns = [m[0] for m in COMPARE_METRICS]
    return _average_metrics(matches_df, columns)


def _verdict(delta: float, direction: str, metric: str, peer: float) -> str:
    """Classify a gap as above/below/inline relative to peers.

    Args:
        delta: Player value minus peer average.
        direction: Whether higher or lower is better.
        metric: Metric key (for thresholds).
        peer: Peer average (for relative thresholds).

    Returns:
        ``"above"``, ``"below"`` or ``"inline"``.
    """
    if metric in ("gd10", "cs10") and abs(delta) < 30:
        return "inline"
    threshold = max(abs(peer) * 0.08, 0.05) if peer else 0.05
    if direction == "higher":
        if delta > threshold:
            return "above"
        if delta < -threshold:
            return "below"
    else:
        if delta < -threshold:
            return "above"
        if delta > threshold:
            return "below"
    return "inline"


def build_comparisons(
    user_avgs: dict[str, float], peer_avgs: dict[str, float]
) -> list[MetricComparison]:
    """Build side-by-side metric comparisons.

    Args:
        user_avgs: Player averages.
        peer_avgs: Peer/benchmark averages.

    Returns:
        List of :class:`~models.MetricComparison` rows.
    """
    comparisons: list[MetricComparison] = []
    for key, label, direction in COMPARE_METRICS:
        if key not in user_avgs or key not in peer_avgs:
            continue
        yours = float(user_avgs[key])
        peer = float(peer_avgs[key])
        delta = yours - peer
        delta_pct = round(delta / peer * 100, 1) if peer else None
        comparisons.append(
            MetricComparison(
                metric=key,
                label=label,
                yours=round(yours, 3),
                peer_avg=round(peer, 3),
                delta=round(delta, 3),
                delta_pct=delta_pct,
                direction=direction,
                verdict=_verdict(delta, direction, key, peer),
            )
        )
    return comparisons


def _comparison_summary_line(comp: MetricComparison) -> str:
    """Format a one-line strength/weakness summary for a comparison row.

    Args:
        comp: A single metric comparison.

    Returns:
        Human-readable summary; uses absolute delta when % is undefined.
    """
    if comp.delta_pct is not None:
        return f"{comp.label}: {comp.yours} vs {comp.peer_avg} ({comp.delta_pct:+.0f}%)"
    return f"{comp.label}: {comp.yours} vs {comp.peer_avg} ({comp.delta:+.1f})"


def peer_recommendations(
    comparisons: list[MetricComparison],
    rank_label: str,
    peer_games: int,
    *,
    build_label: str,
) -> list[Recommendation]:
    """Generate coaching tips from the largest peer gaps.

    Args:
        comparisons: Metric comparison rows.
        rank_label: Player rank string for messaging.
        peer_games: Peer sample size backing the baseline.
        build_label: Champion + lane label (e.g. ``Ahri mid``).

    Returns:
        Up to five ranked recommendations.
    """
    tips: list[tuple[float, Recommendation]] = []
    peer_name = build_label

    def add_weakness(
        comp: MetricComparison, title: str, detail: str, priority_boost: float = 1.0
    ) -> None:
        """Queue a weakness recommendation if the gap is material."""
        if comp.verdict != "below":
            return
        gap = abs(comp.delta_pct or 0.0)
        if gap < GAP_THRESHOLD_PCT and comp.metric not in ("deaths", "deaths_pre14", "vspm"):
            return
        priority = round(gap / 25.0 * priority_boost + 1.0, 3)
        tips.append(
            (
                priority,
                Recommendation(
                    category="Rank peer",
                    title=title,
                    detail=detail,
                    evidence=(
                        f"You: {comp.yours} vs {rank_label} {peer_name} avg {comp.peer_avg} "
                        f"({comp.delta_pct:+.0f}%)" if comp.delta_pct is not None else
                        f"You: {comp.yours} vs peer avg {comp.peer_avg}"
                    ),
                    priority=priority,
                    sample_size=peer_games,
                ),
            )
        )

    by_key = {c.metric: c for c in comparisons}
    if "deaths" in by_key:
        c = by_key["deaths"]
        add_weakness(
            c,
            f"You die more than peer {peer_name} players",
            f"Average {c.yours:.1f} deaths vs {c.peer_avg:.1f} for {rank_label} {peer_name}. "
            "Tighten map awareness after shoves and track enemy jungle pathing before extending.",
            priority_boost=1.3,
        )
    if "deaths_pre14" in by_key:
        c = by_key["deaths_pre14"]
        add_weakness(
            c,
            "Early deaths lag behind rank peers",
            f"You average {c.yours:.1f} deaths before 14 min vs {c.peer_avg:.1f} for peers. "
            "Respect level 2-3 all-ins and avoid trading without minion cover.",
        )
    if "cspm" in by_key:
        c = by_key["cspm"]
        add_weakness(
            c,
            f"Farming below rank-average {peer_name}",
            f"Your {c.yours:.1f} CS/min trails the {rank_label} {peer_name} average of "
            f"{c.peer_avg:.1f}. Catch every cannon and secure ranged minions under tower.",
        )
    if "cs10" in by_key:
        c = by_key["cs10"]
        add_weakness(
            c,
            f"CS @10 behind same-rank {peer_name}",
            f"{c.yours:.0f} CS @10 vs peer average {c.peer_avg:.0f}. Prioritise wave control "
            "over roams in the first 10 minutes unless the roam is guaranteed.",
        )
    if "gd10" in by_key:
        c = by_key["gd10"]
        add_weakness(
            c,
            "Laning gold deficit vs rank peers",
            f"{c.yours:+.0f} gold @10 vs peer average {c.peer_avg:+.0f}. Trade when your runes "
            "are up and avoid losing XP for bad harass.",
        )
    if "vspm" in by_key:
        c = by_key["vspm"]
        add_weakness(
            c,
            f"Vision below peer {peer_name}",
            f"{c.yours:.2f} VS/min vs peer {c.peer_avg:.2f}. Buy a control ward every recall "
            "after 14 minutes and sweep before objectives.",
            priority_boost=1.1,
        )
    if "control_wards" in by_key:
        c = by_key["control_wards"]
        add_weakness(
            c,
            "Under-investing in control wards",
            f"You buy {c.yours:.1f} control wards/game vs {c.peer_avg:.1f} for peers. "
            f"{peer_name.title()} wins objective fights when the pit is warded — match peer investment.",
        )
    if "dpm" in by_key:
        c = by_key["dpm"]
        add_weakness(
            c,
            "Damage output trails rank peers",
            f"{c.yours:.0f} DPM vs peer {c.peer_avg:.0f}. Look for more poke before fights "
            "and maximise combos in teamfights rather than holding for perfect angles.",
        )
    if "kill_participation" in by_key:
        c = by_key["kill_participation"]
        add_weakness(
            c,
            "Lower kill participation than peers",
            f"{c.yours:.0%} KP vs peer {c.peer_avg:.0%}. Roam on cannon waves when you have "
            "priority and arrive before objectives with your team.",
        )

    tips.sort(key=lambda item: item[0], reverse=True)
    return [rec for _, rec in tips[:5]]


def peer_comparison_for_window(
    base: PeerComparisonResult,
    matches_df: pd.DataFrame,
    records: list[MatchRecord],
) -> PeerComparisonResult:
    """Recompute user-side peer comparisons for a sliced game window.

    Args:
        base: Full-run peer comparison (benchmark and metadata reused).
        matches_df: Filtered per-game table for this window.
        records: Filtered parsed records for this window.

    Returns:
        Updated comparison with window-specific user averages.
    """
    peer_avgs = {comp.metric: comp.peer_avg for comp in base.comparisons}
    user_avgs = _user_averages(matches_df)
    if records:
        for key in ("cs10", "gd10", "deaths_pre14"):
            if key in matches_df.columns and matches_df[key].notna().any():
                user_avgs[key] = float(
                    pd.to_numeric(matches_df[key], errors="coerce").dropna().mean()
                )
    comparisons = build_comparisons(user_avgs, peer_avgs)
    strengths = [
        _comparison_summary_line(comp) for comp in comparisons if comp.verdict == "above"
    ][:4]
    weaknesses = [
        _comparison_summary_line(comp) for comp in comparisons if comp.verdict == "below"
    ][:4]
    return base.model_copy(
        update={
            "comparisons": comparisons,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
    )


def build_peer_comparison(
    client: RiotApiClient,
    store: MatchStore,
    matches_df: pd.DataFrame,
    records: list[MatchRecord],
    user_puuid: str,
    ranked: RankedEntry | None,
    *,
    champion: str,
    role: str,
) -> PeerComparisonResult | None:
    """Build the full rank-peer comparison for the report.

    Args:
        client: Riot API client (for peer rank lookups).
        store: Match store (for scanning peer games in history).
        matches_df: Player's per-game table.
        records: Parsed match records (for timeline-enriched metrics).
        user_puuid: Tracked player PUUID.
        ranked: Player's solo queue rank, if known.
        champion: Riot champion id being analysed.
        role: Normalised team position being analysed.

    Returns:
        Comparison result, or ``None`` when rank cannot be determined.
    """
    log = get_logger("peer_comparison")
    label = build_label(champion, role)
    if ranked is None:
        log.warning(
            "Skipping peer comparison: could not resolve solo queue rank "
            "(unranked, or league-v4 lookup failed — check --platform)"
        )
        return None

    baseline = resolve_peer_baseline(
        client,
        store,
        ranked,
        champion,
        role,
        exclude_puuid=user_puuid,
    )
    if baseline is None:
        log.warning(
            "Skipping peer comparison: no baseline available for %s at %s",
            label,
            ranked.label,
        )
        return None

    final_peer: dict[str, float] = {
        key: float(baseline.metrics[key])
        for key, _, _ in COMPARE_METRICS
        if key in baseline.metrics and baseline.metrics[key] is not None
    }

    history_df = collect_user_history_peers(store, user_puuid, champion, role)
    history_games = len(history_df)
    history_players = int(history_df["puuid"].nunique()) if history_games else 0

    source = baseline.source
    if history_games:
        source += (
            f" ({history_games} other {label} games in your match history from "
            f"{history_players} players.)"
        )

    user_avgs = _user_averages(matches_df)
    if records:
        snap = matches_df
        for key in ("cs10", "gd10", "deaths_pre14"):
            if key in snap.columns and snap[key].notna().any():
                user_avgs[key] = float(pd.to_numeric(snap[key], errors="coerce").dropna().mean())

    comparisons = build_comparisons(user_avgs, final_peer)
    strengths = [
        _comparison_summary_line(c) for c in comparisons if c.verdict == "above"
    ][:4]
    weaknesses = [
        _comparison_summary_line(c) for c in comparisons if c.verdict == "below"
    ][:4]

    return PeerComparisonResult(
        rank_label=ranked.label,
        tier=ranked.tier,
        champion=champion,
        role=role,
        build_label=label,
        source=source,
        peer_games=baseline.games,
        peer_players=baseline.players,
        confidence=baseline.confidence,
        fallback_level=baseline.fallback_level,
        comparisons=comparisons,
        strengths=strengths,
        weaknesses=weaknesses,
    )


def comparisons_dataframe(result: PeerComparisonResult) -> pd.DataFrame:
    """Flatten comparisons for CSV export.

    Args:
        result: Peer comparison output.

    Returns:
        One row per metric.
    """
    return pd.DataFrame([c.model_dump() for c in result.comparisons])
