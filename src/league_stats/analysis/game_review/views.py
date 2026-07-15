"""Build per-queue Game Review payloads."""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.game_review.assemble import assemble_game_detail
from league_stats.analysis.progression.slicing import slice_baseline_exclusive, slice_recent
from league_stats.analysis.statistics import StatisticsEngine
from league_stats.core.config import (
    GAME_REVIEW_BASELINE_M,
    GAME_REVIEW_RECENT_N,
    QUEUE_FILTER_OPTIONS,
    AppConfig,
)
from league_stats.core.models import GameReviewPayload, GameReviewQueueBundle, MatchRecord
from league_stats.pipeline.bundles import filter_records_by_queue
from league_stats.pipeline.frames import AnalysisFrames, build_analysis_frames


def _baseline_means(records: list[MatchRecord]) -> dict[str, float]:
    if not records:
        return {}
    frame = pd.DataFrame([record.to_row() for record in records])
    means: dict[str, float] = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        means[column] = float(series.mean())
    return means


def _baseline_for_game(
    queue_records: list[MatchRecord],
    scored: MatchRecord,
    *,
    recent_n: int,
    baseline_m: int,
) -> dict[str, float]:
    """Personal baseline excluding the recent window; leave-one-out when sample is tiny."""
    if len(queue_records) <= recent_n:
        others = [record for record in queue_records if record.match_id != scored.match_id]
        return _baseline_means(others)
    baseline_records = slice_baseline_exclusive(queue_records, recent_n, baseline_m)
    if baseline_records:
        return _baseline_means(baseline_records)
    others = [record for record in queue_records if record.match_id != scored.match_id]
    return _baseline_means(others)


def build_game_review_views(
    config: AppConfig,
    records: list[MatchRecord],
    frames: AnalysisFrames,
) -> GameReviewPayload:
    """Build last-N game review bundles for each queue filter."""
    recent_n = GAME_REVIEW_RECENT_N
    baseline_m = GAME_REVIEW_BASELINE_M

    queues: dict[str, GameReviewQueueBundle] = {}
    for queue_key in QUEUE_FILTER_OPTIONS:
        queue_records = filter_records_by_queue(records, queue_key)
        queue_frames = frames if queue_key == "all" else build_analysis_frames(queue_records)
        recent_records = slice_recent(queue_records, recent_n)

        label_map: dict[str, str] = {}
        if queue_records:
            labels = StatisticsEngine(queue_frames.matches_df, ".", role=config.role).label_games()
            for match_id, label in zip(queue_frames.matches_df["match_id"], labels, strict=False):
                label_map[str(match_id)] = str(label)

        games = []
        for index, record in enumerate(recent_records, start=1):
            baseline = _baseline_for_game(
                queue_records,
                record,
                recent_n=recent_n,
                baseline_m=baseline_m,
            )
            detail = assemble_game_detail(
                record,
                queue_frames,
                baseline_means=baseline,
                archetype=label_map.get(record.match_id, "Unknown"),
                index=index,
                role=config.role,
            )
            games.append(detail)

        queues[queue_key] = GameReviewQueueBundle(
            available=len(games) > 0,
            games_count=len(games),
            games=games,
        )

    return GameReviewPayload(
        recent_n=recent_n,
        baseline_m=baseline_m,
        scoring="personal",
        queues=queues,
    )


def game_review_to_template_context(
    queue_bundle: GameReviewQueueBundle,
    *,
    recent_n: int,
) -> dict[str, Any]:
    """Map default queue bundle onto Jinja SSR fields."""
    games = [game.model_dump() for game in queue_bundle.games]
    count = queue_bundle.games_count
    subtitle = f"Last {count} game{'s' if count != 1 else ''}" if count else "No games"
    return {
        "game_review_available": queue_bundle.available,
        "game_review_subtitle": subtitle,
        "game_review_recent_n": recent_n,
        "game_review_games": games,
        "game_review_selected_match_id": games[0]["match_id"] if games else None,
    }
