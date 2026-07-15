"""Aggregated summaries, recommendations, and export/chatbot payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from league_stats.analysis.coach.engine import CoachEngine
from league_stats.analysis.deaths import death_summary
from league_stats.analysis.economy import economy_summary, reset_quality
from league_stats.analysis.items import item_summary
from league_stats.analysis.jungle import jungle_summary
from league_stats.analysis.laning import laning_summary
from league_stats.analysis.support import utility_summary
from league_stats.analysis.matchups import matchup_summary
from league_stats.analysis.objectives import objective_summary
from league_stats.analysis.peer import peer_recommendations
from league_stats.analysis.positioning import macro_summary, positioning_summary
from league_stats.analysis.runes import rune_summary
from league_stats.analysis.statistics import ModelResult, StatisticsEngine, WinCorrelation
from league_stats.analysis.teamfights import teamfight_summary
from league_stats.analysis.vision import vision_summary
from league_stats.core.config import AppConfig
from league_stats.core.role_metrics import role_profile
from league_stats.core.models import MatchRecord, PeerComparisonResult, RankedEntry, Recommendation, GameReviewPayload
from league_stats.pipeline.frames import AnalysisFrames, build_overview


@dataclass
class ReportStats:
    """ML and correlation outputs trained once on the full dataset."""

    stats: StatisticsEngine
    corr: pd.DataFrame
    win_corrs: list[WinCorrelation]
    model: ModelResult
    clusters_df: pd.DataFrame


def compute_report_stats(frames: AnalysisFrames, output_dir) -> ReportStats:
    """Train ML and compute correlations once for a report run."""
    stats = StatisticsEngine(frames.matches_df, output_dir)
    return ReportStats(
        stats=stats,
        corr=stats.correlation_matrix(),
        win_corrs=stats.win_correlations(),
        model=stats.train_win_predictor(),
        clusters_df=stats.cluster_games(),
    )


def build_domain_summaries(frames: AnalysisFrames, records: list[MatchRecord]) -> dict[str, Any]:
    """Aggregate summaries used by exports and window bundles."""
    player_role = records[0].role if records else "MIDDLE"
    summaries = {
        "overview": build_overview(frames.matches_df),
        "laning": laning_summary(frames.matches_df),
        "economy": economy_summary(frames.matches_df),
        "resets": reset_quality(records),
        "vision": vision_summary(frames.vision_df),
        "deaths": death_summary(frames.deaths_df),
        "teamfights": teamfight_summary(frames.teamfights_df),
        "positioning": positioning_summary(frames.matches_df, player_role),
        "objectives": objective_summary(frames.objectives_df),
        "macro": macro_summary(records, frames.matches_df),
        "matchups": matchup_summary(frames.matchups_df),
        "items": item_summary(frames.items_df),
        "runes": rune_summary(frames.runes_df),
        "jungle": jungle_summary(frames.matches_df),
        "utility": utility_summary(frames.matches_df),
    }
    return summaries


def generate_recommendations(
    frames: AnalysisFrames,
    stats: StatisticsEngine,
    config: AppConfig,
    *,
    peer_comparison: PeerComparisonResult | None = None,
    records_count: int,
) -> list[Recommendation]:
    """Run the coach engine and merge peer recommendations when present."""
    coach = CoachEngine(
        frames.matches_df,
        frames.deaths_df,
        frames.matchups_df,
        frames.objectives_df,
        stats,
        build_label=config.build_label,
        role=config.role,
    )
    recommendations = coach.generate()
    if peer_comparison is not None:
        peer_recs = peer_recommendations(
            peer_comparison.comparisons,
            peer_comparison.rank_label,
            max(peer_comparison.peer_games, records_count),
            build_label=peer_comparison.build_label,
            role=config.role,
        )
        recommendations = sorted(
            peer_recs + recommendations, key=lambda rec: rec.priority, reverse=True
        )
    return recommendations


def build_export_summary(
    config: AppConfig,
    frames: AnalysisFrames,
    summaries: dict[str, Any],
    report_stats: ReportStats,
    *,
    peer_comparison: PeerComparisonResult | None,
    ranked: RankedEntry | None,
    records_count: int,
    game_review: GameReviewPayload | None = None,
) -> dict[str, Any]:
    """Build the machine-readable summary embedded in exports and the chatbot."""
    summary: dict[str, Any] = {
        "player": config.players_label,
        "champion": config.champion,
        "role": config.role,
        "build_label": config.build_label,
        "games": records_count,
        "early_section_title": role_profile(config.role).early_section_title,
        "overview": summaries["overview"],
        "laning": summaries["laning"],
        "economy": summaries["economy"] | {"resets": summaries["resets"]},
        "vision": summaries["vision"],
        "deaths": summaries["deaths"],
        "teamfights": summaries["teamfights"],
        "positioning": summaries["positioning"],
        "objectives": summaries["objectives"],
        "macro": summaries["macro"],
        "matchups": summaries["matchups"],
        "items": summaries["items"],
        "runes": summaries["runes"],
        "jungle": summaries.get("jungle", {}),
        "utility": summaries.get("utility", {}),
        "win_correlations": [vars(c) for c in report_stats.win_corrs],
        "ml_model": {
            "trained": report_stats.model.trained,
            "cv_auc_mean": report_stats.model.cv_auc_mean,
            "cv_auc_std": report_stats.model.cv_auc_std,
            "n_games": report_stats.model.n_games,
        },
    }
    if ranked is not None:
        summary["rank"] = {
            "label": ranked.label,
            "tier": ranked.tier,
            "wins": ranked.wins,
            "losses": ranked.losses,
        }
    if peer_comparison is not None:
        summary["peer_comparison"] = peer_comparison.model_dump()
    if game_review is not None:
        from league_stats.analysis.game_review.export import game_review_chatbot_export

        summary["recent_games"] = game_review_chatbot_export(game_review, queue_key="all")
    return summary
