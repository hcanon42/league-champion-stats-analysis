"""Queue/window slicing and dashboard bundle assembly."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from league_stats.analysis.coach.engine import VISIBLE_RECOMMENDATIONS
from league_stats.analysis.deaths import blind_spot_zones
from league_stats.analysis.items import build_path_stats
from league_stats.analysis.matchups import matchup_recommendation
from league_stats.analysis.peer import peer_comparison_for_window
from league_stats.analysis.runes import rune_setup_stats
from league_stats.analysis.statistics import StatisticsEngine
from league_stats.core.config import (
    DEFAULT_GAME_WINDOW,
    DEFAULT_QUEUE_FILTER,
    GAME_WINDOW_OPTIONS,
    QUEUE_FILTER_OPTIONS,
    QUEUE_LABELS,
    RANKED_FLEX_QUEUE_ID,
    RANKED_SOLO_QUEUE_ID,
    AppConfig,
)
from league_stats.core.models import MatchRecord, PeerComparisonResult
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.pipeline.frames import AnalysisFrames, build_analysis_frames, build_overview
from league_stats.pipeline.summaries import ReportStats, build_domain_summaries, generate_recommendations
from league_stats.pipeline.view_models import (
    card,
    card_entries,
    overview_card_entries,
    peer_row_display,
    peer_subtitle,
    pct,
)
from league_stats.presentation.graphs import ChartIconResolver, GraphFactory
from league_stats.presentation.report import improvement_score, score_badge
from league_stats.presentation.ui_icons import attach_metric_icon_hrefs, icon_fields_for_label


def filter_records_by_queue(records: list[MatchRecord], key: str) -> list[MatchRecord]:
    """Return records for one queue filter key (``solo``, ``flex``, or ``all``)."""
    if key == "solo":
        return [record for record in records if record.queue_id == RANKED_SOLO_QUEUE_ID]
    if key == "flex":
        return [record for record in records if record.queue_id == RANKED_FLEX_QUEUE_ID]
    return records


def queue_filter_options(solo_count: int, flex_count: int) -> list[dict[str, Any]]:
    """Toggle metadata for the queue filter bar."""
    total = solo_count + flex_count
    return [
        {"key": "solo", "label": QUEUE_LABELS["solo"], "enabled": solo_count > 0},
        {"key": "flex", "label": QUEUE_LABELS["flex"], "enabled": flex_count > 0},
        {"key": "all", "label": QUEUE_LABELS["all"], "enabled": total > 0},
    ]


def default_queue_filter_key(solo_count: int, flex_count: int) -> str:
    """Pick the initial queue filter, preferring solo when available."""
    if solo_count > 0:
        return DEFAULT_QUEUE_FILTER
    if flex_count > 0:
        return "flex"
    return "all"


def slice_records(records: list[MatchRecord], limit: int | None) -> list[MatchRecord]:
    """Return the most recent ``limit`` games, or all when ``limit`` is ``None``."""
    if limit is None:
        return records
    return records[:limit]


def default_game_window_key(total_games: int) -> str:
    """Pick the initial dashboard window."""
    if total_games >= DEFAULT_GAME_WINDOW:
        return str(DEFAULT_GAME_WINDOW)
    return "all"


def serialize_report_views_json(report_views: dict[str, dict[str, Any]]) -> str:
    """JSON-encode queue/window snapshots for safe embedding in a ``<script>`` tag."""
    encoded = json.dumps(report_views, default=str)
    return encoded.replace("</", r"<\/")


def game_window_options(total_games: int) -> list[dict[str, Any]]:
    """Toggle metadata for the report template."""
    options = [
        {"key": str(size), "label": f"Last {size}", "enabled": total_games >= size}
        for size in GAME_WINDOW_OPTIONS
    ]
    options.append({"key": "all", "label": "All", "enabled": True})
    return options


def bundle_to_template_context(
    bundle: dict[str, Any],
    *,
    peer_comparison: PeerComparisonResult | None = None,
) -> dict[str, Any]:
    """Map a window bundle onto the Jinja template field names."""
    context: dict[str, Any] = {
        "total_games": bundle["total_games"],
        "patch_range": bundle["patch_range"],
        "queue_label": bundle.get("queue_label", "ranked solo queue"),
        "overview": bundle["overview"],
        "score": bundle["score"],
        "score_components": bundle["score_components"],
        "figures": bundle["figures"],
        "overview_cards": bundle.get("overview_cards", []),
        "lane_cards": bundle["lane_cards"],
        "economy_cards": bundle["economy_cards"],
        "vision_cards": bundle["vision_cards"],
        "death_cards": bundle["death_cards"],
        "teamfight_cards": bundle["teamfight_cards"],
        "objective_rows": bundle["objective_rows"],
        "objectives_section_icon": bundle.get("objectives_section_icon"),
        "blind_spots": bundle["blind_spots"],
        "build_paths": bundle["build_paths"],
        "rune_rows": bundle["rune_rows"],
        "matchup_rows": bundle["matchup_rows"],
        "positive_recommendations": bundle["positive_recommendations"],
        "negative_recommendations": bundle["negative_recommendations"],
        "has_peer_comparison": peer_comparison is not None,
    }
    if peer_comparison is not None:
        context["peer_comparison"] = peer_comparison
        context["peer_rows"] = [row.model_dump() for row in peer_comparison.comparisons]
    return context


def build_window_bundle(
    config: AppConfig,
    records: list[MatchRecord],
    graphs_dir: Path,
    *,
    peer_comparison: PeerComparisonResult | None = None,
    queue_label: str = "ranked solo queue",
    assets: DDragonAssets | None = None,
    shared_stats: ReportStats | None = None,
) -> dict[str, Any]:
    """Run the analysis pipeline for one game window and return a JSON bundle."""
    empty: dict[str, Any] = {
        "total_games": 0,
        "patch_range": "—",
        "queue_label": queue_label,
        "overview": {},
        "overview_cards": [],
        "score": 0,
        "score_components": [],
        "lane_cards": [],
        "economy_cards": [],
        "vision_cards": [],
        "death_cards": [],
        "teamfight_cards": [],
        "objective_rows": [],
        "blind_spots": [],
        "build_paths": [],
        "rune_rows": [],
        "matchup_rows": [],
        "positive_recommendations": [],
        "negative_recommendations": [],
        "figures": {},
    }
    if not records:
        return empty

    frames = build_analysis_frames(records)
    summaries = build_domain_summaries(frames, records)
    overview = summaries["overview"]
    lane = summaries["laning"]
    economy = summaries["economy"]
    resets = summaries["resets"]
    vision = summaries["vision"]
    deaths_agg = summaries["deaths"]
    fights = summaries["teamfights"]
    objectives_agg = summaries["objectives"]

    slice_stats = StatisticsEngine(frames.matches_df, graphs_dir)
    corr = slice_stats.correlation_matrix()
    win_corrs = slice_stats.win_correlations()
    clusters_df = slice_stats.cluster_games()
    model = shared_stats.model if shared_stats is not None else slice_stats.train_win_predictor()

    window_peer = peer_comparison
    if peer_comparison is not None:
        window_peer = peer_comparison_for_window(peer_comparison, frames.matches_df, records)

    recommendations = generate_recommendations(
        frames,
        slice_stats,
        config,
        peer_comparison=window_peer,
        records_count=len(records),
    )

    score, components = improvement_score(frames.matches_df, role=config.role)
    matchups_export = frames.matchups_df.copy()
    if not matchups_export.empty:
        matchups_export["recommendation"] = matchups_export.apply(matchup_recommendation, axis=1)
    matchup_rows = matchups_export.head(20).to_dict("records") if not matchups_export.empty else []

    icon_resolver = None
    if assets is not None:
        icon_resolver = ChartIconResolver(
            from_dir=graphs_dir.parent,
            champion_href=assets.champion_chart_source,
            item_href=assets.item_chart_source,
            keystone_href=assets.keystone_chart_source,
        )
    graphs = GraphFactory(graphs_dir, icon_resolver=icon_resolver)
    series = [(r.win, r.timeline.gold_series, r.timeline.opp_gold_series) for r in records]
    figures = {
        "winrate_trend": graphs.winrate_trend(frames.matches_df),
        "gold_diff_timeline": graphs.gold_diff_timeline(series),
        "gd10_histogram": graphs.gd10_histogram(frames.matches_df),
        "deaths_box": graphs.deaths_box(frames.matches_df),
        "cs10_violin": graphs.cs10_violin(frames.matches_df),
        "dpm_scatter": graphs.dpm_scatter(frames.matches_df),
        "vision_trend": graphs.vision_trend(frames.matches_df),
        "death_heatmap": graphs.death_heatmap(frames.deaths_df),
        "correlation_heatmap": graphs.correlation_heatmap(corr),
        "win_correlation_bar": graphs.win_correlation_bar(win_corrs),
        "feature_importance": graphs.feature_importance(model),
        "cluster_scatter": graphs.cluster_scatter(clusters_df),
        "matchup_bar": graphs.matchup_bar(frames.matchups_df),
        "item_winrate_bar": graphs.item_winrate_bar(frames.items_df),
        "rune_winrate_bar": graphs.rune_winrate_bar(rune_setup_stats(frames.runes_df)),
        "objective_timing": graphs.objective_timing(frames.objectives_df),
    }

    bundle: dict[str, Any] = {
        "total_games": len(records),
        "patch_range": (
            f"{frames.matches_df['patch'].min()} – {frames.matches_df['patch'].max()}"
            if not frames.matches_df.empty
            else "—"
        ),
        "queue_label": queue_label,
        "overview": overview,
        "overview_cards": overview_card_entries(overview),
        "score": score,
        "score_components": [
            {**asdict(component), **icon_fields_for_label(component.name)}
            for component in components
        ],
        "lane_cards": card_entries(
            [
                ("Gold diff @10", card(lane.get("avg_gd10"))),
                ("CS diff @10", card(lane.get("avg_csd10"))),
                ("XP diff @10", card(lane.get("avg_xpd10"))),
                ("Lane win rate", card(pct(lane.get("lane_win_rate")))),
                ("WR when ahead @10", card(pct(lane.get("winrate_when_ahead_at_10")))),
                ("WR when behind @10", card(pct(lane.get("winrate_when_behind_at_10")))),
                ("Deaths pre-14", card(lane.get("avg_deaths_pre14"))),
                ("Gank deaths (lane)", card(lane.get("avg_gank_deaths_laning"))),
                ("Under own tower (lane)", card(lane.get("avg_under_own_tower_laning_deaths"))),
                ("Under enemy tower (lane)", card(lane.get("avg_under_enemy_tower_laning_deaths"))),
                ("Roams pre-15", card(lane.get("avg_roams_pre15"))),
            ]
        ),
        "economy_cards": card_entries(
            [
                ("GPM", card(economy.get("avg_gpm"))),
                ("CS/min", card(economy.get("avg_cspm"))),
                ("Gold share", card(pct(economy.get("avg_gold_share")))),
                ("Damage per gold", card(economy.get("avg_damage_per_gold"))),
                ("Unspent gold/recall", card(economy.get("avg_unspent_gold_before_recall"), "g")),
                ("First recall", card(resets.get("avg_first_recall_min"), " min")),
                ("Time dead/game", card(economy.get("avg_time_dead_s"), "s")),
            ]
        ),
        "vision_cards": card_entries(
            [
                ("Vision score", card(vision.get("avg_vision_score"))),
                ("VS/min", card(vision.get("avg_vspm"))),
                ("Control wards", card(vision.get("avg_control_wards"))),
                ("CW lifetime", card(vision.get("avg_control_ward_lifetime_s"), "s")),
                ("VS/min in wins", card(vision.get("avg_vspm_wins"))),
                ("VS/min in losses", card(vision.get("avg_vspm_losses"))),
            ]
        ),
        "death_cards": card_entries(
            [
                ("Total deaths", card(deaths_agg.get("total_deaths"))),
                ("Solo deaths", card(pct(deaths_agg.get("solo_death_rate")))),
                ("Gank deaths (lane)", card(pct(deaths_agg.get("gank_death_rate")))),
                ("Under own tower (lane)", card(pct(deaths_agg.get("under_own_tower_laning_death_rate")))),
                ("Under enemy tower (lane)", card(pct(deaths_agg.get("under_enemy_tower_laning_death_rate")))),
                ("Greed deaths", card(pct(deaths_agg.get("greed_death_rate")))),
                ("Side-lane deaths", card(pct(deaths_agg.get("side_lane_death_rate")))),
                ("Before dragon", card(pct(deaths_agg.get("death_before_dragon_rate")))),
                ("Avg death minute", card(deaths_agg.get("avg_death_minute"))),
                ("Top killer", card(deaths_agg.get("most_common_killer"))),
            ]
        ),
        "teamfight_cards": card_entries(
            [
                ("Fights detected", card(fights.get("total_fights"))),
                ("Participation", card(pct(fights.get("participation_rate")))),
                ("Fight win rate", card(pct(fights.get("fight_win_rate")))),
                ("Damage/fight", card(fights.get("avg_damage_per_fight"))),
                ("Death rate in fights", card(pct(fights.get("death_rate_in_fights")))),
                ("Front-to-back", card(fights.get("avg_front_to_back"))),
            ]
        ),
        "objective_rows": [
            {
                "kind": kind,
                "count": row.get("count"),
                "taken_rate": row.get("taken_rate"),
                "presence_rate": row.get("presence_rate"),
                "early_rate": row.get("early_rate"),
                "dead_before_rate": row.get("dead_before_rate"),
                "avg_wards_before": row.get("avg_wards_before"),
            }
            for kind, row in sorted(objectives_agg.get("by_kind", {}).items())
        ],
        "blind_spots": blind_spot_zones(frames.deaths_df),
        "build_paths": build_path_stats(frames.matches_df).head(10).to_dict("records"),
        "rune_rows": rune_setup_stats(frames.runes_df).to_dict("records"),
        "matchup_rows": matchup_rows,
        "positive_recommendations": [
            {**rec.model_dump(), "badge": score_badge(rec)}
            for rec in recommendations
            if rec.tone.value == "positive"
        ],
        "negative_recommendations": [
            {**rec.model_dump(), "badge": score_badge(rec)}
            for rec in recommendations
            if rec.tone.value == "negative"
        ],
        "figures": figures,
    }

    if window_peer is not None:
        figures["peer_comparison"] = graphs.peer_comparison_chart(
            window_peer.comparisons, build_label=window_peer.build_label
        )
        bundle["peer"] = {
            "subtitle": peer_subtitle(window_peer),
            "tier": window_peer.tier,
            "confidence": window_peer.confidence,
            "strengths": window_peer.strengths,
            "weaknesses": window_peer.weaknesses,
            "rows": [peer_row_display(row.model_dump()) for row in window_peer.comparisons],
        }
        bundle["figures"]["peer_comparison"] = figures["peer_comparison"]
        bundle["_peer_result"] = window_peer

    if assets is not None:
        from_dir = graphs_dir.parent
        bundle["rune_rows"] = assets.enrich_rune_rows(bundle["rune_rows"], from_dir=from_dir)
        bundle["matchup_rows"] = assets.enrich_matchup_rows(bundle["matchup_rows"], from_dir=from_dir)
        bundle["objective_rows"] = assets.enrich_objective_rows(bundle["objective_rows"], from_dir=from_dir)
        bundle["objectives_section_icon"] = assets.objective_href("dragon", from_dir=from_dir)
        metric_lists = [
            bundle["overview_cards"],
            bundle["lane_cards"],
            bundle["economy_cards"],
            bundle["vision_cards"],
            bundle["death_cards"],
            bundle["teamfight_cards"],
            bundle["score_components"],
        ]
        peer = bundle.get("peer")
        if peer and peer.get("rows"):
            metric_lists.append(peer["rows"])
        for entries in metric_lists:
            attach_metric_icon_hrefs(entries, assets, from_dir=from_dir)

    return bundle


# Re-export for template context consumers
VISIBLE_RECOMMENDATIONS_COUNT = VISIBLE_RECOMMENDATIONS
QUEUE_FILTER_KEYS = QUEUE_FILTER_OPTIONS
