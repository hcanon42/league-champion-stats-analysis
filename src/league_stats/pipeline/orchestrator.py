"""End-to-end report generation orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from tqdm import tqdm

from league_stats.analysis.coach.engine import VISIBLE_RECOMMENDATIONS
from league_stats.analysis.deaths import deaths_dataframe
from league_stats.analysis.matchups import matchup_recommendation
from league_stats.analysis.peer import build_peer_comparison, comparisons_dataframe
from league_stats.core.config import (
    GAME_WINDOW_OPTIONS,
    QUEUE_FILTER_OPTIONS,
    QUEUE_SUBTITLE_LABELS,
    RANKED_FLEX_QUEUE_ID,
    RANKED_SOLO_QUEUE_ID,
    AppConfig,
)
from league_stats.core.models import MatchRecord, PeerComparisonResult, RankedEntry
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.infra.riot_api import RiotApiClient
from league_stats.ingest.parser import discover_build_pools
from league_stats.pipeline.bundles import (
    build_window_bundle,
    bundle_to_template_context,
    default_game_window_key,
    default_queue_filter_key,
    filter_records_by_queue,
    game_window_options,
    queue_filter_options,
    serialize_report_views_json,
    slice_records,
)
from league_stats.pipeline.fetch import fetch_matches, group_records, load_all_records, resolve_player_contexts
from league_stats.pipeline.frames import build_analysis_frames
from league_stats.pipeline.services import PlayerContext, Services
from league_stats.pipeline.summaries import (
    build_domain_summaries,
    build_export_summary,
    compute_report_stats,
    generate_recommendations,
)
from league_stats.presentation.brand_assets import brand_context
from league_stats.presentation.export import Exporter
from league_stats.presentation.graphs import GraphFactory
from league_stats.presentation.report import (
    ReportBuilder,
    build_manifest_entry,
    build_player_builds_nav,
    refresh_report_indexes,
    write_report_meta,
)
from league_stats.utils import get_logger


def write_full_exports(
    config: AppConfig,
    records: list[MatchRecord],
    run_dir: Path,
    *,
    peer_comparison: PeerComparisonResult | None,
    ranked: RankedEntry | None,
    frames=None,
    report_stats=None,
) -> dict[str, Any]:
    """Write CSV/JSON exports from the full (all games) dataset."""
    analysis_frames = frames or build_analysis_frames(records)
    summaries = build_domain_summaries(analysis_frames, records)
    stats_bundle = report_stats or compute_report_stats(analysis_frames, run_dir)

    summary = build_export_summary(
        config,
        analysis_frames,
        summaries,
        stats_bundle,
        peer_comparison=peer_comparison,
        ranked=ranked,
        records_count=len(records),
    )

    matchups_export = analysis_frames.matchups_df.copy()
    if not matchups_export.empty:
        matchups_export["recommendation"] = matchups_export.apply(matchup_recommendation, axis=1)
    corr_export = (
        stats_bundle.corr.reset_index().rename(columns={"index": "feature"})
        if not stats_bundle.corr.empty
        else stats_bundle.corr
    )

    recommendations = generate_recommendations(
        analysis_frames,
        stats_bundle.stats,
        config,
        peer_comparison=peer_comparison,
        records_count=len(records),
    )

    export_tables: dict[str, pd.DataFrame] = {
        "matches": analysis_frames.matches_df,
        "deaths": analysis_frames.deaths_df,
        "timeline": analysis_frames.timeline_df,
        "matchups": matchups_export,
        "vision": analysis_frames.vision_df,
        "items": analysis_frames.items_df,
        "runes": analysis_frames.runes_df,
        "objectives": analysis_frames.objectives_df,
        "teamfights": analysis_frames.teamfights_df,
        "correlations": corr_export,
    }
    if peer_comparison is not None:
        export_tables["rank_comparison"] = comparisons_dataframe(peer_comparison)

    Exporter(run_dir).write_all(
        tables=export_tables,
        summary=summary,
        recommendations=recommendations,
        build_label=config.build_label,
    )
    return summary


def run_analysis(
    config: AppConfig,
    records: list[MatchRecord],
    *,
    peer_comparison: PeerComparisonResult | None = None,
    ranked: RankedEntry | None = None,
    player_builds: list[dict[str, Any]] | None = None,
    assets: DDragonAssets | None = None,
) -> Path:
    """Run every analysis, write exports and render the report."""
    log = get_logger("pipeline")
    if not records:
        log.error("No qualifying ranked %s games found.", config.build_label)
        raise typer.Exit(code=1)

    records = sorted(records, key=lambda record: record.game_creation_ms, reverse=True)
    total_games = len(records)
    solo_count = sum(1 for record in records if record.queue_id == RANKED_SOLO_QUEUE_ID)
    flex_count = sum(1 for record in records if record.queue_id == RANKED_FLEX_QUEUE_ID)

    run_dir = config.report_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = config.run_graphs_dir
    graphs_dir.mkdir(parents=True, exist_ok=True)

    asset_catalog = assets or DDragonAssets(config)
    asset_catalog.ensure_downloaded()

    full_frames = build_analysis_frames(records)
    report_stats = compute_report_stats(full_frames, run_dir)

    summary = write_full_exports(
        config,
        records,
        run_dir,
        peer_comparison=peer_comparison,
        ranked=ranked,
        frames=full_frames,
        report_stats=report_stats,
    )
    GraphFactory(graphs_dir).death_heatmap_png(deaths_dataframe(records))

    window_specs: list[tuple[str, int | None]] = [
        (str(size), size) for size in GAME_WINDOW_OPTIONS
    ]
    window_specs.append(("all", None))

    report_views: dict[str, dict[str, Any]] = {}
    view_peers: dict[str, dict[str, PeerComparisonResult | None]] = {}
    default_queue = default_queue_filter_key(solo_count, flex_count)
    for queue_key in QUEUE_FILTER_OPTIONS:
        queue_records = filter_records_by_queue(records, queue_key)
        queue_total = len(queue_records)
        queue_peer = peer_comparison if queue_key == "solo" else None
        queue_label = QUEUE_SUBTITLE_LABELS[queue_key]
        windows: dict[str, dict[str, Any]] = {}
        window_peers: dict[str, PeerComparisonResult | None] = {}
        for window_key, limit in window_specs:
            sliced = slice_records(queue_records, limit)
            bundle = build_window_bundle(
                config,
                sliced,
                graphs_dir,
                peer_comparison=queue_peer,
                queue_label=queue_label,
                assets=asset_catalog,
                shared_stats=report_stats,
            )
            window_peers[window_key] = bundle.pop("_peer_result", None)
            serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
            windows[window_key] = serializable
        default_window = default_game_window_key(queue_total)
        report_views[queue_key] = {
            "total_games": queue_total,
            "default_window": default_window,
            "window_options": game_window_options(queue_total),
            "windows": windows,
        }
        view_peers[queue_key] = window_peers

    default_window = report_views[default_queue]["default_window"]
    default_bundle = report_views[default_queue]["windows"][default_window]
    default_peer = view_peers.get(default_queue, {}).get(default_window)

    context: dict[str, Any] = {
        **brand_context(from_dir=run_dir, output_dir=config.output_dir),
        "build_label": config.build_label,
        "champion": config.champion,
        "champion_icon": asset_catalog.champion_href(config.champion, from_dir=run_dir),
        "role_icon_href": asset_catalog.role_href(config.role, from_dir=run_dir),
        "role_display": config.role_display,
        "player_name": config.players_label,
        "recommendation_visible_count": VISIBLE_RECOMMENDATIONS,
        "queue_filter_default": default_queue,
        "queue_filter_options": queue_filter_options(solo_count, flex_count),
        "game_window_default": default_window,
        "game_window_total": report_views[default_queue]["total_games"],
        "game_window_options": report_views[default_queue]["window_options"],
        "queue_label": default_bundle.get("queue_label", QUEUE_SUBTITLE_LABELS[default_queue]),
        "report_views_json": serialize_report_views_json(report_views),
        "chatbot_stats": summary,
        "gemini_api_key": config.gemini_api_key,
    }
    context.update(
        bundle_to_template_context(default_bundle, peer_comparison=default_peer)
    )
    if player_builds:
        context["player_builds"] = build_player_builds_nav(
            player_builds,
            current_champion=config.champion,
            current_role=config.role,
            assets=asset_catalog,
            from_dir=run_dir,
        )

    static_src = config.template_dir / "static"
    if static_src.is_dir():
        shutil.copytree(static_src, run_dir / "static", dirs_exist_ok=True)

    builder = ReportBuilder(config.template_dir)
    report_path = builder.render(run_dir / "report.html", context)
    generated_at = context.get("generated_at", "")
    write_report_meta(
        run_dir,
        {
            "player": config.players_label,
            "riot_id": config.riot_id,
            "tagline": config.tagline,
            "champion": config.champion,
            "role": config.role,
            "role_display": config.role_display,
            "build_label": config.build_label,
            "games": total_games,
            "winrate": default_bundle["overview"]["winrate"],
            "generated_at": generated_at,
        },
    )
    player_label = config.players_label
    global_index, player_hub = refresh_report_indexes(
        config.output_dir,
        config.template_dir,
        player_dir=config.player_reports_dir,
        player_label=player_label,
        assets=asset_catalog,
    )
    if player_hub is not None:
        log.info("Done. Open %s (player hub: %s, index: %s)", report_path, player_hub, global_index)
    else:
        log.info("Done. Open %s (index: %s)", report_path, global_index)
    return report_path


def ensure_platform(client: RiotApiClient, records: list[MatchRecord], config: AppConfig) -> None:
    """Pick the league-v4 platform host from match ids or config."""
    if records:
        inferred = RiotApiClient.infer_platform_from_match_id(records[0].match_id)
        if inferred:
            client.set_platform(inferred)
            return
    if config.platform:
        client.set_platform(config.platform)


def run_with_peer(
    config: AppConfig,
    services: Services,
    puuid: str,
    records: list[MatchRecord],
    *,
    ranked: RankedEntry | None = None,
    player_builds: list[dict[str, Any]] | None = None,
    skip_peer: bool = False,
) -> Path:
    """Fetch rank, optionally build peer comparison and run the analysis pipeline."""
    if ranked is None:
        ensure_platform(services.client, records, config)
        ranked = services.client.fetch_solo_rank(puuid)
    peer = None
    if not skip_peer:
        matches_df = pd.DataFrame([r.to_row() for r in records])
        peer = build_peer_comparison(
            services.client,
            services.store,
            matches_df,
            records,
            puuid,
            ranked,
            champion=config.champion,
            role=config.role,
        )
    return run_analysis(
        config,
        records,
        peer_comparison=peer,
        ranked=ranked,
        player_builds=player_builds,
        assets=services.assets,
    )


def run_all_builds(
    services: Services,
    player_contexts: list[PlayerContext],
    *,
    fetch: bool = False,
    skip_peer: bool = False,
) -> Path:
    """Discover, parse once and analyse every eligible champion+lane build."""
    log = get_logger("pipeline")
    if fetch:
        fetch_matches(services)
        player_contexts = resolve_player_contexts(services)

    puuids = [context.puuid for context in player_contexts]
    primary_puuid = player_contexts[0].puuid

    pools = discover_build_pools(
        services.store,
        puuids,
        services.config,
        min_games=services.config.min_games,
    )
    if services.config.filter_champion:
        pools = [p for p in pools if p.champion == services.config.filter_champion]
    if services.config.filter_role:
        normalized = services.config.filter_role
        pools = [p for p in pools if p.role == normalized]
    if not pools:
        log.error(
            "No champion+lane builds with at least %d ranked games found.",
            services.config.min_games,
        )
        raise typer.Exit(code=1)

    log.info(
        "Found %d eligible build(s) with >= %d games: %s",
        len(pools),
        services.config.min_games,
        ", ".join(pool.build_label for pool in pools),
    )

    services.assets.ensure_downloaded()

    all_records = load_all_records(services, puuids)
    manifest_builds: list[dict[str, Any]] = []
    for pool in pools:
        grouped = group_records(all_records, pool.champion, pool.role)
        winrate = float(sum(r.win for r in grouped) / len(grouped)) if grouped else 0.0
        manifest_builds.append(
            build_manifest_entry(
                champion=pool.champion,
                role=pool.role,
                games=len(grouped),
                winrate=winrate,
            )
        )

    player_label = services.config.players_label
    player_dir = services.config.player_reports_dir

    ranked: RankedEntry | None = None
    last_report: Path | None = None
    for pool in tqdm(pools, desc="Analyzing builds", unit="build"):
        records = group_records(all_records, pool.champion, pool.role)
        if len(records) < services.config.min_games:
            log.warning("Skipping %s: only %d games after parse", pool.build_label, len(records))
            continue
        if ranked is None:
            ensure_platform(services.client, records, services.config)
            ranked = services.client.fetch_solo_rank(primary_puuid)
        build_config = services.config.model_copy(
            update={"champion": pool.champion, "role": pool.role}
        )
        build_config.report_dir.mkdir(parents=True, exist_ok=True)
        build_config.run_graphs_dir.mkdir(parents=True, exist_ok=True)
        last_report = run_with_peer(
            build_config,
            services,
            primary_puuid,
            records,
            ranked=ranked,
            player_builds=manifest_builds,
            skip_peer=skip_peer,
        )

    if last_report is None:
        log.error("No builds could be analysed.")
        raise typer.Exit(code=1)

    global_index, hub_path = refresh_report_indexes(
        services.config.output_dir,
        services.config.template_dir,
        player_dir=player_dir,
        player_label=player_label,
        assets=services.assets,
    )
    hub_path = hub_path or player_dir / "index.html"
    log.info(
        "Generated %d report(s) (≥%d games). Open %s (global index: %s)",
        len(manifest_builds),
        services.config.min_games,
        hub_path,
        global_index,
    )
    return hub_path
