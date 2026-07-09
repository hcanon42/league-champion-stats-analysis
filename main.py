"""Champion coaching analyzer CLI.

Commands:

* ``analyze`` — download matches + analyse every eligible champion/lane build,
* ``fetch``  — download matches into the local store only,
* ``report`` — rebuild all eligible build reports from stored matches,
* ``reports`` — rebuild the global report index,
* ``clear-cache`` — wipe the HTTP cache (stored matches are kept).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from tqdm import tqdm

from analysis.peer_comparison import (
    build_peer_comparison,
    comparisons_dataframe,
    peer_recommendations,
)
from analysis.coach import CoachEngine, VISIBLE_RECOMMENDATIONS
from analysis.deaths import blind_spot_zones, death_summary, deaths_dataframe
from analysis.economy import economy_summary, reset_quality
from analysis.items import build_path_stats, item_summary, items_dataframe
from analysis.laning import laning_summary
from analysis.matchups import matchup_recommendation, matchup_summary, matchups_dataframe
from analysis.objectives import objective_summary, objectives_dataframe
from analysis.positioning import macro_summary
from analysis.runes import rune_setup_stats, rune_summary, runes_dataframe
from analysis.statistics import StatisticsEngine
from analysis.teamfights import teamfight_summary, teamfights_dataframe
from analysis.timeline import timeline_dataframe_rows
from analysis.vision import vision_dataframe, vision_summary
from cache import HttpCache, MatchStore
from config import AppConfig, load_config
from export import Exporter
from graphs import GraphFactory
from models import MatchRecord, PeerComparisonResult, RankedEntry
from parser import BaseMatchFilter, ItemCatalog, MatchParser, discover_build_pools
from report import (
    ReportBuilder,
    build_manifest_entry,
    build_player_builds_nav,
    discover_reports,
    improvement_score,
    refresh_report_indexes,
    score_badge,
    write_report_meta,
)
from riot_api import RiotApiClient
from utils import get_logger, setup_logging

app = typer.Typer(
    help="Ranked solo queue coaching analyzer for any champion + lane (Riot Match-V5 API).",
    no_args_is_help=True,
)


@dataclass
class Services:
    """Wired application services (composition root for DI)."""

    config: AppConfig
    http_cache: HttpCache
    store: MatchStore
    client: RiotApiClient


def _build_services(
    riot_id: str | None,
    tagline: str | None,
    region: str | None,
    platform: str | None,
    api_key: str | None,
    count: int | None,
    min_games: int | None,
    verbose: bool,
) -> Services:
    """Load configuration and construct every service.

    Args:
        riot_id: Riot ID game name (CLI override).
        tagline: Riot ID tagline (CLI override).
        region: Regional or platform routing value (CLI override).
        platform: Platform host for league-v4 (``euw1``, ``na1``, ...).
        api_key: Riot API key (CLI override; falls back to ``RIOT_API_KEY``).
        count: Number of matches to consider (CLI override).
        min_games: Minimum solo/duo games per champion+lane build.
        verbose: Enable debug logging.

    Returns:
        The wired :class:`Services`.
    """
    setup_logging(verbose)
    config = load_config(
        riot_id=riot_id,
        tagline=tagline,
        region=region,
        platform=platform,
        api_key=api_key,
        match_count=count,
        min_games=min_games,
        verbose=verbose,
    )
    config.ensure_directories()
    http_cache = HttpCache(config.http_cache_dir)
    store = MatchStore(config.db_path)
    client = RiotApiClient(config, http_cache, store)
    return Services(config=config, http_cache=http_cache, store=store, client=client)


def _fetch(services: Services) -> str:
    """Resolve the PUUID and download the match history.

    Args:
        services: Wired services.

    Returns:
        The player's PUUID.
    """
    config = services.config
    puuid = services.client.resolve_puuid(config.riot_id, config.tagline)
    match_ids = services.client.fetch_match_ids(puuid, config.match_count)
    services.client.download_matches(puuid, match_ids)
    return puuid


def _load_all_records(services: Services, puuid: str) -> list[MatchRecord]:
    """Parse every stored ranked solo game for the player (all champions/lanes).

    Args:
        services: Wired services.
        puuid: The player's PUUID.

    Returns:
        Parsed match records, most recent first.
    """
    log = get_logger("pipeline")
    catalog = ItemCatalog(services.client.fetch_item_catalog())
    match_filter = BaseMatchFilter(services.config)
    parser = MatchParser(catalog)
    records: list[MatchRecord] = []
    match_ids = list(services.store.iter_match_ids(puuid))
    for match_id in tqdm(match_ids, desc="Parsing matches", unit="match"):
        match = services.store.load_match(match_id)
        timeline = services.store.load_timeline(match_id)
        if not match or not timeline:
            continue
        if not match_filter.accept(match, puuid):
            continue
        try:
            records.append(parser.parse(match, timeline, puuid))
        except Exception as exc:  # one malformed match must not kill the run
            log.warning("Failed to parse %s: %s", match_id, exc)
    records.sort(key=lambda r: r.game_creation_ms, reverse=True)
    log.info("Parsed %d qualifying ranked solo queue games", len(records))
    return records


def _group_records(
    records: list[MatchRecord], champion: str, role: str
) -> list[MatchRecord]:
    """Filter parsed records to one champion + lane build."""
    return [r for r in records if r.champion == champion and r.role == role]


def _card(value: Any, suffix: str = "") -> str:
    """Format a possibly-missing metric for a dashboard card."""
    return "—" if value is None else f"{value}{suffix}"


def run_analysis(
    config: AppConfig,
    records: list[MatchRecord],
    *,
    peer_comparison: PeerComparisonResult | None = None,
    ranked: RankedEntry | None = None,
    player_builds: list[dict[str, Any]] | None = None,
) -> Path:
    """Run every analysis, write exports and render the report.

    Args:
        config: Application configuration (directories, player identity).
        records: Parsed match records.
        peer_comparison: Optional rank-peer comparison block.
        ranked: Player's solo queue rank, if known.

    Returns:
        Path of the rendered ``report.html``.

    Raises:
        typer.Exit: When there are no qualifying games.
    """
    log = get_logger("pipeline")
    if not records:
        log.error("No qualifying ranked solo queue %s games found.", config.build_label)
        raise typer.Exit(code=1)

    run_dir = config.report_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = config.run_graphs_dir
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- Tables
    matches_df = pd.DataFrame([r.to_row() for r in records])
    deaths_df = deaths_dataframe(records)
    tf_df = teamfights_dataframe(records)
    obj_df = objectives_dataframe(records)
    vis_df = vision_dataframe(records)
    runes_df = runes_dataframe(records)
    items_df = items_dataframe(matches_df)
    matchups_df = matchups_dataframe(matches_df)
    timeline_df = pd.DataFrame(
        [row for r in records for row in timeline_dataframe_rows(r.match_id, r.timeline)]
    )

    # ---------------------------------------------------------- Statistics
    stats = StatisticsEngine(matches_df, run_dir)
    corr = stats.correlation_matrix()
    win_corrs = stats.win_correlations()
    model = stats.train_win_predictor()
    clusters_df = stats.cluster_games()

    # --------------------------------------------------------------- Coach
    coach = CoachEngine(
        matches_df, deaths_df, matchups_df, obj_df, stats, build_label=config.build_label
    )
    recommendations = coach.generate()
    if peer_comparison is not None:
        peer_recs = peer_recommendations(
            peer_comparison.comparisons,
            peer_comparison.rank_label,
            max(peer_comparison.peer_games, len(records)),
            build_label=peer_comparison.build_label,
        )
        recommendations = sorted(
            peer_recs + recommendations, key=lambda r: r.priority, reverse=True
        )

    # ------------------------------------------------------------ Summary
    overview = {
        "winrate": round(float(matches_df["win"].mean()), 3),
        "avg_kda": round(float(matches_df["kda"].mean()), 2),
        "avg_dpm": round(float(matches_df["dpm"].mean()), 0),
        "avg_cspm": round(float(matches_df["cspm"].mean()), 2),
        "avg_damage_share": round(float(matches_df["damage_share"].mean()), 3),
        "avg_deaths": round(float(matches_df["deaths"].mean()), 1),
        "avg_vspm": round(float(matches_df["vspm"].mean()), 2),
        "avg_duration": round(float(matches_df["duration_min"].mean()), 1),
    }
    lane = laning_summary(matches_df)
    economy = economy_summary(matches_df)
    resets = reset_quality(records)
    vision = vision_summary(vis_df)
    deaths_agg = death_summary(deaths_df)
    fights = teamfight_summary(tf_df)
    objectives_agg = objective_summary(obj_df)
    macro = macro_summary(records, matches_df)
    summary: dict[str, Any] = {
        "player": f"{config.riot_id}#{config.tagline}",
        "champion": config.champion,
        "role": config.role,
        "build_label": config.build_label,
        "games": len(records),
        "overview": overview,
        "laning": lane,
        "economy": economy | {"resets": resets},
        "vision": vision,
        "deaths": deaths_agg,
        "teamfights": fights,
        "objectives": objectives_agg,
        "macro": macro,
        "matchups": matchup_summary(matchups_df),
        "items": item_summary(items_df),
        "runes": rune_summary(runes_df),
        "win_correlations": [vars(c) for c in win_corrs],
        "ml_model": {
            "trained": model.trained,
            "cv_auc_mean": model.cv_auc_mean,
            "cv_auc_std": model.cv_auc_std,
            "n_games": model.n_games,
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

    # ------------------------------------------------------------- Exports
    matchups_export = matchups_df.copy()
    if not matchups_export.empty:
        matchups_export["recommendation"] = matchups_export.apply(matchup_recommendation, axis=1)
    corr_export = corr.reset_index().rename(columns={"index": "feature"}) if not corr.empty else corr
    exporter = Exporter(run_dir)
    export_tables: dict[str, pd.DataFrame] = {
        "matches": matches_df,
        "deaths": deaths_df,
        "timeline": timeline_df,
        "matchups": matchups_export,
        "vision": vis_df,
        "items": items_df,
        "runes": runes_df,
        "objectives": obj_df,
        "teamfights": tf_df,
        "correlations": corr_export,
    }
    if peer_comparison is not None:
        export_tables["rank_comparison"] = comparisons_dataframe(peer_comparison)
    exporter.write_all(
        tables=export_tables,
        summary=summary,
        recommendations=recommendations,
        build_label=config.build_label,
    )

    # ------------------------------------------------------------- Figures
    graphs = GraphFactory(graphs_dir)
    graphs.death_heatmap_png(deaths_df)
    series = [(r.win, r.timeline.gold_series, r.timeline.opp_gold_series) for r in records]
    figures = {
        "winrate_trend": graphs.winrate_trend(matches_df),
        "gold_diff_timeline": graphs.gold_diff_timeline(series),
        "gd10_histogram": graphs.gd10_histogram(matches_df),
        "deaths_box": graphs.deaths_box(matches_df),
        "cs10_violin": graphs.cs10_violin(matches_df),
        "dpm_scatter": graphs.dpm_scatter(matches_df),
        "vision_trend": graphs.vision_trend(matches_df),
        "death_heatmap": graphs.death_heatmap(deaths_df),
        "correlation_heatmap": graphs.correlation_heatmap(corr),
        "win_correlation_bar": graphs.win_correlation_bar(win_corrs),
        "feature_importance": graphs.feature_importance(model),
        "cluster_scatter": graphs.cluster_scatter(clusters_df),
        "matchup_bar": graphs.matchup_bar(matchups_df),
        "item_winrate_bar": graphs.item_winrate_bar(items_df),
        "rune_winrate_bar": graphs.rune_winrate_bar(rune_setup_stats(runes_df)),
        "objective_timing": graphs.objective_timing(obj_df),
    }
    if peer_comparison is not None:
        figures["peer_comparison"] = graphs.peer_comparison_chart(
            peer_comparison.comparisons, build_label=peer_comparison.build_label
        )

    # -------------------------------------------------------------- Report
    score, components = improvement_score(matches_df)
    matchup_rows = matchups_export.head(20).to_dict("records") if not matchups_export.empty else []
    context: dict[str, Any] = {
        "app_title": "Champion Stats Analyzer",
        "build_label": config.build_label,
        "champion": config.champion,
        "role_display": config.role_display,
        "player_name": f"{config.riot_id}#{config.tagline}",
        "total_games": len(records),
        "patch_range": f"{matches_df['patch'].min()} – {matches_df['patch'].max()}",
        "overview": overview,
        "score": score,
        "score_components": components,
        "figures": figures,
        "lane_cards": [
            ("Gold diff @10", _card(lane.get("avg_gd10"))),
            ("CS diff @10", _card(lane.get("avg_csd10"))),
            ("XP diff @10", _card(lane.get("avg_xpd10"))),
            ("Lane win rate", _card(_pct(lane.get("lane_win_rate")))),
            ("WR when ahead @10", _card(_pct(lane.get("winrate_when_ahead_at_10")))),
            ("WR when behind @10", _card(_pct(lane.get("winrate_when_behind_at_10")))),
            ("Deaths pre-14", _card(lane.get("avg_deaths_pre14"))),
            ("Roams pre-15", _card(lane.get("avg_roams_pre15"))),
        ],
        "economy_cards": [
            ("GPM", _card(economy.get("avg_gpm"))),
            ("CS/min", _card(economy.get("avg_cspm"))),
            ("Gold share", _card(_pct(economy.get("avg_gold_share")))),
            ("Damage per gold", _card(economy.get("avg_damage_per_gold"))),
            ("Unspent gold/recall", _card(economy.get("avg_unspent_gold_before_recall"), "g")),
            ("First recall", _card(resets.get("avg_first_recall_min"), " min")),
            ("Time dead/game", _card(economy.get("avg_time_dead_s"), "s")),
        ],
        "vision_cards": [
            ("Vision score", _card(vision.get("avg_vision_score"))),
            ("VS/min", _card(vision.get("avg_vspm"))),
            ("Control wards", _card(vision.get("avg_control_wards"))),
            ("CW lifetime", _card(vision.get("avg_control_ward_lifetime_s"), "s")),
            ("VS/min in wins", _card(vision.get("avg_vspm_wins"))),
            ("VS/min in losses", _card(vision.get("avg_vspm_losses"))),
        ],
        "death_cards": [
            ("Total deaths", _card(deaths_agg.get("total_deaths"))),
            ("Solo deaths", _card(_pct(deaths_agg.get("solo_death_rate")))),
            ("Greed deaths", _card(_pct(deaths_agg.get("greed_death_rate")))),
            ("Side-lane deaths", _card(_pct(deaths_agg.get("side_lane_death_rate")))),
            ("Before dragon", _card(_pct(deaths_agg.get("death_before_dragon_rate")))),
            ("Avg death minute", _card(deaths_agg.get("avg_death_minute"))),
            ("Top killer", _card(deaths_agg.get("most_common_killer"))),
        ],
        "teamfight_cards": [
            ("Fights detected", _card(fights.get("total_fights"))),
            ("Participation", _card(_pct(fights.get("participation_rate")))),
            ("Fight win rate", _card(_pct(fights.get("fight_win_rate")))),
            ("Damage/fight", _card(fights.get("avg_damage_per_fight"))),
            ("Death rate in fights", _card(_pct(fights.get("death_rate_in_fights")))),
            ("Front-to-back", _card(fights.get("avg_front_to_back"))),
        ],
        "objective_rows": sorted(objectives_agg.get("by_kind", {}).items()),
        "blind_spots": blind_spot_zones(deaths_df),
        "build_paths": build_path_stats(matches_df).head(10).to_dict("records"),
        "rune_rows": rune_setup_stats(runes_df).to_dict("records"),
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
        "recommendation_visible_count": VISIBLE_RECOMMENDATIONS,
    }
    if player_builds:
        context["player_builds"] = build_player_builds_nav(
            player_builds,
            current_champion=config.champion,
            current_role=config.role,
        )
    if peer_comparison is not None:
        context["peer_comparison"] = peer_comparison
        context["peer_rows"] = [c.model_dump() for c in peer_comparison.comparisons]
        context["figures"]["peer_comparison"] = figures.get("peer_comparison", "")
    builder = ReportBuilder(config.template_dir)
    report_path = builder.render(run_dir / "report.html", context)
    generated_at = context.get("generated_at", "")
    write_report_meta(
        run_dir,
        {
            "player": f"{config.riot_id}#{config.tagline}",
            "riot_id": config.riot_id,
            "tagline": config.tagline,
            "champion": config.champion,
            "role": config.role,
            "role_display": config.role_display,
            "build_label": config.build_label,
            "games": len(records),
            "winrate": overview["winrate"],
            "generated_at": generated_at,
        },
    )
    player_label = f"{config.riot_id}#{config.tagline}"
    global_index, player_hub = refresh_report_indexes(
        config.output_dir,
        config.template_dir,
        player_dir=config.player_reports_dir,
        player_label=player_label,
    )
    if player_hub is not None:
        log.info("Done. Open %s (player hub: %s, index: %s)", report_path, player_hub, global_index)
    else:
        log.info("Done. Open %s (index: %s)", report_path, global_index)
    return report_path


def _pct(value: float | None) -> str | None:
    """Format a ratio as a percentage string, keeping ``None``."""
    return None if value is None else f"{value * 100:.0f}%"


def _ensure_platform(client: RiotApiClient, records: list[MatchRecord], config: AppConfig) -> None:
    """Pick the league-v4 platform host from match ids or config."""
    if records:
        inferred = RiotApiClient.infer_platform_from_match_id(records[0].match_id)
        if inferred:
            client.set_platform(inferred)
            return
    if config.platform:
        client.set_platform(config.platform)


def _run_with_peer(
    config: AppConfig,
    services: Services,
    puuid: str,
    records: list[MatchRecord],
    *,
    ranked: RankedEntry | None = None,
    player_builds: list[dict[str, Any]] | None = None,
) -> Path:
    """Fetch rank, build peer comparison and run the analysis pipeline."""
    if ranked is None:
        _ensure_platform(services.client, records, config)
        ranked = services.client.fetch_solo_rank(puuid)
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
    )


def run_all_builds(services: Services, puuid: str, *, fetch: bool = False) -> Path:
    """Discover, parse once and analyse every eligible champion+lane build."""
    log = get_logger("pipeline")
    if fetch:
        _fetch(services)

    pools = discover_build_pools(
        services.store,
        puuid,
        services.config,
        min_games=services.config.min_games,
    )
    if not pools:
        log.error(
            "No champion+lane builds with at least %d ranked solo queue games found.",
            services.config.min_games,
        )
        raise typer.Exit(code=1)

    log.info(
        "Found %d eligible build(s) with >= %d games: %s",
        len(pools),
        services.config.min_games,
        ", ".join(pool.build_label for pool in pools),
    )

    all_records = _load_all_records(services, puuid)
    manifest_builds: list[dict[str, Any]] = []
    for pool in pools:
        grouped = _group_records(all_records, pool.champion, pool.role)
        winrate = float(sum(r.win for r in grouped) / len(grouped)) if grouped else 0.0
        manifest_builds.append(
            build_manifest_entry(
                champion=pool.champion,
                role=pool.role,
                games=len(grouped),
                winrate=winrate,
            )
        )

    player_label = f"{services.config.riot_id}#{services.config.tagline}"
    player_dir = services.config.player_reports_dir

    ranked: RankedEntry | None = None
    last_report: Path | None = None
    for pool in tqdm(pools, desc="Analyzing builds", unit="build"):
        records = _group_records(all_records, pool.champion, pool.role)
        if len(records) < services.config.min_games:
            log.warning("Skipping %s: only %d games after parse", pool.build_label, len(records))
            continue
        if ranked is None:
            _ensure_platform(services.client, records, services.config)
            ranked = services.client.fetch_solo_rank(puuid)
        build_config = services.config.model_copy(
            update={"champion": pool.champion, "role": pool.role}
        )
        build_config.report_dir.mkdir(parents=True, exist_ok=True)
        build_config.run_graphs_dir.mkdir(parents=True, exist_ok=True)
        last_report = _run_with_peer(
            build_config,
            services,
            puuid,
            records,
            ranked=ranked,
            player_builds=manifest_builds,
        )

    if last_report is None:
        log.error("No builds could be analysed.")
        raise typer.Exit(code=1)

    global_index, hub_path = refresh_report_indexes(
        services.config.output_dir,
        services.config.template_dir,
        player_dir=player_dir,
        player_label=player_label,
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


@app.command()
def analyze(
    riot_id: str = typer.Option(None, "--riot-id", help="Riot ID game name (e.g. 'Faker')."),
    tagline: str = typer.Option(None, "--tagline", help="Riot ID tagline without '#'."),
    region: str = typer.Option(None, "--region", help="Routing region (europe/americas/asia/sea) or platform (euw1, na1...)."),
    platform: str = typer.Option(None, "--platform", help="Platform for league-v4 rank lookup (euw1, eun1, na1...). Auto-detected from match ids when omitted."),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY", help="Riot API key."),
    count: int = typer.Option(None, "--count", help="Max matches to scan (default 500)."),
    min_games: int = typer.Option(None, "--min-games", help="Min solo/duo games per champion+lane (default 20)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Run the full pipeline: download, analyse all eligible builds and generate reports."""
    services = _build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose
    )
    try:
        puuid = _fetch(services)
        run_all_builds(services, puuid, fetch=False)
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def fetch(
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option(None, "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    count: int = typer.Option(None, "--count"),
    min_games: int = typer.Option(None, "--min-games"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download matches into the local store without analysing them."""
    services = _build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose
    )
    try:
        _fetch(services)
        get_logger().info("Store now holds %d complete matches.", services.store.count())
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def report(
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option(None, "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    min_games: int = typer.Option(None, "--min-games"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild all eligible build reports from already-downloaded matches."""
    services = _build_services(
        riot_id, tagline, region, platform, api_key, None, min_games, verbose
    )
    try:
        puuid = services.client.resolve_puuid(services.config.riot_id, services.config.tagline)
        run_all_builds(services, puuid, fetch=False)
    finally:
        services.store.close()
        services.http_cache.close()


@app.command("clear-cache")
def clear_cache(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Clear the HTTP response cache (downloaded matches are kept)."""
    setup_logging(verbose)
    config = load_config(api_key="unused", riot_id="unused", tagline="unused")
    cache = HttpCache(config.http_cache_dir)
    cache.clear()
    cache.close()
    get_logger().info("HTTP cache cleared.")


@app.command()
def reports(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild the report index from saved reports (no analysis)."""
    setup_logging(verbose)
    config = load_config(api_key="unused", riot_id="unused", tagline="unused")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = refresh_report_indexes(config.output_dir, config.template_dir)[0]
    count = len(discover_reports(config.output_dir))
    get_logger().info("Index refreshed with %d report(s). Open %s", count, index_path)


if __name__ == "__main__":
    app()
