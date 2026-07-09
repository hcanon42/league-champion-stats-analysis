"""Champion coaching analyzer CLI.

Commands:

* ``analyze`` — download matches + analyse every eligible champion/lane build,
* ``fetch``  — download matches into the local store only,
* ``report`` — rebuild all eligible build reports from stored matches,
* ``reports`` — rebuild the global report index,
* ``clear-cache`` — wipe the HTTP cache (stored matches are kept).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from tqdm import tqdm

from analysis.peer_comparison import (
    build_peer_comparison,
    comparisons_dataframe,
    peer_comparison_for_window,
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
from champions import parse_riot_id
from config import (
    DEFAULT_GAME_WINDOW,
    GAME_WINDOW_OPTIONS,
    AppConfig,
    PlayerIdentity,
    load_config,
)
from export import Exporter
from graphs import GraphFactory
from models import MatchRecord, PeerComparisonResult, RankedEntry
from parser import BaseMatchFilter, ItemCatalog, MatchParser, discover_build_pools
from report import (
    ReportBuilder,
    ScoreComponent,
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


@dataclass
class PlayerContext:
    """Resolved player identity and PUUID."""

    riot_id: str
    tagline: str
    puuid: str

    @property
    def label(self) -> str:
        return f"{self.riot_id}#{self.tagline}"


def _parse_players_cli(
    player_flags: list[str],
    riot_id: str | None,
    tagline: str | None,
) -> list[PlayerIdentity] | None:
    """Resolve CLI player identities from ``--player`` or ``--riot-id``/``--tagline``."""
    if player_flags:
        players: list[PlayerIdentity] = []
        for value in player_flags:
            try:
                name, tag = parse_riot_id(value)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            players.append(PlayerIdentity(riot_id=name, tagline=tag))
        return players
    if riot_id and tagline:
        return [PlayerIdentity(riot_id=riot_id, tagline=tagline)]
    return None


def _build_services(
    riot_id: str | None,
    tagline: str | None,
    region: str | None,
    platform: str | None,
    api_key: str | None,
    count: int | None,
    min_games: int | None,
    verbose: bool,
    *,
    players: list[PlayerIdentity] | None = None,
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
        players: Optional explicit player list (``--player``).

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
        players=players,
    )
    config.ensure_directories()
    http_cache = HttpCache(config.http_cache_dir)
    store = MatchStore(config.db_path)
    client = RiotApiClient(config, http_cache, store)
    return Services(config=config, http_cache=http_cache, store=store, client=client)


def _fetch(services: Services) -> list[PlayerContext]:
    """Resolve every tracked player and download their match histories.

    Args:
        services: Wired services.

    Returns:
        Resolved player contexts with PUUIDs.
    """
    config = services.config
    contexts: list[PlayerContext] = []
    for player in config.players:
        puuid = services.client.resolve_puuid(player.riot_id, player.tagline)
        match_ids = services.client.fetch_match_ids(puuid, config.match_count)
        services.client.download_matches(puuid, match_ids)
        contexts.append(
            PlayerContext(riot_id=player.riot_id, tagline=player.tagline, puuid=puuid)
        )
    return contexts


def _resolve_player_contexts(services: Services) -> list[PlayerContext]:
    """Resolve PUUIDs for every configured player without downloading."""
    return [
        PlayerContext(
            riot_id=player.riot_id,
            tagline=player.tagline,
            puuid=services.client.resolve_puuid(player.riot_id, player.tagline),
        )
        for player in services.config.players
    ]


def _load_all_records(
    services: Services, puuids: str | list[str]
) -> list[MatchRecord]:
    """Parse stored ranked solo games for one or more players.

    Args:
        services: Wired services.
        puuids: A single PUUID or list of PUUIDs whose games are pooled.

    Returns:
        Parsed match records, most recent first.
    """
    if isinstance(puuids, str):
        puuid_list = [puuids]
    else:
        puuid_list = list(puuids)
    log = get_logger("pipeline")
    catalog = ItemCatalog(services.client.fetch_item_catalog())
    match_filter = BaseMatchFilter(services.config)
    parser = MatchParser(catalog)
    records: list[MatchRecord] = []
    for puuid in puuid_list:
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


def _pct(value: float | None) -> str | None:
    """Format a ratio as a percentage string, keeping ``None``."""
    return None if value is None else f"{value * 100:.0f}%"


def _card_entries(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Convert label/value card pairs to JSON-friendly dicts."""
    return [{"label": label, "value": value} for label, value in pairs]


def _overview_card_entries(overview: dict[str, Any]) -> list[dict[str, str]]:
    """Build overview cards with win/loss styling metadata."""
    winrate = float(overview.get("winrate", 0.0))
    return [
        {
            "label": "Win rate",
            "value": f"{winrate * 100:.0f}%",
            "value_class": "win" if winrate >= 0.5 else "loss",
        },
        {"label": "KDA", "value": str(overview.get("avg_kda", "—")), "value_class": ""},
        {"label": "DPM", "value": str(overview.get("avg_dpm", "—")), "value_class": ""},
        {"label": "CS/min", "value": str(overview.get("avg_cspm", "—")), "value_class": ""},
        {
            "label": "Damage share",
            "value": f"{float(overview.get('avg_damage_share', 0)) * 100:.0f}%",
            "value_class": "",
        },
        {"label": "Deaths/game", "value": str(overview.get("avg_deaths", "—")), "value_class": ""},
        {"label": "Vision/min", "value": str(overview.get("avg_vspm", "—")), "value_class": ""},
        {
            "label": "Avg game",
            "value": f"{overview.get('avg_duration', '—')} min",
            "value_class": "",
        },
    ]


def _slice_records(records: list[MatchRecord], limit: int | None) -> list[MatchRecord]:
    """Return the most recent ``limit`` games, or all when ``limit`` is ``None``."""
    if limit is None:
        return records
    return records[:limit]


def _default_game_window_key(total_games: int) -> str:
    """Pick the initial dashboard window."""
    if total_games >= DEFAULT_GAME_WINDOW:
        return str(DEFAULT_GAME_WINDOW)
    return "all"


def _serialize_game_windows_json(game_windows: dict[str, dict[str, Any]]) -> str:
    """JSON-encode window snapshots for safe embedding in a ``<script>`` tag."""
    encoded = json.dumps(game_windows, default=str)
    # Plotly HTML may contain ``</script>``; escape so the tag is not closed early.
    return encoded.replace("</", r"<\/")


def _game_window_options(total_games: int) -> list[dict[str, Any]]:
    """Toggle metadata for the report template."""
    options = [
        {
            "key": str(size),
            "label": f"Last {size}",
            "enabled": total_games >= size,
        }
        for size in GAME_WINDOW_OPTIONS
    ]
    options.append({"key": "all", "label": "All", "enabled": True})
    return options


def _peer_row_display(row: dict[str, Any]) -> dict[str, str]:
    """Format peer comparison row values for HTML/JSON."""
    metric = row.get("metric")
    yours = row.get("yours")
    peer_avg = row.get("peer_avg")
    if metric in {"win", "kill_participation", "damage_share"}:
        yours_display = f"{float(yours) * 100:.0f}%"
        peer_display = f"{float(peer_avg) * 100:.0f}%"
    else:
        yours_display = str(yours)
        peer_display = str(peer_avg)
    delta_pct = row.get("delta_pct")
    delta = row.get("delta")
    if delta_pct is not None:
        gap_display = f"{float(delta_pct):+.0f}%"
    else:
        gap_display = f"{float(delta):+.1f}"
    return {
        "label": str(row.get("label", "")),
        "yours": yours_display,
        "peer_avg": peer_display,
        "gap": gap_display,
        "verdict": str(row.get("verdict", "inline")),
    }


def _build_window_bundle(
    config: AppConfig,
    records: list[MatchRecord],
    graphs_dir: Path,
    *,
    peer_comparison: PeerComparisonResult | None = None,
) -> dict[str, Any]:
    """Run the analysis pipeline for one game window and return a JSON bundle."""
    matches_df = pd.DataFrame([r.to_row() for r in records])
    deaths_df = deaths_dataframe(records)
    tf_df = teamfights_dataframe(records)
    obj_df = objectives_dataframe(records)
    vis_df = vision_dataframe(records)
    runes_df = runes_dataframe(records)
    items_df = items_dataframe(matches_df)
    matchups_df = matchups_dataframe(matches_df)

    stats = StatisticsEngine(matches_df, graphs_dir)
    corr = stats.correlation_matrix()
    win_corrs = stats.win_correlations()
    model = stats.train_win_predictor()
    clusters_df = stats.cluster_games()

    window_peer = peer_comparison
    if peer_comparison is not None:
        window_peer = peer_comparison_for_window(peer_comparison, matches_df, records)

    coach = CoachEngine(
        matches_df, deaths_df, matchups_df, obj_df, stats, build_label=config.build_label
    )
    recommendations = coach.generate()
    if window_peer is not None:
        peer_recs = peer_recommendations(
            window_peer.comparisons,
            window_peer.rank_label,
            max(window_peer.peer_games, len(records)),
            build_label=window_peer.build_label,
        )
        recommendations = sorted(
            peer_recs + recommendations, key=lambda rec: rec.priority, reverse=True
        )

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

    score, components = improvement_score(matches_df)
    matchups_export = matchups_df.copy()
    if not matchups_export.empty:
        matchups_export["recommendation"] = matchups_export.apply(matchup_recommendation, axis=1)
    matchup_rows = matchups_export.head(20).to_dict("records") if not matchups_export.empty else []

    graphs = GraphFactory(graphs_dir)
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

    bundle: dict[str, Any] = {
        "total_games": len(records),
        "patch_range": (
            f"{matches_df['patch'].min()} – {matches_df['patch'].max()}"
            if not matches_df.empty
            else "—"
        ),
        "overview": overview,
        "overview_cards": _overview_card_entries(overview),
        "score": score,
        "score_components": [asdict(component) for component in components],
        "lane_cards": _card_entries(
            [
                ("Gold diff @10", _card(lane.get("avg_gd10"))),
                ("CS diff @10", _card(lane.get("avg_csd10"))),
                ("XP diff @10", _card(lane.get("avg_xpd10"))),
                ("Lane win rate", _card(_pct(lane.get("lane_win_rate")))),
                ("WR when ahead @10", _card(_pct(lane.get("winrate_when_ahead_at_10")))),
                ("WR when behind @10", _card(_pct(lane.get("winrate_when_behind_at_10")))),
                ("Deaths pre-14", _card(lane.get("avg_deaths_pre14"))),
                ("Gank deaths (lane)", _card(lane.get("avg_gank_deaths_laning"))),
                ("Under own tower (lane)", _card(lane.get("avg_under_own_tower_laning_deaths"))),
                ("Under enemy tower (lane)", _card(lane.get("avg_under_enemy_tower_laning_deaths"))),
                ("Roams pre-15", _card(lane.get("avg_roams_pre15"))),
            ]
        ),
        "economy_cards": _card_entries(
            [
                ("GPM", _card(economy.get("avg_gpm"))),
                ("CS/min", _card(economy.get("avg_cspm"))),
                ("Gold share", _card(_pct(economy.get("avg_gold_share")))),
                ("Damage per gold", _card(economy.get("avg_damage_per_gold"))),
                ("Unspent gold/recall", _card(economy.get("avg_unspent_gold_before_recall"), "g")),
                ("First recall", _card(resets.get("avg_first_recall_min"), " min")),
                ("Time dead/game", _card(economy.get("avg_time_dead_s"), "s")),
            ]
        ),
        "vision_cards": _card_entries(
            [
                ("Vision score", _card(vision.get("avg_vision_score"))),
                ("VS/min", _card(vision.get("avg_vspm"))),
                ("Control wards", _card(vision.get("avg_control_wards"))),
                ("CW lifetime", _card(vision.get("avg_control_ward_lifetime_s"), "s")),
                ("VS/min in wins", _card(vision.get("avg_vspm_wins"))),
                ("VS/min in losses", _card(vision.get("avg_vspm_losses"))),
            ]
        ),
        "death_cards": _card_entries(
            [
                ("Total deaths", _card(deaths_agg.get("total_deaths"))),
                ("Solo deaths", _card(_pct(deaths_agg.get("solo_death_rate")))),
                ("Gank deaths (lane)", _card(_pct(deaths_agg.get("gank_death_rate")))),
                ("Under own tower (lane)", _card(_pct(deaths_agg.get("under_own_tower_laning_death_rate")))),
                ("Under enemy tower (lane)", _card(_pct(deaths_agg.get("under_enemy_tower_laning_death_rate")))),
                ("Greed deaths", _card(_pct(deaths_agg.get("greed_death_rate")))),
                ("Side-lane deaths", _card(_pct(deaths_agg.get("side_lane_death_rate")))),
                ("Before dragon", _card(_pct(deaths_agg.get("death_before_dragon_rate")))),
                ("Avg death minute", _card(deaths_agg.get("avg_death_minute"))),
                ("Top killer", _card(deaths_agg.get("most_common_killer"))),
            ]
        ),
        "teamfight_cards": _card_entries(
            [
                ("Fights detected", _card(fights.get("total_fights"))),
                ("Participation", _card(_pct(fights.get("participation_rate")))),
                ("Fight win rate", _card(_pct(fights.get("fight_win_rate")))),
                ("Damage/fight", _card(fights.get("avg_damage_per_fight"))),
                ("Death rate in fights", _card(_pct(fights.get("death_rate_in_fights")))),
                ("Front-to-back", _card(fights.get("avg_front_to_back"))),
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
        "figures": figures,
    }

    if window_peer is not None:
        figures["peer_comparison"] = graphs.peer_comparison_chart(
            window_peer.comparisons, build_label=window_peer.build_label
        )
        bundle["peer"] = {
            "subtitle": (
                f"Your averages vs {config.build_label} at {window_peer.rank_label} · "
                f"{window_peer.source} · {window_peer.peer_games} peer games "
                f"({window_peer.peer_players} players)"
            ),
            "tier": window_peer.tier,
            "strengths": window_peer.strengths,
            "weaknesses": window_peer.weaknesses,
            "rows": [
                _peer_row_display(row.model_dump()) for row in window_peer.comparisons
            ],
        }
        bundle["figures"]["peer_comparison"] = figures["peer_comparison"]
        bundle["_peer_result"] = window_peer

    return bundle


def _bundle_to_template_context(
    bundle: dict[str, Any],
    *,
    peer_comparison: PeerComparisonResult | None = None,
) -> dict[str, Any]:
    """Map a window bundle onto the Jinja template field names."""
    context: dict[str, Any] = {
        "total_games": bundle["total_games"],
        "patch_range": bundle["patch_range"],
        "overview": bundle["overview"],
        "score": bundle["score"],
        "score_components": [
            ScoreComponent(**component) for component in bundle["score_components"]
        ],
        "figures": bundle["figures"],
        "lane_cards": [(card["label"], card["value"]) for card in bundle["lane_cards"]],
        "economy_cards": [(card["label"], card["value"]) for card in bundle["economy_cards"]],
        "vision_cards": [(card["label"], card["value"]) for card in bundle["vision_cards"]],
        "death_cards": [(card["label"], card["value"]) for card in bundle["death_cards"]],
        "teamfight_cards": [
            (card["label"], card["value"]) for card in bundle["teamfight_cards"]
        ],
        "objective_rows": [(row["kind"], row) for row in bundle["objective_rows"]],
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


def _write_full_exports(
    config: AppConfig,
    records: list[MatchRecord],
    run_dir: Path,
    *,
    peer_comparison: PeerComparisonResult | None,
    ranked: RankedEntry | None,
) -> dict[str, Any]:
    """Write CSV/JSON exports from the full (all games) dataset."""
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

    stats = StatisticsEngine(matches_df, run_dir)
    corr = stats.correlation_matrix()
    win_corrs = stats.win_correlations()
    model = stats.train_win_predictor()

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
        "player": config.players_label,
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

    matchups_export = matchups_df.copy()
    if not matchups_export.empty:
        matchups_export["recommendation"] = matchups_export.apply(matchup_recommendation, axis=1)
    corr_export = corr.reset_index().rename(columns={"index": "feature"}) if not corr.empty else corr

    coach = CoachEngine(
        matches_df,
        deaths_df,
        matchups_df,
        obj_df,
        stats,
        build_label=config.build_label,
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
            peer_recs + recommendations, key=lambda rec: rec.priority, reverse=True
        )

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
    return summary


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

    records = sorted(records, key=lambda record: record.game_creation_ms, reverse=True)
    total_games = len(records)

    run_dir = config.report_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = config.run_graphs_dir
    graphs_dir.mkdir(parents=True, exist_ok=True)

    _write_full_exports(
        config,
        records,
        run_dir,
        peer_comparison=peer_comparison,
        ranked=ranked,
    )
    GraphFactory(graphs_dir).death_heatmap_png(deaths_dataframe(records))

    window_specs: list[tuple[str, int | None]] = [
        (str(size), size) for size in GAME_WINDOW_OPTIONS
    ]
    window_specs.append(("all", None))

    game_windows: dict[str, dict[str, Any]] = {}
    window_peers: dict[str, PeerComparisonResult | None] = {}
    for key, limit in window_specs:
        sliced = _slice_records(records, limit)
        bundle = _build_window_bundle(
            config, sliced, graphs_dir, peer_comparison=peer_comparison
        )
        window_peers[key] = bundle.pop("_peer_result", None)
        serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
        game_windows[key] = serializable

    default_window = _default_game_window_key(total_games)
    default_bundle = game_windows[default_window]
    default_peer = window_peers.get(default_window)

    context: dict[str, Any] = {
        "app_title": "Champion Stats Analyzer",
        "build_label": config.build_label,
        "champion": config.champion,
        "role_display": config.role_display,
        "player_name": config.players_label,
        "recommendation_visible_count": VISIBLE_RECOMMENDATIONS,
        "game_window_default": default_window,
        "game_window_total": total_games,
        "game_window_options": _game_window_options(total_games),
        "game_windows_json": _serialize_game_windows_json(game_windows),
    }
    context.update(
        _bundle_to_template_context(
            default_bundle,
            peer_comparison=default_peer,
        )
    )
    if player_builds:
        context["player_builds"] = build_player_builds_nav(
            player_builds,
            current_champion=config.champion,
            current_role=config.role,
        )

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
    )
    if player_hub is not None:
        log.info("Done. Open %s (player hub: %s, index: %s)", report_path, player_hub, global_index)
    else:
        log.info("Done. Open %s (index: %s)", report_path, global_index)
    return report_path


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


def run_all_builds(
    services: Services,
    player_contexts: list[PlayerContext],
    *,
    fetch: bool = False,
) -> Path:
    """Discover, parse once and analyse every eligible champion+lane build."""
    log = get_logger("pipeline")
    if fetch:
        _fetch(services)
        player_contexts = _resolve_player_contexts(services)

    puuids = [context.puuid for context in player_contexts]
    primary_puuid = player_contexts[0].puuid

    pools = discover_build_pools(
        services.store,
        puuids,
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

    all_records = _load_all_records(services, puuids)
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

    player_label = services.config.players_label
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
            ranked = services.client.fetch_solo_rank(primary_puuid)
        build_config = services.config.model_copy(
            update={"champion": pool.champion, "role": pool.role}
        )
        build_config.report_dir.mkdir(parents=True, exist_ok=True)
        build_config.run_graphs_dir.mkdir(parents=True, exist_ok=True)
        last_report = _run_with_peer(
            build_config,
            services,
            primary_puuid,
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
    player: list[str] = typer.Option(
        [],
        "--player",
        help='Riot ID as "Name#Tag". Repeat to pool multiple players into one report.',
    ),
    riot_id: str = typer.Option(None, "--riot-id", help="Riot ID game name (e.g. 'Faker')."),
    tagline: str = typer.Option(None, "--tagline", help="Riot ID tagline without '#'."),
    region: str = typer.Option("europe", "--region", help="Routing region (europe/americas/asia/sea) or platform (euw1, na1...). Default: europe."),
    platform: str = typer.Option(None, "--platform", help="Platform for league-v4 rank lookup (euw1, eun1, na1...). Auto-detected from match ids when omitted."),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY", help="Riot API key."),
    count: int = typer.Option(None, "--count", help="Max matches to scan (default 500)."),
    min_games: int = typer.Option(None, "--min-games", help="Min solo/duo games per champion+lane (default 20)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Run the full pipeline: download, analyse all eligible builds and generate reports."""
    players = _parse_players_cli(player, riot_id, tagline)
    services = _build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose, players=players
    )
    try:
        contexts = _fetch(services)
        run_all_builds(services, contexts, fetch=False)
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def fetch(
    player: list[str] = typer.Option([], "--player", help='Riot ID as "Name#Tag". Repeat for multiple players.'),
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option("europe", "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    count: int = typer.Option(None, "--count"),
    min_games: int = typer.Option(None, "--min-games"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download matches into the local store without analysing them."""
    players = _parse_players_cli(player, riot_id, tagline)
    services = _build_services(
        riot_id, tagline, region, platform, api_key, count, min_games, verbose, players=players
    )
    try:
        _fetch(services)
        get_logger().info("Store now holds %d complete matches.", services.store.count())
    finally:
        services.store.close()
        services.http_cache.close()


@app.command()
def report(
    player: list[str] = typer.Option([], "--player", help='Riot ID as "Name#Tag". Repeat for multiple players.'),
    riot_id: str = typer.Option(None, "--riot-id"),
    tagline: str = typer.Option(None, "--tagline"),
    region: str = typer.Option("europe", "--region"),
    platform: str = typer.Option(None, "--platform"),
    api_key: str = typer.Option(None, "--api-key", envvar="RIOT_API_KEY"),
    min_games: int = typer.Option(None, "--min-games"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild all eligible build reports from already-downloaded matches."""
    players = _parse_players_cli(player, riot_id, tagline)
    services = _build_services(
        riot_id, tagline, region, platform, api_key, None, min_games, verbose, players=players
    )
    try:
        contexts = _resolve_player_contexts(services)
        run_all_builds(services, contexts, fetch=False)
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
