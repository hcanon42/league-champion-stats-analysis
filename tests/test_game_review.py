"""Tests for Game Review per-match deep dive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from league_stats.analysis.game_review.behaviors import evaluate_behaviors
from league_stats.analysis.game_review.compare import compare_to_baseline
from league_stats.analysis.game_review.export import game_review_chatbot_export
from league_stats.analysis.game_review.score import compute_game_score
from league_stats.analysis.game_review.views import build_game_review_views
from league_stats.analysis.progression.slicing import slice_recent
from league_stats.analysis.statistics import StatisticsEngine
from league_stats.core.config import (
    GAME_REVIEW_MAX_BEHAVIORS,
    GAME_REVIEW_MAX_COMPARISONS,
    GAME_REVIEW_RECENT_N,
    RANKED_FLEX_QUEUE_ID,
    RANKED_SOLO_QUEUE_ID,
    AppConfig,
)
from league_stats.core.models import MatchRecord
from league_stats.pipeline.bundles import filter_records_by_queue
from league_stats.pipeline.frames import build_analysis_frames
from league_stats.pipeline.game_review import build_game_review_views as pipeline_build_game_review
from league_stats.pipeline.summaries import build_export_summary, compute_report_stats
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.cli.app import run_analysis
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from tests.test_reports import _config


def _make_config(**overrides: object) -> AppConfig:
    base = {
        "riot_id": "Test",
        "tagline": "EUW",
        "api_key": "RGAPI-test-key-1234567890",
        "champion": "Viktor",
        "role": "MIDDLE",
    }
    base.update(overrides)
    return AppConfig(**base)


def _parse_record(
    *,
    match_id: str = "EUW1_1",
    win: bool = True,
    game_creation_ms: int = 1_700_000_000_000,
    queue_id: int = RANKED_SOLO_QUEUE_ID,
    gd15: float | None = None,
) -> MatchRecord:
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    match = make_match()
    match["metadata"]["matchId"] = match_id
    match["info"]["gameCreation"] = game_creation_ms
    match["info"]["queueId"] = queue_id
    match["info"]["participants"][0]["win"] = win
    if gd15 is not None:
        timeline = make_timeline()
        for frame in timeline["info"]["frames"]:
            if frame.get("timestamp") == 900_000:
                frame["participantFrames"]["1"]["totalGold"] = 5000 + gd15
                frame["participantFrames"]["6"]["totalGold"] = 5000
        return parser.parse(match, timeline, MY_PUUID)
    return parser.parse(match, make_timeline(), MY_PUUID)


def _make_records(n: int, **kwargs: object) -> list[MatchRecord]:
    records = [
        _parse_record(
            match_id=f"EUW1_{index}",
            game_creation_ms=1_700_000_000_000 + index * 3_600_000,
            win=index % 2 == 0,
            **kwargs,
        )
        for index in range(n)
    ]
    return sorted(records, key=lambda record: record.game_creation_ms, reverse=True)


def test_slice_last_five_per_queue() -> None:
    records = _make_records(12)
    records[0] = _parse_record(match_id="EUW1_flex", queue_id=RANKED_FLEX_QUEUE_ID, game_creation_ms=1_800_000_000_000)
    records = sorted(records, key=lambda record: record.game_creation_ms, reverse=True)
    solo = filter_records_by_queue(records, "solo")
    flex = filter_records_by_queue(records, "flex")
    all_q = filter_records_by_queue(records, "all")
    assert len(slice_recent(solo, GAME_REVIEW_RECENT_N)) == 5
    assert len(slice_recent(flex, GAME_REVIEW_RECENT_N)) == 1
    assert len(slice_recent(all_q, GAME_REVIEW_RECENT_N)) == 5


def test_game_score_personal_baseline() -> None:
    record = _parse_record()
    row = record.to_row()
    baseline = {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
    baseline["gd10"] = float(row["gd10"]) - 300
    baseline["deaths"] = float(row["deaths"]) + 2
    score = compute_game_score(row, baseline, role="MIDDLE")
    assert 0 <= score.overall <= 100
    assert score.tier in {"S", "A", "B", "C", "D"}
    assert score.laning >= 50
    assert score.survival >= 50


def test_game_score_fallback_small_sample() -> None:
    records = _make_records(2)
    frames = build_analysis_frames(records)
    payload = build_game_review_views(_make_config(), records, frames)
    game = payload.queues["all"].games[0]
    assert game.score.overall >= 0


def test_behaviors_throw_and_greed() -> None:
    throw = _parse_record(win=False, gd15=900)
    row = throw.to_row()
    row["gd15"] = 900
    deaths_rows = [{"minute": 18.0, "after_greed": True, "before_neutral_objective": False}]
    good, bad = evaluate_behaviors(
        throw,
        row,
        deaths_rows,
        baseline_means={"gd10": 100, "deaths": 4, "vspm": 1.5, "objectives_present_rate": 0.5},
        archetype="Throw",
    )
    titles = {item.title for item in bad}
    assert "Throw" in titles
    assert "Greed deaths" in titles
    assert len(good) <= GAME_REVIEW_MAX_BEHAVIORS
    assert len(bad) <= GAME_REVIEW_MAX_BEHAVIORS


def test_behaviors_cap_at_five() -> None:
    record = _parse_record()
    row = record.to_row()
    row.update(
        {
            "gd10": 800,
            "deaths": 1,
            "deaths_pre14": 0,
            "vspm": 3.0,
            "objectives_present_rate": 0.9,
            "solo_deaths": 0,
            "fights_disadvantaged": 0,
        }
    )
    good, bad = evaluate_behaviors(
        record,
        row,
        [],
        baseline_means={"gd10": 100, "deaths": 5, "vspm": 1.0, "objectives_present_rate": 0.4, "deaths_pre14": 2},
        archetype="Lane stomp win",
    )
    assert len(good) <= GAME_REVIEW_MAX_BEHAVIORS
    assert len(bad) <= GAME_REVIEW_MAX_BEHAVIORS


def test_comparisons_cap_at_five() -> None:
    records = _make_records(10)
    frames = build_analysis_frames(records)
    detail = build_game_review_views(_make_config(), records, frames).queues["all"].games[0]
    assert len(detail.vs_baseline) <= GAME_REVIEW_MAX_COMPARISONS


def test_compare_values_share_precision() -> None:
    record = _parse_record()
    row = record.to_row()
    baseline = {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
    baseline["gd10"] = float(row["gd10"]) + 123.456
    rows = compare_to_baseline(row, baseline, role="MIDDLE")
    gd10 = next(item for item in rows if item.metric == "gd10")
    assert gd10.game_value == round(float(row["gd10"]), 0)
    assert gd10.benchmark_value == round(baseline["gd10"], 0)
    assert gd10.delta == round(float(row["gd10"]) - baseline["gd10"], 0)


def test_death_flags_use_readable_labels() -> None:
    records = _make_records(3)
    frames = build_analysis_frames(records)
    detail = build_game_review_views(_make_config(), records, frames).queues["all"].games[0]
    for death in detail.deaths:
        for flag in death.flags:
            assert flag[0].isupper()
            assert "_" not in flag


def test_build_game_review_views_serializes() -> None:
    records = _make_records(8)
    frames = build_analysis_frames(records)
    payload = build_game_review_views(_make_config(), records, frames)
    dumped = payload.model_dump()
    assert dumped["recent_n"] == GAME_REVIEW_RECENT_N
    assert len(dumped["queues"]["all"]["games"]) == 5
    assert dumped["queues"]["all"]["games"][0]["match_id"]


def test_build_export_summary_includes_recent_games(tmp_path: Path) -> None:
    records = _make_records(8)
    frames = build_analysis_frames(records)
    stats = compute_report_stats(frames, tmp_path)
    game_review = build_game_review_views(_make_config(), records, frames)
    summary = build_export_summary(
        _make_config(),
        frames,
        {"overview": {}, "laning": {}, "economy": {}, "resets": {}, "vision": {}, "deaths": {}, "teamfights": {}, "positioning": {}, "objectives": {}, "macro": {}, "matchups": {}, "items": {}, "runes": {}, "jungle": {}, "utility": {}},
        stats,
        peer_comparison=None,
        ranked=None,
        records_count=len(records),
        game_review=game_review,
    )
    assert "recent_games" in summary
    assert summary["recent_games"]["n"] == GAME_REVIEW_RECENT_N
    assert len(summary["recent_games"]["games"]) == 5
    assert "events_summary" in summary["recent_games"]["games"][0]


def test_game_review_available_with_one_game() -> None:
    records = _make_records(1)
    frames = build_analysis_frames(records)
    bundle = build_game_review_views(_make_config(), records, frames).queues["all"]
    assert bundle.available is True
    assert bundle.games_count == 1
    assert bundle.games[0].index == 1


def test_chatbot_export_omits_full_timeline() -> None:
    records = _make_records(3)
    frames = build_analysis_frames(records)
    payload = pipeline_build_game_review(_make_config(), records, frames, graphs_dir=Path("."))
    exported = game_review_chatbot_export(payload, queue_key="all")
    game = exported["games"][0]
    assert "timeline" not in game
    assert "highlights" in game
    assert len(game["events_summary"]["notable_deaths"]) <= 3


def test_pipeline_enriches_game_review_icons(tmp_path: Path) -> None:
    config = _make_config(output_dir=tmp_path / "output", cache_dir=tmp_path / ".cache")
    records = _make_records(3)
    frames = build_analysis_frames(records)
    assets = DDragonAssets(config)
    assets._champions_dir.mkdir(parents=True)
    assets._runes_dir.mkdir(parents=True)
    assets._items_dir.mkdir(parents=True)
    assets._summoners_dir.mkdir(parents=True)
    assets._rune_trees_dir.mkdir(parents=True)
    (assets._champions_dir / "Viktor.png").write_bytes(b"png")
    (assets._champions_dir / "Syndra.png").write_bytes(b"png")
    (assets._runes_dir / "8229.png").write_bytes(b"png")
    (assets._summoners_dir / "Flash.png").write_bytes(b"png")
    (assets._summoners_dir / "Teleport.png").write_bytes(b"png")
    (assets._rune_trees_dir / "Sorcery.png").write_bytes(b"png")
    (assets._rune_trees_dir / "Inspiration.png").write_bytes(b"png")
    (assets._items_dir / "6655.png").write_bytes(b"png")
    (assets._items_dir / "3157.png").write_bytes(b"png")
    (assets._items_dir / "3020.png").write_bytes(b"png")
    assets._item_name_to_id = {
        "Luden's Companion": 6655,
        "Zhonya's Hourglass": 3157,
        "Sorcerer's Shoes": 3020,
    }
    assets._map_dir.mkdir(parents=True)
    (assets._map_dir / "summoners_rift.png").write_bytes(b"png-map")

    report_dir = config.report_dir
    report_dir.mkdir(parents=True)
    payload = pipeline_build_game_review(
        config,
        records,
        frames,
        assets=assets,
        from_dir=report_dir,
    )
    game = payload.queues["all"].games[0]
    assert game.champion_icon is not None
    assert game.opponent_icon is not None
    assert game.build.keystone_icon is not None
    assert game.build.primary_tree_icon is not None
    assert game.build.secondary_tree_icon is not None
    assert any(icon is not None for icon in game.build.summoner_icons)
    assert any(icon is not None for icon in game.build.item_icons)
    assert isinstance(game.key_moments, list)
    if game.key_moments:
        participant = game.key_moments[0].frames[0].participants[0]
        assert participant.champion_icon is not None
    assert game.map_background is not None


def test_report_has_game_review_category_tab(tmp_path: Path) -> None:
    config = _config(tmp_path)
    records = _make_records(5)
    run_analysis(config, records, peer_comparison=None, ranked=None)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    assert 'id="category-games"' in html
    assert 'data-category="games"' in html
    assert 'id="game-review-data"' in html
    assert 'id="game-review-matchup"' in html
    assert 'data-tab="key-moments"' in html
    assert 'nav-group--games' in html
