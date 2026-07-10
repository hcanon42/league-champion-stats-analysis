"""Tests for multi-report storage and the report index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from league_stats.analysis.peer import build_comparisons
from league_stats.core.champions import player_slug
from league_stats.core.config import AppConfig
from league_stats.cli.app import run_analysis
from league_stats.core.models import MatchRecord, PeerComparisonResult, RankedEntry
from league_stats.presentation.report import discover_reports, group_reports_by_player, refresh_report_indexes
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from league_stats.ingest.parser import ItemCatalog, MatchParser


def _make_records(n: int = 12) -> list[MatchRecord]:
    base = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(make_match(), make_timeline(), MY_PUUID)
    return [
        base.model_copy(
            deep=True,
            update={
                "match_id": f"EUW1_{index}",
                "win": index % 2 == 0,
                "game_creation_ms": 1_700_000_000_000 + index * 3_600_000,
            },
        )
        for index in range(n)
    ]


def _peer(records: list[MatchRecord]) -> PeerComparisonResult:
    peer_metrics = {
        "win": 0.5,
        "kda": 2.4,
        "dpm": 640.0,
        "cspm": 7.0,
        "deaths": 5.0,
        "vspm": 1.0,
        "control_wards": 2.0,
        "kill_participation": 0.6,
        "damage_share": 0.2,
    }
    return PeerComparisonResult(
        rank_label="GOLD II",
        tier="GOLD",
        source="test benchmark",
        peer_games=0,
        peer_players=0,
        comparisons=build_comparisons(
            pd.DataFrame([r.to_row() for r in records]).mean(numeric_only=True).to_dict(),
            peer_metrics,
        ),
    )


def _config(tmp_path: Path, *, champion: str = "Viktor", role: str = "MIDDLE") -> AppConfig:
    config = AppConfig(
        riot_id="Test",
        tagline="EUW",
        region="europe",
        api_key="RGAPI-test",
        champion=champion,
        role=role,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        template_dir=Path(__file__).resolve().parent.parent / "src/league_stats/presentation/templates",
    )
    config.ensure_directories()
    return config


def test_different_champions_create_separate_reports(tmp_path: Path) -> None:
    """Each player/champion/lane combo gets its own directory."""
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)
    records = _make_records()
    peer = _peer(records)

    viktor_config = _config(tmp_path, champion="Viktor", role="MIDDLE")
    ahri_config = _config(tmp_path, champion="Ahri", role="MIDDLE")

    viktor_path = run_analysis(viktor_config, records, peer_comparison=peer, ranked=ranked)
    ahri_path = run_analysis(ahri_config, records, peer_comparison=peer, ranked=ranked)

    assert viktor_path != ahri_path
    assert viktor_path.parent.name == "viktor_middle"
    assert ahri_path.parent.name == "ahri_middle"
    assert viktor_path.exists() and ahri_path.exists()

    entries = discover_reports(viktor_config.output_dir)
    assert len(entries) == 2
    labels = {entry["build_label"] for entry in entries}
    assert labels == {"Viktor mid", "Ahri mid"}


def test_same_combo_overwrites_report(tmp_path: Path) -> None:
    """Re-running the same player/champion/lane replaces that report."""
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)
    config = _config(tmp_path)
    peer = _peer(_make_records(10))

    first_path = run_analysis(config, _make_records(10), peer_comparison=peer, ranked=ranked)
    first_meta = (config.report_dir / "meta.json").read_text(encoding="utf-8")

    second_path = run_analysis(config, _make_records(20), peer_comparison=peer, ranked=ranked)
    second_meta = (config.report_dir / "meta.json").read_text(encoding="utf-8")

    assert first_path == second_path
    assert '"games": 20' in second_meta
    assert '"games": 10' not in second_meta
    assert len(discover_reports(config.output_dir)) == 1


def test_player_slug_sanitizes_special_characters() -> None:
    """Riot IDs with spaces or symbols become safe directory names."""
    assert player_slug("Hide on Bush", "KR1") == "hide_on_bush_kr1"


def test_refresh_index_lists_all_reports(tmp_path: Path) -> None:
    """The index page links to every saved report."""
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)
    records = _make_records()
    peer = _peer(records)

    run_analysis(_config(tmp_path, champion="Viktor"), records, peer_comparison=peer, ranked=ranked)
    run_analysis(_config(tmp_path, champion="Ahri"), records, peer_comparison=peer, ranked=ranked)

    index_path = refresh_report_indexes(tmp_path / "output", _config(tmp_path).template_dir)[0]
    html = index_path.read_text(encoding="utf-8")
    assert "Viktor mid" in html or "Viktor" in html
    assert "Ahri" in html
    assert "reports/" in html
    assert "player-group" in html
    assert 'class="sortable"' in html


def test_group_reports_by_player(tmp_path: Path) -> None:
    """Reports are grouped by player with hub links and default build order."""
    reports = [
        {
            "player": "Beta#EUW",
            "champion": "Ahri",
            "role": "MIDDLE",
            "games": 30,
            "winrate": 0.55,
            "generated_at": "2026-01-02",
            "href": "reports/beta_euw/ahri_middle/report.html",
        },
        {
            "player": "Alpha#EUW",
            "champion": "Viktor",
            "role": "MIDDLE",
            "games": 50,
            "winrate": 0.6,
            "generated_at": "2026-01-01",
            "href": "reports/alpha_euw/viktor_middle/report.html",
        },
        {
            "player": "Beta#EUW",
            "champion": "Viktor",
            "role": "TOP",
            "games": 10,
            "winrate": 0.4,
            "generated_at": "2026-01-03",
            "href": "reports/beta_euw/viktor_top/report.html",
        },
    ]
    groups = group_reports_by_player(reports)
    assert [group["player"] for group in groups] == ["Alpha#EUW", "Beta#EUW"]
    assert groups[1]["hub_href"] == "reports/beta_euw/index.html"
    assert groups[1]["build_count"] == 2
    assert [build["champion"] for build in groups[1]["reports"]] == ["Ahri", "Viktor"]
