"""Tests that report indexes refresh as builds are written."""

from __future__ import annotations

from pathlib import Path

import pytest

from league_stats.core.config import AppConfig
from league_stats.cli.app import run_analysis
from league_stats.core.models import MatchRecord
from league_stats.presentation.report import discover_reports, discover_player_builds, refresh_report_indexes
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from league_stats.ingest.parser import ItemCatalog, MatchParser


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


def _records(n: int = 15) -> list[MatchRecord]:
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


def test_run_analysis_refreshes_global_index(tmp_path: Path) -> None:
    """Each completed report appears on the global index immediately."""
    viktor_config = _config(tmp_path, champion="Viktor", role="MIDDLE")
    run_analysis(viktor_config, _records())

    global_index = viktor_config.output_dir / "index.html"
    assert global_index.exists()
    assert "Viktor" in global_index.read_text(encoding="utf-8")
    assert len(discover_reports(viktor_config.output_dir)) == 1

    ahri_config = _config(tmp_path, champion="Ahri", role="MIDDLE")
    run_analysis(ahri_config, _records())

    html = global_index.read_text(encoding="utf-8")
    assert "Viktor" in html
    assert "Ahri" in html
    assert len(discover_reports(viktor_config.output_dir)) == 2


def test_run_analysis_refreshes_player_hub(tmp_path: Path) -> None:
    """The player hub lists every build as soon as its report is written."""
    run_analysis(_config(tmp_path, champion="Viktor"), _records())
    run_analysis(_config(tmp_path, champion="Ahri"), _records())

    player_dir = _config(tmp_path).player_reports_dir
    hub = player_dir / "index.html"
    assert hub.exists()
    hub_html = hub.read_text(encoding="utf-8")
    assert "Viktor" in hub_html
    assert "Ahri" in hub_html
    assert len(discover_player_builds(player_dir)) == 2


def test_refresh_report_indexes_rebuilds_from_disk(tmp_path: Path) -> None:
    """Manual refresh picks up reports already on disk."""
    config = _config(tmp_path, champion="Zac", role="JUNGLE")
    run_analysis(config, _records())

    global_index, hub = refresh_report_indexes(
        config.output_dir,
        config.template_dir,
        player_dir=config.player_reports_dir,
        player_label="Test#EUW",
    )
    assert global_index.exists()
    assert hub is not None
    assert hub.exists()
