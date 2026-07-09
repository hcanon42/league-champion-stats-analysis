"""End-to-end smoke test: parsed records -> exports + HTML report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.peer_comparison import build_comparisons
from config import AppConfig
from main import run_analysis
from models import MatchRecord, PeerComparisonResult, RankedEntry
from parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline

OPPONENTS = ["Syndra", "Orianna", "Akali", "Ahri", "Zed"]


def _make_records(n: int = 15) -> list[MatchRecord]:
    """Parse the fixture once and derive varied copies.

    Args:
        n: Number of records to produce.

    Returns:
        Records with varied ids, results and opponents.
    """
    base = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(make_match(), make_timeline(), MY_PUUID)
    records: list[MatchRecord] = []
    for index in range(n):
        records.append(
            base.model_copy(
                deep=True,
                update={
                    "match_id": f"EUW1_{index}",
                    "win": index % 3 != 0,
                    "lane_opponent": OPPONENTS[index % len(OPPONENTS)],
                    "game_creation_ms": 1_700_000_000_000 + index * 3_600_000,
                },
            )
        )
    return records


def test_full_pipeline_generates_all_artifacts(tmp_path: Path) -> None:
    """The full analysis produces the report, every CSV and the summary."""
    config = AppConfig(
        riot_id="Test",
        tagline="EUW",
        region="europe",
        api_key="RGAPI-test",
        output_dir=tmp_path / "output",
        graphs_dir=tmp_path / "graphs",
        cache_dir=tmp_path / "cache",
        template_dir=Path(__file__).resolve().parent.parent / "templates",
    )
    config.ensure_directories()
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)
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
    peer = PeerComparisonResult(
        rank_label=ranked.label,
        tier=ranked.tier,
        source="test benchmark",
        peer_games=0,
        peer_players=0,
        comparisons=build_comparisons(
            pd.DataFrame([r.to_row() for r in _make_records()]).mean(numeric_only=True).to_dict(),
            peer_metrics,
        ),
    )
    report_path = run_analysis(
        config, _make_records(), peer_comparison=peer, ranked=ranked
    )

    assert report_path.exists()
    assert report_path == config.report_dir / "report.html"
    html = report_path.read_text(encoding="utf-8")
    assert "Improvement score" in html and "Recommendations" in html
    assert "Rank peer comparison" in html
    assert "Your champions" in html or "All players" in html

    expected = [
        "summary.json", "matches.csv", "deaths.csv", "timeline.csv", "matchups.csv",
        "vision.csv", "items.csv", "runes.csv", "objectives.csv", "teamfights.csv",
        "correlations.csv", "recommendations.md", "rank_comparison.csv", "meta.json",
    ]
    for name in expected:
        assert (config.report_dir / name).exists(), f"missing export: {name}"
    assert (config.run_graphs_dir / "death_heatmap.png").exists()

    index_path = config.output_dir / "index.html"
    assert index_path.exists()
    assert "All players" in html or "Test#EUW" in html
