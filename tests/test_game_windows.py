"""Tests for the report game-window toggle."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from league_stats.analysis.peer import build_comparisons, peer_comparison_for_window
from league_stats.core.config import DEFAULT_GAME_WINDOW
from league_stats.cli.app import _default_game_window_key, run_analysis
from league_stats.core.models import MatchRecord, PeerComparisonResult, RankedEntry
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from tests.test_reports import _config, _peer
from league_stats.ingest.parser import ItemCatalog, MatchParser


def _make_records(n: int, *, recent_wins: bool = False) -> list[MatchRecord]:
    base = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(make_match(), make_timeline(), MY_PUUID)
    records: list[MatchRecord] = []
    for index in range(n):
        win = index % 2 == 0
        if recent_wins and index >= n - 10:
            win = True
        records.append(
            base.model_copy(
                deep=True,
                update={
                    "match_id": f"EUW1_{index}",
                    "win": win,
                    "game_creation_ms": 1_700_000_000_000 + index * 3_600_000,
                },
            )
        )
    return sorted(records, key=lambda record: record.game_creation_ms, reverse=True)


def test_default_game_window_key_prefers_100(tmp_path: Path) -> None:
    """Default window is 100 when enough games exist."""
    assert _default_game_window_key(120) == str(DEFAULT_GAME_WINDOW)
    assert _default_game_window_key(30) == "all"


def test_peer_comparison_for_window_updates_user_side() -> None:
    """Windowed peer comparison recomputes only the player averages."""
    records = _make_records(20)
    matches_df = pd.DataFrame([record.to_row() for record in records])
    base = _peer(records)
    windowed = peer_comparison_for_window(base, matches_df[:5], records[:5])
    assert windowed.comparisons
    assert windowed.rank_label == base.rank_label
    assert windowed.strengths != base.strengths or windowed.weaknesses != base.weaknesses


def test_report_contains_game_window_toggle(tmp_path: Path) -> None:
    """Generated HTML embeds the toggle and all window snapshots."""
    config = _config(tmp_path)
    records = _make_records(25)
    peer = _peer(records)
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)

    run_analysis(config, records, peer_comparison=peer, ranked=ranked)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    assert 'id="game-window-bar"' in html
    assert 'id="report-views-data"' in html
    assert "Last 50" in html
    assert "Last 100" in html
    assert 'data-window="all"' in html

    match = re.search(
        r'<script type="application/json" id="report-views-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    views = json.loads(match.group(1))
    solo_windows = views["solo"]["windows"]
    assert set(solo_windows) == {"50", "100", "all"}
    assert solo_windows["50"]["total_games"] == 25
    assert solo_windows["100"]["total_games"] == 25
    assert solo_windows["all"]["total_games"] == 25
    assert re.search(
        r'class="game-window-btn is-active"[^>]*data-window="all"',
        html.replace("\n", " "),
    )


def test_default_window_active_when_enough_games(tmp_path: Path) -> None:
    """Last 100 is the initial view when at least 100 games exist."""
    config = _config(tmp_path)
    records = _make_records(120)
    peer = _peer(records)

    run_analysis(config, records, peer_comparison=peer, ranked=None)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    assert re.search(
        rf'class="game-window-btn is-active"[^>]*data-window="{DEFAULT_GAME_WINDOW}"',
        html.replace("\n", " "),
    )


def test_window_snapshots_change_winrate(tmp_path: Path) -> None:
    """Recent-window stats can differ from the full-history view."""
    config = _config(tmp_path)
    records = _make_records(60, recent_wins=True)
    peer = PeerComparisonResult(
        rank_label="GOLD II",
        tier="GOLD",
        source="test benchmark",
        peer_games=0,
        peer_players=0,
        comparisons=build_comparisons(
            {"win": 0.5},
            {"win": 0.5},
        ),
    )

    run_analysis(config, records, peer_comparison=peer, ranked=None)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-views-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    views = json.loads(match.group(1))
    solo_windows = views["solo"]["windows"]
    assert solo_windows["50"]["overview"]["winrate"] > solo_windows["all"]["overview"]["winrate"]
