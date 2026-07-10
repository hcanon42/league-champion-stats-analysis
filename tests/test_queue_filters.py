"""Tests for the report queue filter toggle (solo / flex / all)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from league_stats.core.config import DEFAULT_QUEUE_FILTER, RANKED_FLEX_QUEUE_ID
from league_stats.cli.app import (
    _default_queue_filter_key,
    _filter_records_by_queue,
    _queue_filter_options,
    run_analysis,
)
from league_stats.core.models import MatchRecord, RankedEntry
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from tests.test_game_windows import _make_records
from tests.test_reports import _config, _peer


def _make_flex_records(n: int) -> list[MatchRecord]:
    base = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(
        make_match(queue_id=RANKED_FLEX_QUEUE_ID), make_timeline(), MY_PUUID
    )
    records: list[MatchRecord] = []
    for index in range(n):
        records.append(
            base.model_copy(
                deep=True,
                update={
                    "match_id": f"EUW1_flex_{index}",
                    "game_creation_ms": 1_700_000_000_000 + index * 3_600_000,
                },
            )
        )
    return sorted(records, key=lambda record: record.game_creation_ms, reverse=True)


def test_filter_records_by_queue_splits_solo_and_flex() -> None:
    """Queue helpers partition records by queue id."""
    solo = _make_records(3)
    flex = _make_flex_records(2)
    records = solo + flex
    assert len(_filter_records_by_queue(records, "solo")) == 3
    assert len(_filter_records_by_queue(records, "flex")) == 2
    assert len(_filter_records_by_queue(records, "all")) == 5


def test_default_queue_filter_key_prefers_solo() -> None:
    """Solo is the default when both queues have games."""
    assert _default_queue_filter_key(10, 5) == DEFAULT_QUEUE_FILTER
    assert _default_queue_filter_key(0, 5) == "flex"
    assert _default_queue_filter_key(0, 0) == "all"


def test_queue_filter_options_disable_empty_queues() -> None:
    """Toggle metadata disables queues with no games."""
    options = _queue_filter_options(10, 0)
    by_key = {option["key"]: option for option in options}
    assert by_key["solo"]["enabled"] is True
    assert by_key["flex"]["enabled"] is False
    assert by_key["all"]["enabled"] is True


def test_report_contains_queue_filter_toggle(tmp_path: Path) -> None:
    """Generated HTML embeds the queue toggle and nested view snapshots."""
    config = _config(tmp_path)
    records = _make_records(25)
    peer = _peer(records)
    ranked = RankedEntry(tier="GOLD", rank="II", league_points=45, wins=80, losses=75)

    run_analysis(config, records, peer_comparison=peer, ranked=ranked)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    assert 'id="queue-filter-bar"' in html
    assert 'id="report-views-data"' in html
    assert "Solo/Duo" in html
    assert "Flex" in html
    assert 'data-queue="all"' in html

    match = re.search(
        r'<script type="application/json" id="report-views-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    views = json.loads(match.group(1))
    assert set(views) == {"solo", "flex", "all"}
    assert set(views["solo"]["windows"]) == {"50", "100", "all"}
    assert views["solo"]["windows"]["all"]["total_games"] == 25
    assert views["flex"]["windows"]["all"]["total_games"] == 0
    assert re.search(
        r'class="queue-filter-btn game-window-btn is-active"[^>]*data-queue="solo"',
        html.replace("\n", " "),
    )


def test_flex_records_appear_in_flex_view(tmp_path: Path) -> None:
    """Flex-only games populate the flex queue bundles."""
    config = _config(tmp_path)
    records = _make_flex_records(12)
    peer = _peer(records)

    run_analysis(config, records, peer_comparison=peer, ranked=None)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-views-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    views = json.loads(match.group(1))
    assert views["flex"]["windows"]["all"]["total_games"] == 12
    assert views["solo"]["windows"]["all"]["total_games"] == 0
    assert views["flex"]["windows"]["all"].get("peer") is None
    assert re.search(
        r'class="queue-filter-btn game-window-btn is-active"[^>]*data-queue="flex"',
        html.replace("\n", " "),
    )


def test_mixed_queues_have_different_window_options(tmp_path: Path) -> None:
    """Game-window enablement is computed per queue."""
    config = _config(tmp_path)
    records = _make_records(60) + _make_flex_records(25)
    peer = _peer(records)

    run_analysis(config, records, peer_comparison=peer, ranked=None)

    html = (config.report_dir / "report.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="report-views-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    views = json.loads(match.group(1))
    solo_options = {opt["key"]: opt["enabled"] for opt in views["solo"]["window_options"]}
    flex_options = {opt["key"]: opt["enabled"] for opt in views["flex"]["window_options"]}
    assert solo_options["50"] is True
    assert solo_options["100"] is False
    assert flex_options["50"] is False
    assert flex_options["all"] is True
