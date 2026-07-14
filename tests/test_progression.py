"""Tests for Form Tracker progression analysis."""

from __future__ import annotations

import pandas as pd

from league_stats.analysis.progression.diff import build_progression_comparison
from league_stats.analysis.progression.form_score import compute_form_score, trend_from_score
from league_stats.analysis.progression.metrics import progression_metrics_for_role
from league_stats.analysis.progression.slicing import (
    slice_baseline_exclusive,
    slice_baseline_inclusive,
    slice_recent,
)
from league_stats.analysis.progression.stats import proportion_test, welch_test, winrate_significant
from league_stats.core.config import AppConfig
from league_stats.core.models import MatchRecord, MetricDelta
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from league_stats.ingest.parser import ItemCatalog, MatchParser


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


def _parse_records(count: int, *, win: bool = True) -> list[MatchRecord]:
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    records: list[MatchRecord] = []
    for index in range(count):
        match = make_match()
        match["metadata"]["matchId"] = f"EUW1_{10000 + index}"
        match["info"]["gameCreation"] = 1_700_000_000_000 - index * 86_400_000
        me = match["info"]["participants"][0]
        me["win"] = win if index % 2 == 0 else not win
        me["deaths"] = 2 + (index % 3)
        timeline = make_timeline()
        records.append(parser.parse(match, timeline, MY_PUUID))
    return records


def test_slice_baseline_exclusive_no_overlap() -> None:
    records = _parse_records(100)
    recent = slice_recent(records, 20)
    baseline = slice_baseline_exclusive(records, 20, 80)
    assert len(recent) == 20
    assert len(baseline) == 80
    recent_ids = {record.match_id for record in recent}
    baseline_ids = {record.match_id for record in baseline}
    assert recent_ids.isdisjoint(baseline_ids)


def test_slice_baseline_inclusive_overlaps() -> None:
    records = _parse_records(50)
    recent = slice_recent(records, 10)
    baseline = slice_baseline_inclusive(records, 30)
    assert len(recent) == 10
    assert len(baseline) == 30
    assert recent[0].match_id == baseline[0].match_id


def test_winrate_significance_known_proportions() -> None:
    significant, p_value, effect = winrate_significant(15, 20, 5, 20)
    assert p_value is not None
    assert effect is not None
    assert effect > 0
    assert significant or p_value < 0.2


def test_welch_test_small_sample_returns_none() -> None:
    p_value, cohen_d = welch_test(pd.Series([1.0, 2.0]), pd.Series([3.0, 4.0]))
    assert p_value is None
    assert cohen_d is None


def test_form_score_deaths_down_is_positive() -> None:
    deltas = [
        MetricDelta(
            metric="deaths",
            label="Deaths/game",
            section="overview",
            recent=2.0,
            baseline=4.0,
            delta=-2.0,
            delta_pct=-50.0,
            direction="lower",
            verdict="improved",
            significant=True,
            recent_n=20,
            baseline_n=80,
        )
    ]
    score = compute_form_score(deltas, role="MIDDLE")
    assert score > 0
    assert trend_from_score(score) in {"improving", "stable"}


def test_insufficient_sample_returns_insufficient_confidence() -> None:
    config = _make_config()
    recent = _parse_records(3)
    baseline = _parse_records(10)
    comparison = build_progression_comparison(
        config,
        recent,
        baseline,
        preset_key="20_80",
    )
    assert comparison is not None
    assert comparison.snapshot.confidence == "insufficient"
    assert not comparison.deltas


def test_progression_metrics_role_aware() -> None:
    jungle = {spec.metric for spec in progression_metrics_for_role("JUNGLE")}
    support = {spec.metric for spec in progression_metrics_for_role("UTILITY")}
    assert "kill_participation" in jungle
    assert "solo_death_rate" in support or "greed_death_rate" in support


def test_progression_comparison_schema_roundtrip() -> None:
    config = _make_config()
    recent = _parse_records(20)
    baseline = _parse_records(80)
    comparison = build_progression_comparison(
        config,
        recent,
        baseline,
        preset_key="20_80",
    )
    assert comparison is not None
    payload = comparison.model_dump()
    from league_stats.core.models import ProgressionComparison

    restored = ProgressionComparison.model_validate(payload)
    assert restored.preset_key == "20_80"
    assert restored.snapshot.recent_games == 20
