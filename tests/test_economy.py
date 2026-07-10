"""Tests for economy thresholds and recall-gold severity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.coach import CoachEngine
from analysis.economy import recall_gold_severity
from analysis.statistics import StatisticsEngine


@pytest.mark.parametrize(
    ("avg", "expected"),
    [
        (947, None),
        (1200, None),
        (1299, None),
        (1300, 0.0),
        (1500, pytest.approx(0.286, abs=0.01)),
        (2000, 1.0),
        (2500, 1.0),
    ],
)
def test_recall_gold_severity(avg: float, expected: float | None) -> None:
    """Severity is None for healthy backs and scales from component max upward."""
    assert recall_gold_severity(avg) == expected


def test_unspent_gold_rule_skips_healthy_average(tmp_path: Path) -> None:
    """947g average should not trigger the hoarding recommendation."""
    matches = pd.DataFrame(
        {
            "match_id": [f"M{i}" for i in range(10)],
            "win": [1] * 10,
            "avg_unspent_gold": [947.0] * 10,
        }
    )
    stats = StatisticsEngine(matches, tmp_path)
    coach = CoachEngine(matches, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats)
    titles = [r.title for r in coach.generate()]
    assert "Too much gold sitting unspent" not in titles


def test_unspent_gold_rule_fires_for_hoarding(tmp_path: Path) -> None:
    """1600g average should trigger a hoarding recommendation."""
    matches = pd.DataFrame(
        {
            "match_id": [f"M{i}" for i in range(10)],
            "win": [1] * 10,
            "avg_unspent_gold": [1600.0] * 10,
        }
    )
    stats = StatisticsEngine(matches, tmp_path)
    coach = CoachEngine(matches, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats)
    rec = next(r for r in coach.generate() if r.title == "Too much gold sitting unspent")
    assert rec.effect_size == pytest.approx(0.429, abs=0.01)
    assert "1300" in rec.evidence
