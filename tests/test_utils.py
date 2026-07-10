"""Tests for geometry and math utilities."""

from __future__ import annotations

from league_stats.core.models import Position, Zone
from league_stats.utils import (
    classify_zone,
    distance,
    push_progress,
    safe_div,
    wilson_lower_bound,
)


def test_classify_zone_base() -> None:
    """Fountain corners classify as base."""
    assert classify_zone(Position(x=500, y=500)) == Zone.BASE
    assert classify_zone(Position(x=14300, y=14300)) == Zone.BASE


def test_classify_zone_mid_river_jungle() -> None:
    """Mid diagonal, river anti-diagonal and jungle pockets are separated."""
    assert classify_zone(Position(x=7400, y=7400)) == Zone.MID_LANE
    assert classify_zone(Position(x=5000, y=9800)) == Zone.RIVER
    assert classify_zone(Position(x=11000, y=6200)) == Zone.JUNGLE


def test_classify_zone_side_lanes() -> None:
    """Map edges classify as top and bot lanes."""
    assert classify_zone(Position(x=900, y=8000)) == Zone.TOP_LANE
    assert classify_zone(Position(x=8000, y=14000)) == Zone.TOP_LANE
    assert classify_zone(Position(x=8000, y=900)) == Zone.BOT_LANE
    assert classify_zone(Position(x=14000, y=8000)) == Zone.BOT_LANE


def test_push_progress_sides() -> None:
    """Progress is positive past the centre toward the enemy base."""
    forward = Position(x=9000, y=9000)
    assert push_progress(forward, blue_side=True) > 0
    assert push_progress(forward, blue_side=False) < 0


def test_distance() -> None:
    """Euclidean distance is exact on a 3-4-5 triangle."""
    assert distance(Position(x=0, y=0), Position(x=3, y=4)) == 5.0


def test_safe_div() -> None:
    """Division by zero yields the default."""
    assert safe_div(10, 2) == 5
    assert safe_div(1, 0, default=-1.0) == -1.0


def test_wilson_lower_bound_ordering() -> None:
    """More games at the same rate give a higher lower bound."""
    few = wilson_lower_bound(2, 3)
    many = wilson_lower_bound(20, 30)
    assert many > few
    assert wilson_lower_bound(0, 0) == 0.0
