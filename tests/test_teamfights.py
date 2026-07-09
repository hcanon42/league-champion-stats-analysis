"""Tests for teamfight detection."""

from __future__ import annotations

import pytest

from analysis.teamfights import detect_teamfights
from analysis.timeline import build_context
from tests.fixtures import MY_PUUID, make_match, make_timeline


def test_detects_single_teamfight() -> None:
    """Only the 16-minute 4-kill cluster qualifies as a teamfight."""
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    fights = detect_teamfights(ctx)
    assert len(fights) == 1


def test_teamfight_involvement() -> None:
    """The player's kills, assists and the fight result are correct."""
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    fight = detect_teamfights(ctx)[0]
    assert fight.participated is True
    assert fight.kills == 2
    assert fight.assists == 1
    assert fight.died is False
    assert fight.ally_kills == 3
    assert fight.enemy_kills == 1
    assert fight.won is True
    assert fight.damage_dealt == 1500
    assert fight.start_minute == pytest.approx(16.0, abs=0.1)


def test_isolated_kills_are_not_fights() -> None:
    """Single kills (min 6 and min 12) never form a fight cluster."""
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    fights = detect_teamfights(ctx)
    assert all(f.start_minute > 15 for f in fights)
