"""Tests for death contextualisation."""

from __future__ import annotations

import pytest

from models import MatchRecord, Zone
from parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


@pytest.fixture()
def record() -> MatchRecord:
    """A parsed record of the synthetic game."""
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    return parser.parse(make_match(), make_timeline(), MY_PUUID)


def test_two_deaths_extracted(record: MatchRecord) -> None:
    """Both synthetic deaths are captured."""
    assert len(record.deaths) == 2


def test_first_death_context(record: MatchRecord) -> None:
    """The 12-minute jungle death is solo and 60 s before a dragon."""
    death = record.deaths[0]
    assert death.minute == pytest.approx(12.0, abs=0.1)
    assert death.zone == Zone.JUNGLE
    assert death.alone is True
    assert death.before_dragon is True
    assert death.before_baron is False
    assert death.killer_champion == "Vi"
    assert death.ult_available is True  # R skilled at 11 min
    assert death.flash_available is None  # documented API limitation


def test_second_death_context(record: MatchRecord) -> None:
    """The late bot-lane death is a side-lane push death after an objective."""
    death = record.deaths[1]
    assert death.zone == Zone.BOT_LANE
    assert death.side_lane_push is True
    assert death.after_objective is True  # baron 10 s earlier
    assert death.bounty_held is True  # 450 g bounty
    assert death.zhonya_available is True  # bought at 15 min


def test_deaths_dataframe_shape(record: MatchRecord) -> None:
    """The flattened death table carries match context."""
    from analysis.deaths import deaths_dataframe

    frame = deaths_dataframe([record])
    assert len(frame) == 2
    assert set(["match_id", "win", "zone", "minute", "alone"]).issubset(frame.columns)
