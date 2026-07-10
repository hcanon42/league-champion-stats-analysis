"""Tests for death contextualisation."""

from __future__ import annotations

import pytest

from league_stats.core.models import MatchRecord, Zone
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


@pytest.fixture()
def record() -> MatchRecord:
    """A parsed record of the synthetic game."""
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    return parser.parse(make_match(), make_timeline(), MY_PUUID)


def test_three_deaths_extracted(record: MatchRecord) -> None:
    """All synthetic deaths are captured."""
    assert len(record.deaths) == 3


def test_gank_death_under_own_tower_laning(record: MatchRecord) -> None:
    """The 8-minute dive is flagged as a gank and death under own tower."""
    death = next(d for d in record.deaths if d.minute == pytest.approx(8.0, abs=0.1))
    assert death.to_gank is True
    assert death.under_own_tower_laning is True
    assert death.under_enemy_tower_laning is False
    assert death.killer_champion == "Syndra"


def test_first_death_context(record: MatchRecord) -> None:
    """The 12-minute jungle death is solo and 60 s before a dragon, not a lane gank."""
    death = next(d for d in record.deaths if d.minute == pytest.approx(12.0, abs=0.1))
    assert death.zone == Zone.JUNGLE
    assert death.alone is True
    assert death.to_gank is False
    assert death.before_dragon is True
    assert death.before_baron is False
    assert death.killer_champion == "Vi"
    assert death.ult_available is True  # R skilled at 11 min
    assert death.flash_available is None  # documented API limitation


def test_second_death_context(record: MatchRecord) -> None:
    """The late bot-lane death is a side-lane push death after an objective."""
    death = next(d for d in record.deaths if d.minute == pytest.approx(19.17, abs=0.2))
    assert death.zone == Zone.BOT_LANE
    assert death.side_lane_push is True
    assert death.to_gank is False  # after laning phase
    assert death.under_own_tower_laning is False
    assert death.under_enemy_tower_laning is False
    assert death.after_objective is True  # baron 10 s earlier
    assert death.bounty_held is True  # 450 g bounty
    assert death.zhonya_available is True  # bought at 15 min


def test_deaths_dataframe_shape(record: MatchRecord) -> None:
    """The flattened death table carries match context."""
    from league_stats.analysis.deaths import deaths_dataframe

    frame = deaths_dataframe([record])
    assert len(frame) == 3
    assert set(
        ["match_id", "win", "zone", "minute", "alone", "ult_available", "to_gank",
         "under_own_tower_laning", "under_enemy_tower_laning"]
    ).issubset(frame.columns)
