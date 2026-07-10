"""Tests for match filtering and MatchRecord assembly."""

from __future__ import annotations

import pytest

from config import AppConfig
from models import MatchRecord
from parser import BaseMatchFilter, BuildMatchFilter, ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


@pytest.fixture()
def config() -> AppConfig:
    """A valid configuration for tests."""
    return AppConfig(riot_id="Test", tagline="EUW", region="euw1", api_key="RGAPI-test")


@pytest.fixture()
def record() -> MatchRecord:
    """A parsed record of the synthetic 20-minute game."""
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    return parser.parse(make_match(), make_timeline(), MY_PUUID)


def test_filter_accepts_viktor_mid(config: AppConfig) -> None:
    """The synthetic game qualifies for the configured build."""
    assert BuildMatchFilter(config).accept(make_match(), MY_PUUID)


def test_base_filter_accepts_any_champion(config: AppConfig) -> None:
    """Base filter accepts solo queue games regardless of champion/lane."""
    match = make_match()
    match["info"]["participants"][0]["championName"] = "Orianna"
    match["info"]["participants"][0]["teamPosition"] = "TOP"
    assert BaseMatchFilter(config).accept(match, MY_PUUID)
    assert not BuildMatchFilter(config).accept(match, MY_PUUID)


def test_filter_rejects_wrong_queue(config: AppConfig) -> None:
    """Normal draft games are rejected."""
    assert not BaseMatchFilter(config).accept(make_match(queue_id=400), MY_PUUID)


def test_filter_accepts_flex_queue(config: AppConfig) -> None:
    """Ranked flex queue games are accepted."""
    assert BaseMatchFilter(config).accept(make_match(queue_id=440), MY_PUUID)


def test_filter_rejects_remake(config: AppConfig) -> None:
    """Games at or under five minutes are remakes."""
    assert not BaseMatchFilter(config).accept(make_match(duration_s=240), MY_PUUID)


def test_filter_rejects_remake_flag(config: AppConfig) -> None:
    """Riot's misnamed early-surrender flag marks remakes."""
    match = make_match(duration_s=240, game_ended_in_early_surrender=True)
    assert not BaseMatchFilter(config).accept(match, MY_PUUID)


def test_filter_rejects_pre15_surrender(config: AppConfig) -> None:
    """Surrenders before the 15-minute vote are excluded."""
    match = make_match(duration_s=828, game_ended_in_surrender=True)
    assert not BaseMatchFilter(config).accept(match, MY_PUUID)


def test_filter_accepts_15_min_surrender(config: AppConfig) -> None:
    """15-minute surrender votes count as real games."""
    match = make_match(duration_s=900, game_ended_in_surrender=True)
    assert BaseMatchFilter(config).accept(match, MY_PUUID)


def test_filter_rejects_wrong_lane(config: AppConfig) -> None:
    """Build filter rejects games on the champion in another lane."""
    match = make_match()
    match["info"]["participants"][0]["teamPosition"] = "TOP"
    assert not BuildMatchFilter(config).accept(match, MY_PUUID)


def test_config_normalizes_lane_alias() -> None:
    """Lane aliases in config are normalised to Riot values."""
    cfg = AppConfig(riot_id="Test", tagline="EUW", region="euw1", api_key="RGAPI-test", role="mid")
    assert cfg.role == "MIDDLE"
    assert cfg.build_label == "Viktor mid"


def test_filter_rejects_other_champion(config: AppConfig) -> None:
    """Build filter rejects games on another champion."""
    match = make_match()
    match["info"]["participants"][0]["championName"] = "Orianna"
    assert not BuildMatchFilter(config).accept(match, MY_PUUID)


def test_parse_core_fields(record: MatchRecord) -> None:
    """Identity, result and opponent are read correctly."""
    assert record.match_id == "EUW1_9999"
    assert record.patch == "14.23"
    assert record.queue_id == 420
    assert record.champion == "Viktor"
    assert record.role == "MIDDLE"
    assert record.win is True
    assert record.side.value == "blue"
    assert record.lane_opponent == "Syndra"
    assert record.combat.kills == 7 and record.combat.deaths == 2


def test_parse_snapshots_and_diffs(record: MatchRecord) -> None:
    """Checkpoint gold and differentials match the synthetic frames."""
    snap = record.timeline.snapshots
    assert snap.gold[10] == 500 + 10 * 420
    assert snap.gold_diff[10] == 10 * 60
    assert snap.cs_diff[10] == 10


def test_parse_build_timings(record: MatchRecord) -> None:
    """First/second item and boots timings come from the purchase timeline."""
    timings = record.timings
    assert timings.first_item == "Luden's Companion"
    assert timings.first_item_min == pytest.approx(9.03, abs=0.1)
    assert timings.second_item == "Zhonya's Hourglass"
    assert timings.boots_min == pytest.approx(5.0, abs=0.1)
    assert timings.boots == "Sorcerer's Shoes"
    assert timings.elixirs_bought == 1
    assert timings.trinket_swaps == 1


def test_parse_skill_order(record: MatchRecord) -> None:
    """Q is maxed first in the synthetic skill sequence."""
    assert record.skill_order.startswith("Q")
    assert record.skill_sequence[0] == "Q"


def test_parse_runes_and_summoners(record: MatchRecord) -> None:
    """Rune page and summoners resolve to names."""
    assert record.runes.keystone == "Arcane Comet"
    assert record.runes.primary_tree == "Sorcery"
    assert record.runes.secondary_tree == "Inspiration"
    assert record.summoners == ["Flash", "Teleport"]


def test_parse_vision_lifetime(record: MatchRecord) -> None:
    """The control ward placed at 400 s and killed at 500 s lived 100 s."""
    assert record.vision.avg_control_ward_lifetime_s == pytest.approx(100.0, abs=1.0)


def test_shutdown_gold(record: MatchRecord) -> None:
    """Shutdown gold collected is summed from kill events."""
    assert record.shutdown_gold_collected == 150


def test_to_row_is_flat(record: MatchRecord) -> None:
    """to_row produces scalars only (no lists/dicts)."""
    row = record.to_row()
    assert row["win"] == 1
    assert all(not isinstance(v, (list, dict)) for v in row.values())
