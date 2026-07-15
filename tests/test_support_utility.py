"""Tests for support utility composite scoring."""

from __future__ import annotations

import pandas as pd

from league_stats.analysis.improvement import support_utility_impact
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


def _parse_support_row(**participant_overrides) -> dict:
    match = make_match()
    me = match["info"]["participants"][0]
    me["teamPosition"] = "UTILITY"
    me["championName"] = "Thresh"
    me.update(participant_overrides)
    record = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(match, make_timeline(), MY_PUUID)
    return record.to_row()


def test_healing_counts_allies_only() -> None:
    row = _parse_support_row(totalHeal=5000, totalHealsOnTeammates=1200)
    assert row["healing"] == 1200


def test_support_utility_omits_low_shielding_noise() -> None:
    row = _parse_support_row(
        totalHealsOnTeammates=6000,
        totalDamageShieldedOnTeammates=400,
        timeCCingOthers=50,
    )
    df = pd.DataFrame([row])
    gold = {"ccpm": 1.9, "damage_share": 0.08, "damage_taken_share": 0.20, "healing": 7500, "shielding": 3500}
    _, value = support_utility_impact(df, gold)
    assert "shield/min" not in value
    assert "heal/min" in value


def test_support_utility_includes_damage_taken_share() -> None:
    match = make_match()
    me = match["info"]["participants"][0]
    me["teamPosition"] = "UTILITY"
    me["championName"] = "Leona"
    me["totalHealsOnTeammates"] = 0
    me["totalDamageShieldedOnTeammates"] = 0
    me["totalDamageTaken"] = 28000
    me["timeCCingOthers"] = 50
    for ally in match["info"]["participants"][1:5]:
        ally["totalDamageTaken"] = 8000
    record = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(match, make_timeline(), MY_PUUID)
    df = pd.DataFrame([record.to_row()])
    gold = {"ccpm": 1.9, "damage_share": 0.08, "damage_taken_share": 0.20, "healing": 7500, "shielding": 3500}
    _, value = support_utility_impact(df, gold)
    assert "dmg taken" in value
    assert float(df["damage_taken_share"].iloc[0]) > 0.30
