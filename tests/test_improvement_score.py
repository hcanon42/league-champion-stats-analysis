"""Tests for role-aware improvement score."""

from __future__ import annotations

import pandas as pd

from league_stats.presentation.report import improvement_score
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline
from league_stats.ingest.parser import ItemCatalog, MatchParser


def _matches_df(role: str, *, challenges_kp: float | None = 0.55) -> pd.DataFrame:
    match = make_match()
    me = match["info"]["participants"][0]
    me["teamPosition"] = role
    if role == "UTILITY":
        me["championName"] = "Thresh"
        me["totalHealsOnTeammates"] = 6000
        me["totalDamageShieldedOnTeammates"] = 4000
    if role == "JUNGLE":
        me["championName"] = "LeeSin"
    if challenges_kp is None:
        me.pop("challenges", None)
    else:
        me["challenges"] = {"killParticipation": challenges_kp}
    timeline = make_timeline()
    record = MatchParser(ItemCatalog(FAKE_ITEMS)).parse(match, timeline, MY_PUUID)
    return pd.DataFrame([record.to_row()])


def test_support_utility_category_not_zero() -> None:
    _, components = improvement_score(_matches_df("UTILITY"), role="UTILITY")
    by_name = {component.name: component for component in components}
    assert "Utility" in by_name
    assert by_name["Utility"].score > 0
    assert "CC/min" in by_name["Utility"].value
    assert "Resets" not in by_name
    assert "Economy" in by_name
    assert "Fight" in by_name
    assert "Setup" in by_name


def test_jungle_category_scores() -> None:
    _, components = improvement_score(_matches_df("JUNGLE"), role="JUNGLE")
    by_name = {component.name: component for component in components}
    assert "Early game" in by_name
    assert "Fight" in by_name
    assert by_name["Fight"].score > 0
    assert "Resets" not in by_name
    assert "Clear @10" not in by_name
    assert "Map control" not in by_name


def test_mid_category_scores_include_economy_not_resets() -> None:
    _, components = improvement_score(_matches_df("MIDDLE"), role="MIDDLE")
    names = {component.name for component in components}
    assert names == {"Laning", "Economy", "Fight", "Survival", "Vision", "Objectives"}
    assert "Resets" not in names
    assert "Farming" not in names
    assert "Damage" not in names


def test_fight_damage_share_ceiling_clamps_at_role_benchmark_plus_eight_points() -> None:
    df = _matches_df("MIDDLE")
    # Isolate damage by maxing other Fight ingredients.
    df["kill_participation"] = 0.95
    df["tf_participation"] = 0.90
    df["tf_won_share"] = 0.90
    df["damage_share"] = 0.40
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Fight"].score == 100.0

    # Mid GOLD damage_share bench is 0.24 → ceiling 0.32
    df["damage_share"] = 0.32
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Fight"].score == 100.0

    df["damage_share"] = 0.30
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Fight"].score < 100.0
