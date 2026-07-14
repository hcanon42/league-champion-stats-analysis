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


def test_support_utility_score_not_zero() -> None:
    _, components = improvement_score(_matches_df("UTILITY"), role="UTILITY")
    by_name = {component.name: component for component in components}
    assert "Utility" in by_name
    assert by_name["Utility"].score > 0
    assert "CC/min" in by_name["Utility"].value


def test_jungle_impact_score_not_zero() -> None:
    _, components = improvement_score(_matches_df("JUNGLE"), role="JUNGLE")
    by_name = {component.name: component for component in components}
    assert by_name["Impact"].score > 0
    assert "KP" in by_name["Impact"].value
    assert "Early ganks" in by_name
    assert "Objectives" not in by_name


def test_damage_share_ceiling_clamps_at_role_benchmark_plus_six_points() -> None:
    df = _matches_df("MIDDLE")
    df["damage_share"] = 0.40
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Damage"].score == 100.0

    df["damage_share"] = 0.31
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Damage"].score == 100.0

    df["damage_share"] = 0.29
    _, components = improvement_score(df, role="MIDDLE")
    by_name = {component.name: component for component in components}
    assert by_name["Damage"].score < 100.0
