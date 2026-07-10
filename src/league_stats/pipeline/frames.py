"""Shared analysis DataFrames built from parsed match records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from league_stats.analysis.deaths import deaths_dataframe
from league_stats.analysis.items import items_dataframe
from league_stats.analysis.matchups import matchups_dataframe
from league_stats.analysis.objectives import objectives_dataframe
from league_stats.analysis.runes import runes_dataframe
from league_stats.analysis.teamfights import teamfights_dataframe
from league_stats.analysis.timeline import timeline_dataframe_rows
from league_stats.analysis.vision import vision_dataframe
from league_stats.core.models import MatchRecord


@dataclass
class AnalysisFrames:
    """All tabular views of one record set."""

    matches_df: pd.DataFrame
    deaths_df: pd.DataFrame
    teamfights_df: pd.DataFrame
    objectives_df: pd.DataFrame
    vision_df: pd.DataFrame
    runes_df: pd.DataFrame
    items_df: pd.DataFrame
    matchups_df: pd.DataFrame
    timeline_df: pd.DataFrame


def build_analysis_frames(records: list[MatchRecord]) -> AnalysisFrames:
    """Build every analysis dataframe from parsed records."""
    matches_df = pd.DataFrame([r.to_row() for r in records])
    deaths_df = deaths_dataframe(records)
    teamfights_df = teamfights_dataframe(records)
    objectives_df = objectives_dataframe(records)
    vision_df = vision_dataframe(records)
    runes_df = runes_dataframe(records)
    items_df = items_dataframe(matches_df)
    matchups_df = matchups_dataframe(matches_df)
    timeline_df = pd.DataFrame(
        [row for r in records for row in timeline_dataframe_rows(r.match_id, r.timeline)]
    )
    return AnalysisFrames(
        matches_df=matches_df,
        deaths_df=deaths_df,
        teamfights_df=teamfights_df,
        objectives_df=objectives_df,
        vision_df=vision_df,
        runes_df=runes_df,
        items_df=items_df,
        matchups_df=matchups_df,
        timeline_df=timeline_df,
    )


def build_overview(matches_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute headline overview metrics from the matches table."""
    if matches_df.empty:
        return {}
    return {
        "winrate": round(float(matches_df["win"].mean()), 3),
        "avg_kda": round(float(matches_df["kda"].mean()), 2),
        "avg_dpm": round(float(matches_df["dpm"].mean()), 0),
        "avg_cspm": round(float(matches_df["cspm"].mean()), 2),
        "avg_damage_share": round(float(matches_df["damage_share"].mean()), 3),
        "avg_deaths": round(float(matches_df["deaths"].mean()), 1),
        "avg_vspm": round(float(matches_df["vspm"].mean()), 2),
        "avg_duration": round(float(matches_df["duration_min"].mean()), 1),
    }
