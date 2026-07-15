"""Assemble a full GameDetail from one MatchRecord and analysis frames."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from league_stats.analysis.game_review.behaviors import evaluate_behaviors
from league_stats.analysis.game_review.compare import compare_to_baseline, compare_to_peers
from league_stats.analysis.game_review.score import compute_game_score
from league_stats.analysis.timeline import timeline_dataframe_rows
from league_stats.core.config import RANKED_FLEX_QUEUE_ID, RANKED_SOLO_QUEUE_ID
from league_stats.core.models import (
    GameBuildInfo,
    GameDeathRow,
    GameDetail,
    GameFightRow,
    GameObjectiveRow,
    MatchRecord,
    PeerComparisonResult,
)
from league_stats.pipeline.frames import AnalysisFrames


def _queue_label(queue_id: int) -> str:
    if queue_id == RANKED_SOLO_QUEUE_ID:
        return "solo"
    if queue_id == RANKED_FLEX_QUEUE_ID:
        return "flex"
    return "all"


def _iso_date(game_creation_ms: int) -> str:
    dt = datetime.fromtimestamp(game_creation_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _death_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for column, label in (
        ("alone", "alone"),
        ("after_greed", "after_greed"),
        ("before_neutral_objective", "before_neutral_objective"),
        ("to_gank", "to_gank"),
        ("outnumbered", "outnumbered"),
        ("before_dragon", "before_dragon"),
        ("before_baron", "before_baron"),
    ):
        if row.get(column):
            flags.append(label)
    return flags


def _filter_frame(df: pd.DataFrame, match_id: str) -> pd.DataFrame:
    if df.empty or "match_id" not in df.columns:
        return df.iloc[0:0]
    return df[df["match_id"] == match_id]


def _key_stats(game_row: dict[str, Any]) -> dict[str, float | int | None]:
    keys = (
        "gd10",
        "gd15",
        "deaths",
        "deaths_pre14",
        "dpm",
        "kill_participation",
        "damage_share",
        "gold_share",
        "vspm",
        "control_wards",
        "objectives_present_rate",
        "solo_deaths",
        "greed_deaths",
        "fights_disadvantaged",
    )
    return {key: game_row.get(key) for key in keys}


def assemble_game_detail(
    record: MatchRecord,
    frames: AnalysisFrames,
    *,
    baseline_means: dict[str, float],
    peer_comparison: PeerComparisonResult | None,
    archetype: str,
    index: int,
    role: str,
) -> GameDetail:
    """Build one game review detail payload."""
    game_row = record.to_row()
    deaths_df = _filter_frame(frames.deaths_df, record.match_id)
    fights_df = _filter_frame(frames.teamfights_df, record.match_id)
    objectives_df = _filter_frame(frames.objectives_df, record.match_id)

    deaths_rows = deaths_df.to_dict("records") if not deaths_df.empty else []
    good, bad = evaluate_behaviors(
        record,
        game_row,
        deaths_rows,
        baseline_means=baseline_means,
        archetype=archetype,
    )

    timeline = [
        {key: float(row[key]) for key in ("minute", "gold", "xp", "cs", "gold_diff") if key in row}
        for row in timeline_dataframe_rows(record.match_id, record.timeline)
    ]

    deaths = [
        GameDeathRow(
            minute=float(row.get("minute") or 0),
            zone=str(row.get("zone") or "unknown"),
            killer=str(row.get("killer")) if row.get("killer") else None,
            flags=_death_flags(row),
        )
        for row in deaths_rows
    ]

    fights = [
        GameFightRow(
            start_minute=float(row.get("start_minute") or 0),
            kills=int(row.get("kills") or 0),
            deaths=1 if row.get("died") else 0,
            assists=int(row.get("assists") or 0),
            damage=int(row.get("damage_dealt") or 0),
            fight_won=bool(row.get("fight_won")),
            manpower_advantage=(
                int(row["manpower_advantage"]) if pd.notna(row.get("manpower_advantage")) else None
            ),
        )
        for row in (
            fights_df[fights_df["participated"].astype(bool)].to_dict("records")
            if not fights_df.empty and "participated" in fights_df.columns
            else []
        )
    ]

    objectives = [
        GameObjectiveRow(
            kind=str(row.get("kind") or "unknown"),
            minute=float(row.get("minute") or 0),
            present=bool(row.get("present")),
            dead_before=bool(row.get("dead_before")),
            wards_before=int(row.get("wards_before") or 0),
        )
        for row in (objectives_df.to_dict("records") if not objectives_df.empty else [])
    ]

    build = GameBuildInfo(
        keystone=record.runes.keystone,
        primary_tree=record.runes.primary_tree,
        secondary_tree=record.runes.secondary_tree,
        summoners=list(record.summoners),
        skill_order=record.skill_order,
        items=[item for item in record.final_items if item],
    )

    return GameDetail(
        match_id=record.match_id,
        index=index,
        date=_iso_date(record.game_creation_ms),
        queue=_queue_label(record.queue_id),
        result="win" if record.win else "loss",
        duration_min=round(record.duration_min, 1),
        patch=record.patch,
        opponent=record.lane_opponent or "Unknown",
        side=record.side.value,
        kda=f"{record.combat.kills}/{record.combat.deaths}/{record.combat.assists}",
        archetype=archetype,
        score=compute_game_score(game_row, baseline_means, role=role),
        behaviors_good=good,
        behaviors_bad=bad,
        vs_baseline=compare_to_baseline(game_row, baseline_means, role=role),
        vs_peers=compare_to_peers(game_row, peer_comparison),
        key_stats=_key_stats(game_row),
        deaths=deaths,
        fights=fights,
        objectives=objectives,
        build=build,
        timeline=timeline,
        timeline_figure="",
        ai_recap=None,
    )
