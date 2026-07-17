"""Role-aware metric catalog and value resolution for Form Tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from league_stats.analysis.peer.comparison import compare_metrics_for_role
from league_stats.core.role_metrics import role_profile

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class ProgressionMetricSpec:
    """One metric to compare between recent and baseline windows."""

    metric: str
    label: str
    section: str
    direction: Direction
    source: Literal["matches_df", "summary"]
    summary_section: str = ""
    summary_key: str = ""
    is_rate: bool = False


BEHAVIORAL_DEATH_METRICS: tuple[tuple[str, str, Direction], ...] = (
    ("solo_death_rate", "Solo death rate", "lower"),
    ("greed_death_rate", "Greed death rate", "lower"),
    ("gank_death_rate", "Gank death rate", "lower"),
    ("outnumbered_death_rate", "Outnumbered death rate", "lower"),
    ("death_before_neutral_objective_rate", "Death before objective rate", "lower"),
)

_SCORE_METRIC_LABELS: dict[str, str] = {
    "gd10": "Gold diff @10",
    "csd10": "CS diff @10",
    "deaths_pre14": "Deaths pre-14",
    "cs10": "CS @10",
    "gold_share": "Gold share",
    "avg_unspent_gold": "Unspent gold/recall",
    "first_item_min": "First item (min)",
    "damage_share": "Damage share",
    "kill_participation": "Kill participation",
    "tf_participation": "Fight participation",
    "tf_won_share": "Fight win rate",
    "deaths": "Deaths/game",
    "vspm": "Vision/min",
    "control_wards": "Control wards",
    "objectives_present_rate": "Obj. presence",
    "early_ganks": "Early ganks",
    "roams_pre15": "Roams pre-15",
    "lane_priority": "Lane priority",
    "ccpm": "CC/min",
    "damage_taken_share": "Damage taken share",
    "hpm": "Healing/min",
    "spm": "Shielding/min",
}


def progression_metrics_for_role(role: str, *, avg_damage_share: float | None = None) -> list[ProgressionMetricSpec]:
    """Build the full diff metric list for a role."""
    specs: list[ProgressionMetricSpec] = []
    seen: set[str] = set()

    for column, label, direction in compare_metrics_for_role(role, avg_damage_share=avg_damage_share):
        if column in seen:
            continue
        seen.add(column)
        section = "overview" if column in {"win", "kda", "dpm", "cspm", "deaths", "vspm"} else "laning"
        if column in {"kill_participation", "damage_share"}:
            section = "teamfights"
        specs.append(
            ProgressionMetricSpec(
                metric=column,
                label=label,
                section=section,
                direction=direction,
                source="matches_df",
                is_rate=column == "win",
            )
        )

    profile = role_profile(role)
    for score_spec in profile.score_components:
        for metric in score_spec.metrics:
            if metric.column in seen:
                continue
            seen.add(metric.column)
            direction: Direction = metric.direction
            section = "overview"
            if score_spec.name in {"Laning", "Early game", "Setup"}:
                section = "laning"
            elif score_spec.name == "Economy":
                section = "economy"
            elif score_spec.name in {"Fight", "Utility"}:
                section = "teamfights"
            elif score_spec.name == "Vision":
                section = "vision"
            elif score_spec.name == "Objectives":
                section = "objectives"
            if metric.column in {"gd10", "cs10", "csd10", "lane_priority", "roams_pre15", "early_ganks"}:
                section = "laning"
            if metric.column in {"vspm", "control_wards"}:
                section = "vision"
            if metric.column in {"objectives_present_rate"}:
                section = "objectives"
            if metric.column in {"avg_unspent_gold", "gold_share", "first_item_min"}:
                section = "economy"
            specs.append(
                ProgressionMetricSpec(
                    metric=metric.column,
                    label=_SCORE_METRIC_LABELS.get(metric.column, metric.column),
                    section=section,
                    direction=direction,
                    source="matches_df",
                )
            )

    for key, label, direction in BEHAVIORAL_DEATH_METRICS:
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            ProgressionMetricSpec(
                metric=key,
                label=label,
                section="deaths",
                direction=direction,
                source="summary",
                summary_section="deaths",
                summary_key=key,
                is_rate=True,
            )
        )

    return specs


def form_score_metrics(role: str) -> list[ProgressionMetricSpec]:
    """Subset of metrics used for the composite form score."""
    all_specs = progression_metrics_for_role(role)
    profile = role_profile(role)
    score_columns = {
        metric.column
        for score_spec in profile.score_components
        for metric in score_spec.metrics
    }
    score_columns.add("win")
    return [spec for spec in all_specs if spec.metric in score_columns]


def resolve_matches_df_value(matches_df: pd.DataFrame, metric: str) -> float | None:
    """Mean of a per-game column."""
    if matches_df.empty or metric not in matches_df.columns:
        return None
    series = pd.to_numeric(matches_df[metric], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def resolve_summary_value(summaries: dict, spec: ProgressionMetricSpec) -> float | None:
    """Read a metric from domain summaries."""
    bucket = summaries.get(spec.summary_section, {})
    value = bucket.get(spec.summary_key)
    if value is None:
        return None
    return float(value)


def resolve_metric_value(
    spec: ProgressionMetricSpec,
    *,
    matches_df: pd.DataFrame,
    summaries: dict,
) -> float | None:
    """Resolve one metric value for a window."""
    if spec.source == "summary":
        return resolve_summary_value(summaries, spec)
    return resolve_matches_df_value(matches_df, spec.metric)
