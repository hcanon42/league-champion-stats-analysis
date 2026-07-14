"""Behavioral shift detection between recent and baseline windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from league_stats.analysis.items import build_path_stats
from league_stats.analysis.progression.metrics import BEHAVIORAL_DEATH_METRICS
from league_stats.analysis.runes import rune_setup_stats

SHIFT_RATE_THRESHOLD: float = 0.10
SHIFT_SETUP_THRESHOLD: float = 0.25


def _rate_label(key: str) -> str:
    for metric_key, label, _direction in BEHAVIORAL_DEATH_METRICS:
        if metric_key == key:
            return label.replace(" rate", "").lower()
    return key.replace("_", " ")


def detect_death_shifts(recent_deaths: dict[str, Any], baseline_deaths: dict[str, Any]) -> list[str]:
    """Flag large changes in death context rates."""
    shifts: list[str] = []
    for key, _label, _direction in BEHAVIORAL_DEATH_METRICS:
        recent_val = recent_deaths.get(key)
        baseline_val = baseline_deaths.get(key)
        if recent_val is None or baseline_val is None:
            continue
        delta = float(recent_val) - float(baseline_val)
        if abs(delta) < SHIFT_RATE_THRESHOLD:
            continue
        direction_word = "rose" if delta > 0 else "fell"
        shifts.append(
            f"{_rate_label(key).title()} deaths {direction_word} from "
            f"{float(baseline_val) * 100:.0f}% to {float(recent_val) * 100:.0f}% "
            f"({delta * 100:+.0f} pp)"
        )
    return shifts


def detect_zone_shift(recent_deaths: dict[str, Any], baseline_deaths: dict[str, Any]) -> list[str]:
    """Flag when the most common death zone changed."""
    recent_zone = recent_deaths.get("most_common_zone")
    baseline_zone = recent_deaths.get("most_common_zone")
    if not recent_zone or not baseline_zone or recent_zone == baseline_zone:
        return []
    return [f"Most deaths moved from {baseline_zone} to {recent_zone}"]


def _top_setup_share(
    matches_df: pd.DataFrame,
    runes_df: pd.DataFrame,
    *,
    kind: str,
) -> tuple[str | None, float]:
    total_games = float(len(matches_df))
    if total_games <= 0:
        return None, 0.0
    if kind == "rune":
        frame = rune_setup_stats(runes_df)
        if frame.empty:
            return None, 0.0
        top = frame.iloc[0]
        label = str(top.get("keystone", ""))
        games = float(top.get("games", 0))
        return label, games / total_games
    paths = build_path_stats(matches_df)
    if paths.empty:
        return None, 0.0
    top = paths.iloc[0]
    label = str(top.get("core", ""))
    games = float(top.get("games", 0))
    return label, games / total_games


def detect_setup_shifts(
    recent_matches: pd.DataFrame,
    baseline_matches: pd.DataFrame,
    *,
    recent_runes: pd.DataFrame,
    baseline_runes: pd.DataFrame,
) -> list[str]:
    """Flag rune or build path frequency shifts."""
    shifts: list[str] = []
    for kind, label_prefix in (("rune", "Keystone"), ("build", "Build")):
        recent_label, recent_share = _top_setup_share(
            recent_matches,
            recent_runes,
            kind=kind,
        )
        baseline_label, baseline_share = _top_setup_share(
            baseline_matches,
            baseline_runes,
            kind=kind,
        )
        if not recent_label or not baseline_label or recent_label == baseline_label:
            continue
        share_delta = recent_share - baseline_share
        if abs(share_delta) < SHIFT_SETUP_THRESHOLD:
            continue
        shifts.append(
            f"{label_prefix} shifted from {baseline_label} ({baseline_share * 100:.0f}%) "
            f"to {recent_label} ({recent_share * 100:.0f}%)"
        )
    return shifts


def detect_behavioral_shifts(
    recent_summaries: dict[str, Any],
    baseline_summaries: dict[str, Any],
    *,
    recent_matches: pd.DataFrame,
    baseline_matches: pd.DataFrame,
    recent_runes: pd.DataFrame,
    baseline_runes: pd.DataFrame,
) -> list[str]:
    """Collect all behavioral shift lines."""
    recent_deaths = recent_summaries.get("deaths", {})
    baseline_deaths = baseline_summaries.get("deaths", {})
    shifts: list[str] = []
    shifts.extend(detect_death_shifts(recent_deaths, baseline_deaths))
    shifts.extend(detect_zone_shift(recent_deaths, baseline_deaths))
    shifts.extend(
        detect_setup_shifts(
            recent_matches,
            baseline_matches,
            recent_runes=recent_runes,
            baseline_runes=baseline_runes,
        )
    )
    return shifts
