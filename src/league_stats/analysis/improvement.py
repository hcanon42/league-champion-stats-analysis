"""Improvement-score helpers: role-aware bands and support utility composite."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Typical ranked game length used to turn total-heal benchmarks into per-minute refs.
AVG_GAME_MIN: float = 28.0

# Ignore CC/heal/shield in the utility composite when output is below this
# fraction of the role benchmark (avoids noise from incidental W passives, etc.).
UTILITY_CC_NOISE_RATIO: float = 0.25
UTILITY_HEAL_NOISE_RATIO: float = 0.20
UTILITY_SHIELD_NOISE_RATIO: float = 0.20


def column_mean(matches_df: pd.DataFrame, column: str) -> float | None:
    """Column mean, or ``None`` when the column is missing or all NaN."""
    if column not in matches_df.columns:
        return None
    series = pd.to_numeric(matches_df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def clamp_score(value: float, floor: float, ceiling: float) -> float:
    """Map a value linearly onto 0–100 between floor and ceiling."""
    if floor == ceiling:
        return 50.0
    ratio = (value - floor) / (ceiling - floor)
    return round(max(0.0, min(1.0, ratio)) * 100, 1)


def relative_band_score(
    value: float | None,
    benchmark: float,
    *,
    low: float = 0.55,
    high: float = 1.30,
) -> float | None:
    """Score a metric relative to a role benchmark (higher is better)."""
    if value is None or benchmark <= 0:
        return None
    return clamp_score(value, floor=benchmark * low, ceiling=benchmark * high)


def kill_participation_score(
    matches_df: pd.DataFrame, gold: dict[str, Any]
) -> tuple[float, str]:
    """Impact dimension from kill participation with a wide, role-calibrated band."""
    kp = column_mean(matches_df, "kill_participation")
    if kp is None or kp <= 0:
        return 0.0, "—"
    bench = float(gold.get("kill_participation", 0.60))
    floor = max(0.35, bench - 0.20)
    ceiling = min(0.90, bench + 0.15)
    score = clamp_score(kp, floor=floor, ceiling=ceiling)
    return score, f"{kp * 100:.0f}% KP"


def support_utility_impact(
    matches_df: pd.DataFrame, gold: dict[str, Any]
) -> tuple[float, str]:
    """Composite utility output: CC, damage share, damage taken share, ally heal/shield."""
    weighted: list[tuple[float, float]] = []

    cc = column_mean(matches_df, "ccpm")
    cc_bench = float(gold.get("ccpm", 1.9))
    if cc is not None and cc >= cc_bench * UTILITY_CC_NOISE_RATIO:
        cc_score = relative_band_score(cc, cc_bench, low=0.50, high=1.35)
        if cc_score is not None:
            weighted.append((cc_score, 0.30))

    dmg = column_mean(matches_df, "damage_share")
    dmg_bench = float(gold.get("damage_share", 0.08))
    if dmg is not None:
        dmg_score = relative_band_score(dmg, dmg_bench, low=0.45, high=1.65)
        if dmg_score is not None:
            weighted.append((dmg_score, 0.15))

    taken = column_mean(matches_df, "damage_taken_share")
    taken_bench = float(gold.get("damage_taken_share", 0.20))
    if taken is not None:
        taken_score = relative_band_score(taken, taken_bench, low=0.45, high=1.65)
        if taken_score is not None:
            weighted.append((taken_score, 0.15))

    hpm = column_mean(matches_df, "hpm")
    hpm_bench = float(gold.get("healing", 7500)) / AVG_GAME_MIN
    if (
        hpm is not None
        and hpm_bench > 0
        and hpm >= hpm_bench * UTILITY_HEAL_NOISE_RATIO
    ):
        hpm_score = relative_band_score(hpm, hpm_bench, low=0.40, high=1.45)
        if hpm_score is not None:
            weighted.append((hpm_score, 0.275))

    spm = column_mean(matches_df, "spm")
    spm_bench = float(gold.get("shielding", 3500)) / AVG_GAME_MIN
    if (
        spm is not None
        and spm_bench > 0
        and spm >= spm_bench * UTILITY_SHIELD_NOISE_RATIO
    ):
        spm_score = relative_band_score(spm, spm_bench, low=0.40, high=1.45)
        if spm_score is not None:
            weighted.append((spm_score, 0.275))

    if not weighted:
        return 0.0, "—"

    total_weight = sum(weight for _, weight in weighted)
    score = round(sum(part * weight for part, weight in weighted) / total_weight, 1)

    segments: list[str] = []
    if cc is not None and cc >= cc_bench * UTILITY_CC_NOISE_RATIO:
        segments.append(f"{cc:.2f} CC/min")
    if hpm is not None and hpm >= hpm_bench * UTILITY_HEAL_NOISE_RATIO:
        segments.append(f"{hpm:.0f} heal/min")
    if spm is not None and spm >= spm_bench * UTILITY_SHIELD_NOISE_RATIO:
        segments.append(f"{spm:.0f} shield/min")
    if dmg is not None:
        segments.append(f"{dmg * 100:.0f}% dmg")
    if taken is not None:
        segments.append(f"{taken * 100:.0f}% dmg taken")
    return score, " · ".join(segments) if segments else "—"
