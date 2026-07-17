"""Improvement-score helpers: role-aware category bands and support utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from league_stats.analysis.economy import RECALL_GOLD_HEALTHY_AVG, RECALL_GOLD_HOARDING_WARN
from league_stats.core.role_metrics import ScoreMetricSpec, ScoreSpec

# Typical ranked game length used to turn total-heal benchmarks into per-minute refs.
AVG_GAME_MIN: float = 28.0

# Ignore CC/heal/shield in the utility composite when output is below this
# fraction of the role benchmark (avoids noise from incidental W passives, etc.).
UTILITY_CC_NOISE_RATIO: float = 0.25
UTILITY_HEAL_NOISE_RATIO: float = 0.20
UTILITY_SHIELD_NOISE_RATIO: float = 0.20

# Role-typical gold share when peer benchmarks omit the key.
_GOLD_SHARE_BENCH: dict[str, float] = {
    "TOP": 0.22,
    "MIDDLE": 0.24,
    "BOTTOM": 0.26,
    "JUNGLE": 0.20,
    "UTILITY": 0.12,
}

# First completed item minute bands (floor=slow/bad, ceiling=fast/good).
_FIRST_ITEM_BAND: dict[str, tuple[float, float]] = {
    "TOP": (14.0, 9.0),
    "MIDDLE": (13.0, 8.5),
    "BOTTOM": (14.0, 9.5),
    "JUNGLE": (13.0, 9.0),
    "UTILITY": (16.0, 11.0),
}

_VALUE_FRAGMENT_LIMIT: int = 3


@dataclass(frozen=True)
class CategoryScore:
    """Scored improvement-score category."""

    name: str
    score: float
    value: str
    hint: str


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


def _format_metric_value(column: str, value: float) -> str:
    """Short display fragment for one scored metric."""
    if column in {
        "damage_share",
        "gold_share",
        "kill_participation",
        "tf_participation",
        "objectives_present_rate",
        "lane_priority",
        "damage_taken_share",
    }:
        return f"{value * 100:.0f}%"
    if column == "gd10":
        return f"{value:+.0f}g @10"
    if column == "csd10":
        return f"{value:+.0f} CSΔ"
    if column == "cs10":
        return f"{value:.0f} CS @10"
    if column == "deaths":
        return f"{value:.1f} deaths"
    if column == "deaths_pre14":
        return f"{value:.1f} early deaths"
    if column == "vspm":
        return f"{value:.2f} VS/min"
    if column == "control_wards":
        return f"{value:.1f} CW"
    if column == "avg_unspent_gold":
        return f"{value:.0f}g banked"
    if column == "first_item_min":
        return f"{value:.1f}m item"
    if column == "ccpm":
        return f"{value:.2f} CC/min"
    if column == "early_ganks":
        return f"{value:.1f} ganks"
    if column == "roams_pre15":
        return f"{value:.1f} roams"
    if column == "hpm":
        return f"{value:.0f} heal/min"
    if column == "spm":
        return f"{value:.0f} shield/min"
    return f"{value:.1f}"


def score_metric_value(
    column: str,
    value: float | None,
    *,
    gold: dict[str, Any],
    role: str,
) -> tuple[float, str] | None:
    """Score one raw metric against role/Gold bands. Returns ``None`` when unavailable."""
    if value is None:
        return None

    if column == "gd10":
        return clamp_score(value, -800, 800), _format_metric_value(column, value)
    if column == "csd10":
        return clamp_score(value, -30, 30), _format_metric_value(column, value)
    if column == "cs10":
        bench = float(gold.get("cspm", 6.0)) * 10
        return (
            clamp_score(value, bench * 0.75, bench * 1.15),
            _format_metric_value(column, value),
        )
    if column == "deaths":
        deaths_bench = float(gold.get("deaths", 5.0))
        return (
            clamp_score(value, deaths_bench + 2.5, max(2.0, deaths_bench - 1.5)),
            _format_metric_value(column, value),
        )
    if column == "deaths_pre14":
        return clamp_score(value, 3.5, 0.5), _format_metric_value(column, value)
    if column == "damage_share":
        bench = float(gold.get("damage_share", 0.22))
        # Wide band so near-role damage still scores mid-pack rather than tanking Fight.
        return (
            clamp_score(value, max(0.04, bench - 0.10), bench + 0.08),
            f"{value * 100:.0f}% dmg",
        )
    if column == "ccpm":
        bench = float(gold.get("ccpm", 1.2))
        return (
            clamp_score(value, bench * 0.45, bench * 1.40),
            _format_metric_value(column, value),
        )
    if column == "kill_participation":
        bench = float(gold.get("kill_participation", 0.60))
        floor = max(0.30, bench - 0.22)
        ceiling = min(0.95, bench + 0.18)
        return clamp_score(value, floor, ceiling), f"{value * 100:.0f}% KP"
    if column == "tf_participation":
        # Detected-fight join rate; typical ~50–65% should land near mid-score.
        return clamp_score(value, 0.25, 0.75), f"{value * 100:.0f}% fights"
    if column == "tf_won_share":
        return clamp_score(value, 0.30, 0.70), f"{value * 100:.0f}% fight WR"
    if column == "vspm":
        bench = float(gold.get("vspm", 0.8))
        return (
            clamp_score(value, bench * 0.65, bench * 1.35),
            _format_metric_value(column, value),
        )
    if column == "control_wards":
        bench = float(gold.get("control_wards", 1.5))
        return (
            clamp_score(value, bench * 0.45, bench * 1.40),
            _format_metric_value(column, value),
        )
    if column == "objectives_present_rate":
        bench = float(gold.get("objectives_present_rate", 0.55))
        return (
            clamp_score(value, max(0.25, bench - 0.25), min(0.90, bench + 0.20)),
            f"{value * 100:.0f}% presence",
        )
    if column == "avg_unspent_gold":
        return (
            clamp_score(value, RECALL_GOLD_HOARDING_WARN, RECALL_GOLD_HEALTHY_AVG),
            _format_metric_value(column, value),
        )
    if column == "first_item_min":
        floor, ceiling = _FIRST_ITEM_BAND.get(role, (14.0, 9.0))
        return clamp_score(value, floor, ceiling), _format_metric_value(column, value)
    if column == "gold_share":
        bench = float(gold.get("gold_share", _GOLD_SHARE_BENCH.get(role, 0.22)))
        return (
            clamp_score(value, max(0.05, bench - 0.06), bench + 0.06),
            f"{value * 100:.0f}% gold",
        )
    if column == "early_ganks":
        bench = float(gold.get("early_ganks", 1.5))
        return (
            clamp_score(value, bench * 0.45, bench * 1.50),
            _format_metric_value(column, value),
        )
    if column == "roams_pre15":
        bench = float(gold.get("roams_pre15", 1.5))
        return (
            clamp_score(value, bench * 0.45, bench * 1.40),
            _format_metric_value(column, value),
        )
    if column == "lane_priority":
        return clamp_score(value, 0.35, 0.70), f"{value * 100:.0f}% priority"
    if column == "damage_taken_share":
        bench = float(gold.get("damage_taken_share", 0.20))
        scored = relative_band_score(value, bench, low=0.45, high=1.65)
        if scored is None:
            return None
        return scored, f"{value * 100:.0f}% taken"
    if column == "hpm":
        bench = float(gold.get("healing", 7500)) / AVG_GAME_MIN
        if bench <= 0 or value < bench * UTILITY_HEAL_NOISE_RATIO:
            return None
        scored = relative_band_score(value, bench, low=0.40, high=1.45)
        if scored is None:
            return None
        return scored, _format_metric_value(column, value)
    if column == "spm":
        bench = float(gold.get("shielding", 3500)) / AVG_GAME_MIN
        if bench <= 0 or value < bench * UTILITY_SHIELD_NOISE_RATIO:
            return None
        scored = relative_band_score(value, bench, low=0.40, high=1.45)
        if scored is None:
            return None
        return scored, _format_metric_value(column, value)
    return None


def _resolve_metric_column(metric: ScoreMetricSpec, *, use_cc: bool, role: str) -> str:
    """Swap damage→CC for tanky non-support/jungle builds when appropriate."""
    if metric.column == "damage_share" and use_cc and role not in {"UTILITY", "JUNGLE"}:
        return "ccpm"
    return metric.column


def score_category(
    spec: ScoreSpec,
    matches_df: pd.DataFrame,
    *,
    gold: dict[str, Any],
    role: str,
    use_cc: bool = False,
) -> CategoryScore | None:
    """Weighted average of a category's ingredient metrics."""
    if spec.name == "Utility" and role == "UTILITY":
        score, value = support_utility_impact(matches_df, gold)
        return CategoryScore(name=spec.name, score=score, value=value, hint=spec.hint)

    weighted: list[tuple[float, float]] = []
    fragments: list[str] = []
    for metric in spec.metrics:
        column = _resolve_metric_column(metric, use_cc=use_cc, role=role)
        raw = column_mean(matches_df, column)
        scored = score_metric_value(column, raw, gold=gold, role=role)
        if scored is None:
            continue
        part, fragment = scored
        weighted.append((part, metric.weight))
        fragments.append(fragment)

    if not weighted:
        return None

    total_weight = sum(weight for _, weight in weighted)
    score = round(sum(part * weight for part, weight in weighted) / total_weight, 1)
    value = " · ".join(fragments[:_VALUE_FRAGMENT_LIMIT]) if fragments else "—"
    return CategoryScore(name=spec.name, score=score, value=value, hint=spec.hint)


def score_categories(
    specs: tuple[ScoreSpec, ...],
    matches_df: pd.DataFrame,
    *,
    gold: dict[str, Any],
    role: str,
    use_cc: bool = False,
) -> list[CategoryScore]:
    """Score every category that has at least one usable ingredient."""
    components: list[CategoryScore] = []
    for spec in specs:
        scored = score_category(spec, matches_df, gold=gold, role=role, use_cc=use_cc)
        if scored is not None:
            components.append(scored)
    return components
