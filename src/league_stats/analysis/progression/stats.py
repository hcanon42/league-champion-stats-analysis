"""Statistical helpers for Form Tracker significance and confidence."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from league_stats.core.config import FORM_SIGNIFICANCE_ALPHA

ConfidenceTier = Literal["high", "medium", "low", "insufficient"]

MIN_WR_DELTA_PP: float = 8.0
MIN_COHEN_D: float = 0.35
MIN_TTEST_N: int = 8


def wilson_interval(wins: int, total: int, alpha: float = FORM_SIGNIFICANCE_ALPHA) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        return None, None
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    p_hat = wins / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def proportion_test(
    wins_a: int,
    total_a: int,
    wins_b: int,
    total_b: int,
) -> tuple[float | None, float | None]:
    """Two-proportion z-test; returns (p_value, effect_size as pp difference)."""
    if total_a <= 0 or total_b <= 0:
        return None, None
    try:
        p_a = wins_a / total_a
        p_b = wins_b / total_b
        pooled = (wins_a + wins_b) / (total_a + total_b)
        se = np.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
        if se == 0:
            return None, float((p_a - p_b) * 100)
        z_score = (p_a - p_b) / se
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_score))))
        return p_value, float((p_a - p_b) * 100)
    except (ValueError, FloatingPointError):
        return None, None


def welch_test(series_a: pd.Series, series_b: pd.Series) -> tuple[float | None, float | None]:
    """Welch t-test and Cohen's d between two per-game series."""
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if len(a) < MIN_TTEST_N or len(b) < MIN_TTEST_N:
        return None, None
    if a.nunique() < 2 or b.nunique() < 2:
        return None, None
    try:
        _t, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
        pooled_std = float(np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2))
        if pooled_std == 0:
            cohen_d = 0.0
        else:
            cohen_d = float((a.mean() - b.mean()) / pooled_std)
        return float(p_value), cohen_d
    except (ValueError, FloatingPointError):
        return None, None


def winrate_significant(
    recent_wins: int,
    recent_n: int,
    baseline_wins: int,
    baseline_n: int,
    *,
    alpha: float = FORM_SIGNIFICANCE_ALPHA,
) -> tuple[bool, float | None, float | None]:
    """Test whether win-rate change is statistically and practically significant."""
    p_value, effect_pp = proportion_test(recent_wins, recent_n, baseline_wins, baseline_n)
    if p_value is None or effect_pp is None:
        return False, p_value, effect_pp
    significant = p_value < alpha and abs(effect_pp) >= MIN_WR_DELTA_PP
    return significant, p_value, effect_pp


def continuous_significant(
    recent_series: pd.Series,
    baseline_series: pd.Series,
    *,
    alpha: float = FORM_SIGNIFICANCE_ALPHA,
) -> tuple[bool, float | None, float | None]:
    """Test whether a continuous metric change is significant."""
    p_value, cohen_d = welch_test(recent_series, baseline_series)
    if p_value is None or cohen_d is None:
        return False, p_value, cohen_d
    significant = p_value < alpha and abs(cohen_d) >= MIN_COHEN_D
    return significant, p_value, cohen_d


def confidence_tier(
    recent_n: int,
    baseline_n: int,
    *,
    min_recent: int,
    min_baseline: int,
    significant_count: int,
    wr_significant: bool,
) -> ConfidenceTier:
    """Derive an overall confidence tier from sample sizes and significance."""
    if recent_n < 5 or baseline_n < 15:
        return "insufficient"
    if recent_n < min_recent or baseline_n < min_baseline:
        return "low"
    if recent_n >= 15 and baseline_n >= 40 and (wr_significant or significant_count >= 3):
        return "high"
    if recent_n >= min_recent and baseline_n >= min_baseline:
        return "medium"
    return "low"
