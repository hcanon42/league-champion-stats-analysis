"""Orchestrate recent-vs-baseline progression comparisons."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from league_stats.analysis.progression.coach import generate_form_recommendations
from league_stats.analysis.progression.form_score import compute_form_score, trend_from_score
from league_stats.analysis.progression.metrics import (
    ProgressionMetricSpec,
    progression_metrics_for_role,
    resolve_metric_value,
)
from league_stats.analysis.progression.shifts import detect_behavioral_shifts
from league_stats.analysis.progression.stats import (
    confidence_tier,
    continuous_significant,
    proportion_test,
    wilson_interval,
    winrate_significant,
)
from league_stats.core.config import AppConfig
from league_stats.core.models import FormSnapshot, MatchRecord, MetricDelta, ProgressionComparison
from league_stats.pipeline.frames import build_analysis_frames
from league_stats.pipeline.summaries import build_domain_summaries


def _progression_verdict(
    delta: float,
    direction: str,
    metric: str,
    baseline: float,
) -> Literal["improved", "regressed", "inline"]:
    if metric in ("gd10", "cs10") and abs(delta) < 3:
        return "inline"
    threshold = max(abs(baseline) * 0.08, 0.05) if baseline else 0.05
    if direction == "higher":
        if delta > threshold:
            return "improved"
        if delta < -threshold:
            return "regressed"
        return "inline"
    if delta < -threshold:
        return "improved"
    if delta > threshold:
        return "regressed"
    return "inline"


def _current_streak(matches_df: pd.DataFrame) -> str:
    if matches_df.empty or "win" not in matches_df.columns:
        return ""
    ordered = matches_df.sort_values("game_creation_ms", ascending=False)
    first = bool(ordered.iloc[0]["win"])
    count = 0
    for win in ordered["win"]:
        if bool(win) != first:
            break
        count += 1
    return f"{'W' if first else 'L'}{count}"


def _headline(snapshot: FormSnapshot, top_improved: list[MetricDelta], top_regressed: list[MetricDelta]) -> str:
    if snapshot.confidence == "insufficient":
        return "Need more games to measure your recent form reliably."
    parts: list[str] = []
    if snapshot.trend == "improving":
        parts.append("Recent form is trending up")
    elif snapshot.trend == "declining":
        parts.append("Recent form is slipping")
    else:
        parts.append("Recent form is stable vs your baseline")
    if top_improved:
        parts.append(f"best gains in {top_improved[0].label.lower()}")
    if top_regressed:
        parts.append(f"watch {top_regressed[0].label.lower()}")
    return " — ".join(parts[:2]) + ("; " + parts[2] if len(parts) > 2 else "")


def _top_movers(deltas: list[MetricDelta], *, verdict: str, limit: int = 3) -> list[MetricDelta]:
    filtered = [delta for delta in deltas if delta.verdict == verdict]
    filtered.sort(
        key=lambda delta: (
            1.5 if delta.significant else 1.0,
            abs(delta.effect_size or 0),
            abs(delta.delta_pct or 0),
            abs(delta.delta),
        ),
        reverse=True,
    )
    return filtered[:limit]


def _build_metric_delta(
    spec: ProgressionMetricSpec,
    *,
    recent_value: float,
    baseline_value: float,
    recent_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    recent_n: int,
    baseline_n: int,
    alpha: float,
) -> MetricDelta:
    delta = recent_value - baseline_value
    delta_pct = round(delta / baseline_value * 100, 1) if baseline_value else None
    verdict = _progression_verdict(delta, spec.direction, spec.metric, baseline_value)

    p_value: float | None = None
    effect_size: float | None = None
    significant = False

    if spec.metric == "win":
        recent_wins = int(recent_df["win"].sum()) if "win" in recent_df.columns else 0
        baseline_wins = int(baseline_df["win"].sum()) if "win" in baseline_df.columns else 0
        significant, p_value, effect_size = winrate_significant(
            recent_wins, recent_n, baseline_wins, baseline_n, alpha=alpha
        )
    elif spec.is_rate:
        recent_hits = int(round(recent_value * recent_n))
        baseline_hits = int(round(baseline_value * baseline_n))
        p_value, effect_size = proportion_test(recent_hits, recent_n, baseline_hits, baseline_n)
        significant = bool(p_value is not None and p_value < alpha and abs(recent_value - baseline_value) >= 0.10)
    elif spec.metric in recent_df.columns and spec.metric in baseline_df.columns:
        significant, p_value, effect_size = continuous_significant(
            recent_df[spec.metric],
            baseline_df[spec.metric],
            alpha=alpha,
        )
    else:
        significant = False
        p_value = None
        effect_size = None

    return MetricDelta(
        metric=spec.metric,
        label=spec.label,
        section=spec.section,
        recent=round(recent_value, 4),
        baseline=round(baseline_value, 4),
        delta=round(delta, 4),
        delta_pct=delta_pct,
        direction=spec.direction,
        verdict=verdict,
        p_value=round(p_value, 5) if p_value is not None else None,
        effect_size=round(effect_size, 4) if effect_size is not None else None,
        significant=significant,
        recent_n=recent_n,
        baseline_n=baseline_n,
    )


def build_progression_comparison(
    config: AppConfig,
    recent_records: list[MatchRecord],
    baseline_records: list[MatchRecord],
    *,
    preset_key: str,
    overlap_mode: Literal["exclusive", "inclusive"] = "exclusive",
) -> ProgressionComparison | None:
    """Build a full progression comparison from two record slices."""
    recent_n = len(recent_records)
    baseline_n = len(baseline_records)
    min_recent = config.progression_min_recent
    min_baseline = config.progression_min_baseline
    alpha = config.progression_alpha

    if recent_n < 5 or baseline_n < 15:
        snapshot = FormSnapshot(
            form_score=0.0,
            trend="stable",
            confidence="insufficient",
            recent_games=recent_n,
            baseline_games=baseline_n,
            recent_winrate=0.0,
            baseline_winrate=0.0,
            winrate_delta_pp=0.0,
            headline="Need at least 5 recent and 15 baseline games for Form Tracker.",
        )
        return ProgressionComparison(
            preset_key=preset_key,
            overlap_mode=overlap_mode,
            recent_n=recent_n,
            baseline_m=baseline_n,
            role=config.role,
            build_label=config.build_label,
            snapshot=snapshot,
        )

    recent_frames = build_analysis_frames(recent_records)
    baseline_frames = build_analysis_frames(baseline_records)
    recent_summaries = build_domain_summaries(recent_frames, recent_records)
    baseline_summaries = build_domain_summaries(baseline_frames, baseline_records)

    metric_specs = progression_metrics_for_role(
        config.role,
        avg_damage_share=float(recent_summaries.get("overview", {}).get("avg_damage_share") or 0),
    )

    deltas: list[MetricDelta] = []
    for spec in metric_specs:
        recent_value = resolve_metric_value(
            spec,
            matches_df=recent_frames.matches_df,
            summaries=recent_summaries,
        )
        baseline_value = resolve_metric_value(
            spec,
            matches_df=baseline_frames.matches_df,
            summaries=baseline_summaries,
        )
        if recent_value is None or baseline_value is None:
            continue
        deltas.append(
            _build_metric_delta(
                spec,
                recent_value=recent_value,
                baseline_value=baseline_value,
                recent_df=recent_frames.matches_df,
                baseline_df=baseline_frames.matches_df,
                recent_n=recent_n,
                baseline_n=baseline_n,
                alpha=alpha,
            )
        )

    form_score = compute_form_score(deltas, config.role)
    trend = trend_from_score(form_score)

    recent_wr = float(recent_frames.matches_df["win"].mean()) if not recent_frames.matches_df.empty else 0.0
    baseline_wr = float(baseline_frames.matches_df["win"].mean()) if not baseline_frames.matches_df.empty else 0.0
    recent_wins = int(recent_frames.matches_df["win"].sum()) if not recent_frames.matches_df.empty else 0
    wr_sig, wr_p, _ = winrate_significant(
        recent_wins,
        recent_n,
        int(baseline_frames.matches_df["win"].sum()) if not baseline_frames.matches_df.empty else 0,
        baseline_n,
        alpha=alpha,
    )
    ci_low, ci_high = wilson_interval(recent_wins, recent_n, alpha=alpha)
    sig_count = sum(1 for delta in deltas if delta.significant)
    confidence = confidence_tier(
        recent_n,
        baseline_n,
        min_recent=min_recent,
        min_baseline=min_baseline,
        significant_count=sig_count,
        wr_significant=wr_sig,
    )

    top_improved = _top_movers(deltas, verdict="improved")
    top_regressed = _top_movers(deltas, verdict="regressed")
    behavioral_shifts = detect_behavioral_shifts(
        recent_summaries,
        baseline_summaries,
        recent_matches=recent_frames.matches_df,
        baseline_matches=baseline_frames.matches_df,
        recent_runes=recent_frames.runes_df,
        baseline_runes=baseline_frames.runes_df,
    )

    snapshot = FormSnapshot(
        form_score=form_score,
        trend=trend,  # type: ignore[arg-type]
        confidence=confidence,
        recent_games=recent_n,
        baseline_games=baseline_n,
        recent_winrate=recent_wr,
        baseline_winrate=baseline_wr,
        winrate_delta_pp=round((recent_wr - baseline_wr) * 100, 1),
        winrate_ci_low=ci_low,
        winrate_ci_high=ci_high,
        current_streak=_current_streak(recent_frames.matches_df),
        headline="",
    )
    snapshot.headline = _headline(snapshot, top_improved, top_regressed)

    recommendations = generate_form_recommendations(deltas)

    return ProgressionComparison(
        preset_key=preset_key,
        overlap_mode=overlap_mode,
        recent_n=recent_n,
        baseline_m=baseline_n,
        role=config.role,
        build_label=config.build_label,
        snapshot=snapshot,
        deltas=deltas,
        top_improved=top_improved,
        top_regressed=top_regressed,
        behavioral_shifts=behavioral_shifts,
        recommendations=recommendations,
    )
