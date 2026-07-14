"""Diff-specific coaching recommendations triggered by significant metric deltas."""

from __future__ import annotations

from league_stats.core.models import MetricDelta, Recommendation, RecommendationTone


def _priority(delta: MetricDelta) -> float:
    significance = 1.5 if delta.significant else 0.5
    effect = abs(delta.effect_size or delta.delta_pct or delta.delta)
    return round(effect * significance + (delta.recent_n + delta.baseline_n) / 100, 3)


def _find_delta(deltas: list[MetricDelta], metric: str) -> MetricDelta | None:
    for delta in deltas:
        if delta.metric == metric:
            return delta
    return None


def _rule_winrate_improved(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "win" or delta.verdict != "improved" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title="Win rate trending up",
        detail="Your recent games are converting more often than your prior baseline.",
        evidence=(
            f"Win rate moved from {delta.baseline * 100:.0f}% to {delta.recent * 100:.0f}% "
            f"({delta.delta * 100:+.0f} pp, p={delta.p_value:.3f})."
        ),
        tone=RecommendationTone.POSITIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_winrate_regressed(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "win" or delta.verdict != "regressed" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title="Recent win rate dip",
        detail="Recent results are below your longer baseline — review what changed in laning and deaths.",
        evidence=(
            f"Win rate dropped from {delta.baseline * 100:.0f}% to {delta.recent * 100:.0f}% "
            f"({delta.delta * 100:+.0f} pp, p={delta.p_value:.3f})."
        ),
        tone=RecommendationTone.NEGATIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_deaths_regressed(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "deaths" or delta.verdict != "regressed" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title="Deaths creeping up",
        detail="You are dying more often in recent games than in your baseline period.",
        evidence=(
            f"Deaths/game rose from {delta.baseline:.1f} to {delta.recent:.1f} "
            f"(d={delta.effect_size:.2f}, p={delta.p_value:.3f})."
        ),
        tone=RecommendationTone.NEGATIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_laning_improved(delta: MetricDelta) -> Recommendation | None:
    if delta.metric not in {"gd10", "cs10"} or delta.verdict != "improved" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title=f"{delta.label} improved",
        detail="Early game fundamentals are stronger in your recent window — keep the same lane habits.",
        evidence=(
            f"{delta.label} moved from {delta.baseline:+.0f} to {delta.recent:+.0f} "
            f"(p={delta.p_value:.3f})."
        ),
        tone=RecommendationTone.POSITIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_greed_deaths(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "greed_death_rate" or delta.verdict != "regressed":
        return None
    if not delta.significant and abs(delta.delta) < 0.10:
        return None
    return Recommendation(
        category="Form",
        title="More greed deaths lately",
        detail="A higher share of recent deaths follow overextension — tighten reset timing after plates.",
        evidence=(
            f"Greed death rate rose from {delta.baseline * 100:.0f}% to {delta.recent * 100:.0f}%."
        ),
        tone=RecommendationTone.NEGATIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_vision_improved(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "vspm" or delta.verdict != "improved" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title="Vision trending up",
        detail="Recent games show better map information — maintain control ward habits before objectives.",
        evidence=f"VS/min rose from {delta.baseline:.2f} to {delta.recent:.2f}.",
        tone=RecommendationTone.POSITIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_vision_regressed(delta: MetricDelta) -> Recommendation | None:
    if delta.metric != "vspm" or delta.verdict != "regressed" or not delta.significant:
        return None
    return Recommendation(
        category="Form",
        title="Vision dropped recently",
        detail="Recent games have less vision per minute than your baseline — buy more control wards.",
        evidence=f"VS/min fell from {delta.baseline:.2f} to {delta.recent:.2f}.",
        tone=RecommendationTone.NEGATIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta),
        sample_size=delta.recent_n,
    )


def _rule_generic_regression(delta: MetricDelta) -> Recommendation | None:
    if not delta.significant or delta.verdict != "regressed":
        return None
    if delta.metric in {"win", "deaths", "greed_death_rate", "vspm"}:
        return None
    return Recommendation(
        category="Form",
        title=f"{delta.label} slipped",
        detail=f"Recent {delta.label.lower()} is below your baseline — worth reviewing replays from this stretch.",
        evidence=f"Baseline {delta.baseline:.2f} → recent {delta.recent:.2f}.",
        tone=RecommendationTone.NEGATIVE,
        p_value=delta.p_value,
        effect_size=delta.effect_size,
        priority=_priority(delta) * 0.8,
        sample_size=delta.recent_n,
    )


_FORM_RULES = (
    _rule_winrate_improved,
    _rule_winrate_regressed,
    _rule_deaths_regressed,
    _rule_laning_improved,
    _rule_greed_deaths,
    _rule_vision_improved,
    _rule_vision_regressed,
    _rule_generic_regression,
)


def generate_form_recommendations(
    deltas: list[MetricDelta],
    *,
    existing_titles: set[str] | None = None,
    limit: int = 3,
) -> list[Recommendation]:
    """Generate diff-specific coaching tips ranked by priority."""
    seen = existing_titles or set()
    recommendations: list[Recommendation] = []
    for rule in _FORM_RULES:
        for delta in deltas:
            try:
                rec = rule(delta)
            except Exception:
                rec = None
            if rec is None or rec.title in seen:
                continue
            seen.add(rec.title)
            recommendations.append(rec)
    recommendations.sort(key=lambda rec: rec.priority, reverse=True)
    return recommendations[:limit]
