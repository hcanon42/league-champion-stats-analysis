"""Composite form score from weighted metric deltas."""

from __future__ import annotations

from league_stats.analysis.progression.metrics import ProgressionMetricSpec, form_score_metrics
from league_stats.core.models import MetricDelta


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_form_score(deltas: list[MetricDelta], role: str) -> float:
    """Compute a -100..+100 form score from direction-aware normalized deltas."""
    score_specs = {spec.metric: spec for spec in form_score_metrics(role)}
    weighted_sum = 0.0
    weight_total = 0.0

    for delta in deltas:
        spec = score_specs.get(delta.metric)
        if spec is None:
            continue
        scale = abs(delta.baseline)
        if scale < 1e-6:
            scale = max(abs(delta.recent), 1.0)
        signed = delta.delta / scale
        if spec.direction == "lower":
            signed = -signed
        norm = _clamp(signed, -1.0, 1.0)
        weight = 1.5 if delta.significant else 1.0
        weighted_sum += weight * norm
        weight_total += weight

    if weight_total == 0:
        return 0.0
    raw = weighted_sum / weight_total
    return round(_clamp(raw * 100, -100.0, 100.0), 1)


def trend_from_score(form_score: float) -> str:
    """Map form score to a trend label."""
    if form_score >= 15:
        return "improving"
    if form_score <= -15:
        return "declining"
    return "stable"
