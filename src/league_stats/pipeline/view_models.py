"""Dashboard card and peer-row view models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from league_stats.core.models import PeerComparisonResult
from league_stats.core.role_metrics import (
    MetricSpec,
    card_tier_headlines,
    overview_metric_specs,
    resolve_metric_value,
    role_profile,
)
from league_stats.presentation.metric_colors import (
    color_winrate,
    interpolate_metric_color,
    score_death_share,
    score_deaths_per_game,
    score_form_delta,
    score_lane_diff,
    score_peer_gap,
    score_winrate,
)

_SIGNED_LANE_METRICS = frozenset({"gd10", "cs10", "gd15", "xpd10", "csd10"})
_RATE_METRICS = frozenset(
    {
        "win",
        "kill_participation",
        "damage_share",
        "objectives_present_rate",
    }
)
from league_stats.presentation.ui_icons import icon_fields_for_label, with_icons

PRIORITY_LABELS: dict[str, str] = {"high": "High", "medium": "Medium", "low": "Low"}


@dataclass(frozen=True)
class MetricCard:
    """One metric tile in the HTML dashboard."""

    label: str
    value: str
    icon: str | None = None
    iconify: str | None = None
    icon_href: str | None = None
    icon_tone: str = "muted"
    value_class: str = ""
    value_color: str = ""


def card(value: Any, suffix: str = "") -> str:
    """Format a possibly-missing metric for a dashboard card."""
    return "—" if value is None else f"{value}{suffix}"


def pct(value: float | None) -> str | None:
    """Format a ratio as a percentage string, keeping ``None``."""
    return None if value is None else f"{value * 100:.0f}%"


def card_entries(
    pairs: list[tuple[str, str]], *, color_values: bool = True
) -> list[dict[str, str]]:
    """Convert label/value card pairs to JSON-friendly dicts."""
    entries = with_icons([{"label": label, "value": value} for label, value in pairs])
    if color_values:
        for entry in entries:
            enrich_value_semantics(entry)
    return entries


def _format_metric_value(spec: MetricSpec, raw: Any) -> str:
    if raw is None:
        return "—"
    if spec.pct:
        return f"{float(raw) * 100:.0f}%"
    if spec.key == "winrate_pct":
        return f"{float(raw):.0f}%"
    if isinstance(raw, float):
        if raw == int(raw):
            return f"{int(raw)}{spec.suffix}"
        return f"{raw:.2f}{spec.suffix}"
    return f"{raw}{spec.suffix}"


def cards_from_specs(
    specs: tuple[MetricSpec, ...],
    summaries: dict[str, Any],
    *,
    section: str,
    role: str = "MIDDLE",
    avg_damage_share: float | None = None,
) -> list[dict[str, Any]]:
    """Build dashboard cards from role metric specs."""
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        raw = resolve_metric_value(spec, summaries)
        pairs.append((spec.label, _format_metric_value(spec, raw)))
    entries = card_entries(pairs, color_values=section != "deaths")
    return annotate_card_tiers(
        entries,
        section,
        role=role,
        avg_damage_share=avg_damage_share,
    )


def priority_label(badge: str) -> str:
    """Map a recommendation badge class to a player-facing label."""
    return PRIORITY_LABELS.get(badge, "Medium")


def _parse_signed_number(value: str) -> float | None:
    """Extract a leading signed number from a formatted metric value."""
    match = re.search(r"([+-]?\d+(?:\.\d+)?)", str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_percent(value: str) -> float | None:
    """Parse a percentage string such as ``53%``."""
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _apply_value_color(entry: dict[str, Any], score: float | None) -> None:
    """Attach a weighted gradient color when a semantic score is available."""
    if score is None:
        return
    entry["value_color"] = interpolate_metric_color(score)
    if score > 0.15:
        entry["value_class"] = "win"
    elif score < -0.15:
        entry["value_class"] = "loss"
    else:
        entry["value_class"] = ""


def enrich_value_semantics(entry: dict[str, Any]) -> None:
    """Attach weighted metric coloring when a metric has a clear good/bad direction."""
    if entry.get("value_color"):
        return
    label = str(entry.get("label", ""))
    value = str(entry.get("value", ""))
    lower = label.lower()

    if "diff @10" in lower or label.startswith("GD@") or label.startswith("CSD@"):
        number = _parse_signed_number(value)
        if number is not None:
            _apply_value_color(entry, score_lane_diff(number))
        return

    if "win rate" in lower or lower.startswith("wr when"):
        pct_value = _parse_percent(value)
        if pct_value is not None:
            _apply_value_color(entry, score_winrate(pct_value))
        return

    if "deaths" in lower or "death rate" in lower:
        pct_value = _parse_percent(value)
        if pct_value is not None:
            _apply_value_color(entry, score_death_share(pct_value))
            return
        number = _parse_signed_number(value)
        if number is not None and "game" in lower:
            _apply_value_color(entry, score_deaths_per_game(number))


def annotate_card_tiers(
    entries: list[dict[str, Any]],
    section: str,
    *,
    role: str = "MIDDLE",
    avg_damage_share: float | None = None,
) -> list[dict[str, Any]]:
    """Mark cards as headline or secondary stats and order headline metrics first."""
    profile = role_profile(role)
    if section == "overview":
        headlines = card_tier_headlines(profile, "overview", avg_damage_share=avg_damage_share)
    elif section in {"lane", "early"}:
        headlines = card_tier_headlines(profile, "early", avg_damage_share=avg_damage_share)
    else:
        headlines = card_tier_headlines(profile, section, avg_damage_share=avg_damage_share)
    headline_set = set(headlines)
    ordered: list[dict[str, Any]] = []
    for label in headlines:
        for entry in entries:
            if entry.get("label") == label:
                ordered.append({**entry, "tier": "headline"})
                break
    for entry in entries:
        if entry.get("label") not in headline_set:
            ordered.append({**entry, "tier": "more"})
    return ordered


def overview_card_entries(
    overview: dict[str, Any], *, role: str = "MIDDLE"
) -> list[dict[str, str]]:
    """Build overview cards from the role profile."""
    avg_damage_share = overview.get("avg_damage_share")
    if avg_damage_share is not None:
        avg_damage_share = float(avg_damage_share)
    profile = role_profile(role)
    summaries = {"overview": overview}
    specs = overview_metric_specs(profile, avg_damage_share=avg_damage_share)
    entries = cards_from_specs(
        specs,
        summaries,
        section="overview",
        role=role,
        avg_damage_share=avg_damage_share,
    )
    for entry in entries:
        if entry.get("label") == "Win rate":
            winrate = float(overview.get("winrate", 0.0))
            entry["value_color"] = color_winrate(winrate)
            if winrate >= 0.53:
                entry["value_class"] = "win"
            elif winrate <= 0.47:
                entry["value_class"] = "loss"
        if entry.get("label") == "Deaths/game":
            avg_deaths = overview.get("avg_deaths")
            if avg_deaths is not None:
                _apply_value_color(entry, score_deaths_per_game(float(avg_deaths)))
    return entries


def peer_row_display(row: dict[str, Any]) -> dict[str, str]:
    """Format peer comparison row values for HTML/JSON."""
    metric = row.get("metric")
    yours = row.get("yours")
    peer_avg = row.get("peer_avg")
    if metric in {"win", "kill_participation", "damage_share", "objectives_present_rate"}:
        yours_display = f"{float(yours) * 100:.0f}%"
        peer_display = f"{float(peer_avg) * 100:.0f}%"
    else:
        yours_display = str(yours)
        peer_display = str(peer_avg)
    delta_pct = row.get("delta_pct")
    delta = row.get("delta")
    if delta_pct is not None:
        gap_display = f"{float(delta_pct):+.0f}%"
    else:
        gap_display = f"{float(delta):+.1f}"
    gap_score = score_peer_gap(
        metric=str(row.get("metric", "")),
        delta_pct=row.get("delta_pct"),
        delta=float(row.get("delta", 0.0)),
        direction=str(row.get("direction", "higher")),
    )
    gap_color = interpolate_metric_color(gap_score) if gap_score is not None else ""
    return {
        "label": str(row.get("label", "")),
        **icon_fields_for_label(str(row.get("label", ""))),
        "yours": yours_display,
        "peer_avg": peer_display,
        "gap": gap_display,
        "gap_color": gap_color,
        "verdict": str(row.get("verdict", "inline")),
    }


def _improvement_delta(delta: float, direction: str) -> float:
    """Positive value means the player improved on this metric."""
    return delta if direction == "higher" else -delta


def _form_gap_color_score(metric: str, improvement: float) -> float | None:
    """Map an improvement delta to a [-1, 1] color score."""
    return score_form_delta(metric, improvement)


def _form_change_pct(row: dict[str, Any]) -> float | None:
    """Percent change vs baseline; ``None`` for signed lane diffs or unusable baselines."""
    metric = str(row.get("metric", ""))
    if metric in _SIGNED_LANE_METRICS:
        return None
    if row.get("delta_pct") is not None:
        return float(row["delta_pct"])
    baseline = row.get("baseline")
    delta = row.get("delta")
    if baseline is None or delta is None or abs(float(baseline)) < 1e-6:
        return None
    return float(delta) / float(baseline) * 100


def _form_gap_display_and_score(row: dict[str, Any]) -> tuple[str, float | None]:
    """Direction-aware change label and color score for Form Tracker rows."""
    metric = str(row.get("metric", ""))
    delta = float(row.get("delta", 0.0))
    direction = str(row.get("direction", "higher"))
    improvement = _improvement_delta(delta, direction)
    gap_score = _form_gap_color_score(metric, improvement)

    if metric in _SIGNED_LANE_METRICS:
        return f"{delta:+.0f}", gap_score
    pct = _form_change_pct(row)
    if pct is not None:
        return f"{pct:+.0f}%", gap_score
    if metric in {"deaths", "deaths_pre14"}:
        return f"{delta:+.2f}", gap_score
    return f"{delta:+.2f}", gap_score


def form_delta_chart_value(row: dict[str, Any]) -> float:
    """Bar magnitude: % change for most metrics, raw delta for signed lane diffs."""
    metric = str(row.get("metric", ""))
    if metric in _SIGNED_LANE_METRICS:
        return float(row.get("delta", 0.0))
    pct = _form_change_pct(row)
    if pct is not None:
        return pct
    return float(row.get("delta", 0.0))


def form_delta_rank_magnitude(row: dict[str, Any]) -> float:
    """Absolute normalized impact used to rank chart movers."""
    return abs(form_delta_chart_value(row))


def form_row_display(row: dict[str, Any]) -> dict[str, str]:
    """Format Form Tracker delta row values for HTML/JSON."""
    metric = row.get("metric")
    recent = row.get("recent")
    baseline = row.get("baseline")
    if metric in _RATE_METRICS or str(metric).endswith("_rate"):
        recent_display = f"{float(recent) * 100:.0f}%"
        baseline_display = f"{float(baseline) * 100:.0f}%"
    elif metric in _SIGNED_LANE_METRICS:
        recent_display = f"{float(recent):+.0f}"
        baseline_display = f"{float(baseline):+.0f}"
    else:
        recent_display = f"{float(recent):.2f}" if isinstance(recent, float) else str(recent)
        baseline_display = f"{float(baseline):.2f}" if isinstance(baseline, float) else str(baseline)

    gap_display, gap_score = _form_gap_display_and_score(row)
    verdict = str(row.get("verdict", "inline"))
    if verdict == "inline":
        gap_color = ""
    else:
        gap_color = interpolate_metric_color(gap_score) if gap_score is not None else ""
    return {
        "label": str(row.get("label", "")),
        **icon_fields_for_label(str(row.get("label", ""))),
        "recent": recent_display,
        "baseline": baseline_display,
        "gap": gap_display,
        "gap_color": gap_color,
        "verdict": verdict,
        "significant": bool(row.get("significant")),
        "section": str(row.get("section", "")),
    }


def peer_subtitle(peer: PeerComparisonResult) -> str:
    """Build the peer comparison subtitle with sample size and confidence."""
    confidence = peer.confidence.replace("_", " ")
    return (
        f"Your averages vs {peer.build_label} at {peer.rank_label} · "
        f"{peer.source} · {peer.peer_games} peer games "
        f"({peer.peer_players} players, {confidence} confidence)"
    )


def _game_word(count: int) -> str:
    return "game" if count == 1 else "games"


def form_sample_subtitle(
    *,
    recent_games: int,
    baseline_games: int,
    overlap_mode: str = "exclusive",
) -> str:
    """Describe which game windows Form Tracker compares."""
    recent = max(0, recent_games)
    baseline = max(0, baseline_games)
    if recent == 0 and baseline == 0:
        return "Recent form vs your personal baseline."
    if overlap_mode == "inclusive":
        return (
            f"Statistics from your last {recent} {_game_word(recent)} "
            f"compared to your previous {baseline}-game baseline (overlapping windows)."
        )
    if baseline == 0:
        return f"Statistics from your last {recent} {_game_word(recent)}."
    return (
        f"Statistics from your last {recent} {_game_word(recent)} "
        f"compared to the {baseline} {_game_word(baseline)} before that."
    )
