"""Gradient metric colors: loss (red) -> neutral (gold) -> win (green)."""

from __future__ import annotations

LOSS_HEX = "#e05563"
NEUTRAL_HEX = "#c4aa6a"
WIN_HEX = "#3fb68b"

_WINRATE_MID = 50.0
_WINRATE_SPAN = 20.0
_LANE_DIFF_SPAN = 300.0
_DEATHS_MID = 4.5
_DEATHS_SPAN = 1.5
_REFERENCE_GAME_MIN = 30.0
_DEATH_SHARE_MID = 37.5
_DEATH_SHARE_SPAN = 12.5
_PEER_GAP_SPAN = 25.0

# Form Tracker: recent-vs-baseline bar impact (tuned separately from peer cards).
_FORM_RATE_SPAN = 12.0  # pp shift that reaches full bar (15pp WR ~= max)
_FORM_RAW_SPANS: dict[str, float] = {
    "gd10": 300.0,
    "cs10": 15.0,
    "csd10": 15.0,
    "deaths": 2.5,
    "deaths_pre14": 2.0,
    "kda": 1.5,
    "dpm": 200.0,
    "ccpm": 0.6,
    "cspm": 2.0,
    "vspm": 1.5,
    "vision_score": 15.0,
    "kill_participation": 0.15,
    "damage_share": 0.10,
    "gold_share": 0.06,
    "avg_unspent_gold": 400.0,
    "first_item_min": 2.5,
    "control_wards": 2.0,
    "roams_pre15": 2.0,
    "early_ganks": 2.0,
    "tf_participation": 0.20,
    "tf_won_share": 0.20,
    "lane_priority": 0.15,
}

_PEER_RAW_SPANS: dict[str, float] = {
    "gd10": 300.0,
    "cs10": 15.0,
    "deaths": 1.5,
    "kda": 1.0,
    "dpm": 120.0,
    "ccpm": 0.5,
    "cspm": 1.5,
    "vision_score": 10.0,
    "kill_participation": 0.12,
    "damage_share": 0.08,
}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_metric_color(score: float) -> str:
    """Map ``score`` in [-1, 1] to a hex color between loss, neutral, and win."""
    score = _clamp(score)
    if score <= 0:
        t = score + 1.0
        r1, g1, b1 = _hex_to_rgb(LOSS_HEX)
        r2, g2, b2 = _hex_to_rgb(NEUTRAL_HEX)
    else:
        t = score
        r1, g1, b1 = _hex_to_rgb(NEUTRAL_HEX)
        r2, g2, b2 = _hex_to_rgb(WIN_HEX)
    return _rgb_to_hex(
        round(_lerp(r1, r2, t)),
        round(_lerp(g1, g2, t)),
        round(_lerp(b1, b2, t)),
    )


def score_winrate(pct: float) -> float:
    """Win-rate percentage where 50% is neutral."""
    return _clamp((pct - _WINRATE_MID) / _WINRATE_SPAN)


def score_lane_diff(value: float) -> float:
    """Signed lane differential where positive is better."""
    return _clamp(value / _LANE_DIFF_SPAN)


def normalize_deaths_for_duration(deaths: float, duration_min: float) -> float:
    """Express deaths as equivalent deaths in a 30-minute game."""
    return float(deaths) * _REFERENCE_GAME_MIN / max(float(duration_min), 1.0)


def score_deaths_per_game(value: float, *, duration_min: float = _REFERENCE_GAME_MIN) -> float:
    """Death count where lower is better, scaled to a 30-minute game."""
    normalized = normalize_deaths_for_duration(value, duration_min)
    return _clamp((_DEATHS_MID - normalized) / _DEATHS_SPAN)


def score_death_share(pct: float) -> float:
    """Death-share percentage where lower is better."""
    return _clamp((_DEATH_SHARE_MID - pct) / _DEATH_SHARE_SPAN)


def score_peer_gap(
    *,
    metric: str,
    delta_pct: float | None,
    delta: float,
    direction: str,
) -> float | None:
    """Signed peer gap where positive means better than peers."""
    if delta_pct is not None:
        gap = float(delta_pct)
    else:
        span = _PEER_RAW_SPANS.get(metric)
        if span is None or span == 0:
            return None
        gap = float(delta) / span * 100.0
    if direction == "lower":
        gap = -gap
    return _clamp(gap / _PEER_GAP_SPAN)


def score_form_delta(metric: str, improvement: float) -> float | None:
    """Signed recent-vs-baseline impact for Form Tracker bars and colors."""
    if metric in {"gd10", "cs10", "gd15", "xpd10", "csd10"}:
        return score_lane_diff(improvement)
    if metric in {"win", "kill_participation", "damage_share", "objectives_present_rate"}:
        return _clamp(improvement * 100 / _FORM_RATE_SPAN)
    if metric.endswith("_rate"):
        return _clamp(improvement * 100 / _FORM_RATE_SPAN)
    span = _FORM_RAW_SPANS.get(metric)
    if span is None:
        return None
    return _clamp(improvement / span)


def color_winrate(rate: float) -> str:
    """Color for a win-rate ratio in [0, 1]."""
    return interpolate_metric_color(score_winrate(rate * 100.0))


def colors_for_winrates(rates: list[float]) -> list[str]:
    return [color_winrate(rate) for rate in rates]
