"""Dashboard card and peer-row view models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from league_stats.core.models import PeerComparisonResult
from league_stats.presentation.ui_icons import icon_fields_for_label, with_icons


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


def card(value: Any, suffix: str = "") -> str:
    """Format a possibly-missing metric for a dashboard card."""
    return "—" if value is None else f"{value}{suffix}"


def pct(value: float | None) -> str | None:
    """Format a ratio as a percentage string, keeping ``None``."""
    return None if value is None else f"{value * 100:.0f}%"


def card_entries(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Convert label/value card pairs to JSON-friendly dicts."""
    return with_icons([{"label": label, "value": value} for label, value in pairs])


def overview_card_entries(overview: dict[str, Any]) -> list[dict[str, str]]:
    """Build overview cards with win/loss styling metadata."""
    winrate = float(overview.get("winrate", 0.0))
    return with_icons(
        [
            {
                "label": "Win rate",
                "value": f"{winrate * 100:.0f}%",
                "value_class": "win" if winrate >= 0.5 else "loss",
            },
            {"label": "KDA", "value": str(overview.get("avg_kda", "—")), "value_class": ""},
            {"label": "DPM", "value": str(overview.get("avg_dpm", "—")), "value_class": ""},
            {"label": "CS/min", "value": str(overview.get("avg_cspm", "—")), "value_class": ""},
            {
                "label": "Damage share",
                "value": f"{float(overview.get('avg_damage_share', 0)) * 100:.0f}%",
                "value_class": "",
            },
            {"label": "Deaths/game", "value": str(overview.get("avg_deaths", "—")), "value_class": ""},
            {"label": "Vision/min", "value": str(overview.get("avg_vspm", "—")), "value_class": ""},
            {
                "label": "Avg game",
                "value": f"{overview.get('avg_duration', '—')} min",
                "value_class": "",
            },
        ]
    )


def peer_row_display(row: dict[str, Any]) -> dict[str, str]:
    """Format peer comparison row values for HTML/JSON."""
    metric = row.get("metric")
    yours = row.get("yours")
    peer_avg = row.get("peer_avg")
    if metric in {"win", "kill_participation", "damage_share"}:
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
    return {
        "label": str(row.get("label", "")),
        **icon_fields_for_label(str(row.get("label", ""))),
        "yours": yours_display,
        "peer_avg": peer_display,
        "gap": gap_display,
        "verdict": str(row.get("verdict", "inline")),
    }


def peer_subtitle(peer: PeerComparisonResult) -> str:
    """Build the peer comparison subtitle with sample size and confidence."""
    confidence = peer.confidence.replace("_", " ")
    return (
        f"Your averages vs {peer.build_label} at {peer.rank_label} · "
        f"{peer.source} · {peer.peer_games} peer games "
        f"({peer.peer_players} players, {confidence} confidence)"
    )
