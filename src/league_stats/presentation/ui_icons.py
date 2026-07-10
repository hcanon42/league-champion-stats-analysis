"""Map dashboard labels to Iconify icon ids (Lucide + game-icons)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Internal keys -> Iconify ids (https://icon-sets.iconify.design/)
ICONIFY_ICONS: dict[str, str] = {
    "coin": "lucide:coins",
    "minion": "lucide:wheat",
    "skull": "lucide:skull",
    "eye": "lucide:eye",
    "ward": "lucide:shield",
    "trophy": "lucide:trophy",
    "combat": "lucide:swords",
    "flame": "lucide:flame",
    "clock": "lucide:clock",
    "dragon": "game-icons:dragon-head",
    "teamfight": "lucide:users",
    "tower": "lucide:castle",
    "roam": "lucide:footprints",
    "recall": "lucide:home",
    "level": "lucide:trending-up",
    "lane": "lucide:map",
    "chart": "lucide:bar-chart-2",
    "wand": "lucide:wand-sparkles",
    "items": "lucide:wand-sparkles",
    "rune": "lucide:sparkles",
    "bulb": "lucide:lightbulb",
    "kp": "lucide:users-round",
}

# Internal keys backed by local PNG assets instead of Iconify.
ICON_ASSET_FILES: dict[str, str] = {
    "cs": "minions.png",
}

METRIC_ICONS: dict[str, str] = {
    # Overview
    "Win rate": "trophy",
    "KDA": "combat",
    "DPM": "flame",
    "CS/min": "cs",
    "Damage share": "flame",
    "Deaths/game": "skull",
    "Vision/min": "eye",
    "Avg game": "clock",
    # Lane
    "Gold diff @10": "coin",
    "CS diff @10": "cs",
    "XP diff @10": "level",
    "Lane win rate": "trophy",
    "WR when ahead @10": "trophy",
    "WR when behind @10": "trophy",
    "Deaths pre-14": "skull",
    "Gank deaths (lane)": "skull",
    "Under own tower (lane)": "tower",
    "Under enemy tower (lane)": "tower",
    "Roams pre-15": "roam",
    # Economy
    "GPM": "coin",
    "Gold share": "coin",
    "Damage per gold": "flame",
    "Unspent gold/recall": "coin",
    "First recall": "recall",
    "Time dead/game": "clock",
    # Vision
    "Vision score": "eye",
    "VS/min": "eye",
    "Control wards": "ward",
    "CW lifetime": "ward",
    "VS/min in wins": "eye",
    "VS/min in losses": "eye",
    # Deaths
    "Total deaths": "skull",
    "Solo deaths": "skull",
    "Greed deaths": "skull",
    "Side-lane deaths": "skull",
    "Before dragon": "dragon",
    "Avg death minute": "clock",
    "Top killer": "combat",
    # Teamfights
    "Fights detected": "teamfight",
    "Participation": "kp",
    "Fight win rate": "trophy",
    "Damage/fight": "flame",
    "Death rate in fights": "skull",
    "Front-to-back": "teamfight",
    # Peer / score dimensions
    "Laning": "coin",
    "Farming": "cs",
    "Survival": "skull",
    "Damage": "flame",
    "Vision": "eye",
    "Objectives": "dragon",
    "Resets": "recall",
    "Strengths": "trophy",
    "Weaknesses": "skull",
    "Kill participation": "kp",
}

SECTION_ICONS: dict[str, str] = {
    "overview": "chart",
    "score": "chart",
    "rank-peers": "chart",
    "lane": "lane",
    "economy": "coin",
    "vision": "eye",
    "deaths": "skull",
    "teamfights": "teamfight",
    "objectives": "dragon",
    "items": "wand",
    "runes": "rune",
    "matchups": "combat",
    "graphs": "chart",
    "recommendations": "bulb",
}

ICON_TONES: dict[str, str] = {
    "coin": "gold",
    "minion": "green",
    "cs": "green",
    "skull": "danger",
    "eye": "blue",
    "ward": "blue",
    "trophy": "win",
    "combat": "accent",
    "flame": "orange",
    "clock": "muted",
    "dragon": "gold",
    "teamfight": "accent",
    "tower": "muted",
    "roam": "accent",
    "recall": "gold",
    "level": "green",
    "lane": "accent",
    "chart": "accent",
    "wand": "gold",
    "items": "gold",
    "rune": "accent",
    "bulb": "win",
    "kp": "accent",
}


def icon_for_label(label: str) -> str | None:
    """Return an internal icon key for a metric card label, if mapped."""
    return METRIC_ICONS.get(label)


def icon_for_objective(kind: str) -> str | None:
    """Return an internal objective icon key when a scoreboard asset exists."""
    normalized = str(kind).strip().lower()
    if normalized in {"dragon", "elder", "baron", "herald", "grubs"}:
        return normalized
    return None


def icon_for_section(section_id: str) -> str | None:
    """Return an internal icon key for a report section id."""
    return SECTION_ICONS.get(section_id)


def icon_fields_for_label(label: str) -> dict[str, Any]:
    """Resolve icon metadata for a metric label."""
    icon_key = icon_for_label(label)
    if not icon_key:
        return {"icon": None, "iconify": None, "icon_tone": "muted"}
    if icon_key in ICON_ASSET_FILES:
        return {
            "icon": icon_key,
            "icon_asset": ICON_ASSET_FILES[icon_key],
            "iconify": None,
            "icon_tone": icon_tone(icon_key),
        }
    return {
        "icon": icon_key,
        "iconify": iconify_for_key(icon_key),
        "icon_tone": icon_tone(icon_key),
    }


def attach_metric_icon_hrefs(
    entries: list[dict[str, Any]],
    assets: Any,
    *,
    from_dir: Path,
) -> list[dict[str, Any]]:
    """Attach relative ``icon_href`` URLs for metric rows using local PNG assets."""
    for entry in entries:
        asset_file = entry.get("icon_asset")
        if not asset_file or assets is None:
            continue
        href = assets.ui_icon_href(str(asset_file), from_dir=from_dir)
        if href:
            entry["icon_href"] = href
    return entries


def iconify_for_key(icon_key: str | None) -> str | None:
    """Resolve an internal icon key to an Iconify icon id."""
    if not icon_key:
        return None
    return ICONIFY_ICONS.get(icon_key)


def icon_tone(icon_key: str | None) -> str:
    """CSS tone class suffix for an icon key."""
    if not icon_key:
        return "muted"
    return ICON_TONES.get(icon_key, "muted")


def with_icon(card: dict[str, Any]) -> dict[str, Any]:
    """Attach ``icon``, ``iconify`` and ``icon_tone`` to a card dict when mapped."""
    enriched = dict(card)
    enriched.update(icon_fields_for_label(str(enriched.get("label", ""))))
    return enriched


def with_icons(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach icon metadata to every card dict."""
    return [with_icon(card) for card in cards]
