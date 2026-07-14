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
    "target": "lucide:target",
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
    "tower": "tower.png",
}

METRIC_ICONS: dict[str, str] = {
    # Overview
    "Win rate": "trophy",
    "KDA": "combat",
    "DPM": "flame",
    "CC/min": "target",
    "CS/min": "cs",
    "Damage share": "flame",
    "Deaths/game": "skull",
    "Vision/min": "eye",
    "Avg game": "clock",
    "Kill participation": "kp",
    "Obj. presence": "target",
    "Gold @10": "coin",
    "Early ganks": "roam",
    "Gank assists": "teamfight",
    "KP @15": "kp",
    "Roam conversions": "roam",
    "Dist to ADC": "lane",
    "Grouped share": "teamfight",
    "VS/min @10": "eye",
    "Bot lane presence": "lane",
    "Assists/game": "teamfight",
    "Healing/min": "teamfight",
    "Shielding/min": "teamfight",
    "Caught while farming": "skull",
    "Overextended roams": "roam",
    "Facechecks / deep wards": "ward",
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
    # Positioning
    "Grouped with team": "teamfight",
    "Solo on map": "roam",
    "Side-lane time": "lane",
    "Allies nearby": "teamfight",
    "Avg teammate dist": "roam",
    "Dist to top": "lane",
    "Dist to jungle": "roam",
    "Dist to mid": "lane",
    "Dist to bot": "lane",
    "Dist to support": "teamfight",
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
    "Before neutral obj.": "target",
    "Gold at death": "coin",
    "Outnumbered deaths": "skull",
    "Avg death minute": "clock",
    "Top killer": "combat",
    # Teamfights
    "Fights detected": "teamfight",
    "Participation": "kp",
    "Fight win rate": "trophy",
    "Damage/fight": "flame",
    "Death rate in fights": "skull",
    "Front-to-back": "teamfight",
    "Unspent gold/fight": "coin",
    "Advantaged fights": "kp",
    "Disadvantaged fights": "skull",
    "WR advantaged fights": "trophy",
    "WR disadvantaged fights": "trophy",
    # Peer / score dimensions
    "Laning": "coin",
    "Farming": "cs",
    "Survival": "skull",
    "Damage": "flame",
    "CC impact": "target",
    "Utility": "teamfight",
    "Impact": "kp",
    "Map control": "target",
    "Clear @10": "cs",
    "Setup": "roam",
    "Vision": "eye",
    "Objectives": "target",
    "Resets": "recall",
    "Strengths": "trophy",
    "Weaknesses": "skull",
    "Kill participation": "kp",
}

METRIC_TOOLTIPS: dict[str, str] = {
    # Overview
    "Win rate": "Wins divided by total games in the selected window.",
    "KDA": "Average (kills + assists) ÷ deaths per game. Deaths are floored at 1.",
    "DPM": "Average damage to champions per minute: total champ damage ÷ game length.",
    "CC/min": "Average crowd-control time per minute: total seconds CCing enemies ÷ game length.",
    "CS/min": "Average creep score per minute (lane + jungle minions from the match summary).",
    "Damage share": "Average share of your team's total damage to champions each game.",
    "Deaths/game": "Average deaths per game in the window.",
    "Vision/min": "Average vision score per minute (Riot vision score ÷ game length).",
    "Avg game": "Average match duration in minutes.",
    # Lane
    "Gold diff @10": "Your total gold minus your lane opponent's at the 10-minute timeline frame, averaged across games.",
    "CS diff @10": "Your CS minus your lane opponent's at minute 10 (lane + jungle minions).",
    "XP diff @10": "Your XP minus your lane opponent's at the 10-minute timeline frame.",
    "Lane win rate": "Share of games where gold diff @10 is positive.",
    "WR when ahead @10": "Win rate in games where you were ahead in gold at 10 minutes.",
    "WR when behind @10": "Win rate in games where gold diff @10 was negative at 10 minutes.",
    "Deaths pre-14": "Average deaths before minute 14 (end of the laning phase).",
    "Gank deaths (lane)": "Deaths before 14 min in a lane where the killer or an assist was not your lane opponent (e.g. jungler gank).",
    "Under own tower (lane)": "Deaths during the laning phase while near your lane tower.",
    "Under enemy tower (lane)": "Deaths during the laning phase while near the enemy lane tower.",
    "Roams pre-15": "Average roams detected before minute 15 (timeline position shifts away from your lane).",
    # Positioning
    "Grouped with team": "Share of mid/late frames (after 14 min, excluding base) where at least two allies are within 3000 map units.",
    "Solo on map": "Share of those frames with no allies within 3000 map units.",
    "Side-lane time": "Share of mid/late frames spent in a side lane (top or bot, not mid).",
    "Allies nearby": "Average number of allies within 3000 map units per mid/late frame.",
    "Avg teammate dist": "Mean distance to all teammates per frame during mid/late game, in map units. Lower means closer.",
    # Economy
    "GPM": "Gold per minute: total gold earned ÷ game length.",
    "Gold share": "Your gold as a share of team total gold each game.",
    "Damage per gold": "Damage to champions divided by gold earned, averaged per game.",
    "Unspent gold/recall": "Average gold banked on the timeline frame before each inferred recall (burst of item purchases).",
    "First recall": "Average game minute of your first inferred recall.",
    "Time dead/game": "Average seconds spent on death timers per game.",
    # Vision
    "Vision score": "Average Riot vision score per game.",
    "VS/min": "Vision score per minute.",
    "Control wards": "Average control wards bought per game.",
    "CW lifetime": "Average seconds each control ward stayed alive until cleared or game end.",
    "VS/min in wins": "Vision score per minute averaged over wins only.",
    "VS/min in losses": "Vision score per minute averaged over losses only.",
    # Objectives
    "Wards before": "Average wards you placed in the 2 minutes before each objective take. Any ward type counts; map location is not filtered.",
    # Deaths
    "Total deaths": "Total death count across all games in the window (not an average).",
    "Solo deaths": "Share of deaths with no allies within 2200 map units.",
    "Greed deaths": "Share of deaths shortly after deep side-lane pushing without nearby allies.",
    "Side-lane deaths": "Share of deaths while isolated in a side lane after minute 14.",
    "Before dragon": "Share of deaths within 60 seconds before a dragon take.",
    "Gold at death": "Average gold in your inventory at the moment of death.",
    "Outnumbered deaths": "Share of deaths where nearby enemies outnumbered nearby allies.",
    "Avg death minute": "Mean game minute when you died.",
    "Top killer": "Champion that secured the most kills on you in this window.",
    # Teamfights
    "Fights detected": "Clusters of at least three kills within 25 seconds and 4000 map units of each other.",
    "Participation": "Share of detected fights where you killed, assisted, died, or were within 3000 units of the fight.",
    "Fight win rate": "Share of joined fights your team won (more ally than enemy kills in the cluster).",
    "Damage/fight": "Average damage to champions you dealt in fights you joined.",
    "Death rate in fights": "Share of joined fights where you died.",
    "Front-to-back": "Your map push depth minus allies' average at fight start; positive means you were further forward.",
    "Unspent gold/fight": "Average gold in your inventory when a joined fight started.",
    "Advantaged fights": "Joined fights where your team had more nearby champions than the enemy.",
    "Disadvantaged fights": "Joined fights where the enemy had more nearby champions.",
    "WR advantaged fights": "Win rate in joined fights where you had a manpower advantage nearby.",
    "WR disadvantaged fights": "Win rate in joined fights where you were outnumbered nearby.",
    # Improvement score dimensions
    "Laning": "0–100 score from average gold diff @10, mapped between −800g and +800g vs your lane opponent.",
    "Farming": "0–100 score from CS @10 against a role-specific benchmark band.",
    "Survival": "0–100 score from deaths per game; fewer deaths score higher.",
    "Damage": "0–100 score from your average share of team damage to champions.",
    "Vision": "0–100 score from vision score per minute against role benchmarks.",
    "Objectives": "0–100 score from your presence rate at epic monster takes.",
    "Resets": "0–100 score from gold banked before recalls; hoarding 1500g+ before backing scores lower.",
    "Utility": "0–100 composite of CC/min, damage share, healing/min and shielding/min vs support benchmarks.",
    "Impact": "0–100 score from kill participation vs role benchmarks.",
    "Map control": "0–100 score from objective presence at epic monster takes.",
    "Clear @10": "0–100 score from jungle CS @10 vs role clear-speed benchmarks.",
    "Setup": "0–100 score from early roams before 15 minutes.",
    "CC impact": "0–100 score from crowd control time per minute.",
}

DIST_TO_TOOLTIP = (
    "Average distance to your {role} teammate during mid/late game (after 14 min), "
    "from 60-second timeline frames. Lower means you stay closer."
)

SECTION_ICONS: dict[str, str] = {
    "overview": "chart",
    "score": "chart",
    "rank-peers": "chart",
    "lane": "lane",
    "economy": "coin",
    "vision": "eye",
    "deaths": "skull",
    "positioning": "roam",
    "teamfights": "teamfight",
    "objectives": "target",
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
    "target": "gold",
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


def tooltip_for_label(label: str) -> str | None:
    """Return a short calculation note for a metric card label, if defined."""
    if label in METRIC_TOOLTIPS:
        return METRIC_TOOLTIPS[label]
    if label.startswith("Dist to "):
        role = label.removeprefix("Dist to ")
        return DIST_TO_TOOLTIP.format(role=role)
    return None


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
    """Attach icon metadata and an optional calculation tooltip to a card dict."""
    enriched = dict(card)
    label = str(enriched.get("label", ""))
    enriched.update(icon_fields_for_label(label))
    tooltip = tooltip_for_label(label)
    if tooltip:
        enriched["tooltip"] = tooltip
    return enriched


def with_icons(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach icon metadata to every card dict."""
    return [with_icon(card) for card in cards]
