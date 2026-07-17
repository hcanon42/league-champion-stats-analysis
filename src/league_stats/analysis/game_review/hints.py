"""Tooltip and label helpers for Game Review UI."""

from __future__ import annotations

from league_stats.presentation.ui_icons import METRIC_TOOLTIPS, tooltip_for_label

GAME_REVIEW_KEY_STATS: dict[str, tuple[str, str]] = {
    "gd10": ("Gold diff @10", "Gold diff @10"),
    "gd15": ("Gold diff @15", "Your total gold minus lane opponent's at 15 minutes."),
    "deaths": ("Deaths", "Deaths/game"),
    "deaths_pre14": ("Deaths pre-14", "Deaths pre-14"),
    "dpm": ("DPM", "DPM"),
    "kill_participation": ("Kill participation", "Kill participation"),
    "damage_share": ("Damage share", "Damage share"),
    "gold_share": ("Gold share", "Gold share"),
    "vspm": ("VS/min", "VS/min"),
    "control_wards": ("Control wards", "Control wards"),
    "objectives_present_rate": ("Obj. presence", "Obj. presence"),
    "solo_deaths": ("Solo deaths", "Solo deaths"),
    "greed_deaths": ("Greed deaths", "Greed deaths"),
    "fights_disadvantaged": ("Disadvantaged fights", "Disadvantaged fights"),
}

GAME_REVIEW_SCORE_DIMENSIONS: tuple[str, ...] = (
    "Laning",
    "Survival",
    "Impact",
    "Vision",
    "Objectives",
)

OBJECTIVE_DEAD_SETUP_LABEL = "Died in setup window (45s)"

OBJECTIVE_COLUMN_TOOLTIPS: dict[str, str] = {
    "Taken": "Whether your team secured this epic monster.",
    "Present": "You were within range when the objective was taken.",
    OBJECTIVE_DEAD_SETUP_LABEL: (
        "You died within 45 seconds before this objective was taken "
        "(even if you respawned and arrived in time)."
    ),
    "Wards before": METRIC_TOOLTIPS["Wards before"],
}


def _tooltip(label: str) -> str | None:
    return tooltip_for_label(label) or METRIC_TOOLTIPS.get(label)


def game_review_tooltips() -> dict[str, dict[str, str]]:
    """Tooltip map embedded in the report for Game Review panels."""
    score: dict[str, str] = {}
    for label in (*GAME_REVIEW_SCORE_DIMENSIONS, "Game score"):
        hint = _tooltip(label)
        if hint:
            score[label] = hint
    score["Survival"] = (
        "0–100 score from deaths normalized to a 30-minute game; fewer deaths score higher."
    )
    score["Game score"] = (
        "Personal performance vs your baseline for this single game (0–100, letter tier). "
        "Independent of win/loss."
    )
    key_stats: dict[str, str] = {}
    key_stats_labels: dict[str, str] = {}
    for column, (label, tooltip_key) in GAME_REVIEW_KEY_STATS.items():
        key_stats_labels[column] = label
        hint = _tooltip(tooltip_key) or _tooltip(label)
        if hint:
            key_stats[column] = hint
    return {
        "score": score,
        "key_stats": key_stats,
        "key_stats_labels": key_stats_labels,
        "objectives": OBJECTIVE_COLUMN_TOOLTIPS,
        "key_moments": {
            "interpolation": (
                "Minute snapshots from just before the action through just after, "
                "including any minute marks in between. Riot records all ten players "
                "once per minute. Objective icons are bright when up, dim when taken."
            ),
        },
    }
