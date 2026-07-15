"""Rule-based per-game behavior bullets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from league_stats.core.config import GAME_REVIEW_MAX_BEHAVIORS
from league_stats.core.models import GameBehavior, MatchRecord
from league_stats.presentation.metric_colors import normalize_deaths_for_duration

_DEATH_FLAG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("alone", "Solo death"),
    ("after_greed", "Greed death"),
    ("before_neutral_objective", "Dead before objective"),
    ("to_gank", "Gank death"),
    ("outnumbered", "Outnumbered death"),
    ("before_dragon", "Dead before dragon"),
    ("before_baron", "Dead before baron"),
)


@dataclass(frozen=True)
class _Candidate:
    tone: Literal["positive", "negative"]
    title: str
    detail: str
    priority: float
    anchor: str | None = None


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def _baseline(baseline_means: dict[str, float], key: str, default: float = 0.0) -> float:
    return float(baseline_means.get(key, default))


def evaluate_behaviors(
    record: MatchRecord,
    game_row: dict[str, Any],
    deaths_rows: list[dict[str, Any]],
    *,
    baseline_means: dict[str, float],
    archetype: str,
) -> tuple[list[GameBehavior], list[GameBehavior]]:
    """Return up to MAX good and MAX bad behavior bullets for one game."""
    candidates: list[_Candidate] = []
    duration = float(game_row.get("duration_min") or record.duration_min)

    gd10 = _num(game_row, "gd10")
    gd15 = _num(game_row, "gd15")
    deaths = int(game_row.get("deaths") or 0)
    deaths_pre14 = int(game_row.get("deaths_pre14") or 0)
    vspm = _num(game_row, "vspm")
    control_wards = int(game_row.get("control_wards") or 0)
    obj_rate = _num(game_row, "objectives_present_rate")
    fights_disadv = int(game_row.get("fights_disadvantaged") or 0)
    solo_deaths = int(game_row.get("solo_deaths") or 0)

    base_gd10 = _baseline(baseline_means, "gd10")
    base_deaths = _baseline(baseline_means, "deaths", 4.0)
    base_duration = _baseline(baseline_means, "duration_min", duration)
    norm_deaths = normalize_deaths_for_duration(deaths, duration)
    norm_base_deaths = normalize_deaths_for_duration(base_deaths, base_duration)
    base_vspm = _baseline(baseline_means, "vspm", 1.0)
    base_obj = _baseline(baseline_means, "objectives_present_rate", 0.5)

    if gd10 >= base_gd10 + 200:
        candidates.append(
            _Candidate(
                "positive",
                "Strong laning",
                f"GD@10 {gd10:+.0f} vs your avg {base_gd10:+.0f}.",
                8.0 + abs(gd10 - base_gd10) / 100,
                "lane",
            )
        )
    elif gd10 <= base_gd10 - 200:
        candidates.append(
            _Candidate(
                "negative",
                "Weak laning",
                f"GD@10 {gd10:+.0f} vs your avg {base_gd10:+.0f}.",
                8.0 + abs(gd10 - base_gd10) / 100,
                "lane",
            )
        )

    if solo_deaths == 0 and norm_deaths <= max(1.0, norm_base_deaths - 1.0):
        candidates.append(
            _Candidate(
                "positive",
                "Clean survival",
                f"{deaths} deaths in {duration:.0f}m with no solo deaths.",
                7.0,
                "deaths",
            )
        )

    if norm_deaths >= norm_base_deaths + 2.0:
        candidates.append(
            _Candidate(
                "negative",
                "High death count",
                f"{deaths} deaths in {duration:.0f}m vs your avg {base_deaths:.1f}/game.",
                7.5 + norm_deaths - norm_base_deaths,
                "deaths",
            )
        )

    if deaths_pre14 >= 3 or deaths_pre14 >= int(_baseline(baseline_means, "deaths_pre14", 2)) + 2:
        candidates.append(
            _Candidate(
                "negative",
                "Early deaths",
                f"{deaths_pre14} deaths before 14 minutes.",
                7.0 + deaths_pre14,
                "deaths",
            )
        )

    if obj_rate >= 0.75 or obj_rate >= base_obj + 0.15:
        candidates.append(
            _Candidate(
                "positive",
                "Objective presence",
                f"Present for {obj_rate * 100:.0f}% of epic objectives.",
                6.5,
                "objectives",
            )
        )
    elif obj_rate < 0.35 and base_obj >= 0.4:
        candidates.append(
            _Candidate(
                "negative",
                "Low objective presence",
                f"Only {obj_rate * 100:.0f}% objective presence.",
                6.0,
                "objectives",
            )
        )

    if vspm >= base_vspm * 1.15:
        candidates.append(
            _Candidate(
                "positive",
                "Strong vision",
                f"{vspm:.1f} VS/min vs your avg {base_vspm:.1f}.",
                6.0,
                "vision",
            )
        )
    elif control_wards == 0 and duration >= 25:
        candidates.append(
            _Candidate(
                "negative",
                "No control wards",
                f"0 control wards in a {duration:.0f}-minute game.",
                5.5,
                "vision",
            )
        )

    if archetype == "Comeback win":
        candidates.append(
            _Candidate(
                "positive",
                "Comeback win",
                f"Won despite being behind at 15 ({gd15:+.0f} GD@15).",
                9.0,
            )
        )
    elif archetype == "Throw":
        candidates.append(
            _Candidate(
                "negative",
                "Throw",
                f"Lost with a {gd15:+.0f} gold lead at 15 minutes.",
                9.5,
                "lane",
            )
        )
    elif archetype == "Lane stomp win":
        candidates.append(
            _Candidate(
                "positive",
                "Lane stomp win",
                f"Dominant laning (+{gd15:.0f} GD@15) converted to a win.",
                8.5,
                "lane",
            )
        )
    elif archetype == "One-sided loss":
        candidates.append(
            _Candidate(
                "negative",
                "One-sided loss",
                f"Fell behind early ({gd15:+.0f} GD@15) and never recovered.",
                7.0,
            )
        )

    greed_minutes: list[float] = []
    objective_minutes: list[float] = []
    for row in deaths_rows:
        minute = float(row.get("minute") or 0)
        if row.get("after_greed"):
            greed_minutes.append(minute)
        if row.get("before_neutral_objective"):
            objective_minutes.append(minute)

    if greed_minutes:
        label = ", ".join(f"{m:.0f}m" for m in greed_minutes[:3])
        candidates.append(
            _Candidate(
                "negative",
                "Greed deaths",
                f"{len(greed_minutes)} greed death(s) at {label}.",
                8.0 + len(greed_minutes),
                "deaths",
            )
        )

    if objective_minutes:
        label = ", ".join(f"{m:.0f}m" for m in objective_minutes[:3])
        candidates.append(
            _Candidate(
                "negative",
                "Died in setup window",
                f"Died within 45s before an objective at {label}.",
                8.5 + len(objective_minutes),
                "objectives",
            )
        )

    if fights_disadv >= 2:
        candidates.append(
            _Candidate(
                "negative",
                "Disadvantaged fights",
                f"Joined {fights_disadv} fight(s) while outnumbered.",
                6.5 + fights_disadv,
                "teamfights",
            )
        )

    unspent = game_row.get("avg_unspent_gold")
    if unspent is not None and float(unspent) >= 1400:
        candidates.append(
            _Candidate(
                "negative",
                "Gold hoarding",
                f"Avg {float(unspent):.0f}g unspent before recalls.",
                5.5,
                "economy",
            )
        )

    good = sorted(
        (c for c in candidates if c.tone == "positive"),
        key=lambda c: c.priority,
        reverse=True,
    )[:GAME_REVIEW_MAX_BEHAVIORS]
    bad = sorted(
        (c for c in candidates if c.tone == "negative"),
        key=lambda c: c.priority,
        reverse=True,
    )[:GAME_REVIEW_MAX_BEHAVIORS]

    def to_model(items: list[_Candidate]) -> list[GameBehavior]:
        return [
            GameBehavior(tone=item.tone, title=item.title, detail=item.detail, anchor=item.anchor)
            for item in items
        ]

    return to_model(good), to_model(bad)
