"""AI coach: rule-based recommendations ranked by statistical evidence.

Each rule inspects the aggregated data, and when a pattern is both material
(large effect) and statistically supported (low p-value where testable),
emits a :class:`~models.Recommendation`. Recommendations are prioritised by
a combination of effect size and significance.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from scipy import stats as scipy_stats

from analysis.statistics import StatisticsEngine
from models import Recommendation
from utils import get_logger

MIN_GAMES: int = 10
SIGNIFICANT_P: float = 0.05
SUGGESTIVE_P: float = 0.15


def _priority(effect: float, p_value: float | None, sample: int) -> float:
    """Score a recommendation for ranking.

    Larger effects, smaller p-values and bigger samples rank higher.

    Args:
        effect: Effect size on a roughly [0, 1] scale (e.g. win-rate delta).
        p_value: Significance when testable, else ``None``.
        sample: Number of games/events backing the insight.

    Returns:
        A priority score (higher = more important).
    """
    significance = 1.0 if p_value is None else max(0.0, 1.0 - min(1.0, p_value / SUGGESTIVE_P))
    volume = min(1.0, sample / 40.0)
    return round(abs(effect) * 2.0 + significance + volume, 3)


class CoachEngine:
    """Generates and ranks coaching recommendations."""

    def __init__(
        self,
        matches_df: pd.DataFrame,
        deaths_df: pd.DataFrame,
        matchups_df: pd.DataFrame,
        objectives_df: pd.DataFrame,
        stats_engine: StatisticsEngine,
        *,
        build_label: str = "Viktor mid",
    ) -> None:
        """Wire the coach with the aggregated views (dependency injection).

        Args:
            matches_df: Master per-game table.
            deaths_df: Per-death table.
            matchups_df: Per-opponent table.
            objectives_df: Per-objective table.
            stats_engine: Shared statistics engine.
            build_label: Champion + lane label for personalised messaging.
        """
        self._matches = matches_df
        self._deaths = deaths_df
        self._matchups = matchups_df
        self._objectives = objectives_df
        self._stats = stats_engine
        self._build_label = build_label
        self._champion = build_label.split(" ", 1)[0]
        self._log = get_logger("coach")

    def generate(self) -> list[Recommendation]:
        """Run every rule and return recommendations sorted by priority.

        Returns:
            Ranked recommendations (best first).
        """
        if len(self._matches) < 2:
            return []
        rules: list[Callable[[], Recommendation | None]] = [
            self._rule_early_deaths,
            self._rule_unspent_gold,
            self._rule_first_item_timing,
            self._rule_control_wards,
            self._rule_side_lane_deaths,
            self._rule_best_matchup,
            self._rule_worst_matchup,
            self._rule_deaths_before_dragon,
            self._rule_objective_presence,
            self._rule_solo_deaths,
            self._rule_cs10,
            self._rule_lane_priority,
        ]
        recommendations: list[Recommendation] = []
        for rule in rules:
            try:
                result = rule()
            except Exception as exc:  # a single broken rule must not kill the report
                self._log.warning("Coach rule %s failed: %s", rule.__name__, exc)
                continue
            if result is not None:
                recommendations.append(result)
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)

    # ----------------------------------------------------------------- Rules

    def _rule_early_deaths(self) -> Recommendation | None:
        """Win rate impact of dying before 20 minutes."""
        split = self._stats.winrate_split_test("deaths_pre20", 2)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Deaths",
            title="Dying early is costing you games",
            detail=(
                f"You lose {round((1 - split['winrate_high']) * 100)}% of games where you die "
                f"2+ times before 20 minutes, versus "
                f"{round((1 - split['winrate_low']) * 100)}% otherwise. Play the early game "
                "for stability: track the enemy jungler and don't contest without priority."
            ),
            evidence=(
                f"WR {split['winrate_high']:.0%} ({split['n_high']} games with 2+ early deaths) "
                f"vs {split['winrate_low']:.0%} ({split['n_low']} games)"
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_unspent_gold(self) -> Recommendation | None:
        """Flag consistently large gold reserves before recalls."""
        series = pd.to_numeric(self._matches.get("avg_unspent_gold"), errors="coerce").dropna()
        if len(series) < 5:
            return None
        avg = float(series.mean())
        if avg < 650:
            return None
        return Recommendation(
            category="Economy",
            title="Too much gold sitting unspent",
            detail=(
                f"You average {avg:.0f} unspent gold when you recall. That's a permanent item "
                "deficit versus your opponent - plan resets around component breakpoints "
                "(1100 for Lost Chapter, 900 for Blasting Wand)."
            ),
            evidence=f"Mean banked gold before recall: {avg:.0f}g over {len(series)} games",
            p_value=None,
            effect_size=round(min(1.0, (avg - 500) / 1000), 3),
            priority=_priority(min(1.0, (avg - 500) / 1000), None, len(series)),
            sample_size=len(series),
        )

    def _rule_first_item_timing(self) -> Recommendation | None:
        """Test whether slow first items correlate with losing."""
        frame = self._matches[["first_item_min", "win"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna()
        if len(frame) < MIN_GAMES:
            return None
        wins = frame[frame["win"] == 1]["first_item_min"]
        losses = frame[frame["win"] == 0]["first_item_min"]
        if len(wins) < 3 or len(losses) < 3:
            return None
        stat_result = scipy_stats.mannwhitneyu(wins, losses, alternative="two-sided")
        gap = float(losses.mean() - wins.mean())
        if gap < 0.5:
            return None
        return Recommendation(
            category="Items",
            title="Slow first item completions correlate with losses",
            detail=(
                f"Your first item lands {gap:.1f} minutes later in losses "
                f"({losses.mean():.1f} min) than in wins ({wins.mean():.1f} min). "
                "Tighten your first two resets to protect your power spike."
            ),
            evidence=(
                f"First item at {wins.mean():.1f} min in wins vs {losses.mean():.1f} min in "
                f"losses (Mann-Whitney p={stat_result.pvalue:.3f})"
            ),
            p_value=round(float(stat_result.pvalue), 5),
            effect_size=round(min(1.0, gap / 4), 3),
            priority=_priority(min(1.0, gap / 4), float(stat_result.pvalue), len(frame)),
            sample_size=len(frame),
        )

    def _rule_control_wards(self) -> Recommendation | None:
        """Correlate control ward purchases with winning."""
        frame = self._matches[["control_wards", "win"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna()
        if len(frame) < MIN_GAMES or frame["control_wards"].nunique() < 2:
            return None
        corr, p_value = scipy_stats.pointbiserialr(frame["win"], frame["control_wards"])
        if corr < 0.15 or p_value > SUGGESTIVE_P:
            return None
        wins_avg = frame[frame["win"] == 1]["control_wards"].mean()
        losses_avg = frame[frame["win"] == 0]["control_wards"].mean()
        return Recommendation(
            category="Vision",
            title="Control wards are winning you games",
            detail=(
                f"You buy {wins_avg:.1f} control wards in wins but only {losses_avg:.1f} in "
                "losses, and the correlation with winning is positive. Make the control ward "
                "part of every recall after the laning phase."
            ),
            evidence=f"Point-biserial r={corr:.2f}, p={p_value:.3f}, n={len(frame)}",
            p_value=round(float(p_value), 5),
            effect_size=round(float(corr), 3),
            priority=_priority(float(corr), float(p_value), len(frame)),
            sample_size=len(frame),
        )

    def _rule_side_lane_deaths(self) -> Recommendation | None:
        """Flag frequent deep side-lane deaths in the mid/late game."""
        if self._deaths.empty:
            return None
        side = self._deaths[self._deaths["side_lane_push"]]
        games = self._matches["match_id"].nunique()
        if games == 0 or len(side) / games < 0.5:
            return None
        late = side[side["minute"] >= 22]
        loss_share = float((side["win"] == 0).mean()) if len(side) else 0.0
        return Recommendation(
            category="Macro",
            title="You die too often pushing side lanes",
            detail=(
                f"{len(side)} deaths came from side-lane pushes ({len(late)} after 22 min), and "
                f"{loss_share:.0%} of them happened in games you lost. After 22 minutes, only "
                "take a side wave with vision, tempo and Teleport/ult advantage - otherwise "
                f"group as {self._champion}'s teamfight strength is your win condition."
            ),
            evidence=f"{len(side)} side-lane deaths across {games} games",
            p_value=None,
            effect_size=round(min(1.0, len(side) / games / 2), 3),
            priority=_priority(min(1.0, len(side) / games / 2), None, len(side)),
            sample_size=len(side),
        )

    def _rule_best_matchup(self) -> Recommendation | None:
        """Report the statistically strongest matchup."""
        from analysis.matchups import matchup_summary

        summary = matchup_summary(self._matchups)
        if not summary or summary["best_matchup_winrate"] < 0.55:
            return None
        return Recommendation(
            category="Matchups",
            title=f"Your strongest matchup is {summary['best_matchup']}",
            detail=(
                f"You win {summary['best_matchup_winrate']:.0%} of {summary['best_matchup_games']} "
                f"games against {summary['best_matchup']}. Look for it in champ select and play "
                "it aggressively for early leads."
            ),
            evidence=(
                f"{summary['best_matchup_winrate']:.0%} WR over "
                f"{summary['best_matchup_games']} games"
            ),
            p_value=None,
            effect_size=round(summary["best_matchup_winrate"] - 0.5, 3),
            priority=_priority(
                summary["best_matchup_winrate"] - 0.5, None, summary["best_matchup_games"]
            ),
            sample_size=summary["best_matchup_games"],
        )

    def _rule_worst_matchup(self) -> Recommendation | None:
        """Report the worst matchup with a cause when detectable."""
        from analysis.matchups import matchup_summary

        summary = matchup_summary(self._matchups)
        if not summary or summary["worst_matchup_winrate"] > 0.45:
            return None
        cause = ""
        deaths_pre14 = summary.get("worst_matchup_deaths_pre14")
        if deaths_pre14 is not None and deaths_pre14 >= 1.5:
            cause = (
                f" You average {deaths_pre14:.1f} deaths before 14 minutes in this lane - the "
                "games are lost in the laning phase, not later."
            )
        return Recommendation(
            category="Matchups",
            title=f"You struggle most against {summary['worst_matchup']}",
            detail=(
                f"You win only {summary['worst_matchup_winrate']:.0%} of "
                f"{summary['worst_matchup_games']} games against {summary['worst_matchup']}."
                f"{cause} Consider a defensive rune page or banning it."
            ),
            evidence=(
                f"{summary['worst_matchup_winrate']:.0%} WR over "
                f"{summary['worst_matchup_games']} games"
            ),
            p_value=None,
            effect_size=round(0.5 - summary["worst_matchup_winrate"], 3),
            priority=_priority(
                0.5 - summary["worst_matchup_winrate"], None, summary["worst_matchup_games"]
            ),
            sample_size=summary["worst_matchup_games"],
        )

    def _rule_deaths_before_dragon(self) -> Recommendation | None:
        """Impact of dying in the minute before a dragon spawn fight."""
        split = self._stats.winrate_split_test("deaths_before_dragon", 1)
        if split is None or split["n_high"] < 4:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.12:
            return None
        return Recommendation(
            category="Objectives",
            title="Deaths right before dragons are throwing objectives",
            detail=(
                f"You win only {split['winrate_high']:.0%} of games where you die within 60 "
                f"seconds before a dragon is taken, versus {split['winrate_low']:.0%} otherwise. "
                "Reset 90 seconds before spawns, then move with your team."
            ),
            evidence=(
                f"WR {split['winrate_high']:.0%} ({split['n_high']} games) vs "
                f"{split['winrate_low']:.0%} ({split['n_low']} games), p={split['p_value']:.3f}"
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_objective_presence(self) -> Recommendation | None:
        """Flag low presence at epic monster takes."""
        if self._objectives.empty:
            return None
        presence = float(self._objectives["present"].mean())
        if presence >= 0.55 or len(self._objectives) < 15:
            return None
        return Recommendation(
            category="Objectives",
            title="You miss too many objective fights",
            detail=(
                f"You were near the pit for only {presence:.0%} of epic monster takes. "
                f"{self._champion}'s zone control is a huge objective-fight advantage - shove "
                "mid before spawns and rotate first, not last."
            ),
            evidence=f"Present at {presence:.0%} of {len(self._objectives)} objective takes",
            p_value=None,
            effect_size=round(0.6 - presence, 3),
            priority=_priority(0.6 - presence, None, len(self._objectives)),
            sample_size=len(self._objectives),
        )

    def _rule_solo_deaths(self) -> Recommendation | None:
        """Win-rate impact of solo (unassisted-by-team) deaths."""
        split = self._stats.winrate_split_test("solo_deaths", 2)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Deaths",
            title="Solo deaths are your biggest leak",
            detail=(
                f"With 2+ solo deaths your win rate drops to {split['winrate_high']:.0%} "
                f"(vs {split['winrate_low']:.0%}). Most were with little recent team vision - "
                "don't cross the river without a ward and a reason."
            ),
            evidence=(
                f"WR {split['winrate_high']:.0%} ({split['n_high']} games) vs "
                f"{split['winrate_low']:.0%} ({split['n_low']} games), p={split['p_value']:.3f}"
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_cs10(self) -> Recommendation | None:
        """Compare CS at 10 in wins vs losses."""
        frame = self._matches[["cs10", "win"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(frame) < MIN_GAMES:
            return None
        avg = float(frame["cs10"].mean())
        if avg >= 72:
            return None
        return Recommendation(
            category="Laning",
            title="Your CS at 10 minutes has room to grow",
            detail=(
                f"You average {avg:.0f} CS at 10. {self._champion}'s power spikes are gold-bound: pushing this "
                "to 75+ is roughly a free half-item by mid game. Prioritise catching every "
                "cannon and using Q resets on low minions."
            ),
            evidence=f"Mean CS@10 = {avg:.1f} over {len(frame)} games",
            p_value=None,
            effect_size=round(min(1.0, (75 - avg) / 30), 3),
            priority=_priority(min(1.0, (75 - avg) / 30), None, len(frame)),
            sample_size=len(frame),
        )

    def _rule_lane_priority(self) -> Recommendation | None:
        """Correlate lane priority with winning."""
        frame = self._matches[["lane_priority", "win"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna()
        if len(frame) < MIN_GAMES or frame["lane_priority"].nunique() < 2:
            return None
        corr, p_value = scipy_stats.pointbiserialr(frame["win"], frame["lane_priority"])
        if corr < 0.2 or p_value > SUGGESTIVE_P:
            return None
        return Recommendation(
            category="Laning",
            title="Lane priority translates directly into wins for you",
            detail=(
                "Games where you hold mid priority (position past the halfway line) are "
                "significantly more likely to be wins. Keep the wave crashing before every "
                "objective spawn and roam off the shove."
            ),
            evidence=f"Point-biserial r={corr:.2f}, p={p_value:.3f}, n={len(frame)}",
            p_value=round(float(p_value), 5),
            effect_size=round(float(corr), 3),
            priority=_priority(float(corr), float(p_value), len(frame)),
            sample_size=len(frame),
        )


def recommendations_markdown(
    recommendations: list[Recommendation], *, build_label: str = "Viktor mid"
) -> str:
    """Render recommendations as a Markdown document.

    Args:
        recommendations: Ranked recommendations.
        build_label: Champion + lane label for the document title.

    Returns:
        Markdown text for ``recommendations.md``.
    """
    lines = [f"# {build_label.title()} Coaching Recommendations", ""]
    if not recommendations:
        lines.append("_Not enough data to generate recommendations yet._")
        return "\n".join(lines)
    for index, rec in enumerate(recommendations, start=1):
        lines.append(f"## {index}. [{rec.category}] {rec.title}")
        lines.append("")
        lines.append(rec.detail)
        lines.append("")
        lines.append(f"- **Evidence:** {rec.evidence}")
        if rec.p_value is not None:
            lines.append(f"- **p-value:** {rec.p_value:.4f}")
        lines.append(f"- **Priority score:** {rec.priority:.2f}")
        lines.append(f"- **Sample size:** {rec.sample_size}")
        lines.append("")
    return "\n".join(lines)
