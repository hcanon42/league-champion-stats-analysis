"""AI coach: rule-based recommendations ranked by statistical evidence.

Each rule inspects the aggregated data, and when a pattern is both material
(large effect) and statistically supported (low p-value where testable),
emits a :class:`~models.Recommendation`. Recommendations are prioritised by
a combination of effect size and significance.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from scipy import stats as scipy_stats

from analysis.statistics import StatisticsEngine
from models import Recommendation, RecommendationTone
from utils import get_logger

MIN_GAMES: int = 10
SIGNIFICANT_P: float = 0.05
SUGGESTIVE_P: float = 0.15
VISIBLE_RECOMMENDATIONS: int = 5

# Human labels and coaching tips for the personal win-condition rule.
WIN_FEATURE_HINTS: dict[str, tuple[str, str]] = {
    "gd10": (
        "a gold lead at 10 minutes",
        "Trade with cooldown and minion cover, secure cannon waves, and avoid donating XP.",
    ),
    "gd15": (
        "a gold lead at 15 minutes",
        "Convert lane leads into plates, roams, or objective setup before the lead fades.",
    ),
    "xpd10": (
        "an XP lead at 10 minutes",
        "Respect level spikes and press your level advantage with short trades.",
    ),
    "cs10": (
        "strong CS at 10 minutes",
        "Secure cannon minions and don't miss farm under tower.",
    ),
    "csd10": (
        "a CS lead at 10 minutes",
        "Freeze or slow-push when ahead to deny farm and invite jungle pressure.",
    ),
    "kill_participation": (
        "high kill participation",
        "Arrive before objectives with your team and look for plays when your wave is pushed.",
    ),
    "damage_share": (
        "a high damage share",
        "Stay active in skirmishes and teamfights instead of passing on winnable fights.",
    ),
    "dpm": (
        "high damage per minute",
        "Look for poke before fights and maximise your combos when cooldowns are available.",
    ),
    "vspm": (
        "strong vision score",
        "Buy a control ward every recall after 14 minutes and sweep before objectives.",
    ),
    "control_wards": (
        "more control wards",
        "Make control wards part of every recall once laning ends.",
    ),
    "lane_priority": (
        "wave priority in lane",
        "Keep the wave pushed before rotating and roam off the shove.",
    ),
    "roams_pre15": (
        "productive early roams",
        "Roam on cannon waves when you have push and a clear target.",
    ),
    "first_item_min": (
        "fast first-item timing",
        "Tighten your early resets so your first completed item lands sooner.",
    ),
}


def _priority(effect: float, p_value: float | None, sample: int) -> float:
    """Score a recommendation for ranking."""
    significance = 1.0 if p_value is None else max(0.0, 1.0 - min(1.0, p_value / SUGGESTIVE_P))
    volume = min(1.0, sample / 40.0)
    return round(abs(effect) * 2.0 + significance + volume, 3)


def _split_threshold(matches: pd.DataFrame, column: str) -> float:
    """Pick a meaningful high/low split for a win-correlation feature."""
    defaults: dict[str, float] = {
        "gd10": 0.0,
        "gd15": 0.0,
        "xpd10": 0.0,
        "csd10": 0.0,
        "kill_participation": 0.55,
        "damage_share": 0.22,
        "lane_priority": 0.5,
        "first_item_min": 11.0,
    }
    if column in defaults:
        return defaults[column]
    series = pd.to_numeric(matches.get(column), errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.median())


def _winrate_split_on_series(
    matches: pd.DataFrame, values: pd.Series, threshold: float
) -> dict[str, Any] | None:
    """Fisher exact test of win rates above/below a threshold series."""
    wins = pd.to_numeric(matches["win"], errors="coerce")
    values = pd.to_numeric(values, errors="coerce")
    mask = values.notna() & wins.notna()
    high = wins[mask & (values >= threshold)]
    low = wins[mask & (values < threshold)]
    if high.empty or low.empty:
        return None
    table = [
        [int(high.sum()), int(len(high) - high.sum())],
        [int(low.sum()), int(len(low) - low.sum())],
    ]
    odds_ratio, p_value = scipy_stats.fisher_exact(table)
    return {
        "threshold": threshold,
        "winrate_high": round(float(high.mean()), 3),
        "winrate_low": round(float(low.mean()), 3),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "odds_ratio": round(float(odds_ratio), 3),
        "p_value": round(float(p_value), 5),
    }


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
        self._matches = matches_df
        self._deaths = deaths_df
        self._matchups = matchups_df
        self._objectives = objectives_df
        self._stats = stats_engine
        self._build_label = build_label
        self._champion = build_label.split(" ", 1)[0]
        self._log = get_logger("coach")

    def generate(self) -> list[Recommendation]:
        """Run every rule and return recommendations sorted by priority."""
        if len(self._matches) < 2:
            return []
        rules: list[Callable[[], Recommendation | None]] = [
            self._rule_personal_win_condition,
            self._rule_early_deaths,
            self._rule_unspent_gold,
            self._rule_first_item_timing,
            self._rule_control_wards,
            self._rule_side_lane_deaths,
            self._rule_best_matchup,
            self._rule_worst_matchup,
            self._rule_deaths_before_objectives,
            self._rule_objective_presence,
            self._rule_solo_deaths,
            self._rule_greed_deaths,
            self._rule_gank_deaths_laning,
            self._rule_under_own_tower_laning_deaths,
            self._rule_under_enemy_tower_laning_deaths,
            self._rule_shutdown_bounties,
            self._rule_throw_leads,
            self._rule_teamfight_participation,
            self._rule_dead_before_objectives,
            self._rule_cs10,
            self._rule_lane_priority,
        ]
        recommendations: list[Recommendation] = []
        for rule in rules:
            try:
                result = rule()
            except Exception as exc:
                self._log.warning("Coach rule %s failed: %s", rule.__name__, exc)
                continue
            if result is not None:
                recommendations.append(result)
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)

    def _rule_personal_win_condition(self) -> Recommendation | None:
        """Surface the top positive win correlates for this player."""
        corrs = self._stats.win_correlations()
        positive = [c for c in corrs if c.correlation >= 0.18 and c.p_value <= SUGGESTIVE_P]
        if not positive:
            return None

        segments: list[str] = []
        evidence_parts: list[str] = []
        best_delta = 0.0
        best_p: float | None = None
        sample = 0

        for pick in positive[:2]:
            if pick.feature not in WIN_FEATURE_HINTS:
                continue
            threshold = _split_threshold(self._matches, pick.feature)
            split = self._stats.winrate_split_test(pick.feature, threshold)
            if split is None or split["n_high"] < 3 or split["n_low"] < 3:
                continue
            delta = split["winrate_high"] - split["winrate_low"]
            if delta < 0.08:
                continue
            label, tip = WIN_FEATURE_HINTS[pick.feature]
            segments.append(
                f"When you have {label}, you win {split['winrate_high']:.0%} of games "
                f"versus {split['winrate_low']:.0%} otherwise. {tip}"
            )
            evidence_parts.append(
                f"{pick.feature}: r={pick.correlation:.2f}, WR {split['winrate_high']:.0%} "
                f"({split['n_high']}g) vs {split['winrate_low']:.0%} ({split['n_low']}g)"
            )
            best_delta = max(best_delta, delta)
            best_p = pick.p_value if best_p is None else min(best_p, pick.p_value)
            sample += split["n_high"] + split["n_low"]

        if not segments:
            return None

        return Recommendation(
            category="Win condition",
            title="Your personal win conditions",
            detail=" ".join(segments),
            evidence="; ".join(evidence_parts),
            tone=RecommendationTone.POSITIVE,
            p_value=best_p,
            effect_size=round(best_delta, 3),
            priority=_priority(best_delta, best_p, sample),
            sample_size=sample,
        )

    def _rule_early_deaths(self) -> Recommendation | None:
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
                "deficit versus your opponent — plan resets around your next item component "
                "instead of sitting on gold."
            ),
            evidence=f"Mean banked gold before recall: {avg:.0f}g over {len(series)} games",
            p_value=None,
            effect_size=round(min(1.0, (avg - 500) / 1000), 3),
            priority=_priority(min(1.0, (avg - 500) / 1000), None, len(series)),
            sample_size=len(series),
        )

    def _rule_first_item_timing(self) -> Recommendation | None:
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
            tone=RecommendationTone.POSITIVE,
            p_value=round(float(p_value), 5),
            effect_size=round(float(corr), 3),
            priority=_priority(float(corr), float(p_value), len(frame)),
            sample_size=len(frame),
        )

    def _rule_side_lane_deaths(self) -> Recommendation | None:
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
                "take a side wave with vision, tempo and Teleport/ult advantage — otherwise "
                "group and play around your team's win condition."
            ),
            evidence=f"{len(side)} side-lane deaths across {games} games",
            p_value=None,
            effect_size=round(min(1.0, len(side) / games / 2), 3),
            priority=_priority(min(1.0, len(side) / games / 2), None, len(side)),
            sample_size=len(side),
        )

    def _rule_best_matchup(self) -> Recommendation | None:
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
            tone=RecommendationTone.POSITIVE,
            p_value=None,
            effect_size=round(summary["best_matchup_winrate"] - 0.5, 3),
            priority=_priority(
                summary["best_matchup_winrate"] - 0.5, None, summary["best_matchup_games"]
            ),
            sample_size=summary["best_matchup_games"],
        )

    def _rule_worst_matchup(self) -> Recommendation | None:
        from analysis.matchups import matchup_summary

        summary = matchup_summary(self._matchups)
        if not summary or summary["worst_matchup_winrate"] > 0.45:
            return None
        cause = ""
        deaths_pre14 = summary.get("worst_matchup_deaths_pre14")
        if deaths_pre14 is not None and deaths_pre14 >= 1.5:
            cause = (
                f" You average {deaths_pre14:.1f} deaths before 14 minutes in this lane — the "
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

    def _rule_deaths_before_objectives(self) -> Recommendation | None:
        dragon = pd.to_numeric(self._matches.get("deaths_before_dragon"), errors="coerce").fillna(0)
        baron = pd.to_numeric(self._matches.get("deaths_before_baron"), errors="coerce").fillna(0)
        combined = dragon + baron
        split = _winrate_split_on_series(self._matches, combined, 1)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        pre_dragon = int((dragon >= 1).sum())
        pre_baron = int((baron >= 1).sum())
        return Recommendation(
            category="Objectives",
            title="Deaths right before epic monsters are throwing objectives",
            detail=(
                f"You win only {split['winrate_high']:.0%} of games where you die within 60 "
                f"seconds before a dragon or baron is taken, versus {split['winrate_low']:.0%} "
                "otherwise. Reset 90 seconds before spawns, then move with your team."
            ),
            evidence=(
                f"WR {split['winrate_high']:.0%} ({split['n_high']} games) vs "
                f"{split['winrate_low']:.0%} ({split['n_low']} games), "
                f"{pre_dragon} pre-dragon / {pre_baron} pre-baron games, p={split['p_value']:.3f}"
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_objective_presence(self) -> Recommendation | None:
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
                "Being present for objectives matters more than your average game shows — push "
                "your assigned lane before rotating to the pit, and arrive first, not last."
            ),
            evidence=f"Present at {presence:.0%} of {len(self._objectives)} objective takes",
            p_value=None,
            effect_size=round(0.6 - presence, 3),
            priority=_priority(0.6 - presence, None, len(self._objectives)),
            sample_size=len(self._objectives),
        )

    def _rule_solo_deaths(self) -> Recommendation | None:
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
                f"(vs {split['winrate_low']:.0%}). Most were with little recent team vision — "
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

    def _rule_greed_deaths(self) -> Recommendation | None:
        if "greed_deaths" not in self._matches.columns:
            return None
        split = self._stats.winrate_split_test("greed_deaths", 2)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Deaths",
            title="Greed deaths are a recurring pattern",
            detail=(
                f"Games with 2+ greed deaths win only {split['winrate_high']:.0%} "
                f"(vs {split['winrate_low']:.0%}). These are deaths after overextending "
                "without a clear payoff — back off when vision is thin or numbers are even."
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

    def _rule_gank_deaths_laning(self) -> Recommendation | None:
        if "gank_deaths_laning" not in self._matches.columns:
            return None
        split = self._stats.winrate_split_test("gank_deaths_laning", 1)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Laning",
            title="Gank deaths in lane are a pattern",
            detail=(
                f"Games with a gank death before 14 minutes win only {split['winrate_high']:.0%} "
                f"(vs {split['winrate_low']:.0%}). Track the jungler, respect river wards, and "
                "don't push without knowing where the enemy jungler started."
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

    def _rule_under_own_tower_laning_deaths(self) -> Recommendation | None:
        if "under_own_tower_laning_deaths" not in self._matches.columns:
            return None
        split = self._stats.winrate_split_test("under_own_tower_laning_deaths", 1)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Laning",
            title="You're dying under your own tower in lane",
            detail=(
                f"When you die under your own tower before 14 minutes your win rate is only "
                f"{split['winrate_high']:.0%} (vs {split['winrate_low']:.0%}). Respect dive "
                "threats: manage wave state, keep health above dive thresholds, and ping "
                "for jungle help before you get trapped."
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

    def _rule_under_enemy_tower_laning_deaths(self) -> Recommendation | None:
        if "under_enemy_tower_laning_deaths" not in self._matches.columns:
            return None
        split = self._stats.winrate_split_test("under_enemy_tower_laning_deaths", 1)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.10:
            return None
        return Recommendation(
            category="Laning",
            title="Your tower dives in lane are costing games",
            detail=(
                f"When you die under an enemy tower before 14 minutes your win rate is only "
                f"{split['winrate_high']:.0%} (vs {split['winrate_low']:.0%}). Only dive with "
                "clear kill pressure, wave setup, and jungle cover — reset if the trade "
                "isn't guaranteed."
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

    def _rule_shutdown_bounties(self) -> Recommendation | None:
        if "shutdown_given" not in self._matches.columns:
            return None
        split = self._stats.winrate_split_test("shutdown_given", 200)
        if split is None or split["n_high"] < 3:
            return None
        delta = split["winrate_low"] - split["winrate_high"]
        if delta < 0.08:
            return None
        series = pd.to_numeric(self._matches["shutdown_given"], errors="coerce").dropna()
        avg = float(series.mean())
        bounty_deaths = 0
        if not self._deaths.empty and "shutdown_given" in self._deaths.columns:
            bounty_deaths = int(
                (pd.to_numeric(self._deaths["shutdown_given"], errors="coerce") > 0).sum()
            )
        return Recommendation(
            category="Deaths",
            title="You give away too many shutdown bounties",
            detail=(
                f"Games where you give 200+ shutdown gold win only {split['winrate_high']:.0%} "
                f"(vs {split['winrate_low']:.0%}). You average {avg:.0f} shutdown gold given per "
                "game. When ahead, play for safe positioning in fights instead of chasing "
                "low-value kills."
            ),
            evidence=(
                f"WR {split['winrate_high']:.0%} ({split['n_high']} high-bounty games) vs "
                f"{split['winrate_low']:.0%} ({split['n_low']} games)"
                + (f"; {bounty_deaths} bounty deaths logged" if bounty_deaths else "")
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_throw_leads(self) -> Recommendation | None:
        if "gd15" not in self._matches.columns:
            return None
        frame = self._matches[["gd15", "win"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(frame) < MIN_GAMES:
            return None
        ahead = frame[frame["gd15"] >= 750]
        if len(ahead) < 5:
            return None
        ahead_wr = float(ahead["win"].mean())
        if ahead_wr >= 0.62:
            return None
        throws = ahead[ahead["win"] == 0]
        throw_rate = len(throws) / len(frame)
        return Recommendation(
            category="Macro",
            title="You throw leads you build in lane",
            detail=(
                f"You win only {ahead_wr:.0%} of games where you're 750+ gold ahead at 15 "
                f"minutes ({len(throws)} throws in {len(ahead)} ahead games). Convert leads "
                "with objective setup, vision, and grouped fights — don't bleed gold on "
                "greedy side waves or bad skirmishes."
            ),
            evidence=(
                f"{ahead_wr:.0%} WR when ahead at 15 ({len(ahead)} games); "
                f"{throw_rate:.0%} of all games are thrown leads"
            ),
            p_value=None,
            effect_size=round(0.62 - ahead_wr, 3),
            priority=_priority(0.62 - ahead_wr, None, len(ahead)),
            sample_size=len(ahead),
        )

    def _rule_teamfight_participation(self) -> Recommendation | None:
        if "tf_participation" not in self._matches.columns:
            return None
        frame = self._matches[["tf_participation", "win"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna()
        if len(frame) < MIN_GAMES or frame["tf_participation"].nunique() < 2:
            return None
        split = self._stats.winrate_split_test("tf_participation", 0.65)
        if split is None or split["n_low"] < 3:
            return None
        delta = split["winrate_high"] - split["winrate_low"]
        if delta < 0.10:
            return None
        low_avg = float(frame[frame["tf_participation"] < 0.65]["tf_participation"].mean())
        return Recommendation(
            category="Teamfights",
            title="Low teamfight participation is limiting your impact",
            detail=(
                f"When you show up to fewer than 65% of detected teamfights your win rate is "
                f"{split['winrate_low']:.0%} versus {split['winrate_high']:.0%} otherwise "
                f"(you average {low_avg:.0%} participation in those games). Group before "
                "objectives and track fight timers on the map."
            ),
            evidence=(
                f"WR {split['winrate_low']:.0%} ({split['n_low']} low-participation games) vs "
                f"{split['winrate_high']:.0%} ({split['n_high']} games), p={split['p_value']:.3f}"
            ),
            p_value=split["p_value"],
            effect_size=round(delta, 3),
            priority=_priority(delta, split["p_value"], split["n_high"] + split["n_low"]),
            sample_size=split["n_high"] + split["n_low"],
        )

    def _rule_dead_before_objectives(self) -> Recommendation | None:
        if self._objectives.empty or "dead_before" not in self._objectives.columns:
            return None
        dead_rate = float(self._objectives["dead_before"].mean())
        if dead_rate < 0.18 or len(self._objectives) < 12:
            return None
        by_kind = self._objectives.groupby("kind")["dead_before"].mean().sort_values(ascending=False)
        worst_kind = str(by_kind.index[0]) if not by_kind.empty else "objective"
        return Recommendation(
            category="Objectives",
            title="You're dead too often right before objectives",
            detail=(
                f"You were on death timer for {dead_rate:.0%} of epic monster takes "
                f"({worst_kind} is the worst). Start your reset earlier and path toward "
                "the pit 60–90 seconds before spawn so you're alive and in position."
            ),
            evidence=f"Dead before {dead_rate:.0%} of {len(self._objectives)} objective takes",
            p_value=None,
            effect_size=round(dead_rate, 3),
            priority=_priority(dead_rate, None, len(self._objectives)),
            sample_size=len(self._objectives),
        )

    def _rule_cs10(self) -> Recommendation | None:
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
                f"You average {avg:.0f} CS at 10. Your power spikes are gold-bound: pushing this "
                "to 75+ is roughly a free half-item by mid game. Prioritise catching every "
                "cannon and securing ranged minions under tower."
            ),
            evidence=f"Mean CS@10 = {avg:.1f} over {len(frame)} games",
            p_value=None,
            effect_size=round(min(1.0, (75 - avg) / 30), 3),
            priority=_priority(min(1.0, (75 - avg) / 30), None, len(frame)),
            sample_size=len(frame),
        )

    def _rule_lane_priority(self) -> Recommendation | None:
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
                "Games where you hold wave priority in lane are significantly more likely to "
                "be wins. Keep the wave pushed before every objective spawn and roam off the "
                "shove."
            ),
            evidence=f"Point-biserial r={corr:.2f}, p={p_value:.3f}, n={len(frame)}",
            tone=RecommendationTone.POSITIVE,
            p_value=round(float(p_value), 5),
            effect_size=round(float(corr), 3),
            priority=_priority(float(corr), float(p_value), len(frame)),
            sample_size=len(frame),
        )


def recommendations_markdown(
    recommendations: list[Recommendation], *, build_label: str = "Viktor mid"
) -> str:
    """Render recommendations as a Markdown document."""
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
