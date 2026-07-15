"""Role-aware metric profiles: which cards, scores, peers, and coach rules apply per lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from league_stats.analysis.combat import combat_output_metric, prefers_cc_over_dpm
from league_stats.core.champions import normalize_role

MetricDirection = Literal["higher", "lower"]

# Coach rule method names enabled per role (see CoachEngine.generate).
ALL_COACH_RULES: Final[frozenset[str]] = frozenset(
    {
        "_rule_personal_win_condition",
        "_rule_early_deaths",
        "_rule_unspent_gold",
        "_rule_unspent_gold_fights",
        "_rule_gold_at_death",
        "_rule_first_item_timing",
        "_rule_control_wards",
        "_rule_side_lane_deaths",
        "_rule_best_matchup",
        "_rule_worst_matchup",
        "_rule_deaths_before_objectives",
        "_rule_objective_presence",
        "_rule_solo_deaths",
        "_rule_outnumbered_deaths",
        "_rule_greed_deaths",
        "_rule_gank_deaths_laning",
        "_rule_under_own_tower_laning_deaths",
        "_rule_under_enemy_tower_laning_deaths",
        "_rule_shutdown_bounties",
        "_rule_throw_leads",
        "_rule_teamfight_participation",
        "_rule_disadvantaged_fights",
        "_rule_over_grouping",
        "_rule_splitting_for_farm",
        "_rule_ally_proximity",
        "_rule_dead_before_objectives",
        "_rule_cs10",
        "_rule_lane_priority",
        "_rule_low_kill_participation",
        "_rule_low_vision",
        "_rule_low_cc",
    }
)

LANER_COACH_RULES: Final[frozenset[str]] = ALL_COACH_RULES - frozenset(
    {
        "_rule_low_kill_participation",
        "_rule_low_vision",
        "_rule_low_cc",
    }
)

JUNGLE_COACH_RULES: Final[frozenset[str]] = ALL_COACH_RULES - frozenset(
    {
        "_rule_cs10",
        "_rule_lane_priority",
        "_rule_throw_leads",
        "_rule_gank_deaths_laning",
        "_rule_under_own_tower_laning_deaths",
        "_rule_under_enemy_tower_laning_deaths",
        "_rule_greed_deaths",
        "_rule_side_lane_deaths",
        "_rule_splitting_for_farm",
        "_rule_low_vision",
        "_rule_low_cc",
    }
)

UTILITY_COACH_RULES: Final[frozenset[str]] = ALL_COACH_RULES - frozenset(
    {
        "_rule_cs10",
        "_rule_lane_priority",
        "_rule_throw_leads",
        "_rule_gank_deaths_laning",
        "_rule_under_enemy_tower_laning_deaths",
        "_rule_greed_deaths",
        "_rule_side_lane_deaths",
        "_rule_splitting_for_farm",
    }
)


@dataclass(frozen=True)
class MetricSpec:
    """One dashboard card: label plus key in a summary bucket."""

    label: str
    section: str
    key: str
    suffix: str = ""
    pct: bool = False


@dataclass(frozen=True)
class ScoreSpec:
    """One improvement-score dimension."""

    name: str
    column: str
    hint: str
    value_fmt: str  # format string with one `{v}` placeholder


@dataclass(frozen=True)
class RoleMetricProfile:
    """Metrics and coaching scope for one Riot team position."""

    role: str
    early_section_title: str
    overview: tuple[MetricSpec, ...]
    early_game: tuple[MetricSpec, ...]
    early_headlines: tuple[str, ...]
    economy: tuple[MetricSpec, ...]
    vision: tuple[MetricSpec, ...]
    deaths: tuple[MetricSpec, ...]
    teamfights: tuple[MetricSpec, ...]
    score_components: tuple[ScoreSpec, ...]
    peer_metrics: tuple[tuple[str, str, MetricDirection], ...]
    coach_rule_ids: frozenset[str]
    ml_features: tuple[str, ...]
    early_ml_features: tuple[str, ...]
    section_order: tuple[str, ...] = (
        "overview",
        "early",
        "positioning",
        "economy",
        "vision",
        "deaths",
        "teamfights",
        "objectives",
    )


def _laner_overview() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Win rate", "overview", "winrate_pct"),
        MetricSpec("KDA", "overview", "avg_kda"),
        MetricSpec("DPM", "overview", "avg_dpm"),
        MetricSpec("CS/min", "overview", "avg_cspm"),
        MetricSpec("Damage share", "overview", "avg_damage_share", pct=True),
        MetricSpec("Deaths/game", "overview", "avg_deaths"),
        MetricSpec("Vision/min", "overview", "avg_vspm"),
        MetricSpec("Avg game", "overview", "avg_duration", suffix=" min"),
    )


def _laner_early() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Gold diff @10", "laning", "avg_gd10"),
        MetricSpec("CS diff @10", "laning", "avg_csd10"),
        MetricSpec("XP diff @10", "laning", "avg_xpd10"),
        MetricSpec("Lane win rate", "laning", "lane_win_rate", pct=True),
        MetricSpec("WR when ahead @10", "laning", "winrate_when_ahead_at_10", pct=True),
        MetricSpec("WR when behind @10", "laning", "winrate_when_behind_at_10", pct=True),
        MetricSpec("Deaths pre-14", "laning", "avg_deaths_pre14"),
        MetricSpec("Gank deaths (lane)", "laning", "avg_gank_deaths_laning"),
        MetricSpec("Under own tower (lane)", "laning", "avg_under_own_tower_laning_deaths"),
        MetricSpec("Under enemy tower (lane)", "laning", "avg_under_enemy_tower_laning_deaths"),
        MetricSpec("Roams pre-15", "laning", "avg_roams_pre15"),
    )


def _laner_economy() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("GPM", "economy", "avg_gpm"),
        MetricSpec("CS/min", "economy", "avg_cspm"),
        MetricSpec("Gold share", "economy", "avg_gold_share", pct=True),
        MetricSpec("Damage per gold", "economy", "avg_damage_per_gold"),
        MetricSpec("Unspent gold/recall", "economy", "avg_unspent_gold_before_recall", suffix="g"),
        MetricSpec("First recall", "economy", "avg_first_recall_min", suffix=" min"),
        MetricSpec("Time dead/game", "economy", "avg_time_dead_s", suffix="s"),
    )


def _standard_vision() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Vision score", "vision", "avg_vision_score"),
        MetricSpec("VS/min", "vision", "avg_vspm"),
        MetricSpec("Control wards", "vision", "avg_control_wards"),
        MetricSpec("CW lifetime", "vision", "avg_control_ward_lifetime_s", suffix="s"),
        MetricSpec("VS/min in wins", "vision", "avg_vspm_wins"),
        MetricSpec("VS/min in losses", "vision", "avg_vspm_losses"),
    )


def _laner_deaths() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Total deaths", "deaths", "total_deaths"),
        MetricSpec("Solo deaths", "deaths", "solo_death_rate", pct=True),
        MetricSpec("Gank deaths (lane)", "deaths", "gank_death_rate", pct=True),
        MetricSpec("Under own tower (lane)", "deaths", "under_own_tower_laning_death_rate", pct=True),
        MetricSpec("Under enemy tower (lane)", "deaths", "under_enemy_tower_laning_death_rate", pct=True),
        MetricSpec("Greed deaths", "deaths", "greed_death_rate", pct=True),
        MetricSpec("Side-lane deaths", "deaths", "side_lane_death_rate", pct=True),
        MetricSpec("Before neutral obj.", "deaths", "death_before_neutral_objective_rate", pct=True),
        MetricSpec("Gold at death", "deaths", "avg_gold_at_death", suffix="g"),
        MetricSpec("Outnumbered deaths", "deaths", "outnumbered_death_rate", pct=True),
        MetricSpec("Avg death minute", "deaths", "avg_death_minute"),
        MetricSpec("Top killer", "deaths", "most_common_killer"),
    )


def _standard_teamfights() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Fights detected", "teamfights", "total_fights"),
        MetricSpec("Participation", "teamfights", "participation_rate", pct=True),
        MetricSpec("Fight win rate", "teamfights", "fight_win_rate", pct=True),
        MetricSpec("Damage/fight", "teamfights", "avg_damage_per_fight"),
        MetricSpec("Death rate in fights", "teamfights", "death_rate_in_fights", pct=True),
        MetricSpec("Front-to-back", "teamfights", "avg_front_to_back"),
        MetricSpec("Unspent gold/fight", "teamfights", "avg_unspent_gold_per_fight", suffix="g"),
        MetricSpec("Advantaged fights", "teamfights", "fights_advantaged"),
        MetricSpec("Disadvantaged fights", "teamfights", "fights_disadvantaged"),
        MetricSpec("WR advantaged fights", "teamfights", "fight_win_rate_when_advantaged", pct=True),
        MetricSpec("WR disadvantaged fights", "teamfights", "fight_win_rate_when_disadvantaged", pct=True),
    )


def _laner_score() -> tuple[ScoreSpec, ...]:
    return (
        ScoreSpec("Laning", "gd10", "Average gold diff vs lane opponent at 10 min", "{v:+.0f} gold @10"),
        ScoreSpec("Farming", "cs10", "Role benchmark band for CS @10", "{v:.0f} CS @10"),
        ScoreSpec("Survival", "deaths", "Fewer deaths score higher", "{v:.1f} deaths/game"),
        ScoreSpec("Damage", "damage_share", "Share of team damage to champions", "{v:.0%} team damage"),
        ScoreSpec("Vision", "vspm", "Vision score per minute", "{v:.2f} VS/min"),
        ScoreSpec("Objectives", "objectives_present_rate", "Presence at epic monster takes", "{v:.0%} presence"),
        ScoreSpec("Resets", "avg_unspent_gold", "Gold banked before recalls", "{v:.0f}g banked"),
    )


_LANER_PEER: tuple[tuple[str, str, MetricDirection], ...] = (
    ("win", "Win rate", "higher"),
    ("kda", "KDA", "higher"),
    ("dpm", "DPM", "higher"),
    ("cspm", "CS/min", "higher"),
    ("deaths", "Deaths/game", "lower"),
    ("vspm", "Vision/min", "higher"),
    ("control_wards", "Control wards", "higher"),
    ("cs10", "CS @10", "higher"),
    ("gd10", "Gold diff @10", "higher"),
    ("kill_participation", "Kill participation", "higher"),
    ("damage_share", "Damage share", "higher"),
    ("deaths_pre14", "Deaths pre-14", "lower"),
)

_LANER_ML: tuple[str, ...] = (
    "deaths_pre20", "deaths_pre14", "control_wards", "first_item_min", "cs10", "gd10", "gd15",
    "xpd10", "dpm", "vspm", "avg_unspent_gold", "grouped_share", "solo_share", "roams_pre15",
    "lane_priority", "solo_deaths", "kill_participation", "damage_share",
)

_LANER_EARLY_ML: tuple[str, ...] = (
    "gd10", "xpd10", "csd10", "cs10", "deaths_pre14", "first_item_min", "roams_pre15",
    "lane_priority", "gd15",
)


def _jungle_overview() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Win rate", "overview", "winrate_pct"),
        MetricSpec("KDA", "overview", "avg_kda"),
        MetricSpec("Kill participation", "overview", "avg_kill_participation", pct=True),
        MetricSpec("Obj. presence", "overview", "avg_objectives_present_rate", pct=True),
        MetricSpec("Deaths/game", "overview", "avg_deaths"),
        MetricSpec("Vision/min", "overview", "avg_vspm"),
        MetricSpec("DPM", "overview", "avg_dpm"),
        MetricSpec("Avg game", "overview", "avg_duration", suffix=" min"),
    )


def _jungle_early() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Gold @10", "laning", "avg_gold10"),
        MetricSpec("CS @10", "laning", "avg_cs10"),
        MetricSpec("First recall", "resets", "avg_first_recall_min", suffix=" min"),
        MetricSpec("Early ganks", "jungle", "avg_early_ganks"),
        MetricSpec("Gank assists", "jungle", "avg_gank_assists"),
        MetricSpec("Roams pre-15", "laning", "avg_roams_pre15"),
        MetricSpec("Deaths pre-14", "laning", "avg_deaths_pre14"),
        MetricSpec("Solo deaths", "deaths", "solo_death_rate", pct=True),
        MetricSpec("KP @15", "jungle", "avg_kp15", pct=True),
    )


def _jungle_economy() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("GPM", "economy", "avg_gpm"),
        MetricSpec("Gold share", "economy", "avg_gold_share", pct=True),
        MetricSpec("Damage per gold", "economy", "avg_damage_per_gold"),
        MetricSpec("Unspent gold/recall", "economy", "avg_unspent_gold_before_recall", suffix="g"),
        MetricSpec("First recall", "economy", "avg_first_recall_min", suffix=" min"),
        MetricSpec("Time dead/game", "economy", "avg_time_dead_s", suffix="s"),
        MetricSpec("CC/min", "overview", "avg_ccpm"),
    )


def _jungle_deaths() -> tuple[MetricSpec, ...]:
    base = list(_laner_deaths())
    replacements = {
        "Gank deaths (lane)": MetricSpec(
            "Caught while farming", "deaths", "gank_death_rate", pct=True
        ),
    }
    return tuple(replacements.get(spec.label, spec) for spec in base)


def _jungle_teamfights() -> tuple[MetricSpec, ...]:
    return _standard_teamfights()


def _jungle_score() -> tuple[ScoreSpec, ...]:
    return (
        ScoreSpec("Map control", "objectives_present_rate", "Presence at epic monster takes", "{v:.0%} presence"),
        ScoreSpec("Clear @10", "cs10", "Jungle clear speed at 10 minutes", "{v:.0f} CS @10"),
        ScoreSpec("Survival", "deaths", "Fewer deaths score higher", "{v:.1f} deaths/game"),
        ScoreSpec("Impact", "kill_participation", "Share of team kills and assists", "{v:.0%} KP"),
        ScoreSpec("Vision", "vspm", "Vision score per minute", "{v:.2f} VS/min"),
        ScoreSpec("Early ganks", "early_ganks", "Successful early gank pressure", "{v:.1f} early ganks"),
        ScoreSpec("Resets", "avg_unspent_gold", "Gold banked before recalls", "{v:.0f}g banked"),
    )


_JUNGLE_PEER: tuple[tuple[str, str, MetricDirection], ...] = (
    ("win", "Win rate", "higher"),
    ("kda", "KDA", "higher"),
    ("kill_participation", "Kill participation", "higher"),
    ("objectives_present_rate", "Obj. presence", "higher"),
    ("deaths", "Deaths/game", "lower"),
    ("vspm", "Vision/min", "higher"),
    ("control_wards", "Control wards", "higher"),
    ("cspm", "CS/min", "higher"),
    ("dpm", "DPM", "higher"),
    ("cs10", "CS @10", "higher"),
    ("deaths_pre14", "Deaths pre-14", "lower"),
    ("early_ganks", "Early ganks", "higher"),
)

_JUNGLE_ML: tuple[str, ...] = (
    "deaths_pre20", "deaths_pre14", "control_wards", "first_item_min", "cs10",
    "kill_participation", "objectives_present_rate", "early_ganks", "kp15",
    "vspm", "solo_deaths", "dpm", "ccpm", "avg_unspent_gold",
)

_JUNGLE_EARLY_ML: tuple[str, ...] = (
    "cs10", "deaths_pre14", "first_item_min", "roams_pre15", "early_ganks", "kp15",
)


def _utility_overview() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Win rate", "overview", "winrate_pct"),
        MetricSpec("KDA", "overview", "avg_kda"),
        MetricSpec("CC/min", "overview", "avg_ccpm"),
        MetricSpec("Kill participation", "overview", "avg_kill_participation", pct=True),
        MetricSpec("Vision/min", "overview", "avg_vspm"),
        MetricSpec("Control wards", "overview", "avg_control_wards"),
        MetricSpec("Deaths/game", "overview", "avg_deaths"),
        MetricSpec("Avg game", "overview", "avg_duration", suffix=" min"),
    )


def _utility_early() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("Roams pre-15", "laning", "avg_roams_pre15"),
        MetricSpec("Roam conversions", "support", "avg_roam_conversions"),
        MetricSpec("Dist to ADC", "positioning", "dist_bottom"),
        MetricSpec("Grouped share", "positioning", "avg_grouped_share", pct=True),
        MetricSpec("Deaths pre-14", "laning", "avg_deaths_pre14"),
        MetricSpec("KP @15", "support", "avg_kp15", pct=True),
        MetricSpec("VS/min @10", "support", "avg_vspm10"),
        MetricSpec("Bot lane presence", "laning", "avg_lane_priority", pct=True),
    )


def _utility_economy() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("GPM", "economy", "avg_gpm"),
        MetricSpec("Gold share", "economy", "avg_gold_share", pct=True),
        MetricSpec("Assists/game", "utility", "avg_assists"),
        MetricSpec("Healing/min", "utility", "avg_hpm"),
        MetricSpec("Shielding/min", "utility", "avg_spm"),
        MetricSpec("Unspent gold/recall", "economy", "avg_unspent_gold_before_recall", suffix="g"),
        MetricSpec("Time dead/game", "economy", "avg_time_dead_s", suffix="s"),
    )


def _utility_deaths() -> tuple[MetricSpec, ...]:
    specs = list(_laner_deaths())
    replacements = {
        "Gank deaths (lane)": MetricSpec("Overextended roams", "deaths", "gank_death_rate", pct=True),
        "Under own tower (lane)": MetricSpec(
            "Facechecks / deep wards", "deaths", "under_own_tower_laning_death_rate", pct=True
        ),
        "Under enemy tower (lane)": MetricSpec(
            "Overextended roams", "deaths", "under_enemy_tower_laning_death_rate", pct=True
        ),
    }
    return tuple(replacements.get(spec.label, spec) for spec in specs)


def _utility_teamfights() -> tuple[MetricSpec, ...]:
    base = list(_standard_teamfights())
    extra = [
        MetricSpec("CC/min", "overview", "avg_ccpm"),
        MetricSpec("Healing/min", "utility", "avg_hpm"),
        MetricSpec("Shielding/min", "utility", "avg_spm"),
    ]
    return tuple(base + extra)


def _utility_score() -> tuple[ScoreSpec, ...]:
    return (
        ScoreSpec(
            "Utility",
            "utility_impact",
            "CC, poke damage, damage taken, healing and shielding to allies",
            "{v}",
        ),
        ScoreSpec("Vision", "vspm", "Vision score per minute", "{v:.2f} VS/min"),
        ScoreSpec("Impact", "kill_participation", "Share of team kills and assists", "{v:.0%} KP"),
        ScoreSpec("Setup", "roams_pre15", "Early roams and map presence", "{v:.1f} roams pre-15"),
        ScoreSpec("Survival", "deaths", "Fewer deaths score higher", "{v:.1f} deaths/game"),
        ScoreSpec("Objectives", "objectives_present_rate", "Presence at epic monster takes", "{v:.0%} presence"),
        ScoreSpec("Resets", "avg_unspent_gold", "Gold banked before recalls", "{v:.0f}g banked"),
    )


_UTILITY_PEER: tuple[tuple[str, str, MetricDirection], ...] = (
    ("win", "Win rate", "higher"),
    ("kda", "KDA", "higher"),
    ("ccpm", "CC/min", "higher"),
    ("kill_participation", "Kill participation", "higher"),
    ("vspm", "Vision/min", "higher"),
    ("control_wards", "Control wards", "higher"),
    ("deaths", "Deaths/game", "lower"),
    ("assists", "Assists/game", "higher"),
    ("healing", "Healing", "higher"),
    ("shielding", "Shielding", "higher"),
    ("roams_pre15", "Roams pre-15", "higher"),
    ("objectives_present_rate", "Obj. presence", "higher"),
)

_UTILITY_ML: tuple[str, ...] = (
    "deaths_pre20", "deaths_pre14", "control_wards", "ccpm", "vspm", "kill_participation",
    "roams_pre15", "roam_conversions", "kp15", "grouped_share", "dist_bottom", "healing",
    "shielding", "assists", "objectives_present_rate", "tf_participation",
)

_UTILITY_EARLY_ML: tuple[str, ...] = (
    "roams_pre15", "roam_conversions", "kp15", "deaths_pre14", "lane_priority", "vspm10",
)


_PROFILES: dict[str, RoleMetricProfile] = {
    "TOP": RoleMetricProfile(
        role="TOP",
        early_section_title="Laning",
        overview=_laner_overview(),
        early_game=_laner_early(),
        early_headlines=("Gold diff @10", "CS diff @10", "Lane win rate"),
        economy=_laner_economy(),
        vision=_standard_vision(),
        deaths=_laner_deaths(),
        teamfights=_standard_teamfights(),
        score_components=_laner_score(),
        peer_metrics=_LANER_PEER,
        coach_rule_ids=LANER_COACH_RULES,
        ml_features=_LANER_ML,
        early_ml_features=_LANER_EARLY_ML,
    ),
    "MIDDLE": RoleMetricProfile(
        role="MIDDLE",
        early_section_title="Laning",
        overview=_laner_overview(),
        early_game=_laner_early(),
        early_headlines=("Gold diff @10", "CS diff @10", "Lane win rate"),
        economy=_laner_economy(),
        vision=_standard_vision(),
        deaths=_laner_deaths(),
        teamfights=_standard_teamfights(),
        score_components=_laner_score(),
        peer_metrics=_LANER_PEER,
        coach_rule_ids=LANER_COACH_RULES,
        ml_features=_LANER_ML,
        early_ml_features=_LANER_EARLY_ML,
    ),
    "BOTTOM": RoleMetricProfile(
        role="BOTTOM",
        early_section_title="Laning",
        overview=_laner_overview(),
        early_game=_laner_early(),
        early_headlines=("Gold diff @10", "CS diff @10", "Lane win rate"),
        economy=_laner_economy(),
        vision=_standard_vision(),
        deaths=_laner_deaths(),
        teamfights=_standard_teamfights(),
        score_components=_laner_score(),
        peer_metrics=_LANER_PEER,
        coach_rule_ids=LANER_COACH_RULES,
        ml_features=_LANER_ML,
        early_ml_features=_LANER_EARLY_ML,
    ),
    "JUNGLE": RoleMetricProfile(
        role="JUNGLE",
        early_section_title="Early game",
        overview=_jungle_overview(),
        early_game=_jungle_early(),
        early_headlines=("Gold @10", "CS @10", "Early ganks"),
        economy=_jungle_economy(),
        vision=_standard_vision(),
        deaths=_jungle_deaths(),
        teamfights=_jungle_teamfights(),
        score_components=_jungle_score(),
        peer_metrics=_JUNGLE_PEER,
        coach_rule_ids=JUNGLE_COACH_RULES,
        ml_features=_JUNGLE_ML,
        early_ml_features=_JUNGLE_EARLY_ML,
        section_order=(
            "overview", "early", "objectives", "positioning", "economy", "vision", "deaths", "teamfights",
        ),
    ),
    "UTILITY": RoleMetricProfile(
        role="UTILITY",
        early_section_title="Early game",
        overview=_utility_overview(),
        early_game=_utility_early(),
        early_headlines=("Roams pre-15", "Dist to ADC", "KP @15"),
        economy=_utility_economy(),
        vision=_standard_vision(),
        deaths=_utility_deaths(),
        teamfights=_utility_teamfights(),
        score_components=_utility_score(),
        peer_metrics=_UTILITY_PEER,
        coach_rule_ids=UTILITY_COACH_RULES,
        ml_features=_UTILITY_ML,
        early_ml_features=_UTILITY_EARLY_ML,
        section_order=(
            "overview", "early", "vision", "positioning", "economy", "deaths", "teamfights", "objectives",
        ),
    ),
}


def role_profile(role: str) -> RoleMetricProfile:
    """Return the metric profile for a normalised team position."""
    normalized = normalize_role(role)
    return _PROFILES.get(normalized, _PROFILES["MIDDLE"])


def compare_metrics_for_profile(
    profile: RoleMetricProfile,
    *,
    avg_damage_share: float | None = None,
) -> tuple[tuple[str, str, MetricDirection], ...]:
    """Resolve peer comparison rows, applying DPM/CC/min swap for tank builds."""
    use_cc = prefers_cc_over_dpm(profile.role, avg_damage_share=avg_damage_share)
    metrics: list[tuple[str, str, MetricDirection]] = []
    for key, label, direction in profile.peer_metrics:
        if key == "dpm" and use_cc:
            metrics.append(("ccpm", "CC/min", "higher"))
        elif key == "ccpm" and not use_cc and profile.role != "UTILITY":
            metrics.append(("dpm", "DPM", "higher"))
        else:
            metrics.append((key, label, direction))
    return tuple(metrics)


def overview_metric_specs(
    profile: RoleMetricProfile,
    *,
    avg_damage_share: float | None = None,
) -> tuple[MetricSpec, ...]:
    """Overview cards with combat-output swap for non-support tank builds."""
    if profile.role == "UTILITY":
        return profile.overview
    combat_key, combat_label = combat_output_metric(profile.role, avg_damage_share=avg_damage_share)
    result: list[MetricSpec] = []
    for spec in profile.overview:
        if spec.label == "DPM" and combat_key == "ccpm":
            result.append(MetricSpec(combat_label, spec.section, f"avg_{combat_key}"))
        else:
            result.append(spec)
    return tuple(result)


def card_tier_headlines(
    profile: RoleMetricProfile,
    section: str,
    *,
    avg_damage_share: float | None = None,
) -> list[str]:
    """Headline labels for a dashboard section."""
    if section == "overview":
        return [spec.label for spec in overview_metric_specs(profile, avg_damage_share=avg_damage_share)[:4]]
    mapping = {
        "early": list(profile.early_headlines),
        "lane": list(profile.early_headlines),
        "economy": [spec.label for spec in profile.economy[:3]],
        "vision": [spec.label for spec in profile.vision[:3]],
        "deaths": [spec.label for spec in profile.deaths[:3]],
        "teamfight": [spec.label for spec in profile.teamfights[:3]],
    }
    return mapping.get(section, [])


def resolve_metric_value(spec: MetricSpec, summaries: dict[str, Any]) -> Any:
    """Look up a metric value from aggregated summaries."""
    if spec.section == "overview" and spec.key == "winrate_pct":
        winrate = summaries.get("overview", {}).get("winrate")
        return None if winrate is None else winrate * 100
    bucket = summaries.get(spec.section, {})
    if spec.key in bucket:
        return bucket.get(spec.key)
    if spec.section == "overview":
        return summaries.get("overview", {}).get(spec.key)
    return None
