"""Typed domain models for Viktor Analyzer.

Every entity extracted from the Riot Match-V5 API is normalised into one of
the Pydantic models below. ``MatchRecord`` is the aggregate root: one fully
parsed, analysis-ready ranked Viktor game.

Fields that cannot be derived from the public API (e.g. summoner-spell
cooldowns) are typed ``Optional`` and documented as heuristics or unknowns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Side(StrEnum):
    """Map side the player spawned on."""

    BLUE = "blue"
    RED = "red"


class Zone(StrEnum):
    """Coarse Summoner's Rift zone classification."""

    BASE = "base"
    RIVER = "river"
    MID_LANE = "mid"
    TOP_LANE = "top"
    BOT_LANE = "bot"
    JUNGLE = "jungle"


class ObjectiveKind(StrEnum):
    """Epic monster objective types tracked by the analyzer."""

    DRAGON = "dragon"
    ELDER = "elder"
    BARON = "baron"
    HERALD = "herald"
    GRUBS = "grubs"


class Position(BaseModel):
    """A point on the Summoner's Rift map in Riot map units."""

    x: float
    y: float


class RuneSetup(BaseModel):
    """Full rune page used in a game."""

    keystone: str
    primary_tree: str
    secondary_tree: str
    primary_runes: list[str] = Field(default_factory=list)
    secondary_runes: list[str] = Field(default_factory=list)
    shards: list[str] = Field(default_factory=list)


class ItemPurchase(BaseModel):
    """A single item purchase from the match timeline."""

    minute: float
    item_id: int
    item_name: str
    is_completed: bool = False
    is_boots: bool = False
    is_elixir: bool = False
    is_trinket: bool = False


class BuildTimings(BaseModel):
    """Key power-spike timings inferred from the purchase timeline."""

    boots_min: float | None = None
    first_item_min: float | None = None
    second_item_min: float | None = None
    third_item_min: float | None = None
    first_item: str | None = None
    second_item: str | None = None
    third_item: str | None = None
    boots: str | None = None
    elixirs_bought: int = 0
    trinket_swaps: int = 0


class RecallEvent(BaseModel):
    """An inferred recall (shopping trip), derived from purchase clusters.

    ``unspent_gold`` is the player's banked gold on the last timeline frame
    before the shopping trip started.
    """

    minute: float
    unspent_gold: int


class RoamEvent(BaseModel):
    """An inferred early-game roam away from mid lane."""

    start_minute: float
    end_minute: float
    zone: Zone


class SnapshotSet(BaseModel):
    """Per-minute checkpoint metrics and lane differentials.

    Keys are minute marks (5/10/15/20). Diff values are ``None`` when no lane
    opponent could be identified.
    """

    gold: dict[int, int] = Field(default_factory=dict)
    xp: dict[int, int] = Field(default_factory=dict)
    cs: dict[int, int] = Field(default_factory=dict)
    gold_diff: dict[int, int | None] = Field(default_factory=dict)
    xp_diff: dict[int, int | None] = Field(default_factory=dict)
    cs_diff: dict[int, int | None] = Field(default_factory=dict)


class TimelineStats(BaseModel):
    """All metrics derived from the Match-V5 timeline for the player."""

    snapshots: SnapshotSet = Field(default_factory=SnapshotSet)
    recalls: list[RecallEvent] = Field(default_factory=list)
    roams: list[RoamEvent] = Field(default_factory=list)
    first_recall_min: float | None = None
    avg_unspent_gold_before_recall: float | None = None
    time_dead_s: int = 0
    lane_priority: float | None = None
    wave_push_ratio: float | None = None
    gold_series: list[int] = Field(default_factory=list)
    xp_series: list[int] = Field(default_factory=list)
    cs_series: list[int] = Field(default_factory=list)
    opp_gold_series: list[int] = Field(default_factory=list)
    grouped_share: float | None = None
    solo_share: float | None = None
    side_lane_share: float | None = None
    avg_allies_nearby: float | None = None
    avg_teammate_distance: float | None = None
    role_distances: dict[str, float] = Field(default_factory=dict)
    early_ganks: int = 0
    gank_assists: int = 0
    kp15: float | None = None
    roam_conversions: int = 0
    vspm10: float | None = None


class DeathEvent(BaseModel):
    """One death, fully contextualised.

    ``flash_available`` and ``enemy_seen`` cannot be derived from the public
    API and stay ``None``; ``ult_available`` means "R learned by then" and
    ``zhonya_available`` means "Zhonya's/Stopwatch in inventory" (cooldowns
    are not exposed).
    """

    minute: float
    position: Position
    zone: Zone
    near_objective: bool = False
    shutdown_given: int = 0
    bounty_held: bool = False
    flash_available: bool | None = None
    ult_available: bool | None = None
    zhonya_available: bool = False
    alone: bool = False
    outnumbered: bool = False
    team_wards_recent: int = 0
    enemy_seen: bool | None = None
    after_greed: bool = False
    after_tower: bool = False
    after_objective: bool = False
    side_lane_push: bool = False
    before_dragon: bool = False
    before_baron: bool = False
    before_neutral_objective: bool = False
    after_recall: bool = False
    to_gank: bool = False  # laning-phase lane death with a roaming enemy involved
    under_own_tower_laning: bool = False
    under_enemy_tower_laning: bool = False
    killer_champion: str | None = None
    current_gold: int | None = None
    avg_teammate_distance: float | None = None


class TeamfightRecord(BaseModel):
    """A detected teamfight and the player's involvement in it.

    ``damage_taken`` is only known when the player died in the fight (the
    timeline exposes damage received solely on kill events). ``enemies_hit_by_ult``
    is not derivable from the public API and stays ``None``.
    """

    start_minute: float
    end_minute: float
    participated: bool
    kills: int = 0
    assists: int = 0
    died: bool = False
    damage_dealt: int = 0
    damage_taken: int | None = None
    time_alive_s: float = 0.0
    centroid: Position | None = None
    front_to_back: float | None = None
    enemies_hit_by_ult: int | None = None
    ally_kills: int = 0
    enemy_kills: int = 0
    won: bool | None = None
    unspent_gold: int | None = None
    allies_present: int | None = None
    enemies_present: int | None = None
    manpower_advantage: int | None = None
    avg_teammate_distance: float | None = None


class ObjectiveRecord(BaseModel):
    """A dragon/baron/herald/grubs/elder take and the player's context."""

    minute: float
    kind: ObjectiveKind
    taken_by_team: bool
    present: bool = False
    arrived_early: bool = False
    arrived_late: bool = False
    dead_before: bool = False
    wards_before: int = 0
    control_wards_before: int = 0


class CombatStats(BaseModel):
    """Combat output for the game."""

    kills: int
    deaths: int
    assists: int
    kda: float
    damage_to_champions: int
    dpm: float
    damage_share: float
    damage_taken: int
    damage_taken_share: float
    true_damage: int
    physical_damage: int
    magic_damage: int
    healing: int
    shielding: int
    cc_score: int
    largest_killing_spree: int
    double_kills: int
    triple_kills: int
    quadra_kills: int
    penta_kills: int
    kill_participation: float


class EconomyStats(BaseModel):
    """Income and farming for the game."""

    gold: int
    gpm: float
    gold_share: float
    cs: int
    cspm: float
    xp: int


class VisionStats(BaseModel):
    """Vision metrics for the game."""

    vision_score: int
    vision_score_per_min: float
    wards_placed: int
    wards_killed: int
    control_wards_bought: int
    avg_control_ward_lifetime_s: float | None = None


class MatchRecord(BaseModel):
    """A fully parsed, analysis-ready ranked queue game."""

    match_id: str
    patch: str
    game_version: str
    game_creation_ms: int
    queue_id: int
    duration_s: int
    champion: str
    role: str
    win: bool
    side: Side
    lane_opponent: str | None = None
    ally_comp: list[str] = Field(default_factory=list)
    enemy_comp: list[str] = Field(default_factory=list)
    avg_rank: str | None = None

    combat: CombatStats
    economy: EconomyStats
    vision: VisionStats
    runes: RuneSetup
    summoners: list[str] = Field(default_factory=list)
    skill_order: str = ""
    skill_sequence: list[str] = Field(default_factory=list)

    final_items: list[str] = Field(default_factory=list)
    item_path: list[str] = Field(default_factory=list)
    purchases: list[ItemPurchase] = Field(default_factory=list)
    timings: BuildTimings = Field(default_factory=BuildTimings)
    shutdown_gold_collected: int = 0
    shutdown_gold_given: int = 0

    timeline: TimelineStats = Field(default_factory=TimelineStats)
    deaths: list[DeathEvent] = Field(default_factory=list)
    teamfights: list[TeamfightRecord] = Field(default_factory=list)
    objectives: list[ObjectiveRecord] = Field(default_factory=list)
    key_moments: list[KeyMoment] = Field(default_factory=list)

    @property
    def duration_min(self) -> float:
        """Game duration in fractional minutes."""
        return self.duration_s / 60.0

    def deaths_before(self, minute: float) -> int:
        """Count deaths that happened before ``minute``.

        Args:
            minute: Cut-off minute mark.

        Returns:
            Number of deaths strictly before the cut-off.
        """
        return sum(1 for d in self.deaths if d.minute < minute)

    def to_row(self) -> dict[str, Any]:
        """Flatten the record into a single row for tabular analysis.

        Returns:
            A dict of scalar features keyed by column name, suitable for
            building the master :class:`pandas.DataFrame`.
        """
        snap = self.timeline.snapshots
        fights = [f for f in self.teamfights if f.participated]
        row: dict[str, Any] = {
            "match_id": self.match_id,
            "patch": self.patch,
            "game_creation_ms": self.game_creation_ms,
            "queue_id": self.queue_id,
            "duration_min": round(self.duration_min, 2),
            "win": int(self.win),
            "side": self.side.value,
            "opponent": self.lane_opponent or "Unknown",
            "kills": self.combat.kills,
            "deaths": self.combat.deaths,
            "assists": self.combat.assists,
            "kda": round(self.combat.kda, 2),
            "dpm": round(self.combat.dpm, 1),
            "damage": self.combat.damage_to_champions,
            "damage_share": round(self.combat.damage_share, 4),
            "damage_taken": self.combat.damage_taken,
            "damage_taken_share": round(self.combat.damage_taken_share, 4),
            "kill_participation": round(self.combat.kill_participation, 4),
            "cc_score": self.combat.cc_score,
            "ccpm": round(self.combat.cc_score / max(1.0, self.duration_min), 2),
            "healing": self.combat.healing,
            "shielding": self.combat.shielding,
            "true_damage": self.combat.true_damage,
            "physical_damage": self.combat.physical_damage,
            "magic_damage": self.combat.magic_damage,
            "largest_spree": self.combat.largest_killing_spree,
            "multikills": (
                self.combat.double_kills
                + self.combat.triple_kills
                + self.combat.quadra_kills
                + self.combat.penta_kills
            ),
            "gold": self.economy.gold,
            "gpm": round(self.economy.gpm, 1),
            "gold_share": round(self.economy.gold_share, 4),
            "cs": self.economy.cs,
            "cspm": round(self.economy.cspm, 2),
            "xp": self.economy.xp,
            "vision_score": self.vision.vision_score,
            "vspm": round(self.vision.vision_score_per_min, 2),
            "wards_placed": self.vision.wards_placed,
            "wards_killed": self.vision.wards_killed,
            "control_wards": self.vision.control_wards_bought,
            "keystone": self.runes.keystone,
            "secondary_tree": self.runes.secondary_tree,
            "skill_order": self.skill_order,
            "summoners": "+".join(self.summoners),
            "first_item": self.timings.first_item,
            "second_item": self.timings.second_item,
            "third_item": self.timings.third_item,
            "boots": self.timings.boots,
            "first_item_min": self.timings.first_item_min,
            "second_item_min": self.timings.second_item_min,
            "third_item_min": self.timings.third_item_min,
            "boots_min": self.timings.boots_min,
            "elixirs": self.timings.elixirs_bought,
            "trinket_swaps": self.timings.trinket_swaps,
            "shutdown_collected": self.shutdown_gold_collected,
            "shutdown_given": self.shutdown_gold_given,
            "recalls": len(self.timeline.recalls),
            "first_recall_min": self.timeline.first_recall_min,
            "avg_unspent_gold": self.timeline.avg_unspent_gold_before_recall,
            "roams_pre15": len(self.timeline.roams),
            "time_dead_s": self.timeline.time_dead_s,
            "lane_priority": self.timeline.lane_priority,
            "wave_push_ratio": self.timeline.wave_push_ratio,
            "grouped_share": self.timeline.grouped_share,
            "solo_share": self.timeline.solo_share,
            "side_lane_share": self.timeline.side_lane_share,
            "avg_teammate_distance": self.timeline.avg_teammate_distance,
            "dist_top": self.timeline.role_distances.get("TOP"),
            "dist_jungle": self.timeline.role_distances.get("JUNGLE"),
            "dist_middle": self.timeline.role_distances.get("MIDDLE"),
            "dist_bottom": self.timeline.role_distances.get("BOTTOM"),
            "dist_support": self.timeline.role_distances.get("UTILITY"),
            "avg_gold_at_death": (
                round(sum(d.current_gold for d in self.deaths if d.current_gold is not None) / len(self.deaths), 0)
                if self.deaths and any(d.current_gold is not None for d in self.deaths)
                else None
            ),
            "deaths_pre14": self.deaths_before(14),
            "deaths_pre20": self.deaths_before(20),
            "solo_deaths": sum(1 for d in self.deaths if d.alone),
            "outnumbered_deaths": sum(1 for d in self.deaths if d.outnumbered),
            "greed_deaths": sum(1 for d in self.deaths if d.after_greed),
            "gank_deaths_laning": sum(1 for d in self.deaths if d.to_gank),
            "under_own_tower_laning_deaths": sum(
                1 for d in self.deaths if d.under_own_tower_laning
            ),
            "under_enemy_tower_laning_deaths": sum(
                1 for d in self.deaths if d.under_enemy_tower_laning
            ),
            "side_lane_deaths": sum(1 for d in self.deaths if d.side_lane_push),
            "deaths_before_neutral_objective": sum(
                1 for d in self.deaths if d.before_neutral_objective
            ),
            "teamfights": len(self.teamfights),
            "tf_participation": (
                round(len(fights) / len(self.teamfights), 3) if self.teamfights else None
            ),
            "tf_won_share": (
                round(sum(1 for f in fights if f.won) / len(fights), 3) if fights else None
            ),
            "avg_unspent_gold_per_fight": (
                round(
                    sum(f.unspent_gold for f in fights if f.unspent_gold is not None) / len(fights),
                    0,
                )
                if fights and any(f.unspent_gold is not None for f in fights)
                else None
            ),
            "fights_advantaged": sum(
                1 for f in fights if f.manpower_advantage is not None and f.manpower_advantage > 0
            ),
            "fights_disadvantaged": sum(
                1 for f in fights if f.manpower_advantage is not None and f.manpower_advantage < 0
            ),
            "pct_advantaged_fights": (
                round(
                    sum(1 for f in fights if f.manpower_advantage is not None and f.manpower_advantage > 0)
                    / len(fights),
                    3,
                )
                if fights
                else None
            ),
            "objectives_present_rate": (
                round(
                    sum(1 for o in self.objectives if o.present) / len(self.objectives), 3
                )
                if self.objectives
                else None
            ),
            "early_ganks": self.timeline.early_ganks,
            "gank_assists": self.timeline.gank_assists,
            "kp15": self.timeline.kp15,
            "roam_conversions": self.timeline.roam_conversions,
            "vspm10": self.timeline.vspm10,
            "avg_wards_before_objective": (
                round(
                    sum(o.wards_before for o in self.objectives) / len(self.objectives), 1
                )
                if self.objectives
                else None
            ),
            "hpm": round(self.combat.healing / max(1.0, self.duration_min), 1),
            "spm": round(self.combat.shielding / max(1.0, self.duration_min), 1),
        }
        for minute in (5, 10, 15, 20):
            row[f"gold{minute}"] = snap.gold.get(minute)
            row[f"gd{minute}"] = snap.gold_diff.get(minute)
        for minute in (5, 10, 15):
            row[f"xp{minute}"] = snap.xp.get(minute)
            row[f"xpd{minute}"] = snap.xp_diff.get(minute)
            row[f"cs{minute}"] = snap.cs.get(minute)
            row[f"csd{minute}"] = snap.cs_diff.get(minute)
        return row


class RecommendationTone(StrEnum):
    """Whether a coaching tip highlights a strength or a weakness."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Recommendation(BaseModel):
    """A single coaching recommendation ranked by statistical evidence."""

    category: str
    title: str
    detail: str
    evidence: str
    tone: RecommendationTone = RecommendationTone.NEGATIVE
    p_value: float | None = None
    effect_size: float | None = None
    priority: float = 0.0
    sample_size: int = 0


class RankedEntry(BaseModel):
    """Ranked solo queue standing from league-v4."""

    tier: str
    rank: str
    league_points: int
    wins: int
    losses: int

    @property
    def label(self) -> str:
        """Human-readable rank label (e.g. ``Gold II``)."""
        if self.tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            return f"{self.tier.title()} {self.league_points} LP"
        return f"{self.tier.title()} {self.rank}"


class MetricComparison(BaseModel):
    """One metric compared between the player and rank peers."""

    metric: str
    label: str
    yours: float
    peer_avg: float
    delta: float
    delta_pct: float | None
    direction: str  # "higher" or "lower" (whether bigger is better)
    verdict: str  # "above", "below", "inline"


class PeerComparisonResult(BaseModel):
    """Full rank-peer comparison for the report and exports."""

    rank_label: str
    tier: str
    champion: str = "Viktor"
    role: str = "MIDDLE"
    build_label: str = "Viktor mid"
    source: str
    peer_games: int
    peer_players: int
    confidence: str = "high"
    fallback_level: int = 0
    comparisons: list[MetricComparison] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class MetricDelta(BaseModel):
    """One metric compared between recent form and a personal baseline window."""

    metric: str
    label: str
    section: str
    recent: float
    baseline: float
    delta: float
    delta_pct: float | None
    direction: Literal["higher", "lower"]
    verdict: Literal["improved", "regressed", "inline"]
    p_value: float | None = None
    effect_size: float | None = None
    significant: bool = False
    recent_n: int = 0
    baseline_n: int = 0


class FormSnapshot(BaseModel):
    """Headline progression summary for the Form Tracker hero panel."""

    form_score: float
    trend: Literal["improving", "declining", "stable"]
    confidence: Literal["high", "medium", "low", "insufficient"]
    recent_games: int
    baseline_games: int
    recent_winrate: float
    baseline_winrate: float
    winrate_delta_pp: float
    winrate_ci_low: float | None = None
    winrate_ci_high: float | None = None
    current_streak: str = ""
    headline: str = ""


class ProgressionComparison(BaseModel):
    """Full recent-vs-baseline comparison for one build and preset."""

    preset_key: str
    overlap_mode: Literal["exclusive", "inclusive"]
    recent_n: int
    baseline_m: int
    role: str
    build_label: str
    snapshot: FormSnapshot
    deltas: list[MetricDelta] = Field(default_factory=list)
    top_improved: list[MetricDelta] = Field(default_factory=list)
    top_regressed: list[MetricDelta] = Field(default_factory=list)
    behavioral_shifts: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


class GameScoreBreakdown(BaseModel):
    """Per-game score across role-aware dimensions (0–100 each)."""

    overall: int
    tier: str
    laning: int
    survival: int
    impact: int
    vision: int
    objectives: int


class GameBehavior(BaseModel):
    """One positive or negative per-game coaching bullet."""

    tone: Literal["positive", "negative"]
    title: str
    detail: str
    anchor: str | None = None


class GameComparisonRow(BaseModel):
    """One metric compared for a single game."""

    metric: str
    label: str
    game_value: float
    benchmark_value: float
    delta: float
    verdict: str


class GameDeathRow(BaseModel):
    """One death event in a game review detail panel."""

    minute: float
    zone: str
    killer: str | None
    killer_icon: str | None = None
    flags: list[str] = Field(default_factory=list)


class GameFightRow(BaseModel):
    """One teamfight the player participated in."""

    start_minute: float
    kills: int
    deaths: int
    assists: int
    damage: int
    fight_won: bool
    allies_present: int | None = None
    enemies_present: int | None = None
    manpower_advantage: int | None = None


class GameObjectiveRow(BaseModel):
    """One epic objective spawn and player presence."""

    kind: str
    minute: float
    taken_by_team: bool
    present: bool
    dead_before: bool
    wards_before: int
    objective_icon: str | None = None


class GameBuildInfo(BaseModel):
    """Build snapshot for one game."""

    keystone: str
    primary_tree: str = ""
    secondary_tree: str
    summoners: list[str] = Field(default_factory=list)
    skill_order: str = ""
    items: list[str] = Field(default_factory=list)
    keystone_icon: str | None = None
    primary_tree_icon: str | None = None
    secondary_tree_icon: str | None = None
    summoner_icons: list[str | None] = Field(default_factory=list)
    item_icons: list[str | None] = Field(default_factory=list)


class MapParticipantPin(BaseModel):
    """One champion pin on the minimap scrubber."""

    participant_id: int
    champion: str
    team_id: int
    x: float
    y: float
    dead: bool = False
    champion_icon: str | None = None


class MapObjectivePin(BaseModel):
    """Objective pit marker on the minimap scrubber."""

    kind: str
    x: float
    y: float
    highlighted: bool = False
    available: bool = True
    objective_icon: str | None = None


class KeyMomentFrame(BaseModel):
    """One discrete map snapshot at a known timeline timestamp."""

    timestamp_ms: int
    label: str = ""
    participants: list[MapParticipantPin] = Field(default_factory=list)
    objectives: list[MapObjectivePin] = Field(default_factory=list)


class KeyMoment(BaseModel):
    """A high-impact team moment with an interactive map scrub window."""

    id: str
    kind: str
    headline: str
    beneficiary: Literal["ally", "enemy"]
    gold_swing: int | None = None
    anchor_ms: int
    anchor_minute: float
    window_start_ms: int
    window_end_ms: int
    frames: list[KeyMomentFrame] = Field(default_factory=list)


class GameDetail(BaseModel):
    """Full deep-dive payload for one ranked game."""

    match_id: str
    index: int
    date: str
    queue: str
    result: Literal["win", "loss"]
    duration_min: float
    patch: str
    opponent: str
    side: str
    kda: str
    archetype: str
    champion_icon: str | None = None
    opponent_icon: str | None = None
    score: GameScoreBreakdown
    behaviors_good: list[GameBehavior] = Field(default_factory=list)
    behaviors_bad: list[GameBehavior] = Field(default_factory=list)
    vs_baseline: list[GameComparisonRow] = Field(default_factory=list)
    key_stats: dict[str, float | int | None] = Field(default_factory=dict)
    deaths: list[GameDeathRow] = Field(default_factory=list)
    fights: list[GameFightRow] = Field(default_factory=list)
    objectives: list[GameObjectiveRow] = Field(default_factory=list)
    build: GameBuildInfo
    timeline: list[dict[str, float]] = Field(default_factory=list)
    timeline_figure: str = ""
    key_moments: list[KeyMoment] = Field(default_factory=list)
    map_background: str | None = None
    ai_recap: str | None = None


class GameReviewQueueBundle(BaseModel):
    """Last-N games for one queue filter."""

    available: bool
    games_count: int
    games: list[GameDetail] = Field(default_factory=list)


class GameReviewPayload(BaseModel):
    """Per-queue game review bundles embedded in the report."""

    recent_n: int = 5
    baseline_m: int = 80
    scoring: Literal["personal"] = "personal"
    queues: dict[str, GameReviewQueueBundle] = Field(default_factory=dict)
