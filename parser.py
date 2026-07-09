"""Match parsing: filtering, name resolution and MatchRecord assembly.

``MatchFilter`` decides which raw matches qualify (ranked solo queue,
configured champion + lane, no remakes). ``MatchParser`` turns a qualifying match + timeline
pair into a fully populated :class:`~models.MatchRecord`, delegating
timeline-level extraction to the ``analysis`` package.
"""

from __future__ import annotations

from typing import Any, Final

from analysis.deaths import extract_deaths
from analysis.objectives import extract_objectives
from analysis.positioning import extract_positioning
from analysis.teamfights import detect_teamfights
from analysis.timeline import TimelineContext, build_context, extract_timeline_stats
from analysis.vision import extract_control_ward_lifetime
from config import AppConfig, REMAKE_MAX_DURATION_S
from models import (
    BuildTimings,
    CombatStats,
    EconomyStats,
    ItemPurchase,
    MatchRecord,
    RuneSetup,
    Side,
    VisionStats,
)
from utils import get_logger, ms_to_min, safe_div

SUMMONER_SPELLS: Final[dict[int, str]] = {
    1: "Cleanse", 3: "Exhaust", 4: "Flash", 6: "Ghost", 7: "Heal", 11: "Smite",
    12: "Teleport", 13: "Clarity", 14: "Ignite", 21: "Barrier", 32: "Snowball",
}

RUNE_STYLES: Final[dict[int, str]] = {
    8000: "Precision", 8100: "Domination", 8200: "Sorcery",
    8300: "Inspiration", 8400: "Resolve",
}

PERK_NAMES: Final[dict[int, str]] = {
    # Keystones
    8005: "Press the Attack", 8008: "Lethal Tempo", 8021: "Fleet Footwork",
    8010: "Conqueror", 8112: "Electrocute", 8128: "Dark Harvest",
    9923: "Hail of Blades", 8214: "Summon Aery", 8229: "Arcane Comet",
    8230: "Phase Rush", 8437: "Grasp of the Undying", 8439: "Aftershock",
    8465: "Guardian", 8351: "Glacial Augment", 8360: "Unsealed Spellbook",
    8369: "First Strike",
    # Common minor runes
    8009: "Presence of Mind", 9101: "Absorb Life", 9111: "Triumph",
    9104: "Legend: Alacrity", 9105: "Legend: Haste", 9103: "Legend: Bloodline",
    8014: "Coup de Grace", 8017: "Cut Down", 8299: "Last Stand",
    8126: "Cheap Shot", 8139: "Taste of Blood", 8143: "Sudden Impact",
    8137: "Sixth Sense", 8140: "Grisly Mementos", 8141: "Deep Ward",
    8135: "Treasure Hunter", 8105: "Relentless Hunter", 8106: "Ultimate Hunter",
    8224: "Nullifying Orb", 8226: "Manaflow Band", 8275: "Nimbus Cloak",
    8210: "Transcendence", 8234: "Celerity", 8233: "Absolute Focus",
    8237: "Scorch", 8232: "Waterwalking", 8236: "Gathering Storm",
    8306: "Hextech Flashtraption", 8304: "Magical Footwear", 8321: "Cash Back",
    8313: "Triple Tonic", 8352: "Time Warp Tonic", 8345: "Biscuit Delivery",
    8347: "Cosmic Insight", 8410: "Approach Velocity", 8316: "Jack of All Trades",
    8446: "Demolish", 8463: "Font of Life", 8401: "Shield Bash",
    8429: "Conditioning", 8444: "Second Wind", 8473: "Bone Plating",
    8451: "Overgrowth", 8453: "Revitalize", 8242: "Unflinching",
    # Stat shards
    5001: "Health Scaling", 5002: "Armor", 5003: "Magic Resist",
    5005: "Attack Speed", 5007: "Ability Haste", 5008: "Adaptive Force",
    5010: "Move Speed", 5011: "Health", 5013: "Tenacity & Slow Resist",
}

ELIXIR_IDS: Final[frozenset[int]] = frozenset({2138, 2139, 2140})
TRINKET_IDS: Final[frozenset[int]] = frozenset({3340, 3363, 3364, 3330})
COMPLETED_GOLD_THRESHOLD: Final[int] = 2200
SKILL_LETTERS: Final[dict[int, str]] = {1: "Q", 2: "W", 3: "E", 4: "R"}


def perk_name(perk_id: int) -> str:
    """Human-readable name for a rune/perk id.

    Args:
        perk_id: Riot perk id.

    Returns:
        The known name or ``"Perk <id>"`` for unmapped ids.
    """
    return PERK_NAMES.get(perk_id, f"Perk {perk_id}")


class ItemCatalog:
    """Item metadata lookup built from Data Dragon's ``item.json``."""

    def __init__(self, raw: dict[int, dict[str, Any]]) -> None:
        """Create the catalogue.

        Args:
            raw: Mapping of item id to raw Data Dragon item definition.
        """
        self._raw = raw

    def name(self, item_id: int) -> str:
        """Item display name, falling back to ``Item <id>``.

        Args:
            item_id: Riot item id.

        Returns:
            The item name.
        """
        data = self._raw.get(item_id)
        return str(data["name"]) if data else f"Item {item_id}"

    def is_boots(self, item_id: int) -> bool:
        """Whether an item carries the ``Boots`` tag.

        Args:
            item_id: Riot item id.

        Returns:
            ``True`` for any boots tier.
        """
        data = self._raw.get(item_id)
        return bool(data and "Boots" in data.get("tags", []))

    def is_completed(self, item_id: int) -> bool:
        """Whether an item is a completed (legendary-tier) item.

        Heuristic: builds into nothing, costs at least
        :data:`COMPLETED_GOLD_THRESHOLD` gold and is not boots.

        Args:
            item_id: Riot item id.

        Returns:
            ``True`` for completed items.
        """
        data = self._raw.get(item_id)
        if not data:
            return False
        gold_total = int(data.get("gold", {}).get("total", 0))
        return not data.get("into") and gold_total >= COMPLETED_GOLD_THRESHOLD and not self.is_boots(item_id)


class MatchFilter:
    """Filters raw matches down to ranked solo queue games on the configured champion + lane."""

    def __init__(self, config: AppConfig) -> None:
        """Create the filter.

        Args:
            config: Application configuration (champion, role, queue).
        """
        self._config = config
        self._log = get_logger("filter")

    def find_participant(self, match: dict[str, Any], puuid: str) -> dict[str, Any] | None:
        """Locate the tracked player's participant entry.

        Args:
            match: Raw match document.
            puuid: The player's PUUID.

        Returns:
            The participant dict or ``None``.
        """
        return next(
            (p for p in match["info"]["participants"] if p.get("puuid") == puuid), None
        )

    def accept(self, match: dict[str, Any], puuid: str) -> bool:
        """Whether a match qualifies for analysis.

        Requirements: ranked solo queue, the tracked player on the configured
        champion in the configured role, not a remake, no early surrender.

        Args:
            match: Raw match document.
            puuid: The player's PUUID.

        Returns:
            ``True`` when the match should be analysed.
        """
        info = match.get("info", {})
        if int(info.get("queueId", 0)) != self._config.queue_id:
            return False
        duration_s = int(info.get("gameDuration", 0))
        if duration_s > 100_000:
            duration_s //= 1000
        if duration_s <= REMAKE_MAX_DURATION_S:
            return False
        me = self.find_participant(match, puuid)
        if me is None:
            return False
        if me.get("gameEndedInEarlySurrender"):
            return False
        return (
            str(me.get("championName", "")) == self._config.champion
            and str(me.get("teamPosition", "")) == self._config.role
        )


class MatchParser:
    """Assembles :class:`~models.MatchRecord` objects from raw documents."""

    def __init__(self, catalog: ItemCatalog) -> None:
        """Create the parser.

        Args:
            catalog: Item metadata catalogue (injected dependency).
        """
        self._catalog = catalog
        self._log = get_logger("parser")

    # ------------------------------------------------------------- Assembly

    def parse(self, match: dict[str, Any], timeline: dict[str, Any], puuid: str) -> MatchRecord:
        """Parse one qualifying match + timeline pair.

        Args:
            match: Raw match-v5 match document.
            timeline: Raw match-v5 timeline document.
            puuid: The tracked player's PUUID.

        Returns:
            A fully populated :class:`~models.MatchRecord`.
        """
        info = match["info"]
        ctx = build_context(match, timeline, puuid)
        participants: list[dict[str, Any]] = info["participants"]
        me = next(p for p in participants if p["puuid"] == puuid)
        allies = [p for p in participants if p["teamId"] == me["teamId"]]
        enemies = [p for p in participants if p["teamId"] != me["teamId"]]
        opponent = (
            ctx.id_to_champion.get(ctx.opponent_id) if ctx.opponent_id is not None else None
        )

        purchases = self._purchases(ctx)
        timings = self._timings(purchases)
        skill_sequence, skill_order = self._skills(ctx)
        ult_learned_min = self._ult_learned_min(ctx)
        timeline_stats = extract_timeline_stats(ctx, int(me.get("totalTimeSpentDead", 0)))
        positioning = extract_positioning(ctx)
        timeline_stats.grouped_share = positioning["grouped_share"]
        timeline_stats.solo_share = positioning["solo_share"]
        timeline_stats.side_lane_share = positioning["side_lane_share"]
        timeline_stats.avg_allies_nearby = positioning["avg_allies_nearby"]
        deaths = extract_deaths(ctx, purchases, timeline_stats.recalls, ult_learned_min)
        shutdown_collected = sum(
            int(e.get("shutdownBounty", 0))
            for e in ctx.events_of("CHAMPION_KILL")
            if int(e.get("killerId", 0)) == ctx.participant_id
        )

        version = str(info.get("gameVersion", "0.0"))
        return MatchRecord(
            match_id=str(match["metadata"]["matchId"]),
            patch=".".join(version.split(".")[:2]),
            game_version=version,
            game_creation_ms=int(info.get("gameCreation", 0)),
            duration_s=ctx.duration_s,
            win=bool(me["win"]),
            side=Side.BLUE if me["teamId"] == 100 else Side.RED,
            lane_opponent=opponent,
            ally_comp=[str(p["championName"]) for p in allies],
            enemy_comp=[str(p["championName"]) for p in enemies],
            avg_rank=None,  # match-v5 does not expose participant ranks
            combat=self._combat(me, allies, ctx),
            economy=self._economy(me, allies, ctx),
            vision=self._vision(me, ctx),
            runes=self._runes(me),
            summoners=[
                SUMMONER_SPELLS.get(int(me.get(f"summoner{i}Id", 0)), "Unknown") for i in (1, 2)
            ],
            skill_order=skill_order,
            skill_sequence=skill_sequence,
            final_items=[
                self._catalog.name(int(me[f"item{i}"]))
                for i in range(6)
                if int(me.get(f"item{i}", 0)) > 0
            ],
            purchases=purchases,
            timings=timings,
            shutdown_gold_collected=shutdown_collected,
            shutdown_gold_given=sum(d.shutdown_given for d in deaths),
            timeline=timeline_stats,
            deaths=deaths,
            teamfights=detect_teamfights(ctx),
            objectives=extract_objectives(ctx),
        )

    # ------------------------------------------------------------ Sub-parts

    def _combat(
        self, me: dict[str, Any], allies: list[dict[str, Any]], ctx: TimelineContext
    ) -> CombatStats:
        """Build combat statistics from the participant document."""
        minutes = max(1.0, ctx.duration_s / 60.0)
        team_damage = sum(int(p.get("totalDamageDealtToChampions", 0)) for p in allies)
        team_kills = sum(int(p.get("kills", 0)) for p in allies)
        kills, deaths, assists = int(me["kills"]), int(me["deaths"]), int(me["assists"])
        challenges = me.get("challenges", {}) or {}
        return CombatStats(
            kills=kills,
            deaths=deaths,
            assists=assists,
            kda=(kills + assists) / max(1, deaths),
            damage_to_champions=int(me.get("totalDamageDealtToChampions", 0)),
            dpm=int(me.get("totalDamageDealtToChampions", 0)) / minutes,
            damage_share=safe_div(int(me.get("totalDamageDealtToChampions", 0)), team_damage),
            true_damage=int(me.get("trueDamageDealtToChampions", 0)),
            physical_damage=int(me.get("physicalDamageDealtToChampions", 0)),
            magic_damage=int(me.get("magicDamageDealtToChampions", 0)),
            healing=int(me.get("totalHealsOnTeammates", 0)) + int(me.get("totalHeal", 0)),
            shielding=int(me.get("totalDamageShieldedOnTeammates", 0)),
            cc_score=int(me.get("timeCCingOthers", 0)),
            largest_killing_spree=int(me.get("largestKillingSpree", 0)),
            double_kills=int(me.get("doubleKills", 0)),
            triple_kills=int(me.get("tripleKills", 0)),
            quadra_kills=int(me.get("quadraKills", 0)),
            penta_kills=int(me.get("pentaKills", 0)),
            kill_participation=float(
                challenges.get("killParticipation", safe_div(kills + assists, team_kills))
            ),
        )

    def _economy(
        self, me: dict[str, Any], allies: list[dict[str, Any]], ctx: TimelineContext
    ) -> EconomyStats:
        """Build economy statistics from the participant document."""
        minutes = max(1.0, ctx.duration_s / 60.0)
        team_gold = sum(int(p.get("goldEarned", 0)) for p in allies)
        cs = int(me.get("totalMinionsKilled", 0)) + int(me.get("neutralMinionsKilled", 0))
        return EconomyStats(
            gold=int(me.get("goldEarned", 0)),
            gpm=int(me.get("goldEarned", 0)) / minutes,
            gold_share=safe_div(int(me.get("goldEarned", 0)), team_gold),
            cs=cs,
            cspm=cs / minutes,
            xp=int(me.get("champExperience", 0)),
        )

    def _vision(self, me: dict[str, Any], ctx: TimelineContext) -> VisionStats:
        """Build vision statistics from the participant document + timeline."""
        minutes = max(1.0, ctx.duration_s / 60.0)
        return VisionStats(
            vision_score=int(me.get("visionScore", 0)),
            vision_score_per_min=int(me.get("visionScore", 0)) / minutes,
            wards_placed=int(me.get("wardsPlaced", 0)),
            wards_killed=int(me.get("wardsKilled", 0)),
            control_wards_bought=int(me.get("visionWardsBoughtInGame", 0)),
            avg_control_ward_lifetime_s=extract_control_ward_lifetime(ctx),
        )

    def _runes(self, me: dict[str, Any]) -> RuneSetup:
        """Build the rune page from the participant ``perks`` block."""
        perks = me.get("perks", {}) or {}
        styles = perks.get("styles", []) or []
        primary = styles[0] if styles else {}
        secondary = styles[1] if len(styles) > 1 else {}
        primary_sel = [int(s["perk"]) for s in primary.get("selections", [])]
        secondary_sel = [int(s["perk"]) for s in secondary.get("selections", [])]
        stat_perks = perks.get("statPerks", {}) or {}
        return RuneSetup(
            keystone=perk_name(primary_sel[0]) if primary_sel else "Unknown",
            primary_tree=RUNE_STYLES.get(int(primary.get("style", 0)), "Unknown"),
            secondary_tree=RUNE_STYLES.get(int(secondary.get("style", 0)), "Unknown"),
            primary_runes=[perk_name(p) for p in primary_sel[1:]],
            secondary_runes=[perk_name(p) for p in secondary_sel],
            shards=[
                perk_name(int(stat_perks.get(slot, 0)))
                for slot in ("offense", "flex", "defense")
                if stat_perks.get(slot)
            ],
        )

    def _purchases(self, ctx: TimelineContext) -> list[ItemPurchase]:
        """Reconstruct the purchase timeline, honouring undo events."""
        purchases: list[ItemPurchase] = []
        for event in ctx.events:
            if int(event.get("participantId", 0)) != ctx.participant_id:
                continue
            if event.get("type") == "ITEM_PURCHASED":
                item_id = int(event.get("itemId", 0))
                purchases.append(
                    ItemPurchase(
                        minute=ms_to_min(int(event["timestamp"])),
                        item_id=item_id,
                        item_name=self._catalog.name(item_id),
                        is_completed=self._catalog.is_completed(item_id),
                        is_boots=self._catalog.is_boots(item_id),
                        is_elixir=item_id in ELIXIR_IDS,
                        is_trinket=item_id in TRINKET_IDS,
                    )
                )
            elif event.get("type") == "ITEM_UNDO":
                undone = int(event.get("beforeId", 0))
                for index in range(len(purchases) - 1, -1, -1):
                    if purchases[index].item_id == undone:
                        purchases.pop(index)
                        break
        return purchases

    def _timings(self, purchases: list[ItemPurchase]) -> BuildTimings:
        """Derive power-spike timings from the purchase timeline."""
        completed = [p for p in purchases if p.is_completed]
        boots = [p for p in purchases if p.is_boots]
        ordered = completed[:3]
        return BuildTimings(
            boots_min=boots[0].minute if boots else None,
            boots=boots[-1].item_name if boots else None,
            first_item_min=ordered[0].minute if len(ordered) > 0 else None,
            first_item=ordered[0].item_name if len(ordered) > 0 else None,
            second_item_min=ordered[1].minute if len(ordered) > 1 else None,
            second_item=ordered[1].item_name if len(ordered) > 1 else None,
            third_item_min=ordered[2].minute if len(ordered) > 2 else None,
            third_item=ordered[2].item_name if len(ordered) > 2 else None,
            elixirs_bought=sum(1 for p in purchases if p.is_elixir),
            trinket_swaps=sum(1 for p in purchases if p.is_trinket and p.minute > 2),
        )

    def _skills(self, ctx: TimelineContext) -> tuple[list[str], str]:
        """Derive the raw skill sequence and the max order (e.g. ``Q>E>W``)."""
        events = [
            e
            for e in ctx.events_of("SKILL_LEVEL_UP")
            if int(e.get("participantId", 0)) == ctx.participant_id
        ]
        sequence = [SKILL_LETTERS.get(int(e.get("skillSlot", 0)), "?") for e in events]
        points: dict[str, int] = {"Q": 0, "W": 0, "E": 0}
        max_order: list[str] = []
        for letter in sequence:
            if letter not in points:
                continue
            points[letter] += 1
            if points[letter] == 5 and letter not in max_order:
                max_order.append(letter)
        for letter, _ in sorted(points.items(), key=lambda kv: -kv[1]):
            if letter not in max_order:
                max_order.append(letter)
        return sequence, ">".join(max_order)

    def _ult_learned_min(self, ctx: TimelineContext) -> float | None:
        """Minute the ultimate (slot 4) was first skilled, if ever."""
        for event in ctx.events_of("SKILL_LEVEL_UP"):
            if (
                int(event.get("participantId", 0)) == ctx.participant_id
                and int(event.get("skillSlot", 0)) == 4
            ):
                return ms_to_min(int(event["timestamp"]))
        return None
