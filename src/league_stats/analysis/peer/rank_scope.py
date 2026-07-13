"""Rank window helpers for peer baseline filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from league_stats.analysis.peer.benchmarks import adjacent_tiers
from league_stats.core.models import RankedEntry

MASTER_PLUS: Final[frozenset[str]] = frozenset({"MASTER", "GRANDMASTER", "CHALLENGER"})
DIVISIONS: Final[tuple[str, ...]] = ("I", "II", "III", "IV")


@dataclass(frozen=True)
class RankScope:
    """Defines which peer ranks are accepted for a baseline lookup."""

    target: RankedEntry
    widened: bool
    extra_tiers: frozenset[str] = field(default_factory=frozenset)

    @property
    def allowed_tiers(self) -> set[str]:
        """Tier names included in this scope."""
        tiers = {self.target.tier.upper()}
        if self.widened:
            tiers |= adjacent_tiers(self.target.tier)
        tiers |= self.extra_tiers
        return tiers


def build_exact_scope(ranked: RankedEntry) -> RankScope:
    """Same tier (all divisions) as the tracked player."""
    return RankScope(target=ranked, widened=False)


def build_widened_scope(ranked: RankedEntry) -> RankScope:
    """Same tier plus immediately adjacent tiers (±1)."""
    return RankScope(target=ranked, widened=True)


def build_wider_scope(ranked: RankedEntry) -> RankScope:
    """Same tier plus up to two adjacent tiers in each direction (±2).

    Used as a last-resort store fallback before static JSON benchmarks.
    """
    first_ring = adjacent_tiers(ranked.tier)
    second_ring: set[str] = set()
    for t in first_ring:
        second_ring |= adjacent_tiers(t)
    extra = frozenset(second_ring - first_ring - {ranked.tier.upper()})
    return RankScope(target=ranked, widened=True, extra_tiers=extra)


def rank_matches(peer_tier: str, peer_rank: str, scope: RankScope) -> bool:
    """Return whether a peer's rank falls inside the scope."""
    tier = peer_tier.upper()
    if tier not in scope.allowed_tiers:
        return False

    target_tier = scope.target.tier.upper()
    if tier in MASTER_PLUS and target_tier in MASTER_PLUS:
        if scope.widened or tier in scope.extra_tiers:
            return True
        return tier == target_tier

    if tier == target_tier:
        return True

    return scope.widened or tier in scope.extra_tiers


def league_lookup_pairs(scope: RankScope) -> list[tuple[str, str]]:
    """Return (tier, division) pairs to query via league-v4.

    The player's exact tier+division is placed first so seed PUUIDs are
    rank-relevant from the start and fewer match downloads are wasted on
    out-of-scope players.
    """
    target_tier = scope.target.tier.upper()
    target_div = (scope.target.rank or "").upper()

    pairs: list[tuple[str, str]] = []

    # Player's exact division first (most relevant seeds)
    if target_tier not in MASTER_PLUS and target_div:
        pairs.append((target_tier, target_div))

    # Remaining divisions of the target tier
    for div in DIVISIONS:
        if div != target_div:
            if target_tier not in MASTER_PLUS:
                pairs.append((target_tier, div))

    # Master+ exact tier
    if target_tier in MASTER_PLUS:
        pairs.append((target_tier, ""))

    # Adjacent / extra tiers (widened and wider scope)
    other_tiers = sorted(scope.allowed_tiers - {target_tier})
    for tier in other_tiers:
        if tier in MASTER_PLUS:
            pairs.append((tier, ""))
        else:
            for div in DIVISIONS:
                pairs.append((tier, div))

    return pairs
