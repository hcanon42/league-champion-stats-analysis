"""Rank window helpers for peer baseline filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from analysis.benchmarks import adjacent_tiers
from models import RankedEntry

MASTER_PLUS: Final[frozenset[str]] = frozenset({"MASTER", "GRANDMASTER", "CHALLENGER"})
DIVISIONS: Final[tuple[str, ...]] = ("I", "II", "III", "IV")


@dataclass(frozen=True)
class RankScope:
    """Defines which peer ranks are accepted for a baseline lookup."""

    target: RankedEntry
    widened: bool

    @property
    def allowed_tiers(self) -> set[str]:
        """Tier names included in this scope."""
        tiers = {self.target.tier.upper()}
        if self.widened:
            tiers |= adjacent_tiers(self.target.tier)
        return tiers


def build_exact_scope(ranked: RankedEntry) -> RankScope:
    """Same tier (all divisions) as the tracked player."""
    return RankScope(target=ranked, widened=False)


def build_widened_scope(ranked: RankedEntry) -> RankScope:
    """Same tier plus immediately adjacent tiers."""
    return RankScope(target=ranked, widened=True)


def rank_matches(peer_tier: str, peer_rank: str, scope: RankScope) -> bool:
    """Return whether a peer's rank falls inside the scope."""
    tier = peer_tier.upper()
    if tier not in scope.allowed_tiers:
        return False

    target_tier = scope.target.tier.upper()
    if tier in MASTER_PLUS and target_tier in MASTER_PLUS:
        if scope.widened:
            return True
        return tier == target_tier

    if tier == target_tier:
        return True

    return scope.widened


def league_lookup_pairs(scope: RankScope) -> list[tuple[str, str]]:
    """Return (tier, division) pairs to query via league-v4."""
    pairs: list[tuple[str, str]] = []
    for tier in sorted(scope.allowed_tiers):
        if tier in MASTER_PLUS:
            pairs.append((tier, ""))
        else:
            for division in DIVISIONS:
                pairs.append((tier, division))
    return pairs
