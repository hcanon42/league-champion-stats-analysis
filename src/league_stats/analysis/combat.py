"""Combat output metrics: DPM vs crowd-control score."""

from __future__ import annotations

from typing import Final, Literal

TANK_DAMAGE_SHARE_MAX: Final[float] = 0.15


def prefers_cc_over_dpm(role: str, *, avg_damage_share: float | None = None) -> bool:
    """Whether CC/min is a better headline combat metric than DPM.

    Supports always prefer CC. Other roles use CC when damage share is tank-like.
    """
    if role.upper() == "UTILITY":
        return True
    if avg_damage_share is not None and avg_damage_share <= TANK_DAMAGE_SHARE_MAX:
        return True
    return False


def combat_output_metric(
    role: str, *, avg_damage_share: float | None = None
) -> tuple[Literal["dpm", "ccpm"], Literal["DPM", "CC/min"]]:
    """Return the primary combat output metric key and dashboard label."""
    if prefers_cc_over_dpm(role, avg_damage_share=avg_damage_share):
        return ("ccpm", "CC/min")
    return ("dpm", "DPM")
