"""Shared utilities: logging, Summoner's Rift geometry, math helpers.

All map geometry is expressed in Riot's map units. Summoner's Rift spans
roughly ``(0, 0)`` (blue fountain corner) to ``(14870, 14870)`` (red fountain
corner). Mid lane runs along the main diagonal ``y = x`` and the river along
the anti-diagonal ``x + y = MAP_SIZE``.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Final

from league_stats.core.models import Position, Zone

MAP_SIZE: Final[float] = 14870.0
DRAGON_PIT: Final[Position] = Position(x=9866, y=4414)
BARON_PIT: Final[Position] = Position(x=5007, y=10471)
BLUE_FOUNTAIN: Final[Position] = Position(x=554, y=581)
RED_FOUNTAIN: Final[Position] = Position(x=14300, y=14391)

RIVER_HALF_WIDTH: Final[float] = 950.0
# The river only spans the central part of the anti-diagonal, roughly between
# the two pits; without this bound the corners of the map would classify as river.
RIVER_SPAN_MIN: Final[float] = 3200.0
RIVER_SPAN_MAX: Final[float] = 11670.0
MID_HALF_WIDTH: Final[float] = 1150.0
LANE_EDGE: Final[float] = 1650.0
BASE_RADIUS: Final[float] = 3200.0
OBJECTIVE_RADIUS: Final[float] = 2500.0
LANING_PHASE_END_MIN: Final[float] = 14.0
TOWER_RADIUS: Final[float] = 1150.0
# Outer, inner and inhibitor turrets on Summoner's Rift (map units).
BLUE_LANE_TOWER_POSITIONS: Final[tuple[Position, ...]] = (
    Position(x=10504, y=1029),
    Position(x=10505, y=2546),
    Position(x=10481, y=4610),
    Position(x=5846, y=6396),
    Position(x=5048, y=4812),
    Position(x=3651, y=3696),
    Position(x=992, y=10441),
    Position(x=2546, y=10504),
    Position(x=4610, y=10481),
)
RED_LANE_TOWER_POSITIONS: Final[tuple[Position, ...]] = (
    Position(x=13866, y=4505),
    Position(x=13327, y=8220),
    Position(x=13696, y=10504),
    Position(x=8955, y=8510),
    Position(x=9767, y=10113),
    Position(x=11134, y=11223),
    Position(x=4318, y=13875),
    Position(x=7943, y=13411),
    Position(x=10481, y=13650),
)
LANE_TOWER_POSITIONS: Final[tuple[Position, ...]] = (
    *BLUE_LANE_TOWER_POSITIONS,
    *RED_LANE_TOWER_POSITIONS,
)

LOGGER_NAME: Final[str] = "league_champion_analyzer"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        verbose: When ``True`` the log level is DEBUG, otherwise INFO.

    Returns:
        The configured application-wide logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def get_logger(child: str | None = None) -> logging.Logger:
    """Return the application logger or one of its children.

    Args:
        child: Optional child logger suffix (e.g. ``"riot_api"``).

    Returns:
        A :class:`logging.Logger` instance.
    """
    name = LOGGER_NAME if child is None else f"{LOGGER_NAME}.{child}"
    return logging.getLogger(name)


def distance(a: Position, b: Position) -> float:
    """Euclidean distance between two map positions.

    Args:
        a: First position.
        b: Second position.

    Returns:
        Distance in map units.
    """
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_zone(pos: Position) -> Zone:
    """Classify a map position into a coarse Summoner's Rift zone.

    Classification priority: base > mid lane > river > side lanes > jungle.
    Mid lane wins over river where they cross (the centre of the map), and
    the river band is limited to the central span between the two pits.

    Args:
        pos: The position to classify.

    Returns:
        The :class:`~models.Zone` the position belongs to.
    """
    if distance(pos, BLUE_FOUNTAIN) < BASE_RADIUS or distance(pos, RED_FOUNTAIN) < BASE_RADIUS:
        return Zone.BASE
    if abs(pos.x - pos.y) < MID_HALF_WIDTH:
        return Zone.MID_LANE
    if (
        abs(pos.x + pos.y - MAP_SIZE) < RIVER_HALF_WIDTH
        and RIVER_SPAN_MIN < pos.x < RIVER_SPAN_MAX
    ):
        return Zone.RIVER
    if pos.x < LANE_EDGE or pos.y > MAP_SIZE - LANE_EDGE:
        return Zone.TOP_LANE
    if pos.y < LANE_EDGE or pos.x > MAP_SIZE - LANE_EDGE:
        return Zone.BOT_LANE
    return Zone.JUNGLE


def is_side_lane(zone: Zone) -> bool:
    """Whether a zone is one of the two side lanes.

    Args:
        zone: Zone to check.

    Returns:
        ``True`` for top or bot lane.
    """
    return zone in (Zone.TOP_LANE, Zone.BOT_LANE)


def near_lane_tower(pos: Position, radius: float = TOWER_RADIUS) -> bool:
    """Whether a position lies within turret range of any lane tower.

    Args:
        pos: Position to test.
        radius: Distance threshold in map units.

    Returns:
        ``True`` when within ``radius`` of any lane turret platform.
    """
    return any(distance(pos, tower) < radius for tower in LANE_TOWER_POSITIONS)


def near_own_lane_tower(pos: Position, blue_side: bool, radius: float = TOWER_RADIUS) -> bool:
    """Whether a position lies within range of one of the player's lane turrets."""
    towers = BLUE_LANE_TOWER_POSITIONS if blue_side else RED_LANE_TOWER_POSITIONS
    return any(distance(pos, tower) < radius for tower in towers)


def near_enemy_lane_tower(pos: Position, blue_side: bool, radius: float = TOWER_RADIUS) -> bool:
    """Whether a position lies within range of an enemy lane turret."""
    towers = RED_LANE_TOWER_POSITIONS if blue_side else BLUE_LANE_TOWER_POSITIONS
    return any(distance(pos, tower) < radius for tower in towers)


def near_major_objective(pos: Position) -> bool:
    """Whether a position lies near the dragon or baron/herald pit.

    Args:
        pos: Position to test.

    Returns:
        ``True`` when within :data:`OBJECTIVE_RADIUS` of either pit.
    """
    return (
        distance(pos, DRAGON_PIT) < OBJECTIVE_RADIUS
        or distance(pos, BARON_PIT) < OBJECTIVE_RADIUS
    )


def push_progress(pos: Position, blue_side: bool) -> float:
    """Progress of a position toward the enemy base along the mid-lane axis.

    Args:
        pos: Position to project.
        blue_side: ``True`` if the player plays on blue side (bottom-left base).

    Returns:
        A value in map units; ``0`` at map centre, positive when past the
        centre toward the enemy base, negative when on the player's own half.
    """
    projection = (pos.x + pos.y) - MAP_SIZE
    return projection if blue_side else -projection


def normalize_coords_to_blue_side(x: float, y: float, *, side: str) -> tuple[float, float]:
    """Mirror map coordinates so every game reads from the blue-side perspective.

    Red-side deaths are reflected across the river diagonal (``x + y = MAP_SIZE``),
    the line perpendicular to mid lane. That keeps bot-lane deaths on the bottom
    edge instead of rotating them to the opposite corner.
    """
    if side.lower() == "red":
        return MAP_SIZE - y, MAP_SIZE - x
    return x, y


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` on a zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when ``denominator`` is zero.

    Returns:
        ``numerator / denominator`` or ``default``.
    """
    return numerator / denominator if denominator else default


def wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial win rate.

    Used to rank matchups robustly when sample sizes differ.

    Args:
        wins: Number of successes.
        games: Number of trials.
        z: Z-score for the confidence level (1.96 ~ 95%).

    Returns:
        Lower bound of the win-rate confidence interval in ``[0, 1]``.
    """
    if games == 0:
        return 0.0
    p = wins / games
    denom = 1 + z * z / games
    centre = p + z * z / (2 * games)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games)
    return max(0.0, (centre - margin) / denom)


def ms_to_min(timestamp_ms: int | float) -> float:
    """Convert a millisecond timestamp to fractional minutes.

    Args:
        timestamp_ms: Timestamp in milliseconds.

    Returns:
        Time in minutes.
    """
    return timestamp_ms / 60_000.0


def fmt_minutes(minutes: float) -> str:
    """Format fractional minutes as ``m:ss``.

    Args:
        minutes: Time in fractional minutes.

    Returns:
        A human-readable ``m:ss`` string.
    """
    total_seconds = int(round(minutes * 60))
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"
