"""Tier benchmarks for champion + lane in ranked solo queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from champions import champion_slug

BENCHMARKS_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent / "data" / "benchmarks"
)
VALID_TIERS: Final[frozenset[str]] = frozenset(
    {
        "IRON",
        "BRONZE",
        "SILVER",
        "GOLD",
        "PLATINUM",
        "EMERALD",
        "DIAMOND",
        "MASTER",
        "GRANDMASTER",
        "CHALLENGER",
    }
)
TIER_ORDER: Final[list[str]] = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]


def benchmark_paths(champion: str, role: str) -> list[Path]:
    """Return benchmark file candidates, most specific first.

    Lookup order:

    1. ``{champion}_{role}.json`` (e.g. ``viktor_middle.json``),
    2. ``_{role}.json`` role-level fallback (e.g. ``_middle.json``).

    Args:
        champion: Riot champion id.
        role: Normalised team position.

    Returns:
        Ordered list of paths to try.
    """
    slug = champion_slug(champion, role)
    return [
        BENCHMARKS_DIR / f"{slug}.json",
        BENCHMARKS_DIR / f"_{role.lower()}.json",
    ]


def resolve_benchmark_path(champion: str, role: str) -> Path:
    """Pick the first existing benchmark file for a champion + lane.

    Args:
        champion: Riot champion id.
        role: Normalised team position.

    Returns:
        Path to the benchmark JSON file.

    Raises:
        FileNotFoundError: When no benchmark file exists for the pair.
    """
    for path in benchmark_paths(champion, role):
        if path.is_file():
            return path
    tried = ", ".join(str(p.name) for p in benchmark_paths(champion, role))
    raise FileNotFoundError(
        f"No benchmark data for {champion} {role}. "
        f"Expected one of: {tried} under {BENCHMARKS_DIR}."
    )


def load_benchmarks(champion: str = "Viktor", role: str = "MIDDLE") -> dict[str, dict[str, float]]:
    """Load per-tier benchmark metrics from JSON.

    Args:
        champion: Riot champion id.
        role: Normalised team position.

    Returns:
        Mapping of tier name to metric averages.
    """
    path = resolve_benchmark_path(champion, role)
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)
    return {tier: values for tier, values in raw.items() if tier in VALID_TIERS}


def tier_benchmark(
    tier: str, champion: str = "Viktor", role: str = "MIDDLE"
) -> dict[str, float]:
    """Return benchmark metrics for a tier, falling back to GOLD when unknown.

    Normalises ``winrate`` from JSON into ``win`` for comparison code.

    Args:
        tier: Riot tier string (e.g. ``"PLATINUM"``).
        champion: Riot champion id.
        role: Normalised team position.

    Returns:
        Average metric values for the champion + lane at that tier.
    """
    benchmarks = load_benchmarks(champion, role)
    key = tier.upper() if tier else "GOLD"
    if key not in benchmarks:
        row = dict(benchmarks["GOLD"])
    else:
        row = dict(benchmarks[key])
    if "winrate" in row:
        row["win"] = float(row["winrate"])
    return row


def try_static_benchmark(
    tier: str, champion: str, role: str
) -> dict[str, float] | None:
    """Return champion-specific static benchmarks when available."""
    try:
        return tier_benchmark(tier, champion, role)
    except FileNotFoundError:
        return None


def try_role_benchmark(tier: str, role: str) -> dict[str, float] | None:
    """Return role-only static benchmarks when available."""
    path = BENCHMARKS_DIR / f"_{role.lower()}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)
    benchmarks = {key: values for key, values in raw.items() if key in VALID_TIERS}
    key = tier.upper() if tier else "GOLD"
    row = dict(benchmarks.get(key, benchmarks.get("GOLD", {})))
    if not row:
        return None
    if "winrate" in row:
        row["win"] = float(row["winrate"])
    return row


def adjacent_tiers(tier: str) -> set[str]:
    """Return the tier and its immediate neighbours on the ladder.

    Args:
        tier: Riot tier string.

    Returns:
        Set of tier names including ``tier`` and adjacent ranks.
    """
    key = tier.upper()
    if key not in TIER_ORDER:
        return {key} if key in VALID_TIERS else {"GOLD"}
    index = TIER_ORDER.index(key)
    neighbours = {key}
    if index > 0:
        neighbours.add(TIER_ORDER[index - 1])
    if index < len(TIER_ORDER) - 1:
        neighbours.add(TIER_ORDER[index + 1])
    return neighbours
