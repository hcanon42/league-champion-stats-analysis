"""Form Tracker: recent-vs-baseline progression analysis."""

from league_stats.analysis.progression.diff import build_progression_comparison
from league_stats.analysis.progression.slicing import (
    slice_baseline,
    slice_baseline_exclusive,
    slice_baseline_inclusive,
    slice_recent,
)

__all__ = [
    "build_progression_comparison",
    "slice_recent",
    "slice_baseline",
    "slice_baseline_exclusive",
    "slice_baseline_inclusive",
]
