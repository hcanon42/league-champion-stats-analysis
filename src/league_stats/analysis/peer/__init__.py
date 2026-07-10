"""Rank-peer comparison facade.

Import from this module rather than individual peer submodules.
"""

from league_stats.analysis.peer.baseline import resolve_peer_baseline
from league_stats.analysis.peer.comparison import (
    build_comparisons,
    build_peer_comparison,
    comparisons_dataframe,
    peer_comparison_for_window,
    peer_recommendations,
)
from league_stats.analysis.peer.ingest import backfill_all_matches, ingest_match

__all__ = [
    "backfill_all_matches",
    "build_comparisons",
    "build_peer_comparison",
    "comparisons_dataframe",
    "ingest_match",
    "peer_comparison_for_window",
    "peer_recommendations",
    "resolve_peer_baseline",
]
