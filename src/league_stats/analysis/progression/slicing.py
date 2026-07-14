"""Window slicing helpers for Form Tracker comparisons."""

from __future__ import annotations

from league_stats.core.models import MatchRecord


def slice_recent(records: list[MatchRecord], n: int) -> list[MatchRecord]:
    """Return the most recent ``n`` games (records are newest-first)."""
    return records[:n]


def slice_baseline_exclusive(
    records: list[MatchRecord],
    recent_n: int,
    baseline_m: int,
) -> list[MatchRecord]:
    """Return games ``(recent_n+1)`` through ``(recent_n+baseline_m)``."""
    return records[recent_n : recent_n + baseline_m]


def slice_baseline_inclusive(records: list[MatchRecord], baseline_m: int) -> list[MatchRecord]:
    """Return the last ``baseline_m`` games (overlaps with recent window)."""
    return records[:baseline_m]


def slice_baseline(
    records: list[MatchRecord],
    recent_n: int,
    baseline_m: int,
    *,
    overlap: bool,
) -> list[MatchRecord]:
    """Slice the baseline window in exclusive or inclusive mode."""
    if overlap:
        return slice_baseline_inclusive(records, baseline_m)
    return slice_baseline_exclusive(records, recent_n, baseline_m)
