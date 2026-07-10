"""Tests for peer baseline resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from league_stats.analysis.peer.baseline import resolve_peer_baseline
from league_stats.analysis.peer.ingest import ingest_match
from league_stats.infra.cache import MatchStore
from league_stats.core.models import RankedEntry
from tests.fixtures import make_match


@pytest.fixture
def ranked() -> RankedEntry:
    return RankedEntry(tier="EMERALD", rank="II", league_points=45, wins=10, losses=10)


def test_resolve_peer_baseline_uses_static_fallback(tmp_path, ranked: RankedEntry) -> None:
    """When store and live sampling are empty, static benchmarks are used."""
    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")

    baseline = resolve_peer_baseline(
        client,
        store,
        ranked,
        "Ornn",
        "TOP",
        exclude_puuid="puuid-me",
    )
    assert baseline is not None
    assert baseline.fallback_level in {3, 4}
    assert baseline.confidence == "low"
    assert baseline.metrics["dpm"] > 0


def test_resolve_peer_baseline_uses_role_only_when_champion_missing(
    tmp_path, ranked: RankedEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Role-only static benchmarks are the last fallback."""
    import league_stats.analysis.peer.baseline as peer_baseline

    monkeypatch.setattr(peer_baseline, "try_static_benchmark", lambda *args, **kwargs: None)
    store = MatchStore(tmp_path / "matches.sqlite")
    client = MagicMock()
    client.configure_mock(platform="euw1")

    baseline = resolve_peer_baseline(
        client,
        store,
        ranked,
        "Ornn",
        "TOP",
        exclude_puuid="puuid-me",
    )
    assert baseline is not None
    assert baseline.fallback_level == 4


def test_resolve_peer_baseline_uses_store_when_enough_games(
    tmp_path, ranked: RankedEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact-rank store samples are preferred when the target count is met."""
    import league_stats.analysis.peer.baseline as peer_baseline

    monkeypatch.setattr(peer_baseline, "MIN_EXACT_GAMES", 2)
    store = MatchStore(tmp_path / "matches.sqlite")
    for index in range(2):
        match = make_match()
        match["info"]["participants"][1]["puuid"] = f"peer-{index}"
        ingest_match(store, f"EUW1_{index}", match, "euw1")
        store.set_puuid_rank(f"peer-{index}", "EMERALD", "II")

    client = MagicMock()
    client.configure_mock(platform="euw1")
    client.fetch_solo_rank.return_value = ranked

    baseline = resolve_peer_baseline(
        client,
        store,
        ranked,
        "LeeSin",
        "JUNGLE",
        exclude_puuid="puuid-me",
    )
    assert baseline is not None
    assert baseline.fallback_level == 0
    assert baseline.confidence == "high"
    assert baseline.games >= 2
