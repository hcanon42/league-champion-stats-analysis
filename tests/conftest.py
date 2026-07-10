"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from league_stats.infra.ddragon_assets import DDragonAssets


@pytest.fixture(autouse=True)
def _skip_ddragon_downloads(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid bulk Data Dragon downloads during tests."""
    if request.module.__name__.endswith("test_ddragon_assets"):
        return
    monkeypatch.setattr(DDragonAssets, "ensure_downloaded", lambda self, force=False: "")
