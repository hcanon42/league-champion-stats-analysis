"""Tests for champion and lane normalization."""

from __future__ import annotations

import pytest

from league_stats.core.champions import (
    build_champion_catalog,
    build_label,
    champion_display_name,
    champion_slug,
    normalize_role,
    player_slug,
    resolve_champion_name,
)


def test_normalize_role_aliases() -> None:
    """Lane aliases map to Riot team positions."""
    assert normalize_role("mid") == "MIDDLE"
    assert normalize_role("support") == "UTILITY"
    assert normalize_role("jg") == "JUNGLE"
    assert normalize_role("MIDDLE") == "MIDDLE"


def test_normalize_role_rejects_unknown() -> None:
    """Unknown lanes raise a clear error."""
    with pytest.raises(ValueError, match="Unknown lane"):
        normalize_role("midlane")


def test_build_label() -> None:
    """Build label combines champion and lane display."""
    assert build_label("Ahri", "MIDDLE") == "Ahri mid"
    assert build_label("LeeSin", "JUNGLE") == "LeeSin jungle"
    assert build_label("MonkeyKing", "TOP") == "Wukong top"
    assert build_label("DrMundo", "TOP") == "Dr. Mundo top"


def test_champion_display_name() -> None:
    """Riot ids with different display names are mapped for UI."""
    assert champion_display_name("MonkeyKing") == "Wukong"
    assert champion_display_name("DrMundo") == "Dr. Mundo"
    assert champion_display_name("Ahri") == "Ahri"


def test_champion_slug() -> None:
    """Benchmark slugs are lowercase champion_role."""
    assert champion_slug("Ahri", "MIDDLE") == "ahri_middle"


def test_player_slug() -> None:
    """Player slugs combine riot id and tagline."""
    assert player_slug("Faker", "KR1") == "faker_kr1"


def test_resolve_champion_name_from_catalog() -> None:
    """Champion lookup is case- and space-insensitive."""
    catalog = build_champion_catalog(
        {
            "Ahri": {"id": "Ahri", "name": "Ahri"},
            "LeeSin": {"id": "LeeSin", "name": "Lee Sin"},
            "MonkeyKing": {"id": "MonkeyKing", "name": "Wukong"},
            "DrMundo": {"id": "DrMundo", "name": "Dr. Mundo"},
        }
    )
    assert resolve_champion_name("ahri", catalog) == "Ahri"
    assert resolve_champion_name("lee sin", catalog) == "LeeSin"
    assert resolve_champion_name("LeeSin", catalog) == "LeeSin"
    assert resolve_champion_name("wukong", catalog) == "MonkeyKing"
    assert resolve_champion_name("dr. mundo", catalog) == "DrMundo"


def test_resolve_champion_name_unknown() -> None:
    """Unknown champions raise ValueError."""
    with pytest.raises(ValueError, match="Unknown champion"):
        resolve_champion_name("NotAChamp", {})
