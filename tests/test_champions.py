"""Tests for champion and lane normalization."""

from __future__ import annotations

import pytest

from champions import (
    build_champion_catalog,
    build_label,
    champion_slug,
    normalize_role,
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


def test_champion_slug() -> None:
    """Benchmark slugs are lowercase champion_role."""
    assert champion_slug("Ahri", "MIDDLE") == "ahri_middle"


def test_resolve_champion_name_from_catalog() -> None:
    """Champion lookup is case- and space-insensitive."""
    catalog = build_champion_catalog(
        {
            "Ahri": {"id": "Ahri", "name": "Ahri"},
            "LeeSin": {"id": "LeeSin", "name": "Lee Sin"},
        }
    )
    assert resolve_champion_name("ahri", catalog) == "Ahri"
    assert resolve_champion_name("lee sin", catalog) == "LeeSin"
    assert resolve_champion_name("LeeSin", catalog) == "LeeSin"


def test_resolve_champion_name_unknown() -> None:
    """Unknown champions raise ValueError."""
    with pytest.raises(ValueError, match="Unknown champion"):
        resolve_champion_name("NotAChamp", {})
