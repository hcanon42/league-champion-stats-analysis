"""Tests for metric icon label mappings."""

from __future__ import annotations

from ui_icons import (
    ICONIFY_ICONS,
    icon_fields_for_label,
    icon_for_label,
    icon_for_objective,
    icon_tone,
    iconify_for_key,
    with_icon,
    with_icons,
)


def test_icon_for_common_labels() -> None:
    assert icon_for_label("CS/min") == "cs"
    assert icon_for_label("CS diff @10") == "cs"
    assert icon_for_label("Farming") == "cs"
    assert icon_for_label("GPM") == "coin"
    assert icon_for_label("Deaths/game") == "skull"
    assert icon_for_label("Vision/min") == "eye"
    assert icon_for_label("Unknown metric") is None


def test_iconify_uses_library_ids() -> None:
    assert iconify_for_key("skull") == "lucide:skull"
    assert iconify_for_key("dragon") == "game-icons:dragon-head"
    assert iconify_for_key("kp") == "lucide:users-round"
    assert all(":" in value for value in ICONIFY_ICONS.values())


def test_icon_for_objectives() -> None:
    assert icon_for_objective("dragon") == "dragon"
    assert icon_for_objective("baron") == "baron"
    assert icon_for_objective("grubs") == "grubs"
    assert icon_for_objective("tower") is None


def test_cs_min_uses_local_asset() -> None:
    fields = icon_fields_for_label("CS/min")
    assert fields["icon"] == "cs"
    assert fields["icon_asset"] == "minions.png"
    assert fields["iconify"] is None
    assert fields["icon_tone"] == "green"


def test_with_icon_enriches_card() -> None:
    card = with_icon({"label": "GPM", "value": "420"})
    assert card["icon"] == "coin"
    assert card["iconify"] == "lucide:coins"
    assert card["icon_tone"] == "gold"


def test_kill_participation_icon() -> None:
    assert icon_for_label("Kill participation") == "kp"
    assert iconify_for_key("kp") == "lucide:users-round"


def test_with_icons_preserves_order() -> None:
    cards = with_icons([{"label": "KDA", "value": "3.1"}, {"label": "DPM", "value": "700"}])
    assert cards[0]["iconify"] == "lucide:swords"
    assert cards[1]["iconify"] == "lucide:flame"


def test_icon_tone_defaults_to_muted() -> None:
    assert icon_tone(None) == "muted"
    assert icon_tone("skull") == "danger"
