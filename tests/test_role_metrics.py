"""Tests for role-aware metric profiles."""

from __future__ import annotations

from league_stats.core.role_metrics import compare_metrics_for_profile, role_profile
from league_stats.pipeline.view_models import overview_card_entries


def test_utility_profile_excludes_laning_peer_metrics() -> None:
    profile = role_profile("UTILITY")
    keys = {item[0] for item in profile.peer_metrics}
    assert "gd10" not in keys
    assert "cs10" not in keys
    assert "ccpm" in keys
    assert "assists" in keys


def test_jungle_profile_emphasizes_map_impact() -> None:
    profile = role_profile("JUNGLE")
    overview_labels = [spec.label for spec in profile.overview]
    assert "Kill participation" in overview_labels
    assert "Obj. presence" in overview_labels
    assert profile.early_section_title == "Early game"
    assert "_rule_cs10" not in profile.coach_rule_ids


def test_utility_overview_hides_damage_and_cs() -> None:
    cards = overview_card_entries(
        {
            "winrate": 0.52,
            "avg_kda": "3.0",
            "avg_ccpm": "2.0",
            "avg_kill_participation": 0.61,
            "avg_vspm": 2.0,
            "avg_control_wards": 2.5,
            "avg_deaths": 4.5,
            "avg_duration": 30,
            "avg_damage_share": 0.08,
            "avg_cspm": 1.2,
        },
        role="UTILITY",
    )
    labels = {card["label"] for card in cards}
    assert "CC/min" in labels
    assert "Kill participation" in labels
    assert "DPM" not in labels
    assert "CS/min" not in labels
    assert "Damage share" not in labels


def test_compare_metrics_for_profile_swaps_cc_on_support() -> None:
    profile = role_profile("UTILITY")
    metrics = compare_metrics_for_profile(profile)
    assert ("ccpm", "CC/min", "higher") in metrics
    assert ("dpm", "DPM", "higher") not in metrics
