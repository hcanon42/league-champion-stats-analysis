"""Tests for combat output metric selection."""

from __future__ import annotations

from league_stats.analysis.combat import combat_output_metric, prefers_cc_over_dpm


def test_support_prefers_cc_over_dpm() -> None:
    assert prefers_cc_over_dpm("UTILITY")
    assert combat_output_metric("UTILITY") == ("ccpm", "CC/min")


def test_tank_damage_share_prefers_cc() -> None:
    assert prefers_cc_over_dpm("TOP", avg_damage_share=0.12)
    assert combat_output_metric("JUNGLE", avg_damage_share=0.11) == ("ccpm", "CC/min")


def test_carry_builds_keep_dpm() -> None:
    assert not prefers_cc_over_dpm("MIDDLE", avg_damage_share=0.24)
    assert combat_output_metric("TOP", avg_damage_share=0.22) == ("dpm", "DPM")
