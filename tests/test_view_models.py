"""Tests for dashboard view-model helpers."""

from __future__ import annotations

from league_stats.pipeline.view_models import (
    annotate_card_tiers,
    enrich_value_semantics,
    overview_card_entries,
    priority_label,
)
from league_stats.presentation.metric_colors import interpolate_metric_color


def test_priority_label_maps_badge_classes() -> None:
    assert priority_label("high") == "High"
    assert priority_label("medium") == "Medium"
    assert priority_label("low") == "Low"


def test_annotate_card_tiers_orders_headline_metrics_first() -> None:
    entries = [
        {"label": "Roams pre-15", "value": "1.2"},
        {"label": "Gold diff @10", "value": "+120"},
        {"label": "Lane win rate", "value": "55%"},
    ]
    ordered = annotate_card_tiers(entries, "lane")
    assert [entry["label"] for entry in ordered[:3]] == [
        "Gold diff @10",
        "Lane win rate",
        "Roams pre-15",
    ]
    assert ordered[0]["tier"] == "headline"
    assert ordered[-1]["tier"] == "more"


def test_enrich_value_semantics_colors_diff_and_win_rate() -> None:
    gd = {"label": "Gold diff @10", "value": "+250", "value_class": ""}
    wr = {"label": "Lane win rate", "value": "42%", "value_class": ""}
    mid_wr = {"label": "Lane win rate", "value": "50%", "value_class": ""}
    enrich_value_semantics(gd)
    enrich_value_semantics(wr)
    enrich_value_semantics(mid_wr)
    assert gd["value_class"] == "win"
    assert wr["value_class"] == "loss"
    assert mid_wr["value_class"] == ""
    assert gd["value_color"] != wr["value_color"]
    assert mid_wr["value_color"] == interpolate_metric_color(0.0)


def test_overview_card_entries_include_tiers() -> None:
    cards = overview_card_entries(
        {
            "winrate": 0.53,
            "avg_kda": "3.1",
            "avg_dpm": "640",
            "avg_cspm": "7.2",
            "avg_damage_share": 0.24,
            "avg_deaths": 4.2,
            "avg_vspm": 1.1,
            "avg_duration": 28,
        }
    )
    headline = [card for card in cards if card.get("tier") == "headline"]
    assert len(headline) == 4
    assert headline[0]["label"] == "Win rate"
    assert headline[2]["label"] == "DPM"


def test_overview_card_entries_use_cc_for_support() -> None:
    cards = overview_card_entries(
        {
            "winrate": 0.51,
            "avg_kda": "2.8",
            "avg_ccpm": "2.1",
            "avg_damage_share": 0.08,
            "avg_deaths": 4.8,
            "avg_vspm": 1.9,
            "avg_duration": 29,
        },
        role="UTILITY",
    )
    headline = [card for card in cards if card.get("tier") == "headline"]
    assert headline[2]["label"] == "CC/min"
    assert headline[2]["value"] == "2.1"
    labels = {card["label"] for card in cards}
    assert "DPM" not in labels
    assert "CS/min" not in labels
    assert "Damage share" not in labels
