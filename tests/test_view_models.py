"""Tests for dashboard view-model helpers."""

from __future__ import annotations

from league_stats.pipeline.view_models import (
    annotate_card_tiers,
    cards_from_specs,
    enrich_value_semantics,
    form_delta_chart_value,
    form_row_display,
    form_sample_subtitle,
    overview_card_entries,
    priority_label,
)
from league_stats.core.role_metrics import role_profile
from league_stats.presentation.metric_colors import interpolate_metric_color, LOSS_HEX, score_form_delta


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


def test_death_section_cards_skip_value_colors() -> None:
    profile = role_profile("MIDDLE")
    summaries = {
        "deaths": {
            "solo_death_rate": 0.50,
            "gank_death_rate": 0.20,
        }
    }
    cards = cards_from_specs(profile.deaths[:2], summaries, section="deaths")
    for card in cards:
        assert not card.get("value_color")
        assert not card.get("value_class")


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


def test_form_row_display_gd10_negative_baseline_shows_positive_improvement() -> None:
    """GD@10 from -58 to +90 should read as +148 improvement in green."""
    row = form_row_display(
        {
            "metric": "gd10",
            "label": "Gold diff @10",
            "recent": 90.0,
            "baseline": -58.0,
            "delta": 148.0,
            "delta_pct": -254.8,
            "direction": "higher",
            "verdict": "improved",
            "significant": True,
        }
    )
    assert row["gap"] == "+148"
    assert row["verdict"] == "improved"
    assert row["gap_color"] != LOSS_HEX
    assert row["gap_color"] == interpolate_metric_color(148 / 300)


def test_form_row_display_deaths_improved_shows_negative_gap_in_green() -> None:
    """Fewer deaths should show a negative percent change with green coloring."""
    row = form_row_display(
        {
            "metric": "deaths",
            "label": "Deaths/game",
            "recent": 4.0,
            "baseline": 5.0,
            "delta": -1.0,
            "delta_pct": -20.0,
            "direction": "lower",
            "verdict": "improved",
            "significant": True,
        }
    )
    assert row["gap"] == "-20%"
    assert row["gap_color"] != LOSS_HEX
    assert row["gap_color"] == interpolate_metric_color(1 / 2.5)


def test_form_row_display_death_rate_improved_shows_negative_percent() -> None:
    """Lower greed death rate should show negative % vs baseline."""
    row = form_row_display(
        {
            "metric": "greed_death_rate",
            "label": "Greed death rate",
            "recent": 0.20,
            "baseline": 0.30,
            "delta": -0.10,
            "delta_pct": -33.3,
            "direction": "lower",
            "verdict": "improved",
            "significant": True,
        }
    )
    assert row["gap"] == "-33%"
    assert row["gap_color"] != LOSS_HEX


def test_form_sample_subtitle_describes_exclusive_windows() -> None:
    text = form_sample_subtitle(recent_games=20, baseline_games=80)
    assert text == "Statistics from your last 20 games compared to the 80 games before that."


def test_form_sample_subtitle_singular_game() -> None:
    text = form_sample_subtitle(recent_games=1, baseline_games=1)
    assert text == "Statistics from your last 1 game compared to the 1 game before that."


def test_form_delta_chart_value_uses_percent_or_raw_lane_diff() -> None:
    gd_row = {"metric": "gd10", "delta": 148.0, "direction": "higher"}
    deaths_row = {
        "metric": "deaths",
        "delta": -1.0,
        "direction": "lower",
        "baseline": 5.0,
        "delta_pct": -20.0,
    }
    assert form_delta_chart_value(gd_row) == 148.0
    assert form_delta_chart_value(deaths_row) == -20.0


def test_form_impact_calibration_weights_win_rate_above_small_death_shifts() -> None:
    win_score = score_form_delta("win", 0.15)
    death_regression = score_form_delta("deaths", -0.44)
    assert win_score == 1.0
    assert death_regression == -0.44 / 2.5
    assert win_score > abs(death_regression)


def test_form_row_display_inline_verdict_uses_neutral_gap_color() -> None:
    """Small DPM drift within the inline band should not flash red."""
    row = form_row_display(
        {
            "metric": "dpm",
            "label": "DPM",
            "recent": 1015.89,
            "baseline": 1056.33,
            "delta": -40.43,
            "delta_pct": -3.8,
            "direction": "higher",
            "verdict": "inline",
            "significant": False,
        }
    )
    assert row["verdict"] == "inline"
    assert row["gap"] == "-4%"
    assert row["gap_color"] == ""
