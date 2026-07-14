"""Tests for Plotly chart generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from league_stats.core.config import AppConfig
from league_stats.core.models import MetricDelta
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.presentation.graphs import ChartIconResolver, GraphFactory
from league_stats.presentation.metric_colors import LOSS_HEX


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        api_key="test",
        riot_id="player",
        tagline="euw",
        output_dir=tmp_path / "output",
    )


def test_icon_bar_charts_embed_data_uris(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._champions_dir.mkdir(parents=True)
    assets._runes_dir.mkdir(parents=True)
    assets._items_dir.mkdir(parents=True)
    assets._item_name_to_id = {"Lich Bane": 3100}
    (assets._champions_dir / "Ahri.png").write_bytes(b"png-champion")
    (assets._runes_dir / "8112.png").write_bytes(b"png-rune")
    (assets._items_dir / "3100.png").write_bytes(b"png-item")

    graphs_dir = config.output_dir / "reports" / "player" / "ahri_mid" / "graphs"
    graphs_dir.mkdir(parents=True)
    resolver = ChartIconResolver(
        from_dir=graphs_dir.parent,
        champion_href=assets.champion_chart_source,
        item_href=assets.item_chart_source,
        keystone_href=assets.keystone_chart_source,
    )
    factory = GraphFactory(graphs_dir, icon_resolver=resolver)

    matchup_html = factory.matchup_bar(
        pd.DataFrame({"opponent": ["Ahri"], "winrate": [0.6], "games": [5]})
    )
    item_html = factory.item_winrate_bar(
        pd.DataFrame({"slot": ["first_item"], "item": ["Lich Bane"], "winrate": [0.55], "games": [10]})
    )
    rune_html = factory.rune_winrate_bar(
        pd.DataFrame({"keystone": ["Electrocute"], "winrate": [0.5], "games": [8]})
    )

    for html in (matchup_html, item_html, rune_html):
        assert "data:image" in html and "base64," in html
        assert "../../../assets/" not in html


def test_form_metric_delta_bar_uses_normalized_bar_length_with_raw_labels(tmp_path: Path) -> None:
    factory = GraphFactory(tmp_path / "graphs")
    html = factory.form_metric_delta_bar(
        [
            MetricDelta(
                metric="gd10",
                label="Gold diff @10",
                section="laning",
                recent=90.0,
                baseline=-58.0,
                delta=148.0,
                delta_pct=-254.8,
                direction="higher",
                verdict="improved",
                significant=True,
                recent_n=10,
                baseline_n=30,
            ),
            MetricDelta(
                metric="deaths",
                label="Deaths/game",
                section="overview",
                recent=4.0,
                baseline=5.0,
                delta=-1.0,
                delta_pct=-20.0,
                direction="lower",
                verdict="improved",
                significant=True,
                recent_n=10,
                baseline_n=30,
            ),
        ]
    )
    assert '"x":[148,-20]' in html or '"x":[148.0,-20.0]' in html
    assert '"text":["+148","-20%"]' in html
    assert LOSS_HEX not in html
