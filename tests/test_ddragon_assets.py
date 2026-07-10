"""Tests for Data Dragon asset downloads and icon href resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig
from ddragon_assets import DDragonAssets, _relative_href, path_to_data_uri


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        api_key="RGAPI-test",
        riot_id="Player",
        tagline="TAG",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / ".cache",
    )


def test_relative_href_from_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "output" / "reports" / "player" / "ahri_middle"
    asset = tmp_path / "output" / "assets" / "champions" / "Ahri.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"png")
    href = _relative_href(report_dir, asset)
    assert href == "../../../assets/champions/Ahri.png"


def test_champion_and_keystone_hrefs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._champions_dir.mkdir(parents=True)
    assets._runes_dir.mkdir(parents=True)
    (assets._champions_dir / "Ahri.png").write_bytes(b"png")
    (assets._runes_dir / "8112.png").write_bytes(b"png")

    report_dir = config.output_dir / "reports" / "player" / "ahri_middle"
    report_dir.mkdir(parents=True)

    assert assets.champion_href("Ahri", from_dir=report_dir) == "../../../assets/champions/Ahri.png"
    assert assets.keystone_href("Electrocute", from_dir=report_dir) == "../../../assets/runes/8112.png"
    assert assets.champion_href("Missing", from_dir=report_dir) is None


def test_ui_icon_href(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._ui_dir.mkdir(parents=True)
    (assets._ui_dir / "minions.png").write_bytes(b"png")
    report_dir = config.output_dir / "reports" / "player" / "ahri_middle"
    report_dir.mkdir(parents=True)
    assert assets.ui_icon_href("minions.png", from_dir=report_dir) == "../../../assets/ui/minions.png"


def test_objective_href(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._objectives_dir.mkdir(parents=True)
    (assets._objectives_dir / "dragon.png").write_bytes(b"png")
    report_dir = config.output_dir / "reports" / "player" / "ahri_middle"
    report_dir.mkdir(parents=True)
    assert assets.objective_href("dragon", from_dir=report_dir) == "../../../assets/objectives/dragon.png"
    assert assets.objective_href("unknown", from_dir=report_dir) is None


def test_crop_minion_icon(tmp_path: Path) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    from ddragon_assets import _crop_top_half_png, _needs_minion_crop, _png_dimensions

    source = tmp_path / "source.png"
    destination = tmp_path / "minions.png"
    stacked = np.zeros((112, 52, 4), dtype=np.float32)
    stacked[:56, :, 3] = 1.0
    stacked[56:, :, 3] = 0.5
    plt.imsave(source, stacked)
    _crop_top_half_png(source, destination)
    assert _png_dimensions(source) == (52, 112)
    assert _png_dimensions(destination) == (52, 56)
    assert _needs_minion_crop(source, destination) is False
    assert _needs_minion_crop(source, source) is True


def test_role_href(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._roles_dir.mkdir(parents=True)
    (assets._roles_dir / "JUNGLE.png").write_bytes(b"png")
    report_dir = config.output_dir / "reports" / "player" / "nidalee_jungle"
    report_dir.mkdir(parents=True)

    assert assets.role_href("JUNGLE", from_dir=report_dir) == "../../../assets/roles/JUNGLE.png"
    assert assets.role_href("jungle", from_dir=report_dir) == "../../../assets/roles/JUNGLE.png"
    assert assets.role_href("UNKNOWN", from_dir=report_dir) is None


def test_enrich_rows_adds_icon_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._champions_dir.mkdir(parents=True)
    assets._runes_dir.mkdir(parents=True)
    (assets._champions_dir / "Darius.png").write_bytes(b"png")
    (assets._runes_dir / "8010.png").write_bytes(b"png")
    from_dir = config.output_dir / "reports" / "player" / "aatrox_top"
    from_dir.mkdir(parents=True)

    rune_rows = assets.enrich_rune_rows([{"keystone": "Conqueror"}], from_dir=from_dir)
    matchup_rows = assets.enrich_matchup_rows([{"opponent": "Darius"}], from_dir=from_dir)

    assert rune_rows[0]["keystone_icon"].endswith("assets/runes/8010.png")
    assert matchup_rows[0]["opponent_icon"].endswith("assets/champions/Darius.png")


def test_path_to_data_uri(tmp_path: Path) -> None:
    png = tmp_path / "icon.png"
    png.write_bytes(b"\x89PNG\r\n")
    uri = path_to_data_uri(png)
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")


def test_chart_sources_use_data_uris(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    assets._champions_dir.mkdir(parents=True)
    assets._runes_dir.mkdir(parents=True)
    assets._items_dir.mkdir(parents=True)
    assets._item_name_to_id = {"Boots": 1001}
    (assets._champions_dir / "Ahri.png").write_bytes(b"png")
    (assets._runes_dir / "8112.png").write_bytes(b"png")
    (assets._items_dir / "1001.png").write_bytes(b"png")

    assert assets.champion_chart_source("Ahri").startswith("data:image/png;base64,")
    assert assets.keystone_chart_source("Electrocute").startswith("data:image/png;base64,")
    assert assets.item_chart_source("Boots").startswith("data:image/png;base64,")


@pytest.mark.integration
def test_download_assets_from_ddragon(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = DDragonAssets(config)
    version = assets.ensure_downloaded(force=True)
    assert version
    assert assets.champion_icon_path("Ahri") is not None
    assert assets.keystone_icon_path("Electrocute") is not None
