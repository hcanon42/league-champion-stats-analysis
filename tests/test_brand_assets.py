"""Tests for project branding assets."""

from __future__ import annotations

from pathlib import Path

from brand_assets import APP_TITLE, brand_context, brand_href, ensure_brand_assets, refresh_saved_report_branding


def test_ensure_brand_assets_copies_logo(tmp_path: Path) -> None:
    brand_dir = ensure_brand_assets(tmp_path)
    assert (brand_dir / "logo.png").is_file()
    assert (brand_dir / "favicon.png").is_file()


def test_brand_href_is_relative_to_html_dir(tmp_path: Path) -> None:
    ensure_brand_assets(tmp_path)
    report_dir = tmp_path / "reports" / "player" / "ahri_middle"
    report_dir.mkdir(parents=True)

    assert brand_href("logo.png", from_dir=tmp_path, output_dir=tmp_path) == "assets/brand/logo.png"
    assert (
        brand_href("favicon.png", from_dir=report_dir, output_dir=tmp_path)
        == "../../../assets/brand/favicon.png"
    )


def test_brand_context_includes_title_and_links(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    from_dir = output_dir / "reports" / "player" / "ahri_middle"
    from_dir.mkdir(parents=True)

    context = brand_context(from_dir=from_dir, output_dir=output_dir)

    assert context["app_title"] == APP_TITLE
    assert context["logo_href"] == "../../../assets/brand/logo.png"
    assert context["favicon_href"] == "../../../assets/brand/favicon.png"


def test_refresh_saved_report_branding_patches_legacy_html(tmp_path: Path) -> None:
    ensure_brand_assets(tmp_path)
    report_dir = tmp_path / "reports" / "player" / "ahri_middle"
    report_dir.mkdir(parents=True)
    report_html = report_dir / "report.html"
    report_html.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test</title>
<style>
nav h1 { font-size: 18px; padding: 0 20px 16px; color: var(--accent); }
</style>
</head>
<body>
<nav>
  <h1>Champion Stats Analyzer</h1>
</nav>
</body>
</html>
""",
        encoding="utf-8",
    )

    updated = refresh_saved_report_branding(tmp_path)

    text = report_html.read_text(encoding="utf-8")
    assert updated == 1
    assert 'rel="icon"' in text
    assert "app-brand--nav" in text
    assert "<h1>Champion Stats Analyzer</h1>" not in text
