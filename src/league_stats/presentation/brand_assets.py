"""Project branding assets (logo and favicon) for generated HTML reports."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BRAND_SOURCE_DIR = _PACKAGE_ROOT / "assets" / "brand"
LOGO_FILENAME = "logo.png"
FAVICON_FILENAME = "favicon.png"
APP_TITLE = "Champion Stats Analyzer"


def ensure_brand_assets(output_dir: Path) -> Path:
    """Copy bundled brand images into ``output/assets/brand/``.

    Args:
        output_dir: Root output directory for generated reports.

    Returns:
        The brand assets directory under ``output/assets/brand/``.
    """
    source_logo = BRAND_SOURCE_DIR / LOGO_FILENAME
    if not source_logo.is_file():
        raise FileNotFoundError(f"Missing brand logo at {source_logo}")

    brand_dir = output_dir / "assets" / "brand"
    brand_dir.mkdir(parents=True, exist_ok=True)

    logo_dest = brand_dir / LOGO_FILENAME
    favicon_dest = brand_dir / FAVICON_FILENAME
    shutil.copy2(source_logo, logo_dest)
    shutil.copy2(source_logo, favicon_dest)
    return brand_dir


def brand_href(filename: str, *, from_dir: Path, output_dir: Path) -> str | None:
    """Relative URL from an HTML directory to a brand asset."""
    asset_path = output_dir / "assets" / "brand" / filename
    if not asset_path.is_file():
        return None
    return Path(os.path.relpath(asset_path.resolve(), from_dir.resolve())).as_posix()


def brand_context(*, from_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Template context fields for favicon and header logo links."""
    ensure_brand_assets(output_dir)
    logo_href = brand_href(LOGO_FILENAME, from_dir=from_dir, output_dir=output_dir)
    favicon_href = brand_href(FAVICON_FILENAME, from_dir=from_dir, output_dir=output_dir)
    return {
        "app_title": APP_TITLE,
        "logo_href": logo_href,
        "favicon_href": favicon_href,
    }


def refresh_saved_report_branding(output_dir: Path) -> int:
    """Inject favicon and sidebar logo into existing ``report.html`` files.

    Older reports rendered before branding was added are patched in place so
    users do not need to re-run a full analysis.
    """
    ensure_brand_assets(output_dir)
    reports_root = output_dir / "reports"
    if not reports_root.is_dir():
        return 0

    favicon_markup = (
        '<link rel="icon" type="image/png" href="../../../assets/brand/favicon.png">\n'
        '<link rel="apple-touch-icon" href="../../../assets/brand/favicon.png">\n'
    )
    brand_css = (
        "nav h1 { font-size: 18px; padding: 0 20px 16px; color: var(--accent); }\n"
        ".app-brand { display: flex; align-items: center; gap: 10px; padding: 0 20px 16px; }\n"
        ".app-brand--nav .app-brand-title { font-size: 18px; font-weight: 700; "
        "color: var(--accent); line-height: 1.2; }\n"
        ".app-brand--nav .app-logo { width: 34px; height: 34px; border-radius: 8px; "
        "flex-shrink: 0; object-fit: cover; box-shadow: 0 2px 10px rgba(0, 0, 0, .35); }\n"
    )
    brand_markup = (
        '<div class="app-brand app-brand--nav">'
        '<img src="../../../assets/brand/logo.png" alt="" class="app-logo" aria-hidden="true">'
        f'<span class="app-brand-title">{APP_TITLE}</span></div>'
    )
    updated = 0

    for report_html in reports_root.glob("*/*/report.html"):
        text = report_html.read_text(encoding="utf-8")
        original = text

        if 'rel="icon"' not in text:
            text = text.replace(
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                + favicon_markup,
                1,
            )

        if ".app-brand--nav" not in text:
            text = text.replace(
                "nav h1 { font-size: 18px; padding: 0 20px 16px; color: var(--accent); }\n",
                brand_css,
                1,
            )

        text = text.replace(f"<h1>{APP_TITLE}</h1>", brand_markup, 1)

        if text != original:
            report_html.write_text(text, encoding="utf-8")
            updated += 1

    return updated
