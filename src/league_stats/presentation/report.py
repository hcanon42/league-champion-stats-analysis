"""HTML report generation: improvement score and Jinja2 dashboard rendering."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from league_stats.analysis.improvement import column_mean, score_categories
from league_stats.core.champions import (
    build_label,
    champion_display_name,
    champion_slug,
    role_display,
)
from league_stats.core.models import Recommendation
from league_stats.presentation.brand_assets import brand_context, refresh_saved_report_branding
from league_stats.presentation.ui_icons import iconify_for_key, tooltip_for_label
from league_stats.utils import get_logger

if TYPE_CHECKING:
    from league_stats.infra.ddragon_assets import DDragonAssets


def utc_now_iso() -> str:
    """UTC timestamp for ``generated_at`` (ISO-8601, sortable, JS-parseable)."""
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_generated_at_ts(raw: str | None) -> int:
    """Return UTC epoch milliseconds for sorting; ``0`` when unparseable."""
    if not raw:
        return 0
    legacy = re.match(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})\s+UTC$", raw)
    if legacy:
        dt = datetime(
            int(legacy.group(1)),
            int(legacy.group(2)),
            int(legacy.group(3)),
            int(legacy.group(4)),
            int(legacy.group(5)),
            tzinfo=timezone.utc,
        )
        return int(dt.timestamp() * 1000)
    date_only = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if date_only:
        dt = datetime(
            int(date_only.group(1)),
            int(date_only.group(2)),
            int(date_only.group(3)),
            tzinfo=timezone.utc,
        )
        return int(dt.timestamp() * 1000)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return 0


def is_group_player_label(player: str) -> bool:
    """True when the label represents a pooled multi-account run."""
    return "," in player


def enrich_index_report(report: dict[str, Any]) -> dict[str, Any]:
    """Add index-page fields to one report metadata dict."""
    player = str(report.get("player", ""))
    generated_at = str(report.get("generated_at", ""))
    report["is_group"] = is_group_player_label(player)
    report["updated_ts"] = parse_generated_at_ts(generated_at)
    report["default_href"] = str(report.get("href", ""))
    return report


@dataclass(frozen=True)
class ScoreComponent:
    """One category of the improvement score."""

    name: str
    score: float  # 0-100
    value: str
    hint: str


def improvement_score(
    matches_df: pd.DataFrame, *, role: str = "MIDDLE"
) -> tuple[float, list[ScoreComponent]]:
    """Compute the composite improvement score (0-100) and its category components.

    Each component is a lane-dependent category score (Economy, Fight, Laning, …)
    built from several underlying metrics. Benchmarks are role-aware targets
    derived from static tier data; the score tracks progress between runs for
    the same build, not cross-player comparison.

    Args:
        matches_df: Master per-game table.
        role: Riot team position (``TOP``, ``MIDDLE``, ``JUNGLE``, ...).

    Returns:
        Tuple of overall score and the per-category components.
    """
    if matches_df.empty:
        return 0.0, []

    from league_stats.analysis.combat import prefers_cc_over_dpm
    from league_stats.analysis.peer.benchmarks import try_role_benchmark
    from league_stats.core.role_metrics import role_profile

    profile = role_profile(role)
    gold = try_role_benchmark("GOLD", role) or {}
    use_cc = prefers_cc_over_dpm(
        role, avg_damage_share=column_mean(matches_df, "damage_share")
    )

    categories = score_categories(
        profile.score_components,
        matches_df,
        gold=gold,
        role=profile.role,
        use_cc=use_cc,
    )
    components = [
        ScoreComponent(name=c.name, score=c.score, value=c.value, hint=c.hint)
        for c in categories
    ]
    overall = round(sum(c.score for c in components) / len(components), 1) if components else 0.0
    return overall, components


class ReportBuilder:
    """Renders the final HTML dashboard via Jinja2."""

    def __init__(self, template_dir: Path) -> None:
        """Create the builder.

        Args:
            template_dir: Directory containing ``report.html``.
        """
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
        )
        self._env.globals["iconify"] = iconify_for_key
        self._env.globals["metric_tooltip"] = tooltip_for_label
        self._log = get_logger("report")

    def render(self, output_path: Path, context: dict[str, Any]) -> Path:
        """Render the dashboard to disk.

        Args:
            output_path: Destination ``report.html`` path.
            context: Template context (sections, figures, tables, score...).

        Returns:
            The written path.
        """
        template = self._env.get_template("report.html")
        context.setdefault("generated_at", utc_now_iso())
        output_path.write_text(template.render(**context), encoding="utf-8")
        self._log.info("Report written to %s", output_path)
        return output_path

    def render_index(self, output_dir: Path, reports: list[dict[str, Any]]) -> Path:
        """Render the report switcher index page.

        Args:
            output_dir: Root output directory (``index.html`` is written here).
            reports: Metadata dicts for each saved report (newest first).

        Returns:
            Path of ``index.html``.
        """
        template = self._env.get_template("index.html")
        output_path = output_dir / "index.html"
        enriched = [enrich_index_report(dict(report)) for report in reports]
        players = group_reports_by_player(enriched)
        flat_reports = sorted(
            enriched,
            key=lambda entry: (entry.get("games", 0), entry.get("updated_ts", 0)),
            reverse=True,
        )
        context = {
            **brand_context(from_dir=output_dir, output_dir=output_dir),
            "players": players,
            "reports": flat_reports,
            "report_count": len(enriched),
            "generated_at": utc_now_iso(),
        }
        output_path.write_text(template.render(**context), encoding="utf-8")
        self._log.info(
            "Report index written to %s (%d reports, %d players)",
            output_path,
            len(reports),
            len(players),
        )
        return output_path

    def render_player_hub(self, player_dir: Path, manifest: dict[str, Any]) -> Path:
        """Render the per-player champion switcher landing page.

        Args:
            player_dir: ``output/reports/{player}/`` directory.
            manifest: Player manifest with ``builds`` and ``default_href``.

        Returns:
            Path of ``index.html`` inside ``player_dir``.
        """
        template = self._env.get_template("player_hub.html")
        output_path = player_dir / "index.html"
        context = {
            **brand_context(from_dir=player_dir, output_dir=player_dir.parent.parent),
            "player": manifest.get("player", ""),
            "builds": manifest.get("builds", []),
            "default_href": manifest.get("default_href", ""),
            "default_report_href": manifest.get("default_href", ""),
            "generated_at": utc_now_iso(),
        }
        output_path.write_text(template.render(**context), encoding="utf-8")
        self._log.info("Player hub written to %s", output_path)
        return output_path


def build_player_builds_nav(
    builds: list[dict[str, Any]],
    *,
    current_champion: str,
    current_role: str,
    assets: "DDragonAssets | None" = None,
    from_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build sidebar champion links relative to the current report directory."""
    current_slug = champion_slug(current_champion, current_role)
    nav: list[dict[str, Any]] = []
    for build in builds:
        slug = champion_slug(str(build["champion"]), str(build["role"]))
        winrate = float(build.get("winrate", 0.0))
        riot_id = str(build["champion"])
        icon_href = None
        role_icon = None
        if assets is not None and from_dir is not None:
            icon_href = assets.champion_href(riot_id, from_dir=from_dir)
            role_icon = assets.role_href(str(build["role"]), from_dir=from_dir)
        nav.append(
            {
                "label": (
                    f'{build["build_label"]} · {build["games"]}g · '
                    f"{winrate * 100:.0f}% WR"
                ),
                "build_label": str(build["build_label"]),
                "champion": champion_display_name(riot_id),
                "role": str(build["role"]),
                "role_display": str(build.get("role_display", role_display(str(build["role"])))),
                "games": int(build.get("games", 0)),
                "winrate": winrate,
                "href": f"../{slug}/report.html",
                "selected": slug == current_slug,
                "champion_icon": icon_href,
                "role_icon": role_icon,
            }
        )
    return nav


def write_player_manifest(player_dir: Path, manifest: dict[str, Any]) -> Path:
    """Persist the player-level build manifest."""
    player_dir.mkdir(parents=True, exist_ok=True)
    path = player_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def build_manifest_entry(
    *,
    champion: str,
    role: str,
    games: int,
    winrate: float,
) -> dict[str, Any]:
    """Create one manifest build entry with a report-relative href."""
    slug = champion_slug(champion, role)
    return {
        "champion": champion,
        "role": role,
        "role_display": role_display(role),
        "build_label": build_label(champion, role),
        "games": games,
        "winrate": round(winrate, 3),
        "href": f"{slug}/report.html",
    }


def write_report_meta(report_dir: Path, meta: dict[str, Any]) -> Path:
    """Persist report metadata beside ``report.html``.

    Args:
        report_dir: Directory for this player/champion/lane run.
        meta: Serializable metadata (player, champion, lane, stats...).

    Returns:
        Path of ``meta.json``.
    """
    path = report_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def discover_player_builds(player_dir: Path) -> list[dict[str, Any]]:
    """Scan a player directory for completed build reports.

    Args:
        player_dir: ``output/reports/{player}/`` directory.

    Returns:
        Build metadata dicts sorted by game count (most played first).
        Each entry includes an ``href`` relative to ``player_dir``.
    """
    if not player_dir.is_dir():
        return []

    builds: list[dict[str, Any]] = []
    for meta_path in sorted(player_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report_html = meta_path.parent / "report.html"
        if not report_html.is_file():
            continue
        slug = meta_path.parent.name
        meta["href"] = f"{slug}/report.html"
        builds.append(meta)

    builds.sort(key=lambda entry: (entry.get("games", 0), entry.get("generated_at", "")), reverse=True)
    return builds


def refresh_player_hub(
    player_dir: Path,
    template_dir: Path,
    *,
    player_label: str | None = None,
    assets: "DDragonAssets | None" = None,
) -> Path | None:
    """Rebuild ``output/reports/{player}/index.html`` from on-disk build metadata.

    Args:
        player_dir: Player reports root.
        template_dir: Directory containing ``player_hub.html``.
        player_label: Display label (``Name#TAG``); inferred from builds when omitted.

    Returns:
        Path of the player hub, or ``None`` when no builds exist yet.
    """
    builds = discover_player_builds(player_dir)
    if not builds:
        return None

    label = player_label or str(builds[0].get("player", ""))
    if assets is not None:
        for build in builds:
            riot_id = str(build.get("champion", ""))
            build["champion_icon"] = assets.champion_href(
                riot_id,
                from_dir=player_dir,
            )
            build["champion"] = champion_display_name(riot_id)
            build["role_icon"] = assets.role_href(
                str(build.get("role", "")),
                from_dir=player_dir,
            )
    manifest = {
        "player": label,
        "builds": builds,
        "default_href": builds[0]["href"],
    }
    write_player_manifest(player_dir, manifest)
    return ReportBuilder(template_dir).render_player_hub(player_dir, manifest)


def refresh_all_player_hubs(
    output_dir: Path,
    template_dir: Path,
    *,
    assets: "DDragonAssets | None" = None,
) -> list[Path]:
    """Rebuild every player hub under ``output/reports/``."""
    reports_root = output_dir / "reports"
    if not reports_root.is_dir():
        return []

    hubs: list[Path] = []
    for player_dir in sorted(reports_root.iterdir()):
        if not player_dir.is_dir():
            continue
        hub = refresh_player_hub(player_dir, template_dir, assets=assets)
        if hub is not None:
            hubs.append(hub)
    return hubs


def refresh_report_indexes(
    output_dir: Path,
    template_dir: Path,
    *,
    player_dir: Path | None = None,
    player_label: str | None = None,
    assets: "DDragonAssets | None" = None,
) -> tuple[Path, Path | None]:
    """Rebuild global and optional player report index pages.

    Call after each report is written so indexes stay current during batch runs.

    Args:
        output_dir: Root output directory.
        template_dir: Template directory.
        player_dir: Optional player reports root for the player hub.
        player_label: Optional player display label for the hub.

    Returns:
        Tuple of global index path and optional player hub path.
    """
    global_index = refresh_report_index(output_dir, template_dir, assets=assets)
    player_hub = None
    if player_dir is not None:
        player_hub = refresh_player_hub(
            player_dir, template_dir, player_label=player_label, assets=assets
        )
    return global_index, player_hub


def discover_reports(output_dir: Path) -> list[dict[str, Any]]:
    """Scan ``output/reports/`` for saved report metadata.

    Args:
        output_dir: Root output directory.

    Returns:
        Report metadata dicts sorted by ``generated_at`` (newest first).
        Each entry includes an ``href`` relative to ``output_dir``.
    """
    reports_root = output_dir / "reports"
    if not reports_root.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for meta_path in reports_root.glob("*/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report_html = meta_path.parent / "report.html"
        if not report_html.is_file():
            continue
        meta["href"] = report_html.relative_to(output_dir).as_posix()
        entries.append(meta)

    entries.sort(key=lambda entry: entry.get("generated_at", ""), reverse=True)
    return entries


def group_reports_by_player(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat report metadata into per-player summary cards for the global index.

    Args:
        reports: Metadata dicts from :func:`discover_reports` (enriched for index).

    Returns:
        Player groups sorted by most recent update. Each group contains ``player``,
        ``default_href``, ``build_count``, ``total_games``, ``last_updated``, and
        ``reports`` (builds sorted by games).
    """
    by_player: dict[str, list[dict[str, Any]]] = {}

    for report in reports:
        player = str(report.get("player", ""))
        by_player.setdefault(player, []).append(report)

    groups: list[dict[str, Any]] = []
    for player, builds in by_player.items():
        builds.sort(
            key=lambda entry: (entry.get("games", 0), entry.get("updated_ts", 0)),
            reverse=True,
        )
        total_games = sum(int(entry.get("games", 0)) for entry in builds)
        last_updated_ts = max(int(entry.get("updated_ts", 0)) for entry in builds)
        last_updated = max(
            (str(entry.get("generated_at", "")) for entry in builds),
            key=lambda value: parse_generated_at_ts(value),
            default="",
        )
        groups.append(
            {
                "player": player,
                "is_group": is_group_player_label(player),
                "default_href": str(builds[0].get("href", "")),
                "build_count": len(builds),
                "total_games": total_games,
                "last_updated": last_updated,
                "last_updated_ts": last_updated_ts,
                "reports": builds,
            }
        )
    groups.sort(key=lambda group: group.get("last_updated_ts", 0), reverse=True)
    return groups


def copy_index_static(template_dir: Path, output_dir: Path) -> Path:
    """Copy shared index stylesheet into the output assets tree."""
    src = template_dir / "static" / "index.css"
    dest_dir = output_dir / "assets" / "static"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "index.css"
    shutil.copy2(src, dest)
    return dest


def refresh_report_index(
    output_dir: Path,
    template_dir: Path,
    *,
    assets: "DDragonAssets | None" = None,
) -> Path:
    """Rebuild ``output/index.html`` from on-disk report metadata.

    Args:
        output_dir: Root output directory.
        template_dir: Directory containing ``index.html``.
        assets: Optional icon catalog for champion images.

    Returns:
        Path of the rendered index page.
    """
    reports = discover_reports(output_dir)
    if assets is not None:
        for report in reports:
            riot_id = str(report.get("champion", ""))
            report["champion_icon"] = assets.champion_href(
                riot_id,
                from_dir=output_dir,
            )
            report["champion"] = champion_display_name(riot_id)
            report["role_icon"] = assets.role_href(
                str(report.get("role", "")),
                from_dir=output_dir,
            )
    builder = ReportBuilder(template_dir)
    copy_index_static(template_dir, output_dir)
    index_path = builder.render_index(output_dir, reports)
    refresh_saved_report_branding(output_dir)
    return index_path


def score_badge(recommendation: Recommendation) -> str:
    """CSS badge class for a recommendation's priority.

    Args:
        recommendation: The recommendation.

    Returns:
        One of ``high``/``medium``/``low``.
    """
    if recommendation.priority >= 2.0:
        return "high"
    if recommendation.priority >= 1.2:
        return "medium"
    return "low"
