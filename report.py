"""HTML report generation: improvement score and Jinja2 dashboard rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from analysis.economy import RECALL_GOLD_HEALTHY_AVG, RECALL_GOLD_HOARDING_WARN
from champions import build_label, champion_slug, role_display
from models import Recommendation
from utils import get_logger


@dataclass(frozen=True)
class ScoreComponent:
    """One dimension of the improvement score."""

    name: str
    score: float  # 0-100
    value: str
    hint: str


def _clamp_score(value: float, floor: float, ceiling: float) -> float:
    """Map a value linearly onto 0-100 between a floor and a ceiling.

    Args:
        value: Observed metric value.
        floor: Value mapping to 0.
        ceiling: Value mapping to 100.

    Returns:
        Score in [0, 100]; the scale inverts automatically when
        ``floor > ceiling`` (lower-is-better metrics).
    """
    if floor == ceiling:
        return 50.0
    ratio = (value - floor) / (ceiling - floor)
    return round(max(0.0, min(1.0, ratio)) * 100, 1)


def improvement_score(matches_df: pd.DataFrame) -> tuple[float, list[ScoreComponent]]:
    """Compute the composite improvement score (0-100) and its components.

    Benchmarks are fixed, documented targets for a strong Viktor mid player;
    the score is meant to track progress between runs, not to compare
    players.

    Args:
        matches_df: Master per-game table.

    Returns:
        Tuple of overall score and the per-dimension components.
    """
    if matches_df.empty:
        return 0.0, []

    def mean(column: str, default: float = 0.0) -> float:
        """Column mean with NaN safety."""
        series = pd.to_numeric(matches_df.get(column), errors="coerce")
        if series is None:
            return default
        series = series.dropna()
        return float(series.mean()) if not series.empty else default

    components = [
        ScoreComponent(
            "Laning", _clamp_score(mean("gd10"), -800, 800),
            f"{mean('gd10'):+.0f} gold @10", "Average gold diff vs lane opponent at 10 min",
        ),
        ScoreComponent(
            "Farming", _clamp_score(mean("cs10"), 55, 85),
            f"{mean('cs10'):.0f} CS @10", "Benchmark: 55 (weak) to 85 (excellent)",
        ),
        ScoreComponent(
            "Survival", _clamp_score(mean("deaths"), 7.5, 3.0),
            f"{mean('deaths'):.1f} deaths/game", "Fewer deaths score higher (7.5 -> 3.0)",
        ),
        ScoreComponent(
            "Damage", _clamp_score(mean("damage_share"), 0.18, 0.32),
            f"{mean('damage_share') * 100:.0f}% team damage", "Share of team damage to champions",
        ),
        ScoreComponent(
            "Vision", _clamp_score(mean("vspm"), 0.8, 2.0),
            f"{mean('vspm'):.2f} VS/min", "Vision score per minute (0.8 -> 2.0)",
        ),
        ScoreComponent(
            "Objectives", _clamp_score(mean("objectives_present_rate", 0.0), 0.30, 0.75),
            f"{mean('objectives_present_rate') * 100:.0f}% presence", "Presence at epic monster takes",
        ),
        ScoreComponent(
            "Resets",
            _clamp_score(mean("avg_unspent_gold", 800), RECALL_GOLD_HOARDING_WARN, RECALL_GOLD_HEALTHY_AVG),
            f"{mean('avg_unspent_gold', 800):.0f}g banked",
            f"Component backs land around 800–1300g; {RECALL_GOLD_HOARDING_WARN}g+ before resets scores lower",
        ),
    ]
    overall = round(sum(c.score for c in components) / len(components), 1)
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
        context.setdefault("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
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
        context = {
            "app_title": "Champion Stats Analyzer",
            "reports": reports,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        output_path.write_text(template.render(**context), encoding="utf-8")
        self._log.info("Report index written to %s (%d reports)", output_path, len(reports))
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
            "app_title": "Champion Stats Analyzer",
            "player": manifest.get("player", ""),
            "builds": manifest.get("builds", []),
            "default_href": manifest.get("default_href", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        output_path.write_text(template.render(**context), encoding="utf-8")
        self._log.info("Player hub written to %s", output_path)
        return output_path


def build_player_builds_nav(
    builds: list[dict[str, Any]],
    *,
    current_champion: str,
    current_role: str,
) -> list[dict[str, Any]]:
    """Build sidebar dropdown entries relative to the current report directory."""
    current_slug = champion_slug(current_champion, current_role)
    nav: list[dict[str, Any]] = []
    for build in builds:
        slug = champion_slug(str(build["champion"]), str(build["role"]))
        winrate = float(build.get("winrate", 0.0))
        nav.append(
            {
                "label": (
                    f'{build["build_label"]} · {build["games"]}g · '
                    f"{winrate * 100:.0f}% WR"
                ),
                "href": f"../{slug}/report.html",
                "selected": slug == current_slug,
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
    manifest = {
        "player": label,
        "builds": builds,
        "default_href": builds[0]["href"],
    }
    write_player_manifest(player_dir, manifest)
    return ReportBuilder(template_dir).render_player_hub(player_dir, manifest)


def refresh_report_indexes(
    output_dir: Path,
    template_dir: Path,
    *,
    player_dir: Path | None = None,
    player_label: str | None = None,
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
    global_index = refresh_report_index(output_dir, template_dir)
    player_hub = None
    if player_dir is not None:
        player_hub = refresh_player_hub(
            player_dir, template_dir, player_label=player_label
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


def refresh_report_index(output_dir: Path, template_dir: Path) -> Path:
    """Rebuild ``output/index.html`` from on-disk report metadata.

    Args:
        output_dir: Root output directory.
        template_dir: Directory containing ``index.html``.

    Returns:
        Path of the rendered index page.
    """
    builder = ReportBuilder(template_dir)
    return builder.render_index(output_dir, discover_reports(output_dir))


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
