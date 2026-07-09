"""HTML report generation: improvement score and Jinja2 dashboard rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

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
            "Resets", _clamp_score(mean("avg_unspent_gold", 800), 1300, 350),
            f"{mean('avg_unspent_gold', 800):.0f}g banked", "Less unspent gold before recalls scores higher",
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
