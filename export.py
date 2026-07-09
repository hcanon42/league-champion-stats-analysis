"""Export layer: CSV/JSON/Markdown artefacts written to ``output/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.coach import recommendations_markdown
from models import Recommendation
from utils import get_logger


class Exporter:
    """Writes every tabular and textual artefact of an analysis run."""

    def __init__(self, output_dir: Path) -> None:
        """Create the exporter.

        Args:
            output_dir: Destination directory (created if missing).
        """
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log = get_logger("export")

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        """Write a dataframe as CSV (also for empty frames, header only).

        Args:
            name: File name without extension (e.g. ``matches``).
            frame: The dataframe to write.

        Returns:
            The written path.
        """
        path = self._dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        self._log.debug("Wrote %s (%d rows)", path, len(frame))
        return path

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """Write the aggregated summary as pretty-printed JSON.

        Args:
            summary: The nested summary dict.

        Returns:
            The written path.
        """
        path = self._dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path

    def write_recommendations(
        self, recommendations: list[Recommendation], *, build_label: str = "Viktor mid"
    ) -> Path:
        """Write the coaching recommendations as Markdown.

        Args:
            recommendations: Ranked recommendations.
            build_label: Champion + lane label for the document title.

        Returns:
            The written path.
        """
        path = self._dir / "recommendations.md"
        path.write_text(
            recommendations_markdown(recommendations, build_label=build_label),
            encoding="utf-8",
        )
        return path

    def write_all(
        self,
        tables: dict[str, pd.DataFrame],
        summary: dict[str, Any],
        recommendations: list[Recommendation],
        *,
        build_label: str = "Viktor mid",
    ) -> list[Path]:
        """Write every export in one call.

        Args:
            tables: Mapping of file base name to dataframe.
            summary: Aggregated summary for ``summary.json``.
            recommendations: Ranked recommendations for ``recommendations.md``.
            build_label: Champion + lane label for recommendations export.

        Returns:
            Paths of every written file.
        """
        paths = [self.write_csv(name, frame) for name, frame in tables.items()]
        paths.append(self.write_summary(summary))
        paths.append(self.write_recommendations(recommendations, build_label=build_label))
        return paths
