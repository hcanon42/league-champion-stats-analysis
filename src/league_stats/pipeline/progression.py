"""Form Tracker pipeline: build progression views and exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from league_stats.analysis.coach.engine import VISIBLE_RECOMMENDATIONS
from league_stats.analysis.progression.diff import build_progression_comparison
from league_stats.analysis.progression.export import progression_to_markdown
from league_stats.analysis.progression.slicing import slice_baseline, slice_recent
from league_stats.core.config import (
    PROGRESSION_PRESETS_V1,
    AppConfig,
    QUEUE_FILTER_OPTIONS,
)
from league_stats.core.models import MatchRecord, ProgressionComparison
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.pipeline.bundles import filter_records_by_queue
from league_stats.pipeline.view_models import form_row_display, form_sample_subtitle
from league_stats.presentation.graphs import GraphFactory


def preset_key(recent_n: int, baseline_m: int) -> str:
    """Filesystem/JSON key for a recent+baseline preset."""
    return f"{recent_n}_{baseline_m}"


def _recommendation_payload(rec: Any) -> dict[str, Any]:
    from league_stats.pipeline.bundles import _recommendation_payload as bundle_rec_payload

    return bundle_rec_payload(rec)


def _sample_subtitle(comparison: ProgressionComparison | None) -> str:
    if comparison is None:
        return form_sample_subtitle(recent_games=0, baseline_games=0)
    snap = comparison.snapshot
    return form_sample_subtitle(
        recent_games=snap.recent_games,
        baseline_games=snap.baseline_games,
        overlap_mode=comparison.overlap_mode,
    )


def _serialize_comparison(
    comparison: ProgressionComparison | None,
    *,
    graphs: GraphFactory,
    all_records: list[MatchRecord],
    recent_n: int,
) -> dict[str, Any]:
    """Convert a comparison to a JSON-friendly preset bundle."""
    sample_subtitle = _sample_subtitle(comparison)
    if comparison is None:
        return {
            "available": False,
            "insufficient_reason": "Could not build comparison.",
            "sample_subtitle": sample_subtitle,
            "snapshot": {},
            "delta_rows": [],
            "top_improved": [],
            "top_regressed": [],
            "behavioral_shifts": [],
            "recommendations": [],
            "figures": {},
            "comparison": None,
        }

    snap = comparison.snapshot
    if snap.confidence == "insufficient":
        return {
            "available": False,
            "insufficient_reason": snap.headline,
            "sample_subtitle": sample_subtitle,
            "snapshot": snap.model_dump(),
            "delta_rows": [],
            "top_improved": [],
            "top_regressed": [],
            "behavioral_shifts": [],
            "recommendations": [],
            "figures": {},
            "comparison": comparison.model_dump(),
        }

    delta_rows = [form_row_display(delta.model_dump()) for delta in comparison.deltas]
    top_improved = [form_row_display(delta.model_dump()) for delta in comparison.top_improved]
    top_regressed = [form_row_display(delta.model_dump()) for delta in comparison.top_regressed]
    recommendations = [_recommendation_payload(rec) for rec in comparison.recommendations]

    from league_stats.pipeline.frames import build_analysis_frames

    frames = build_analysis_frames(all_records)
    figures = {
        "form_rolling_wr": graphs.form_rolling_wr(frames.matches_df, recent_n=recent_n),
        "form_metric_delta_bar": graphs.form_metric_delta_bar(comparison.deltas),
    }

    return {
        "available": True,
        "insufficient_reason": None,
        "sample_subtitle": sample_subtitle,
        "snapshot": snap.model_dump(),
        "delta_rows": delta_rows,
        "top_improved": top_improved,
        "top_regressed": top_regressed,
        "behavioral_shifts": comparison.behavioral_shifts,
        "recommendations": recommendations,
        "figures": figures,
        "comparison": comparison.model_dump(),
    }


def build_progression_views(
    config: AppConfig,
    records: list[MatchRecord],
    graphs_dir: Path,
    *,
    assets: DDragonAssets | None = None,
) -> dict[str, Any]:
    """Build per-queue Form Tracker preset bundles."""
    _ = assets
    graphs = GraphFactory(graphs_dir)
    presets = PROGRESSION_PRESETS_V1
    if (config.progression_recent_n, config.progression_baseline_m) not in presets:
        presets = (*presets, (config.progression_recent_n, config.progression_baseline_m))

    views: dict[str, Any] = {}
    for queue_key in QUEUE_FILTER_OPTIONS:
        queue_records = filter_records_by_queue(records, queue_key)
        preset_bundles: dict[str, Any] = {}
        default_key = preset_key(config.progression_recent_n, config.progression_baseline_m)

        for recent_n, baseline_m in presets:
            key = preset_key(recent_n, baseline_m)
            recent_records = slice_recent(queue_records, recent_n)
            baseline_records = slice_baseline(
                queue_records,
                recent_n,
                baseline_m,
                overlap=config.progression_overlap,
            )
            comparison = build_progression_comparison(
                config,
                recent_records,
                baseline_records,
                preset_key=key,
                overlap_mode="inclusive" if config.progression_overlap else "exclusive",
            )
            preset_bundles[key] = _serialize_comparison(
                comparison,
                graphs=graphs,
                all_records=queue_records,
                recent_n=recent_n,
            )

        views[queue_key] = {
            "default_preset": default_key if default_key in preset_bundles else preset_key(*presets[0]),
            "presets": preset_bundles,
            "total_games": len(queue_records),
        }
    return views


def progression_to_template_context(preset_bundle: dict[str, Any]) -> dict[str, Any]:
    """Map a default preset bundle onto Jinja template fields."""
    snap = preset_bundle.get("snapshot") or {}
    return {
        "form_available": preset_bundle.get("available", False),
        "form_insufficient_reason": preset_bundle.get("insufficient_reason"),
        "form_sample_subtitle": preset_bundle.get("sample_subtitle"),
        "form_snapshot": snap,
        "form_delta_rows": preset_bundle.get("delta_rows", []),
        "form_top_improved": preset_bundle.get("top_improved", []),
        "form_top_regressed": preset_bundle.get("top_regressed", []),
        "form_behavioral_shifts": preset_bundle.get("behavioral_shifts", []),
        "form_recommendations": preset_bundle.get("recommendations", []),
        "form_figures": preset_bundle.get("figures", {}),
        "form_recommendation_visible_count": VISIBLE_RECOMMENDATIONS,
    }


def write_progression_exports(
    run_dir: Path,
    comparison: ProgressionComparison | None,
) -> None:
    """Write progression.json and progression.md to the report directory."""
    if comparison is None:
        return
    payload = comparison.model_dump()
    (run_dir / "progression.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "progression.md").write_text(
        progression_to_markdown(comparison),
        encoding="utf-8",
    )
