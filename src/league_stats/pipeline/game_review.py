"""Game Review pipeline wiring."""

from __future__ import annotations

from pathlib import Path

from league_stats.analysis.game_review.views import build_game_review_views as _build_payload
from league_stats.analysis.game_review.views import game_review_to_template_context
from league_stats.core.config import AppConfig
from league_stats.core.models import (
    GameBuildInfo,
    GameDeathRow,
    GameDetail,
    GameObjectiveRow,
    GameReviewPayload,
    GameReviewQueueBundle,
    MatchRecord,
    PeerComparisonResult,
)
from league_stats.infra.ddragon_assets import DDragonAssets
from league_stats.pipeline.frames import AnalysisFrames
from league_stats.presentation.graphs import GraphFactory


def _enrich_game_detail(
    detail: GameDetail,
    *,
    assets: DDragonAssets,
    from_dir: Path,
    champion: str,
) -> GameDetail:
    """Attach relative icon hrefs for champions, runes, items, and objectives."""
    deaths = [
        death.model_copy(
            update={
                "killer_icon": assets.champion_href(death.killer, from_dir=from_dir)
                if death.killer
                else None,
            }
        )
        for death in detail.deaths
    ]
    objectives = [
        objective.model_copy(
            update={
                "objective_icon": assets.objective_href(objective.kind, from_dir=from_dir),
            }
        )
        for objective in detail.objectives
    ]
    build = detail.build
    enriched_build = GameBuildInfo(
        keystone=build.keystone,
        primary_tree=build.primary_tree,
        secondary_tree=build.secondary_tree,
        summoners=list(build.summoners),
        skill_order=build.skill_order,
        items=list(build.items),
        keystone_icon=assets.keystone_href(build.keystone, from_dir=from_dir),
        primary_tree_icon=assets.rune_tree_href(build.primary_tree, from_dir=from_dir),
        secondary_tree_icon=assets.rune_tree_href(build.secondary_tree, from_dir=from_dir),
        summoner_icons=[
            assets.summoner_href(spell_name, from_dir=from_dir) for spell_name in build.summoners
        ],
        item_icons=[
            assets.item_href_by_name(item_name, from_dir=from_dir) for item_name in build.items
        ],
    )
    return detail.model_copy(
        update={
            "champion_icon": assets.champion_href(champion, from_dir=from_dir),
            "opponent_icon": assets.champion_href(detail.opponent, from_dir=from_dir),
            "deaths": deaths,
            "objectives": objectives,
            "build": enriched_build,
        }
    )


def build_game_review_views(
    config: AppConfig,
    records: list[MatchRecord],
    frames: AnalysisFrames,
    peer_comparison: PeerComparisonResult | None,
    *,
    graphs_dir: Path | None = None,
    assets: DDragonAssets | None = None,
    from_dir: Path | None = None,
) -> GameReviewPayload:
    """Build game review payload with timeline figures and UI icon hrefs."""
    payload = _build_payload(config, records, frames, peer_comparison)
    graphs = GraphFactory(graphs_dir) if graphs_dir is not None else None

    queues: dict[str, GameReviewQueueBundle] = {}
    for queue_key, bundle in payload.queues.items():
        games: list[GameDetail] = []
        for detail in bundle.games:
            enriched = detail
            if assets is not None and from_dir is not None:
                enriched = _enrich_game_detail(
                    detail,
                    assets=assets,
                    from_dir=from_dir,
                    champion=config.champion,
                )
            if graphs is not None and enriched.timeline:
                death_mins = [death.minute for death in enriched.deaths]
                enriched = enriched.model_copy(
                    update={"timeline_figure": graphs.game_gold_timeline(enriched.timeline, death_mins)}
                )
            games.append(enriched)
        queues[queue_key] = bundle.model_copy(update={"games": games})
    return payload.model_copy(update={"queues": queues})


__all__ = [
    "build_game_review_views",
    "game_review_to_template_context",
]
