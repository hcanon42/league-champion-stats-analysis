"""Compact chatbot export for recent games."""

from __future__ import annotations

from typing import Any

from league_stats.core.models import GameDetail, GameReviewPayload


def _notable_deaths(detail: GameDetail, *, limit: int = 3) -> list[dict[str, Any]]:
    flagged = [death for death in detail.deaths if death.flags]
    if not flagged:
        flagged = detail.deaths
    rows: list[dict[str, Any]] = []
    for death in flagged[:limit]:
        rows.append(
            {
                "minute": death.minute,
                "zone": death.zone,
                "flags": death.flags,
            }
        )
    return rows


def _slim_game(detail: GameDetail) -> dict[str, Any]:
    fights_won = sum(1 for fight in detail.fights if fight.fight_won)
    obj_present = sum(1 for obj in detail.objectives if obj.present)
    obj_total = len(detail.objectives)
    obj_dead = sum(1 for obj in detail.objectives if obj.dead_before)
    return {
        "index": detail.index,
        "match_id": detail.match_id,
        "date": detail.date,
        "result": detail.result,
        "opponent": detail.opponent,
        "kda": detail.kda,
        "archetype": detail.archetype,
        "score": detail.score.model_dump(),
        "highlights": {
            "good": [f"{b.title}: {b.detail}" for b in detail.behaviors_good],
            "bad": [f"{b.title}: {b.detail}" for b in detail.behaviors_bad],
        },
        "key_stats": detail.key_stats,
        "vs_baseline": [row.model_dump() for row in detail.vs_baseline],
        "events_summary": {
            "deaths_count": len(detail.deaths),
            "notable_deaths": _notable_deaths(detail),
            "teamfights": {
                "participated": len(detail.fights),
                "won": fights_won,
            },
            "objectives": {
                "present": obj_present,
                "total": obj_total,
                "dead_before": obj_dead,
            },
            "build": {
                "keystone": detail.build.keystone,
                "core_items": detail.build.items[:3],
            },
        },
    }


def game_review_chatbot_export(
    payload: GameReviewPayload,
    *,
    queue_key: str = "all",
) -> dict[str, Any]:
    """Return a token-aware recent_games block for summary.json."""
    bundle = payload.queues.get(queue_key)
    if bundle is None:
        bundle = next(iter(payload.queues.values()), None)
    if bundle is None:
        return {
            "n": payload.recent_n,
            "scoring": payload.scoring,
            "baseline_window": f"games {payload.recent_n + 1}–{payload.recent_n + payload.baseline_m}",
            "games": [],
        }
    return {
        "n": payload.recent_n,
        "scoring": payload.scoring,
        "baseline_window": f"games {payload.recent_n + 1}–{payload.recent_n + payload.baseline_m}",
        "games": [_slim_game(game) for game in bundle.games],
    }
