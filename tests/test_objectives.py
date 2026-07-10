"""Tests for objective setup analysis."""

from __future__ import annotations

from league_stats.analysis.objectives import extract_objectives
from league_stats.analysis.timeline import build_context
from league_stats.core.models import ObjectiveKind
from tests.fixtures import MY_PUUID, make_match, make_timeline


def test_objectives_extracted() -> None:
    """One dragon and one baron are extracted with correct ownership."""
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    objectives = extract_objectives(ctx)
    kinds = {o.kind for o in objectives}
    assert kinds == {ObjectiveKind.DRAGON, ObjectiveKind.BARON}
    dragon = next(o for o in objectives if o.kind == ObjectiveKind.DRAGON)
    baron = next(o for o in objectives if o.kind == ObjectiveKind.BARON)
    assert dragon.taken_by_team is False
    assert baron.taken_by_team is True


def test_dragon_vision_setup() -> None:
    """The support's control ward 20 s before the dragon is counted."""
    ctx = build_context(make_match(), make_timeline(), MY_PUUID)
    dragon = next(
        o for o in extract_objectives(ctx) if o.kind == ObjectiveKind.DRAGON
    )
    assert dragon.team_wards_before >= 1
    assert dragon.control_wards_before >= 1
