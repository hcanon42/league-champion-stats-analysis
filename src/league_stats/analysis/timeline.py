"""Timeline parsing: frame context, checkpoint snapshots, recalls and roams.

The :class:`TimelineContext` built here is the shared input for every other
timeline-level extractor (deaths, teamfights, objectives, vision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from league_stats.core.models import Position, RecallEvent, RoamEvent, SnapshotSet, TimelineStats, Zone
from league_stats.utils import classify_zone, ms_to_min, push_progress

SNAPSHOT_MINUTES: tuple[int, ...] = (5, 10, 15, 20)
STARTING_SHOP_CUTOFF_MS: int = 45_000
SHOPPING_WINDOW_MS: int = 30_000
ROAM_DISTANCE_FROM_MID: float = 2_500.0
LANE_PHASE_END_MIN: int = 14


@dataclass(frozen=True)
class TimelineContext:
    """Pre-indexed view of a Match-V5 timeline for a single tracked player."""

    participant_id: int
    opponent_id: int | None
    team_ids: frozenset[int]
    enemy_ids: frozenset[int]
    blue_side: bool
    duration_s: int
    frames: list[dict[str, Any]]
    events: list[dict[str, Any]]
    id_to_champion: dict[int, str]

    def events_of(self, *types: str) -> list[dict[str, Any]]:
        """Return all timeline events of the given types, in time order.

        Args:
            *types: Event ``type`` strings (e.g. ``"CHAMPION_KILL"``).

        Returns:
            Matching events sorted by timestamp.
        """
        wanted = set(types)
        return [e for e in self.events if e.get("type") in wanted]

    def frame_at_minute(self, minute: int) -> dict[str, Any] | None:
        """Return the frame closest to a minute mark, if within the game.

        Args:
            minute: Minute mark (frames are emitted every 60 s).

        Returns:
            The frame dict or ``None`` when the game ended earlier.
        """
        if minute >= len(self.frames):
            return None
        return self.frames[minute]

    def participant_frame(self, frame: dict[str, Any], pid: int) -> dict[str, Any] | None:
        """Return a participant's frame data within a timeline frame.

        Args:
            frame: A timeline frame.
            pid: Participant id (1-10).

        Returns:
            The participant frame dict or ``None``.
        """
        return frame.get("participantFrames", {}).get(str(pid))

    def position_at_ms(self, pid: int, timestamp_ms: int) -> Position | None:
        """Approximate a participant's position at a timestamp.

        Uses the nearest timeline frame (frames are 60 s apart, so this is a
        coarse approximation, documented as such).

        Args:
            pid: Participant id.
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            The approximate :class:`~models.Position`, or ``None``.
        """
        if not self.frames:
            return None
        index = min(int(round(timestamp_ms / 60_000)), len(self.frames) - 1)
        pframe = self.participant_frame(self.frames[index], pid)
        if not pframe or "position" not in pframe:
            return None
        return Position(**pframe["position"])


def build_context(match: dict[str, Any], timeline: dict[str, Any], puuid: str) -> TimelineContext:
    """Build a :class:`TimelineContext` for the tracked player.

    Args:
        match: Raw match-v5 match document.
        timeline: Raw match-v5 timeline document.
        puuid: PUUID of the tracked player.

    Returns:
        A fully indexed timeline context.

    Raises:
        ValueError: If the player is not part of the match.
    """
    participants = match["info"]["participants"]
    me = next((p for p in participants if p["puuid"] == puuid), None)
    if me is None:
        raise ValueError(f"PUUID {puuid} not found in match {match['metadata']['matchId']}")

    my_id = int(me["participantId"])
    my_team = int(me["teamId"])
    team_ids = frozenset(int(p["participantId"]) for p in participants if p["teamId"] == my_team)
    enemy_ids = frozenset(int(p["participantId"]) for p in participants if p["teamId"] != my_team)
    opponent = next(
        (
            p
            for p in participants
            if p["teamId"] != my_team and p.get("teamPosition") == me.get("teamPosition")
        ),
        None,
    )
    frames = timeline["info"]["frames"]
    events = sorted(
        (e for frame in frames for e in frame.get("events", [])),
        key=lambda e: e.get("timestamp", 0),
    )
    duration_s = int(match["info"].get("gameDuration", 0))
    if duration_s > 100_000:  # legacy matches report milliseconds
        duration_s //= 1000
    return TimelineContext(
        participant_id=my_id,
        opponent_id=int(opponent["participantId"]) if opponent else None,
        team_ids=team_ids,
        enemy_ids=enemy_ids,
        blue_side=my_team == 100,
        duration_s=duration_s,
        frames=frames,
        events=events,
        id_to_champion={int(p["participantId"]): str(p["championName"]) for p in participants},
    )


def _cs_of(pframe: dict[str, Any]) -> int:
    """Total CS (lane + jungle minions) of a participant frame."""
    return int(pframe.get("minionsKilled", 0)) + int(pframe.get("jungleMinionsKilled", 0))


def _snapshots(ctx: TimelineContext) -> SnapshotSet:
    """Compute gold/XP/CS checkpoints and lane differentials.

    Args:
        ctx: Timeline context.

    Returns:
        The populated :class:`~models.SnapshotSet`.
    """
    snap = SnapshotSet()
    for minute in SNAPSHOT_MINUTES:
        frame = ctx.frame_at_minute(minute)
        if frame is None:
            continue
        mine = ctx.participant_frame(frame, ctx.participant_id)
        if mine is None:
            continue
        snap.gold[minute] = int(mine.get("totalGold", 0))
        snap.xp[minute] = int(mine.get("xp", 0))
        snap.cs[minute] = _cs_of(mine)
        theirs = (
            ctx.participant_frame(frame, ctx.opponent_id) if ctx.opponent_id is not None else None
        )
        snap.gold_diff[minute] = (
            snap.gold[minute] - int(theirs.get("totalGold", 0)) if theirs else None
        )
        snap.xp_diff[minute] = snap.xp[minute] - int(theirs.get("xp", 0)) if theirs else None
        snap.cs_diff[minute] = snap.cs[minute] - _cs_of(theirs) if theirs else None
    return snap


def extract_recalls(ctx: TimelineContext) -> list[RecallEvent]:
    """Infer recalls from clusters of item purchases.

    The Match-V5 timeline has no recall event; a shopping trip (a burst of
    purchases after the starting shop) is used as a documented proxy. The
    unspent gold is the player's banked gold on the last frame before the
    trip started.

    Args:
        ctx: Timeline context.

    Returns:
        Inferred recall events in chronological order.
    """
    purchases = [
        e
        for e in ctx.events_of("ITEM_PURCHASED")
        if int(e.get("participantId", 0)) == ctx.participant_id
        and int(e.get("timestamp", 0)) > STARTING_SHOP_CUTOFF_MS
    ]
    recalls: list[RecallEvent] = []
    window_start: int | None = None
    last_ts: int | None = None
    for event in purchases:
        ts = int(event["timestamp"])
        if last_ts is None or ts - last_ts > SHOPPING_WINDOW_MS:
            if window_start is not None:
                recalls.append(_recall_from_window(ctx, window_start))
            window_start = ts
        last_ts = ts
    if window_start is not None:
        recalls.append(_recall_from_window(ctx, window_start))
    return recalls


def _recall_from_window(ctx: TimelineContext, window_start_ms: int) -> RecallEvent:
    """Build a recall event from the start of a shopping window.

    Args:
        ctx: Timeline context.
        window_start_ms: Timestamp of the first purchase in the window.

    Returns:
        The recall event with the banked gold before shopping.
    """
    frame_idx = max(0, min(int(window_start_ms // 60_000), len(ctx.frames) - 1))
    pframe = ctx.participant_frame(ctx.frames[frame_idx], ctx.participant_id)
    unspent = int(pframe.get("currentGold", 0)) if pframe else 0
    return RecallEvent(minute=ms_to_min(window_start_ms), unspent_gold=max(0, unspent))


def extract_roams(ctx: TimelineContext) -> list[RoamEvent]:
    """Infer early-game roams from frame positions away from mid lane.

    A roam is one or more consecutive pre-15-minute frames where the player
    is far from the mid-lane axis and not in a base.

    Args:
        ctx: Timeline context.

    Returns:
        Inferred roam events.
    """
    roams: list[RoamEvent] = []
    current: RoamEvent | None = None
    last_minute = min(LANE_PHASE_END_MIN + 1, len(ctx.frames) - 1)
    for minute in range(3, max(3, last_minute) + 1):
        frame = ctx.frame_at_minute(minute)
        pframe = ctx.participant_frame(frame, ctx.participant_id) if frame else None
        if not pframe or "position" not in pframe:
            continue
        pos = Position(**pframe["position"])
        zone = classify_zone(pos)
        off_mid = abs(pos.x - pos.y) / 2**0.5 > ROAM_DISTANCE_FROM_MID
        roaming = off_mid and zone not in (Zone.BASE, Zone.MID_LANE)
        if roaming and current is None:
            current = RoamEvent(start_minute=float(minute), end_minute=float(minute), zone=zone)
        elif roaming and current is not None:
            current.end_minute = float(minute)
        elif not roaming and current is not None:
            roams.append(current)
            current = None
    if current is not None:
        roams.append(current)
    return roams


def _lane_priority(ctx: TimelineContext) -> float | None:
    """Share of laning-phase frames spent past the map centre.

    A proxy for lane priority: how often the player's position projects
    beyond the mid point of the map toward the enemy base.

    Args:
        ctx: Timeline context.

    Returns:
        A ratio in ``[0, 1]``, or ``None`` with no usable frames.
    """
    ahead = 0
    total = 0
    for minute in range(2, min(LANE_PHASE_END_MIN, len(ctx.frames) - 1) + 1):
        frame = ctx.frame_at_minute(minute)
        pframe = ctx.participant_frame(frame, ctx.participant_id) if frame else None
        if not pframe or "position" not in pframe:
            continue
        pos = Position(**pframe["position"])
        if classify_zone(pos) == Zone.BASE:
            continue
        total += 1
        if push_progress(pos, ctx.blue_side) > 0:
            ahead += 1
    return ahead / total if total else None


def _wave_push_ratio(ctx: TimelineContext) -> float | None:
    """Share of mid-lane frames where the wave position implies pushing.

    Wave state is inferred from the player's own position along the mid-lane
    axis (minion positions are not exposed by the API), as documented.

    Args:
        ctx: Timeline context.

    Returns:
        Ratio of "pushing" frames among mid-lane frames, or ``None``.
    """
    pushing = 0
    total = 0
    for minute in range(2, min(LANE_PHASE_END_MIN, len(ctx.frames) - 1) + 1):
        frame = ctx.frame_at_minute(minute)
        pframe = ctx.participant_frame(frame, ctx.participant_id) if frame else None
        if not pframe or "position" not in pframe:
            continue
        pos = Position(**pframe["position"])
        if classify_zone(pos) != Zone.MID_LANE:
            continue
        total += 1
        if push_progress(pos, ctx.blue_side) > 400:
            pushing += 1
    return pushing / total if total else None


def extract_timeline_stats(ctx: TimelineContext, time_dead_s: int) -> TimelineStats:
    """Compute the full set of timeline-derived statistics.

    Args:
        ctx: Timeline context.
        time_dead_s: Total seconds spent dead (from the match document).

    Returns:
        The populated :class:`~models.TimelineStats`.
    """
    gold_series: list[int] = []
    xp_series: list[int] = []
    cs_series: list[int] = []
    opp_gold_series: list[int] = []
    for frame in ctx.frames:
        mine = ctx.participant_frame(frame, ctx.participant_id)
        if mine is None:
            continue
        gold_series.append(int(mine.get("totalGold", 0)))
        xp_series.append(int(mine.get("xp", 0)))
        cs_series.append(_cs_of(mine))
        theirs = (
            ctx.participant_frame(frame, ctx.opponent_id) if ctx.opponent_id is not None else None
        )
        opp_gold_series.append(int(theirs.get("totalGold", 0)) if theirs else 0)

    recalls = extract_recalls(ctx)
    unspent = [r.unspent_gold for r in recalls]
    return TimelineStats(
        snapshots=_snapshots(ctx),
        recalls=recalls,
        roams=extract_roams(ctx),
        avg_unspent_gold_before_recall=(sum(unspent) / len(unspent)) if unspent else None,
        time_dead_s=time_dead_s,
        lane_priority=_lane_priority(ctx),
        wave_push_ratio=_wave_push_ratio(ctx),
        gold_series=gold_series,
        xp_series=xp_series,
        cs_series=cs_series,
        opp_gold_series=opp_gold_series,
    )


def timeline_dataframe_rows(match_id: str, stats: TimelineStats) -> list[dict[str, Any]]:
    """Explode per-minute series into rows for the timeline CSV export.

    Args:
        match_id: Match id the series belong to.
        stats: The parsed timeline statistics.

    Returns:
        One dict per minute with gold/XP/CS and the gold differential.
    """
    rows: list[dict[str, Any]] = []
    for minute, (gold, xp, cs) in enumerate(
        zip(stats.gold_series, stats.xp_series, stats.cs_series)
    ):
        opp_gold = stats.opp_gold_series[minute] if minute < len(stats.opp_gold_series) else 0
        rows.append(
            {
                "match_id": match_id,
                "minute": minute,
                "gold": gold,
                "xp": xp,
                "cs": cs,
                "gold_diff": gold - opp_gold if opp_gold else None,
            }
        )
    return rows
