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
NEARBY_RADIUS: float = 2_200.0


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
    id_to_role: dict[int, str]

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


def participant_team_id(ctx: TimelineContext, pid: int) -> int:
    """Return Riot team id (100 blue / 200 red) for a participant."""
    if pid in ctx.team_ids:
        return 100 if ctx.blue_side else 200
    return 200 if ctx.blue_side else 100


def team_gold_series(ctx: TimelineContext) -> tuple[list[int], list[int], list[int]]:
    """Per-frame team total gold for blue and red.

    Returns:
        ``(timestamps_ms, blue_totals, red_totals)`` aligned by frame index.
    """
    timestamps: list[int] = []
    blue_totals: list[int] = []
    red_totals: list[int] = []
    for frame in ctx.frames:
        ts = int(frame.get("timestamp", 0))
        blue = 0
        red = 0
        for pid in range(1, 11):
            pframe = ctx.participant_frame(frame, pid)
            if not pframe:
                continue
            gold = int(pframe.get("totalGold", 0))
            if participant_team_id(ctx, pid) == 100:
                blue += gold
            else:
                red += gold
        timestamps.append(ts)
        blue_totals.append(blue)
        red_totals.append(red)
    return timestamps, blue_totals, red_totals


def _frame_index_at_or_before(ctx: TimelineContext, timestamp_ms: int) -> int:
    index = 0
    for i, frame in enumerate(ctx.frames):
        if int(frame.get("timestamp", 0)) <= timestamp_ms:
            index = i
        else:
            break
    return index


_SAMPLE_PRIORITY: dict[str, int] = {
    "kill": 4,
    "kill_assist": 3,
    "objective": 3,
    "objective_assist": 3,
    "plate": 2,
    "plate_assist": 2,
    "ward": 3,
    "frame": 1,
}
# Only interpolate between adjacent real samples when they are close in time
# (e.g. back-to-back kills in a teamfight). Wider gaps hold position.
MAX_INTERPOLATION_GAP_MS: int = 20_000
# Scrubber uses exact samples only — no interpolation between keyframes.
EXACT_SNAPSHOT_GAP_MS: int = 0


def _participant_level_at_ms(ctx: TimelineContext, pid: int, timestamp_ms: int) -> int:
    index = _frame_index_at_or_before(ctx, timestamp_ms)
    pframe = ctx.participant_frame(ctx.frames[index], pid)
    if not pframe:
        return 1
    return max(1, int(pframe.get("level", 1)))


def respawn_duration_ms(level: int) -> int:
    """Approximate Summoner's Rift respawn duration for a champion level."""
    clamped = max(1, min(18, level))
    return int((10.0 + clamped * 2.5) * 1000)


def _append_position_sample(
    tracks: dict[int, list[tuple[int, Position, str]]],
    pid: int,
    ts: int,
    pos: Position,
    source: str,
) -> None:
    if 1 <= pid <= 10:
        tracks[pid].append((ts, pos, source))


def _append_event_participants(
    tracks: dict[int, list[tuple[int, Position, str]]],
    event: dict[str, Any],
    pos: Position,
    ts: int,
    *,
    source: str,
) -> None:
    killer = int(event.get("killerId", 0))
    if killer:
        _append_position_sample(tracks, killer, ts, pos, source)
    for assist_id in event.get("assistingParticipantIds", []) or []:
        pid = int(assist_id)
        if pid:
            _append_position_sample(tracks, pid, ts, pos, f"{source}_assist")


def build_position_tracks(ctx: TimelineContext) -> dict[int, list[tuple[int, Position, str]]]:
    """Collect timestamped positions from minute frames and map events.

    The Match-V5 API only snapshots all ten players once per minute in
    ``participantFrames``. Kill, objective, plate, and ward events carry exact
    positions at event time and are merged here with higher priority than
    minute snapshots.
    """
    tracks: dict[int, list[tuple[int, Position, str]]] = {pid: [] for pid in range(1, 11)}

    for frame in ctx.frames:
        ts = int(frame.get("timestamp", 0))
        for pid in range(1, 11):
            pframe = ctx.participant_frame(frame, pid)
            if pframe and "position" in pframe:
                tracks[pid].append((ts, Position(**pframe["position"]), "frame"))

    for event in ctx.events:
        ts = int(event.get("timestamp", 0))
        event_type = str(event.get("type", ""))
        if not event.get("position"):
            continue
        pos = Position(**event["position"])
        if event_type == "CHAMPION_KILL":
            victim = int(event.get("victimId", 0))
            _append_event_participants(tracks, event, pos, ts, source="kill")
            if victim:
                _append_position_sample(tracks, victim, ts, pos, "kill")
        elif event_type in {"CHAMPION_SPECIAL_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"}:
            _append_event_participants(tracks, event, pos, ts, source="objective")
        elif event_type == "TURRET_PLATE_DESTROYED":
            _append_event_participants(tracks, event, pos, ts, source="plate")
        elif event_type == "WARD_PLACED":
            creator = int(event.get("creatorId", 0))
            if creator:
                _append_position_sample(tracks, creator, ts, pos, "ward")

    deduped: dict[int, list[tuple[int, Position, str]]] = {}
    for pid, samples in tracks.items():
        by_ts: dict[int, tuple[int, Position, str]] = {}
        for sample in samples:
            ts, pos, source = sample
            existing = by_ts.get(ts)
            if existing is None or _SAMPLE_PRIORITY.get(source, 0) > _SAMPLE_PRIORITY.get(existing[2], 0):
                by_ts[ts] = sample
        deduped[pid] = sorted(by_ts.values(), key=lambda item: item[0])
    return deduped


def build_death_intervals(ctx: TimelineContext) -> dict[int, list[tuple[int, int]]]:
    """Return per-participant death windows ``(start_ms, end_ms)`` from kills."""
    intervals: dict[int, list[tuple[int, int]]] = {pid: [] for pid in range(1, 11)}
    for event in ctx.events_of("CHAMPION_KILL"):
        ts = int(event.get("timestamp", 0))
        victim = int(event.get("victimId", 0))
        if not victim:
            continue
        level = _participant_level_at_ms(ctx, victim, ts)
        intervals[victim].append((ts, ts + respawn_duration_ms(level)))
    return intervals


def _position_from_track(
    samples: list[tuple[int, Position, str]],
    timestamp_ms: int,
    *,
    max_gap_ms: int = MAX_INTERPOLATION_GAP_MS,
) -> Position | None:
    if not samples:
        return None
    before: tuple[int, Position] | None = None
    after: tuple[int, Position] | None = None
    for ts, pos, _source in samples:
        if ts <= timestamp_ms:
            before = (ts, pos)
        elif after is None:
            after = (ts, pos)
            break
    if before is None:
        return samples[0][1]
    if after is None:
        return before[1]
    gap = after[0] - before[0]
    if gap > max_gap_ms or gap <= 0:
        return before[1]
    ratio = (timestamp_ms - before[0]) / gap
    return Position(
        x=before[1].x + (after[1].x - before[1].x) * ratio,
        y=before[1].y + (after[1].y - before[1].y) * ratio,
    )


def positions_at_ms(ctx: TimelineContext, timestamp_ms: int) -> dict[int, Position]:
    """Return participant positions at ``timestamp_ms`` from real timeline samples.

    Positions come from minute snapshots plus exact kill, objective, plate, and
    ward events. Values interpolate only between adjacent samples within
    :data:`MAX_INTERPOLATION_GAP_MS`.
    """
    if not ctx.frames:
        return {}
    last_ts = int(ctx.frames[-1].get("timestamp", 0))
    ts = max(0, min(timestamp_ms, last_ts))
    tracks = build_position_tracks(ctx)
    result: dict[int, Position] = {}
    for pid in range(1, 11):
        pos = _position_from_track(tracks.get(pid, []), ts)
        if pos is not None:
            result[pid] = pos
    return result


def is_participant_dead_at_ms(
    intervals: dict[int, list[tuple[int, int]]],
    pid: int,
    timestamp_ms: int,
) -> bool:
    return any(start <= timestamp_ms < end for start, end in intervals.get(pid, []))


def participant_states_at_ms(
    ctx: TimelineContext,
    timestamp_ms: int,
    *,
    tracks: dict[int, list[tuple[int, Position, str]]] | None = None,
    death_intervals: dict[int, list[tuple[int, int]]] | None = None,
    max_interpolation_gap_ms: int = MAX_INTERPOLATION_GAP_MS,
) -> dict[int, tuple[Position, bool]]:
    """Return ``{pid: (position, is_dead)}`` using real timeline samples only."""
    if not ctx.frames:
        return {}
    last_ts = int(ctx.frames[-1].get("timestamp", 0))
    ts = max(0, min(timestamp_ms, last_ts))
    track_data = tracks if tracks is not None else build_position_tracks(ctx)
    deaths = death_intervals if death_intervals is not None else build_death_intervals(ctx)
    states: dict[int, tuple[Position, bool]] = {}
    for pid in range(1, 11):
        pos = _position_from_track(
            track_data.get(pid, []),
            ts,
            max_gap_ms=max_interpolation_gap_ms,
        )
        if pos is None:
            continue
        states[pid] = (pos, is_participant_dead_at_ms(deaths, pid, ts))
    return states


def current_gold_at_ms(ctx: TimelineContext, timestamp_ms: int) -> int | None:
    """Return the player's banked gold at a timestamp from the nearest frame."""
    if not ctx.frames:
        return None
    index = min(int(round(timestamp_ms / 60_000)), len(ctx.frames) - 1)
    pframe = ctx.participant_frame(ctx.frames[index], ctx.participant_id)
    if not pframe:
        return None
    return max(0, int(pframe.get("currentGold", 0)))


def headcount_near(
    ctx: TimelineContext,
    pos: Position,
    timestamp_ms: int,
    *,
    radius: float = NEARBY_RADIUS,
) -> tuple[int, int]:
    """Count allies (excluding the player) and enemies near a position."""
    from league_stats.utils import distance

    allies = 0
    enemies = 0
    for pid in ctx.team_ids | ctx.enemy_ids:
        if pid == ctx.participant_id:
            continue
        other = ctx.position_at_ms(pid, timestamp_ms)
        if other is None or distance(pos, other) > radius:
            continue
        if pid in ctx.team_ids:
            allies += 1
        else:
            enemies += 1
    return allies, enemies


def avg_teammate_distance_at_ms(ctx: TimelineContext, timestamp_ms: int) -> float | None:
    """Mean distance from the player to every living teammate at a timestamp."""
    from league_stats.utils import distance

    my_pos = ctx.position_at_ms(ctx.participant_id, timestamp_ms)
    if my_pos is None:
        return None
    distances: list[float] = []
    for pid in ctx.team_ids:
        if pid == ctx.participant_id:
            continue
        other = ctx.position_at_ms(pid, timestamp_ms)
        if other is not None:
            distances.append(distance(my_pos, other))
    if not distances:
        return None
    return sum(distances) / len(distances)


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
        id_to_role={int(p["participantId"]): str(p.get("teamPosition", "")) for p in participants},
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
        first_recall_min=recalls[0].minute if recalls else None,
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
