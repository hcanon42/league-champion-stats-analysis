"""Detect high-impact team moments for the Game Review map scrubber."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from league_stats.analysis.teamfights import _cluster_kills
from league_stats.analysis.timeline import (
    TimelineContext,
    build_death_intervals,
    is_participant_dead_at_ms,
    participant_team_id,
    team_gold_series,
)
from league_stats.core.models import (
    KeyMoment,
    KeyMomentFrame,
    MapObjectivePin,
    MapParticipantPin,
    Position,
)
from league_stats.utils import BARON_PIT, DRAGON_PIT, distance, ms_to_min

TEAM_GAIN_THRESHOLD: int = 2_000
LEAD_SWING_THRESHOLD: int = 2_500
MERGE_WINDOW_MS: int = 30_000
DEDUP_WINDOW_MS: int = 60_000
SCRUB_BEFORE_MS: int = 45_000
SCRUB_AFTER_MS: int = 15_000
# Back-compat alias for tests that referenced the old symmetric window.
SCRUB_HALF_WINDOW_MS: int = SCRUB_AFTER_MS
MAX_MOMENTS: int = 4
SPLIT_PUSH_KILL_WINDOW_MS: int = 45_000
SPLIT_PUSH_RADIUS: float = 4_000.0

# Summoner's Rift spawn windows (approximate; kill events refine respawn state).
DRAGON_FIRST_SPAWN_MS: int = 5 * 60_000
DRAGON_RESPAWN_MS: int = 5 * 60_000
GRUBS_FIRST_SPAWN_MS: int = 5 * 60_000
GRUBS_DESPAWN_MS: int = 13 * 60_000 + 45_000
GRUBS_MAX_KILLS: int = 6
HERALD_SPAWN_MS: int = 14 * 60_000
HERALD_DESPAWN_MS: int = 19 * 60_000 + 45_000
BARON_SPAWN_MS: int = 20 * 60_000
BARON_RESPAWN_MS: int = 6 * 60_000

BASE_WEIGHTS: dict[str, float] = {
    "elder": 10.0,
    "baron": 9.0,
    "dragon_soul": 8.5,
    "nexus": 10.0,
    "inhibitor": 8.0,
    "gold_swing": 6.0,
    "teamfight": 5.0,
    "split_push": 3.5,
    "first_tower": 4.0,
    "dragon": 3.0,
    "herald": 2.5,
    "grubs": 2.0,
}

OBJECTIVE_PIT_KINDS: dict[str, Position] = {
    "dragon": DRAGON_PIT,
    "elder": DRAGON_PIT,
    "baron": BARON_PIT,
    "herald": BARON_PIT,
    "grubs": BARON_PIT,
}


@dataclass
class _ObjectiveSchedule:
    dragon_kills: list[tuple[int, str]]
    horde_kills: list[int]
    herald_kills: list[int]
    baron_kills: list[int]
    elder_taken: bool

    @classmethod
    def from_context(cls, ctx: TimelineContext) -> _ObjectiveSchedule:
        dragon_kills: list[tuple[int, str]] = []
        horde_kills: list[int] = []
        herald_kills: list[int] = []
        baron_kills: list[int] = []
        elder_taken = False
        for event in ctx.events_of("ELITE_MONSTER_KILL"):
            ts = int(event.get("timestamp", 0))
            monster = str(event.get("monsterType", ""))
            subtype = str(event.get("monsterSubType", ""))
            if monster == "DRAGON":
                dragon_kills.append((ts, subtype))
                if subtype == "ELDER_DRAGON":
                    elder_taken = True
            elif monster == "HORDE":
                horde_kills.append(ts)
            elif monster == "RIFTHERALD":
                herald_kills.append(ts)
            elif monster == "BARON_NASHOR":
                baron_kills.append(ts)
        return cls(
            dragon_kills=sorted(dragon_kills),
            horde_kills=sorted(horde_kills),
            herald_kills=sorted(herald_kills),
            baron_kills=sorted(baron_kills),
            elder_taken=elder_taken,
        )

    def _last_before(self, timestamps: list[int], ts: int) -> int | None:
        last: int | None = None
        for value in timestamps:
            if value <= ts:
                last = value
            else:
                break
        return last

    def _last_dragon_before(self, ts: int) -> tuple[int, str] | None:
        last: tuple[int, str] | None = None
        for kill_ts, subtype in self.dragon_kills:
            if kill_ts <= ts:
                last = (kill_ts, subtype)
            else:
                break
        return last

    def dragon_available(self, ts: int) -> tuple[bool, str]:
        if self.elder_taken:
            last = self._last_dragon_before(ts)
            if last and last[1] == "ELDER_DRAGON" and ts >= last[0]:
                return False, "dragon"
        if ts < DRAGON_FIRST_SPAWN_MS:
            return False, "dragon"
        last = self._last_dragon_before(ts)
        if last is None:
            return True, "dragon"
        kill_ts, subtype = last
        if subtype == "ELDER_DRAGON":
            return False, "elder"
        if ts >= kill_ts + DRAGON_RESPAWN_MS:
            return True, "dragon"
        return False, "dragon"

    def grubs_available(self, ts: int) -> bool:
        if ts < GRUBS_FIRST_SPAWN_MS or ts >= GRUBS_DESPAWN_MS:
            return False
        kills_before = sum(1 for kill_ts in self.horde_kills if kill_ts <= ts)
        return kills_before < GRUBS_MAX_KILLS

    def herald_available(self, ts: int) -> bool:
        if ts < HERALD_SPAWN_MS or ts >= HERALD_DESPAWN_MS:
            return False
        if any(kill_ts <= ts for kill_ts in self.herald_kills):
            return False
        return True

    def baron_available(self, ts: int) -> bool:
        if ts < BARON_SPAWN_MS:
            return False
        last = self._last_before(self.baron_kills, ts)
        if last is None:
            return True
        return ts >= last + BARON_RESPAWN_MS

    def baron_pit_kind(self, ts: int) -> str | None:
        if self.grubs_available(ts):
            return "grubs"
        if self.herald_available(ts):
            return "herald"
        if ts >= BARON_SPAWN_MS:
            return "baron"
        return None


@dataclass
class _Candidate:
    anchor_ms: int
    kind: str
    headline: str
    beneficiary_team: int
    impact_score: float
    gold_swing: int | None = None
    highlight_objective: str | None = None
    labels: list[str] = field(default_factory=list)


def detect_key_moments(ctx: TimelineContext, match: dict[str, Any]) -> list[KeyMoment]:
    """Pick up to four team-impact moments and build scrub frames for each."""
    del match  # reserved for future participant metadata; ctx is sufficient today
    spikes = _detect_gold_spikes(ctx)
    candidates = (
        _event_candidates(ctx)
        + _teamfight_candidates(ctx)
        + _gold_swing_candidates(ctx, spikes)
    )
    candidates = _merge_spikes_into_candidates(candidates, spikes)
    candidates = _deduplicate_candidates(candidates)
    candidates.sort(key=lambda c: c.impact_score, reverse=True)
    max_moments = MAX_MOMENTS if ctx.duration_s >= 600 else min(2, MAX_MOMENTS)
    selected = candidates[:max_moments]
    selected.sort(key=lambda c: c.anchor_ms)
    return [_candidate_to_moment(ctx, candidate) for candidate in selected]


def _format_gold(amount: int) -> str:
    if abs(amount) >= 1000:
        return f"{amount / 1000:.1f}k"
    return str(amount)


def _beneficiary_label(ctx: TimelineContext, team_id: int) -> Literal["ally", "enemy"]:
    my_team = 100 if ctx.blue_side else 200
    return "ally" if team_id == my_team else "enemy"


def _team_gold_at_ms(ctx: TimelineContext, team_id: int, timestamp_ms: int) -> int:
    total = 0
    for pid in range(1, 11):
        if participant_team_id(ctx, pid) != team_id:
            continue
        idx = min(int(round(timestamp_ms / 60_000)), len(ctx.frames) - 1)
        pframe = ctx.participant_frame(ctx.frames[idx], pid)
        if pframe:
            total += int(pframe.get("totalGold", 0))
    return total


def _gold_swing_in_window(
    ctx: TimelineContext,
    team_id: int,
    anchor_ms: int,
    *,
    half_window_ms: int = MERGE_WINDOW_MS // 2,
) -> int:
    start = max(0, anchor_ms - half_window_ms)
    end = anchor_ms + half_window_ms
    return _team_gold_at_ms(ctx, team_id, end) - _team_gold_at_ms(ctx, team_id, start)


def _detect_gold_spikes(ctx: TimelineContext) -> list[dict[str, Any]]:
    timestamps, blue_totals, red_totals = team_gold_series(ctx)
    spikes: list[dict[str, Any]] = []
    for index in range(1, len(timestamps)):
        if timestamps[index] < 60_000:
            continue
        blue_gain = blue_totals[index] - blue_totals[index - 1]
        red_gain = red_totals[index] - red_totals[index - 1]
        lead_swing = (blue_totals[index] - red_totals[index]) - (
            blue_totals[index - 1] - red_totals[index - 1]
        )
        if blue_gain >= TEAM_GAIN_THRESHOLD:
            spikes.append(
                {
                    "anchor_ms": timestamps[index],
                    "beneficiary_team": 100,
                    "gold_swing": blue_gain,
                }
            )
        if red_gain >= TEAM_GAIN_THRESHOLD:
            spikes.append(
                {
                    "anchor_ms": timestamps[index],
                    "beneficiary_team": 200,
                    "gold_swing": red_gain,
                }
            )
        if lead_swing >= LEAD_SWING_THRESHOLD:
            spikes.append(
                {
                    "anchor_ms": timestamps[index],
                    "beneficiary_team": 100,
                    "gold_swing": lead_swing,
                }
            )
        elif lead_swing <= -LEAD_SWING_THRESHOLD:
            spikes.append(
                {
                    "anchor_ms": timestamps[index],
                    "beneficiary_team": 200,
                    "gold_swing": abs(lead_swing),
                }
            )
    return spikes


def _nearest_spike(spikes: list[dict[str, Any]], anchor_ms: int) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_delta = MERGE_WINDOW_MS + 1
    for spike in spikes:
        delta = abs(int(spike["anchor_ms"]) - anchor_ms)
        if delta <= MERGE_WINDOW_MS and delta < best_delta:
            best = spike
            best_delta = delta
    return best


def _score(kind: str, gold_swing: int | None) -> float:
    base = BASE_WEIGHTS.get(kind, 3.0)
    bonus = min((gold_swing or 0) / 1000.0, 5.0)
    return base + bonus


def _event_candidates(ctx: TimelineContext) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    dragon_counts: dict[int, int] = {100: 0, 200: 0}
    first_tower_seen = False

    for event in ctx.events:
        event_type = str(event.get("type", ""))
        ts = int(event.get("timestamp", 0))

        if event_type == "ELITE_MONSTER_KILL":
            monster = str(event.get("monsterType", ""))
            subtype = str(event.get("monsterSubType", ""))
            killer_team = int(event.get("killerTeamId", 0))
            if monster == "BARON_NASHOR":
                swing = _gold_swing_in_window(ctx, killer_team, ts)
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="baron",
                        headline="Baron secured",
                        beneficiary_team=killer_team,
                        impact_score=_score("baron", swing),
                        gold_swing=swing or None,
                        highlight_objective="baron",
                    )
                )
            elif monster == "DRAGON" and subtype == "ELDER_DRAGON":
                swing = _gold_swing_in_window(ctx, killer_team, ts)
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="elder",
                        headline="Elder Dragon secured",
                        beneficiary_team=killer_team,
                        impact_score=_score("elder", swing),
                        gold_swing=swing or None,
                        highlight_objective="elder",
                    )
                )
            elif monster == "DRAGON":
                dragon_counts[killer_team] = dragon_counts.get(killer_team, 0) + 1
                count = dragon_counts[killer_team]
                if count == 4:
                    swing = _gold_swing_in_window(ctx, killer_team, ts)
                    candidates.append(
                        _Candidate(
                            anchor_ms=ts,
                            kind="dragon_soul",
                            headline="Dragon soul secured",
                            beneficiary_team=killer_team,
                            impact_score=_score("dragon_soul", swing),
                            gold_swing=swing or None,
                            highlight_objective="dragon",
                        )
                    )
                elif count <= 3:
                    swing = _gold_swing_in_window(ctx, killer_team, ts)
                    candidates.append(
                        _Candidate(
                            anchor_ms=ts,
                            kind="dragon",
                            headline=f"Dragon #{count} secured",
                            beneficiary_team=killer_team,
                            impact_score=_score("dragon", swing),
                            gold_swing=swing or None,
                            highlight_objective="dragon",
                        )
                    )
            elif monster == "RIFTHERALD":
                swing = _gold_swing_in_window(ctx, killer_team, ts)
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="herald",
                        headline="Rift Herald secured",
                        beneficiary_team=killer_team,
                        impact_score=_score("herald", swing),
                        gold_swing=swing or None,
                        highlight_objective="herald",
                    )
                )
            elif monster == "HORDE":
                swing = _gold_swing_in_window(ctx, killer_team, ts)
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="grubs",
                        headline="Void grubs secured",
                        beneficiary_team=killer_team,
                        impact_score=_score("grubs", swing),
                        gold_swing=swing or None,
                        highlight_objective="grubs",
                    )
                )

        elif event_type == "BUILDING_KILL":
            building = str(event.get("buildingType", ""))
            destroyed_team = int(event.get("teamId", 0))
            beneficiary_team = 200 if destroyed_team == 100 else 100
            pos = Position(**event.get("position", {"x": 0, "y": 0}))
            swing = _gold_swing_in_window(ctx, beneficiary_team, ts)

            if building == "TOWER_BUILDING" and not first_tower_seen:
                first_tower_seen = True
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="first_tower",
                        headline="First tower destroyed",
                        beneficiary_team=beneficiary_team,
                        impact_score=_score("first_tower", swing),
                        gold_swing=swing or None,
                    )
                )
                if _is_split_push(ctx, ts, pos):
                    candidates.append(
                        _Candidate(
                            anchor_ms=ts,
                            kind="split_push",
                            headline="Split push tower",
                            beneficiary_team=beneficiary_team,
                            impact_score=_score("split_push", swing),
                            gold_swing=swing or None,
                        )
                    )
            elif "INHIBITOR" in building:
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="inhibitor",
                        headline="Inhibitor destroyed",
                        beneficiary_team=beneficiary_team,
                        impact_score=_score("inhibitor", swing),
                        gold_swing=swing or None,
                    )
                )
            elif "NEXUS" in building:
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="nexus",
                        headline="Nexus destroyed",
                        beneficiary_team=beneficiary_team,
                        impact_score=_score("nexus", swing),
                        gold_swing=swing or None,
                    )
                )
            elif building == "TOWER_BUILDING" and _is_split_push(ctx, ts, pos):
                candidates.append(
                    _Candidate(
                        anchor_ms=ts,
                        kind="split_push",
                        headline="Split push tower",
                        beneficiary_team=beneficiary_team,
                        impact_score=_score("split_push", swing),
                        gold_swing=swing or None,
                    )
                )

    return candidates


def _is_split_push(ctx: TimelineContext, timestamp_ms: int, pos: Position) -> bool:
    kills_near = 0
    for kill in ctx.events_of("CHAMPION_KILL"):
        kill_ts = int(kill.get("timestamp", 0))
        if abs(kill_ts - timestamp_ms) > SPLIT_PUSH_KILL_WINDOW_MS:
            continue
        kill_pos = Position(**kill.get("position", {"x": 0, "y": 0}))
        if distance(kill_pos, pos) <= SPLIT_PUSH_RADIUS:
            kills_near += 1
    return kills_near <= 1


def _teamfight_candidates(ctx: TimelineContext) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    kills = ctx.events_of("CHAMPION_KILL")
    for cluster in _cluster_kills(kills):
        start_ts = int(cluster[0]["timestamp"])
        end_ts = int(cluster[-1]["timestamp"])
        anchor_ms = (start_ts + end_ts) // 2
        ally_kills = sum(
            1 for kill in cluster if int(kill.get("killerId", 0)) in ctx.team_ids
        )
        enemy_kills = sum(
            1 for kill in cluster if int(kill.get("killerId", 0)) in ctx.enemy_ids
        )
        if ally_kills == enemy_kills:
            beneficiary_team = 100 if ctx.blue_side else 200
        else:
            beneficiary_team = (
                (100 if ctx.blue_side else 200)
                if ally_kills > enemy_kills
                else (200 if ctx.blue_side else 100)
            )
        swing = _gold_swing_in_window(ctx, beneficiary_team, anchor_ms)
        won = ally_kills > enemy_kills
        headline = "Teamfight"
        if swing:
            headline += f" — +{_format_gold(swing)} gold swing"
        if won:
            headline += " (won)"
        elif enemy_kills > ally_kills:
            headline += " (lost)"
        candidates.append(
            _Candidate(
                anchor_ms=anchor_ms,
                kind="teamfight",
                headline=headline,
                beneficiary_team=beneficiary_team,
                impact_score=_score("teamfight", swing),
                gold_swing=swing or None,
            )
        )
    return candidates


def _gold_swing_candidates(
    ctx: TimelineContext,
    spikes: list[dict[str, Any]],
) -> list[_Candidate]:
    return []  # standalone spikes added in merge step


def _merge_spikes_into_candidates(
    candidates: list[_Candidate],
    spikes: list[dict[str, Any]],
) -> list[_Candidate]:
    merged = list(candidates)
    used_spikes: set[int] = set()

    for candidate in merged:
        spike = _nearest_spike(spikes, candidate.anchor_ms)
        if spike is None:
            continue
        spike_idx = int(spike["anchor_ms"])
        used_spikes.add(spike_idx)
        swing = int(spike["gold_swing"])
        if candidate.gold_swing is None or swing > candidate.gold_swing:
            candidate.gold_swing = swing
        candidate.impact_score = _score(candidate.kind, candidate.gold_swing)
        candidate.anchor_ms = spike_idx
        if swing and f"+{_format_gold(swing)}" not in candidate.headline:
            candidate.headline = f"{candidate.headline} — +{_format_gold(swing)} gold swing"

    for spike in spikes:
        spike_idx = int(spike["anchor_ms"])
        if spike_idx in used_spikes:
            continue
        if any(abs(c.anchor_ms - spike_idx) <= MERGE_WINDOW_MS for c in merged):
            continue
        team_id = int(spike["beneficiary_team"])
        swing = int(spike["gold_swing"])
        merged.append(
            _Candidate(
                anchor_ms=spike_idx,
                kind="gold_swing",
                headline=f"+{_format_gold(swing)} gold swing",
                beneficiary_team=team_id,
                impact_score=_score("gold_swing", swing),
                gold_swing=swing,
            )
        )
    return merged


def _deduplicate_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.impact_score, reverse=True)
    kept: list[_Candidate] = []
    for candidate in ordered:
        duplicate = next(
            (
                existing
                for existing in kept
                if abs(existing.anchor_ms - candidate.anchor_ms) <= DEDUP_WINDOW_MS
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        if candidate.impact_score > duplicate.impact_score:
            duplicate.anchor_ms = candidate.anchor_ms
            duplicate.kind = candidate.kind
            duplicate.headline = candidate.headline
            duplicate.beneficiary_team = candidate.beneficiary_team
            duplicate.impact_score = candidate.impact_score
            duplicate.gold_swing = candidate.gold_swing
            duplicate.highlight_objective = candidate.highlight_objective or duplicate.highlight_objective
        elif candidate.headline not in duplicate.headline:
            duplicate.headline = f"{duplicate.headline} / {candidate.headline}"
            if candidate.gold_swing and (
                duplicate.gold_swing is None or candidate.gold_swing > duplicate.gold_swing
            ):
                duplicate.gold_swing = candidate.gold_swing
                duplicate.impact_score = _score(duplicate.kind, duplicate.gold_swing)
    return kept


def _objective_pins_at_ms(
    schedule: _ObjectiveSchedule,
    timestamp_ms: int,
    highlight: str | None,
) -> list[MapObjectivePin]:
    pins: list[MapObjectivePin] = []
    dragon_up, dragon_kind = schedule.dragon_available(timestamp_ms)
    if dragon_up or highlight in {"dragon", "elder"}:
        kind = "elder" if highlight == "elder" else dragon_kind
        pos = OBJECTIVE_PIT_KINDS[kind]
        pins.append(
            MapObjectivePin(
                kind=kind,
                x=pos.x,
                y=pos.y,
                highlighted=highlight == kind or (highlight == "elder" and kind == "dragon"),
                available=dragon_up,
            )
        )

    pit_kind = schedule.baron_pit_kind(timestamp_ms)
    if pit_kind is None and highlight in {"baron", "herald", "grubs"}:
        pit_kind = highlight
    if pit_kind:
        pos = OBJECTIVE_PIT_KINDS[pit_kind]
        available = {
            "grubs": schedule.grubs_available(timestamp_ms),
            "herald": schedule.herald_available(timestamp_ms),
            "baron": schedule.baron_available(timestamp_ms),
        }[pit_kind]
        show = available or highlight == pit_kind
        if show:
            pins.append(
                MapObjectivePin(
                    kind=pit_kind,
                    x=pos.x,
                    y=pos.y,
                    highlighted=highlight == pit_kind,
                    available=available,
                )
            )
    return pins


def _before_after_minute_frames(
    ctx: TimelineContext,
    anchor_ms: int,
) -> list[dict[str, Any]]:
    """Return exactly two minute frames: one before and one after the action."""
    if not ctx.frames:
        return []

    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    for frame in ctx.frames:
        ts = int(frame.get("timestamp", 0))
        if ts < anchor_ms:
            before = frame
        elif ts > anchor_ms:
            after = frame
            break

    if before is None and after is None:
        # Anchor landed exactly on a minute mark — use neighbours around it.
        for index, frame in enumerate(ctx.frames):
            if int(frame.get("timestamp", 0)) != anchor_ms:
                continue
            before = ctx.frames[index - 1] if index > 0 else frame
            after = ctx.frames[index + 1] if index + 1 < len(ctx.frames) else frame
            break
        if before is None:
            before = ctx.frames[0]
            after = ctx.frames[1] if len(ctx.frames) > 1 else ctx.frames[0]
    elif before is None:
        before = ctx.frames[0]
        if after is before and len(ctx.frames) > 1:
            after = ctx.frames[1]
    elif after is None:
        after = ctx.frames[-1]
        if before is after and len(ctx.frames) > 1:
            before = ctx.frames[-2]

    if before is after and len(ctx.frames) > 1:
        ts = int(before.get("timestamp", 0))
        for index, frame in enumerate(ctx.frames):
            if int(frame.get("timestamp", 0)) != ts:
                continue
            before = ctx.frames[max(0, index - 1)]
            after = ctx.frames[min(len(ctx.frames) - 1, index + 1)]
            if before is after:
                if index + 1 < len(ctx.frames):
                    before, after = ctx.frames[index], ctx.frames[index + 1]
                elif index > 0:
                    before, after = ctx.frames[index - 1], ctx.frames[index]
            break

    return [before, after]


def _scrub_minute_frames(
    ctx: TimelineContext,
    anchor_ms: int,
) -> list[dict[str, Any]]:
    """Minute frames from before the action through after, including any in between."""
    endpoints = _before_after_minute_frames(ctx, anchor_ms)
    if not endpoints:
        return []
    before, after = endpoints[0], endpoints[-1]
    before_ts = int(before.get("timestamp", 0))
    after_ts = int(after.get("timestamp", 0))
    if after_ts < before_ts:
        before_ts, after_ts = after_ts, before_ts
    return [
        frame
        for frame in ctx.frames
        if before_ts <= int(frame.get("timestamp", 0)) <= after_ts
    ]


def _build_scrub_frames(
    ctx: TimelineContext,
    window_start_ms: int,
    window_end_ms: int,
    highlight_objective: str | None,
    schedule: _ObjectiveSchedule,
    *,
    anchor_ms: int,
) -> list[KeyMomentFrame]:
    del window_start_ms, window_end_ms
    death_intervals = build_death_intervals(ctx)
    selected = _scrub_minute_frames(ctx, anchor_ms)
    frames: list[KeyMomentFrame] = []
    last_index = len(selected) - 1
    for index, frame in enumerate(selected):
        ts = int(frame.get("timestamp", 0))
        if index == 0:
            phase = "Before"
        elif index == last_index:
            phase = "After"
        else:
            phase = "Between"
        participants: list[MapParticipantPin] = []
        for pid in range(1, 11):
            pframe = ctx.participant_frame(frame, pid)
            if not pframe or "position" not in pframe:
                continue
            pos = Position(**pframe["position"])
            participants.append(
                MapParticipantPin(
                    participant_id=pid,
                    champion=ctx.id_to_champion.get(pid, "Unknown"),
                    team_id=participant_team_id(ctx, pid),
                    x=pos.x,
                    y=pos.y,
                    dead=is_participant_dead_at_ms(death_intervals, pid, ts),
                )
            )
        frames.append(
            KeyMomentFrame(
                timestamp_ms=ts,
                label=f"{phase} · minute {ts // 60_000}",
                participants=participants,
                objectives=_objective_pins_at_ms(schedule, ts, highlight_objective),
            )
        )
    return frames


def _candidate_to_moment(ctx: TimelineContext, candidate: _Candidate) -> KeyMoment:
    anchor_ms = candidate.anchor_ms
    window_start = max(0, anchor_ms - SCRUB_BEFORE_MS)
    window_end = min(ctx.duration_s * 1000, anchor_ms + SCRUB_AFTER_MS)
    schedule = _ObjectiveSchedule.from_context(ctx)
    return KeyMoment(
        id=f"{candidate.kind}-{anchor_ms}",
        kind=candidate.kind,
        headline=candidate.headline,
        beneficiary=_beneficiary_label(ctx, candidate.beneficiary_team),
        gold_swing=candidate.gold_swing,
        anchor_ms=anchor_ms,
        anchor_minute=round(ms_to_min(anchor_ms), 1),
        window_start_ms=window_start,
        window_end_ms=window_end,
        frames=_build_scrub_frames(
            ctx,
            window_start,
            window_end,
            candidate.highlight_objective,
            schedule,
            anchor_ms=anchor_ms,
        ),
    )
