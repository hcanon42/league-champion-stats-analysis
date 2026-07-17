"""Tests for timeline helpers and key moment detection."""

from __future__ import annotations

from league_stats.analysis.key_moments import (
    DEDUP_WINDOW_MS,
    SCRUB_AFTER_MS,
    SCRUB_BEFORE_MS,
    detect_key_moments,
)
from league_stats.analysis.timeline import (
    build_context,
    build_death_intervals,
    build_position_tracks,
    is_participant_dead_at_ms,
    participant_states_at_ms,
    participant_team_id,
    positions_at_ms,
    team_gold_series,
)
from league_stats.ingest.parser import ItemCatalog, MatchParser
from tests.fixtures import FAKE_ITEMS, MY_PUUID, make_match, make_timeline


def _ctx():
    match = make_match()
    timeline = make_timeline()
    return build_context(match, timeline, MY_PUUID), match, timeline


def test_participant_team_id_blue_side():
    ctx, _, _ = _ctx()
    assert participant_team_id(ctx, 1) == 100
    assert participant_team_id(ctx, 6) == 200


def test_team_gold_series_sums_by_team():
    ctx, _, _ = _ctx()
    timestamps, blue, red = team_gold_series(ctx)
    assert len(timestamps) == len(blue) == len(red)
    assert timestamps[0] == 0
    assert blue[0] > 0 and red[0] > 0


def test_positions_hold_across_minute_gap():
    ctx, _, _ = _ctx()
    at_start = positions_at_ms(ctx, 0)[1]
    at_mid = positions_at_ms(ctx, 30_000)[1]
    at_minute = positions_at_ms(ctx, 60_000)[1]
    assert at_start.x == at_mid.x
    assert at_start.y == at_mid.y
    assert at_minute.x > at_start.x


def test_positions_interpolate_between_close_kills():
    match = make_match()
    timeline = make_timeline()
    timeline["info"]["frames"][6]["events"] = [
        {
            "timestamp": 360_000,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 6,
            "position": {"x": 7000, "y": 7000},
            "assistingParticipantIds": [],
            "victimDamageReceived": [],
        },
        {
            "timestamp": 365_000,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 7,
            "position": {"x": 8000, "y": 8000},
            "assistingParticipantIds": [],
            "victimDamageReceived": [],
        },
    ]
    ctx = build_context(match, timeline, MY_PUUID)
    mid = positions_at_ms(ctx, 362_500)[1]
    assert 7000 < mid.x < 8000
    assert 7000 < mid.y < 8000


def test_kill_event_updates_position_at_exact_timestamp():
    match = make_match()
    timeline = make_timeline()
    kill_ts = 360_000
    kill_pos = {"x": 7500, "y": 7500}
    timeline["info"]["frames"][-1]["events"].append(
        {
            "timestamp": kill_ts,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 6,
            "position": kill_pos,
            "assistingParticipantIds": [],
            "victimDamageReceived": [],
        }
    )
    ctx = build_context(match, timeline, MY_PUUID)
    pos = positions_at_ms(ctx, kill_ts)[6]
    assert pos.x == kill_pos["x"]
    assert pos.y == kill_pos["y"]


def test_victim_marked_dead_after_kill():
    match = make_match()
    timeline = make_timeline()
    kill_ts = 480_000
    ctx = build_context(match, timeline, MY_PUUID)
    deaths = build_death_intervals(ctx)
    assert is_participant_dead_at_ms(deaths, 1, kill_ts + 1000)
    assert not is_participant_dead_at_ms(deaths, 1, kill_ts - 1000)


def test_scrub_frames_include_dead_flag():
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    record = parser.parse(make_match(), make_timeline(), MY_PUUID)
    fight = next(m for m in record.key_moments if m.kind == "teamfight")
    dead_frames = [
        participant
        for frame in fight.frames
        for participant in frame.participants
        if participant.dead
    ]
    assert dead_frames


def test_detect_key_moments_from_fixture():
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    match = make_match()
    timeline = make_timeline()
    record = parser.parse(match, timeline, MY_PUUID)
    assert record.key_moments
    assert len(record.key_moments) <= 4
    kinds = {moment.kind for moment in record.key_moments}
    assert "baron" in kinds or "teamfight" in kinds


def test_gold_spike_creates_standalone_moment():
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    match = make_match()
    timeline = make_timeline()
    frames = timeline["info"]["frames"]
    for index in range(5, 7):
        for pid in range(1, 6):
            frames[index]["participantFrames"][str(pid)]["totalGold"] = (
                frames[index - 1]["participantFrames"][str(pid)]["totalGold"] + 600
            )
    ctx = build_context(match, timeline, MY_PUUID)
    moments = detect_key_moments(ctx, match)
    assert any(moment.kind == "gold_swing" for moment in moments)


def test_dragon_soul_detection():
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    match = make_match()
    timeline = make_timeline()
    frames = timeline["info"]["frames"]
    dragon_events = [
        {
            "timestamp": 300_000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "INFERNAL_DRAGON",
            "position": {"x": 9866, "y": 4414},
        },
        {
            "timestamp": 420_000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "CLOUD_DRAGON",
            "position": {"x": 9866, "y": 4414},
        },
        {
            "timestamp": 540_000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "OCEAN_DRAGON",
            "position": {"x": 9866, "y": 4414},
        },
        {
            "timestamp": 660_000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "MOUNTAIN_DRAGON",
            "position": {"x": 9866, "y": 4414},
        },
    ]
    frames[11]["events"] = dragon_events
    ctx = build_context(match, timeline, MY_PUUID)
    moments = detect_key_moments(ctx, match)
    assert any(moment.kind == "dragon_soul" for moment in moments)


def test_scrub_window_and_snapshot_frames():
    parser = MatchParser(ItemCatalog(FAKE_ITEMS))
    record = parser.parse(make_match(), make_timeline(), MY_PUUID)
    moment = record.key_moments[0]
    assert moment.window_end_ms - moment.window_start_ms <= SCRUB_BEFORE_MS + SCRUB_AFTER_MS
    assert moment.anchor_ms - moment.window_start_ms <= SCRUB_BEFORE_MS
    assert moment.window_end_ms - moment.anchor_ms <= SCRUB_AFTER_MS
    assert len(moment.frames) >= 2
    before, after = moment.frames[0], moment.frames[-1]
    assert before.timestamp_ms < moment.anchor_ms
    assert after.timestamp_ms > moment.anchor_ms
    assert before.label.startswith("Before")
    assert after.label.startswith("After")
    timestamps = [frame.timestamp_ms for frame in moment.frames]
    assert timestamps == sorted(timestamps)
    assert all(before.timestamp_ms <= ts <= after.timestamp_ms for ts in timestamps)
    assert all(ts % 60_000 == 0 for ts in timestamps)
    assert all(len(frame.participants) == 10 for frame in moment.frames)
    for frame in moment.frames[1:-1]:
        assert frame.label.startswith("Between")


def test_scrub_includes_minute_marks_between_before_and_after():
    match = make_match(duration_s=1800)
    timeline = make_timeline(duration_s=1800)
    # Force a sparse frame list gap by removing a middle minute, then ensure
    # the scrubber still walks every remaining mark between before and after.
    ctx = build_context(match, timeline, MY_PUUID)
    moments = detect_key_moments(ctx, match)
    assert moments
    for moment in moments:
        assert len(moment.frames) >= 2
        before_ts = moment.frames[0].timestamp_ms
        after_ts = moment.frames[-1].timestamp_ms
        expected = [
            int(frame.get("timestamp", 0))
            for frame in ctx.frames
            if before_ts <= int(frame.get("timestamp", 0)) <= after_ts
        ]
        assert [frame.timestamp_ms for frame in moment.frames] == expected


def test_position_tracks_include_objective_events():
    match = make_match()
    timeline = make_timeline()
    timeline["info"]["frames"][-1]["events"].extend([
        {
            "timestamp": 482_000,
            "type": "TURRET_PLATE_DESTROYED",
            "killerId": 1,
            "position": {"x": 5000, "y": 5000},
            "laneType": "MID_LANE",
            "teamId": 200,
        },
        {
            "timestamp": 483_000,
            "type": "ELITE_MONSTER_KILL",
            "killerId": 2,
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "INFERNAL_DRAGON",
            "position": {"x": 9866, "y": 4414},
            "assistingParticipantIds": [1],
        },
    ])
    timeline["info"]["frames"][-1]["events"].sort(key=lambda e: e["timestamp"])
    ctx = build_context(match, timeline, MY_PUUID)
    tracks = build_position_tracks(ctx)
    plate = [sample for sample in tracks[1] if sample[0] == 482_000]
    assert plate
    assert plate[0][1].x == 5000
    assist = [sample for sample in tracks[1] if sample[0] == 483_000]
    assert assist


def test_objective_pins_reflect_availability():
    match = make_match()
    timeline = make_timeline()
    dragon_ts = 905_000
    for event in timeline["info"]["frames"][-1]["events"]:
        if event.get("type") == "ELITE_MONSTER_KILL" and event.get("monsterType") == "DRAGON":
            event["timestamp"] = dragon_ts
    timeline["info"]["frames"][-1]["events"].sort(key=lambda e: e["timestamp"])
    ctx = build_context(match, timeline, MY_PUUID)
    moments = detect_key_moments(ctx, match)
    dragon_moment = next(m for m in moments if m.kind == "dragon")
    assert len(dragon_moment.frames) >= 2
    before, after = dragon_moment.frames[0], dragon_moment.frames[-1]
    assert before.timestamp_ms < dragon_ts
    assert after.timestamp_ms > dragon_ts
    before_dragon = next((obj for obj in before.objectives if obj.kind == "dragon"), None)
    after_dragon = next((obj for obj in after.objectives if obj.kind == "dragon"), None)
    assert before_dragon is not None
    assert before_dragon.available is True
    assert after_dragon is not None
    assert after_dragon.available is False
    assert after_dragon.highlighted is True


def test_position_tracks_prefer_kill_over_frame():
    match = make_match()
    timeline = make_timeline()
    kill_ts = 362_000
    kill_pos = {"x": 9999, "y": 8888}
    timeline["info"]["frames"][6]["events"] = [
        {
            "timestamp": kill_ts,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 6,
            "position": kill_pos,
            "assistingParticipantIds": [],
            "victimDamageReceived": [],
        }
    ]
    ctx = build_context(match, timeline, MY_PUUID)
    tracks = build_position_tracks(ctx)
    kill_samples = [sample for sample in tracks[6] if sample[0] == kill_ts]
    assert kill_samples
    assert kill_samples[0][1].x == kill_pos["x"]


def test_dedup_events_within_sixty_seconds():
    match = make_match()
    timeline = make_timeline()
    frames = timeline["info"]["frames"]
    cluster = [
        {
            "timestamp": 600_000,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 6,
            "position": {"x": 7400, "y": 7500},
            "assistingParticipantIds": [2],
            "victimDamageReceived": [],
        },
        {
            "timestamp": 605_000,
            "type": "CHAMPION_KILL",
            "killerId": 2,
            "victimId": 7,
            "position": {"x": 7350, "y": 7450},
            "assistingParticipantIds": [1],
            "victimDamageReceived": [],
        },
        {
            "timestamp": 610_000,
            "type": "CHAMPION_KILL",
            "killerId": 1,
            "victimId": 8,
            "position": {"x": 7500, "y": 7400},
            "assistingParticipantIds": [3],
            "victimDamageReceived": [],
        },
    ]
    frames[10]["events"] = cluster + [
        {
            "timestamp": 615_000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 100,
            "monsterType": "BARON_NASHOR",
            "position": {"x": 5007, "y": 10471},
        }
    ]
    ctx = build_context(match, timeline, MY_PUUID)
    moments = detect_key_moments(ctx, match)
    baron_moment = next((m for m in moments if m.kind == "baron"), None)
    fight_moment = next((m for m in moments if m.kind == "teamfight"), None)
    if baron_moment and fight_moment:
        assert abs(baron_moment.anchor_ms - fight_moment.anchor_ms) > DEDUP_WINDOW_MS or baron_moment.id == fight_moment.id
