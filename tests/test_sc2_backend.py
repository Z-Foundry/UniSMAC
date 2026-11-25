"""SC2 backend tests. Tests marked ``sc2`` require a real SC2 client (``pytest -m sc2``);
the rest use a fake session.

* Contract conformance: two-agent duel on ``new_maps/3m`` (``SMAC_Maps/*`` only allows 1 user
  slot, so two Participants return ``NoSpaceForUser``; built-in AI controls Player 2 and does
  not follow NOOP semantics);
* Duel smoke test: two controllers stepping in lockstep, both issuing commands, merged truth;
* map_trigger smoke test: ``SMAC_Maps/3m`` + ComputerPlayer;
* explicit + ComputerPlayer smoke test: ``SMAC_Maps/32x32_flat``;
* Two backends constructed sequentially in the same process;
* SMAC-Hard extensions: ``slot_order='tag'`` (slots sorted by ascending tag) and
  ``query_abilities``;
* Two-phase debug request ordering for explicit spawns (fake session, no SC2 required).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unismac.backends.sc2 import Sc2Backend
from unismac.contract import (
    AgentPlayer,
    Backend,
    Command,
    CommandKind,
    ComputerPlayer,
    ContractError,
    Scenario,
    Spawn,
    Variant,
    conformance,
)

DUEL = replace(
    conformance.SCENARIO,
    map_id='new_maps/3m',
    players=(AgentPlayer('terran'), AgentPlayer('terran')),
)
VS_AI = (AgentPlayer('terran'), ComputerPlayer('terran', 'very_easy'))
SMAC_3M = Scenario('SMAC_Maps/3m', players=VS_AI, spawn_mode='map_trigger', n_slots=6)


@pytest.fixture(scope='module')
def duel() -> Iterator[Sc2Backend]:
    backend = Sc2Backend()
    yield backend
    backend.close()


@pytest.mark.sc2
def test_backend_protocol() -> None:
    backend = Sc2Backend()
    assert isinstance(backend, Backend)
    assert backend.batch_size == 1
    backend.capabilities.require(engine_ai=True, map_trigger=True, pathing='engine', fog=True)
    assert backend.capabilities.max_batch == 1 and not backend.capabilities.deterministic


@pytest.mark.sc2
@pytest.mark.parametrize('check', conformance.CHECKS, ids=lambda c: c.__name__)
def test_conformance_duel(duel: Sc2Backend, check) -> None:
    check(duel, DUEL)


@pytest.mark.sc2
def test_duel_lockstep_commands_and_merged_truth(duel: Sc2Backend) -> None:
    state = duel.reset([DUEL])
    session = duel.session
    assert session is not None and session.n_agents == 2 and not session.has_computer
    # Both sides are 8 apart (< Marine sight 9): mutually visible, all merged fields finite.
    assert state.visible[0].all()
    assert np.isfinite(state.weapon_cooldown).all()

    # Both sides issue a command; both controllers advance together.
    cmd = Command.noop(1, 3)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(10.0, 8.0))
    cmd.set(0, 2, CommandKind.MOVE, target_xy=(29.0, 24.0))
    after = duel.step(cmd, 16)
    assert after.game_loop[0] - state.game_loop[0] == 16
    assert duel.last_action_results == [(0, CommandKind.MOVE, 1), (2, CommandKind.MOVE, 1)]
    assert after.order_kind[0, 0] == after.order_kind[0, 2] == CommandKind.MOVE
    assert (after.order_x[0, 2], after.order_y[0, 2]) == (29.0, 24.0)
    assert after.y[0, 0] < state.y[0, 0] - 1.5
    assert after.x[0, 2] > state.x[0, 2] + 1.0
    for _ in range(8):
        after = duel.step(Command.noop(1, 3), 16)
    # After slot 2 walks away, Player 1 loses sight while Player 2 still sees its own unit.
    assert not after.visible[0, 2, 0] and after.visible[0, 2, 1]
    assert np.isfinite(after.weapon_cooldown[0, 2])

    # Engine rejects ATTACK_UNIT on invisible target: no exception, logged in last_action_results.
    rejected = duel.step(Command.noop(1, 3).set(0, 1, CommandKind.ATTACK_UNIT, target_slot=2), 1)
    assert duel.last_action_results == [(1, CommandKind.ATTACK_UNIT, 2)]  # NotSupported
    assert rejected.order_kind[0, 1] == CommandKind.NOOP

    # Player 2 attacks a visible Player 1 unit: target maps back to slot. In two-Participant
    # mode commands take 2 loops to appear in observations (vs 1 in single-process), so step 4.
    attack = duel.step(Command.noop(1, 3).set(0, 1, CommandKind.MOVE, target_xy=(24.0, 20.0)), 96)
    assert attack.visible[0, 2, 0], 'Player 1 should see slot 2 after slot 1 moves closer'
    attack = duel.step(Command.noop(1, 3).set(0, 2, CommandKind.ATTACK_UNIT, target_slot=1), 4)
    assert duel.last_action_results == [(2, CommandKind.ATTACK_UNIT, 1)]
    assert attack.order_kind[0, 2] == CommandKind.ATTACK_UNIT
    assert attack.order_target_slot[0, 2] == 1


@pytest.mark.sc2
def test_duel_rejects_bad_targets(duel: Sc2Backend) -> None:
    duel.reset([DUEL])
    for cmd in (
        Command.noop(1, 3).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=7),
        Command.noop(1, 3).set(0, 0, CommandKind.MOVE),  # missing target_xy
        Command.noop(1, 3).set(0, 0, CommandKind.ABILITY),  # missing ability
    ):
        with pytest.raises(ContractError):
            duel.step(cmd, 1)


@pytest.mark.sc2
def test_map_trigger_smac_3m() -> None:
    backend = Sc2Backend()
    try:
        state = backend.reset([SMAC_3M])
        assert state.n_slots == 6 and state.alive.all()
        assert sorted(state.owner[0].tolist()) == [1, 1, 1, 2, 2, 2]
        assert (state.unit_type == 48).all()
        ours = state.owner[0] == 1
        # Player 1 uses map-private Marine_RL, normalized to (48, RL).
        assert (state.variant[0, ours] == Variant.RL).all()
        assert (state.variant[0, ~ours] == Variant.STD).all()
        # Computer orders/cooldowns are unavailable (sentinels); SMAC disables fog (all visible).
        assert np.isnan(state.weapon_cooldown[0, ~ours]).all()
        assert (state.order_target_slot[0, ~ours] == -1).all()
        assert state.visible[0].all()

        enemy = int(np.flatnonzero(~ours)[0])
        with pytest.raises(ContractError):
            backend.step(Command.noop(1, 6).set(0, enemy, CommandKind.STOP), 1)
        cmd = Command.noop(1, 6)
        for slot in np.flatnonzero(ours):
            cmd.set(0, int(slot), CommandKind.ATTACK_UNIT, target_slot=enemy)
        state = backend.step(cmd, 8)
        for _ in range(10):
            state = backend.step(Command.noop(1, 6), 8)
        assert state.game_loop[0] > 80

        again = backend.reset([SMAC_3M])
        assert again.alive.all()
        assert sorted(again.owner[0].tolist()) == [1, 1, 1, 2, 2, 2]
        assert np.array_equal(again.health, again.health_max)
    finally:
        backend.close()


@pytest.mark.sc2
def test_slot_order_tag_and_query_abilities_mmm_duel() -> None:
    players = (AgentPlayer('terran'), AgentPlayer('terran'))
    scenario = Scenario('new_maps/MMM', players=players, spawn_mode='map_trigger', n_slots=20)
    backend = Sc2Backend(slot_order='tag')
    assert backend.capabilities.ability_query
    try:
        for episode in range(2):  # 2nd episode uses kill + map trigger respawn
            state = backend.reset([scenario])
            assert state.alive.all()
            owner = state.owner[0]
            tags = np.array(backend._tags)  # noqa: SLF001 - read internal backend tags for test
            for p in (1, 2):
                mine = tags[owner == p]
                assert (np.diff(mine) > 0).all(), f'ep {episode} player {p} slots not sorted by tag'
            assert (owner[:10] == 1).all() and (owner[10:] == 2).all()

        abilities = backend.query_abilities()[0]
        for slot in range(state.n_slots):
            got = abilities[slot]
            assert got == sorted(got)
            assert not {16, 23, 4, 3674, 3794, 3665} & set(got), 'must exclude basic commands'
            unit_type = int(state.unit_type[0, slot])
            # new_maps/MMM only researches Stim for Player 1 (Player 2 Marine/Marauder have none).
            if int(owner[slot]) == 1:
                expected = {48: {380}, 51: {253}, 54: {386, 394, 2116}}[unit_type]
                assert expected <= set(got), (slot, unit_type, got)
            elif unit_type == 54:
                assert {386, 394, 2116} <= set(got), (slot, got)
            else:
                assert got == [], (slot, unit_type, got)

        # After killing a unit, its slot returns an empty list; query does not advance game loop.
        from s2clientprotocol import debug_pb2 as d_pb

        session = backend.session
        assert session is not None
        session.debug([d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=[backend._tags[0]]))])  # noqa: SLF001
        state = backend.step(Command.noop(1, 20), 8)  # debug commands are delayed in duel mode
        loop = int(state.game_loop[0])
        assert not state.alive[0, 0]
        assert backend.query_abilities([0])[0][0] == []
        assert backend.step(Command.noop(1, 20), 1).game_loop[0] == loop + 1
    finally:
        backend.close()


@pytest.mark.sc2
def test_order_kind_table_covers_pysc2_remaps() -> None:
    """All pysc2 abilities with general_id for Attack/Move/Stop/HoldPosition are in the table."""
    from pysc2.lib import actions

    from unismac.backends.sc2 import backend as sc2_backend

    general = {3674: -1, 3794: CommandKind.MOVE, 3665: CommandKind.STOP, 3793: CommandKind.HOLD}
    for fn in actions.RAW_FUNCTIONS:
        if fn.general_id in general:
            got = sc2_backend._ORDER_KIND.get(int(fn.ability_id))  # noqa: SLF001
            assert got == general[fn.general_id], (fn.name, fn.ability_id, got)


@pytest.mark.sc2
def test_battlecruiser_orders_map_to_basic_kinds() -> None:
    """SC2 4.10 BC uses dedicated abilities (attack 3771, move 3776), mapped to ATTACK_*/MOVE."""
    scenario = Scenario(
        'new_maps/3m',
        players=(AgentPlayer('terran'), AgentPlayer('terran')),
        spawns=(Spawn(1, 57, 10.0, 16.0), Spawn(2, 109, 15.0, 16.0)),  # Battlecruiser, Ultralisk
    )
    backend = Sc2Backend()
    try:
        backend.reset([scenario])
        hold = Command.noop(1, 2).set(0, 1, CommandKind.HOLD)
        backend.step(hold, 1)
        state = backend.step(
            Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1), 4
        )
        assert state.order_ability[0, 0] == 3771
        assert state.order_kind[0, 0] == CommandKind.ATTACK_UNIT
        assert state.order_target_slot[0, 0] == 1
        state = backend.step(
            Command.noop(1, 2).set(0, 0, CommandKind.MOVE, target_xy=(6.0, 16.0)), 4
        )
        assert state.order_ability[0, 0] == 3776
        assert state.order_kind[0, 0] == CommandKind.MOVE
        cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_MOVE, target_xy=(20.0, 16.0))
        state = backend.step(cmd, 4)
        assert state.order_ability[0, 0] == 3771
        assert state.order_kind[0, 0] in (CommandKind.ATTACK_MOVE, CommandKind.ATTACK_UNIT)
        state = backend.step(Command.noop(1, 2).set(0, 0, CommandKind.HOLD), 4)
        assert state.order_kind[0, 0] in (CommandKind.HOLD, CommandKind.NOOP), int(
            state.order_ability[0, 0]
        )
        state = backend.step(Command.noop(1, 2).set(0, 0, CommandKind.STOP), 4)
        assert state.order_kind[0, 0] != CommandKind.ABILITY, int(state.order_ability[0, 0])
    finally:
        backend.close()


@pytest.mark.sc2
def test_slot_order_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        Sc2Backend(slot_order='random')  # type: ignore[arg-type]


@pytest.mark.sc2
def test_explicit_vs_computer_flat() -> None:
    scenario = replace(conformance.SCENARIO, players=VS_AI)
    backend = Sc2Backend()
    try:
        state = backend.reset([scenario])
        assert state.alive.all() and state.owner[0].tolist() == [1, 1, 2]
        assert (
            np.isnan(state.weapon_cooldown[0, 2])
            and np.isfinite(state.weapon_cooldown[0, :2]).all()
        )
        cmd = Command.noop(1, 3).set(0, 0, CommandKind.MOVE, target_xy=(10.0, 8.0))
        after = backend.step(cmd, 16)
        assert after.order_kind[0, 0] == CommandKind.MOVE
        assert backend.last_action_results == [(0, CommandKind.MOVE, 1)]
        assert after.y[0, 0] == pytest.approx(state.y[0, 0] - 2.25, abs=0.2)
        with pytest.raises(ContractError):
            backend.step(Command.noop(1, 3).set(0, 2, CommandKind.STOP), 1)
    finally:
        backend.close()


@pytest.mark.sc2
def test_two_backends_in_one_process() -> None:
    """Construct and use two backends (SMAC_Maps/3m and 32x32_flat) concurrently in one process."""
    first = Sc2Backend()
    second = Sc2Backend()
    try:
        a = first.reset([SMAC_3M])
        b = second.reset([replace(conformance.SCENARIO, players=VS_AI)])
        assert a.n_slots == 6 and b.n_slots == 3
        a = first.step(Command.noop(1, 6), 8)
        b = second.step(Command.noop(1, 3), 8)
        assert a.alive.sum() >= 1 and b.alive.all()
    finally:
        first.close()
        second.close()


def _component(pathing: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """8-connected pathable component of ``pathing[iy, ix]`` starting from ``start=(iy, ix)``."""
    h, w = pathing.shape
    seen = np.zeros_like(pathing, dtype=np.bool_)
    stack = [start]
    seen[start] = True
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and pathing[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return seen


@pytest.mark.sc2
def test_map_pathing_matches_query_pathing() -> None:
    """Exported ``MapInfo.pathing`` matches SC2, indexed ``grid[iy, ix]`` with row 0 at y in [0, 1).

    1. Re-decoding ``game_info`` matches the stored asset cell-for-cell;
    2. On vertically asymmetric ``SMAC_Maps/corridor``, sampling 400 cells from a point in the
       main component via ``RequestQueryPathing`` agrees >= 0.95 with the asset component and
       far exceeds the y-flipped grid (measured 0.99 vs 0.43 in extract.py).
    """
    from s2clientprotocol import common_pb2, query_pb2

    from unismac.assets import load_map_info
    from unismac.backends.sc2 import _runtime
    from unismac.backends.sc2.extract import map_info_from_game_info
    from unismac.backends.sc2.session import Session

    map_id = 'SMAC_Maps/corridor'
    info = load_map_info(map_id)
    session = Session(map_id, VS_AI)
    try:
        fresh = map_info_from_game_info(map_id, session.game_info, _runtime.map_sha256(map_id))
        assert fresh.sha256 == info.sha256
        assert np.array_equal(fresh.pathing, info.pathing)
        assert np.array_equal(fresh.placement, info.placement)

        x0, y0, x1, y1 = info.playable
        cells = np.argwhere(info.pathing[y0:y1, x0:x1]) + (y0, x0)
        # Main component: largest connected pathable region; pick a start cell inside it
        best = max(
            (
                _component(info.pathing, (int(c[0]), int(c[1])))
                for c in cells[:: max(1, len(cells) // 8)]
            ),
            key=lambda m: int(m.sum()),
        )
        sy, sx = (int(v) for v in np.argwhere(best)[len(np.argwhere(best)) // 2])
        rng = np.random.default_rng(0)
        ys = rng.integers(y0, y1, 400)
        xs = rng.integers(x0, x1, 400)
        query = query_pb2.RequestQuery()
        for y, x in zip(ys, xs, strict=True):
            query.pathing.add(
                start_pos=common_pb2.Point2D(x=sx + 0.5, y=sy + 0.5),
                end_pos=common_pb2.Point2D(x=float(x) + 0.5, y=float(y) + 0.5),
            )
        result = session.controllers[0].query(query)
        reach = np.array([p.distance > 0 for p in result.pathing])
        reach |= (ys == sy) & (xs == sx)
        agree = float(np.mean(reach == best[ys, xs]))
        flipped = best[::-1]
        agree_flipped = float(np.mean(reach == flipped[ys, xs]))
        assert agree >= 0.95, (agree, agree_flipped)
        assert agree > agree_flipped + 0.2, (agree, agree_flipped)
    finally:
        session.close()


# ---------------------------------------------------------------- Explicit spawns (fake session)


def _ns(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


class _FakeSession:
    """Fake SC2 session recording debug requests. Models observed timings: DebugKillUnit takes
    2 loops and DebugCreateUnit takes 1 loop; ground units spawned within 0.5 of an existing
    ground unit are nudged -0.3135 in y, while Overlord (106) ignores ground collisions."""

    closed = False
    has_computer = False

    def __init__(self, units: list[tuple[int, int, float, float]], playable=(2, 2, 30, 30)):
        self.players = (AgentPlayer('terran'), AgentPlayer('terran'))
        self.agent_ids = [1, 2]
        self.loop = 0
        self.next_tag = 1
        self.units: dict[int, list] = {}  # tag -> [owner, type, x, y]
        for u in units:
            self._add(*u)
        self.pending: list[tuple[int, str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        p0, p1 = _ns(x=playable[0], y=playable[1]), _ns(x=playable[2], y=playable[3])
        self.game_info = _ns(
            start_raw=_ns(playable_area=_ns(p0=p0, p1=p1), map_size=_ns(x=32, y=32))
        )

    def _add(self, owner: int, utype: int, x: float, y: float) -> int:
        tag = self.next_tag
        self.next_tag += 1
        self.units[tag] = [owner, utype, x, y]
        return tag

    def debug(self, commands: list[Any]) -> None:
        call: dict[str, Any] = {
            'loop': self.loop,
            'alive': set(self.units),
            'kill': [],
            'create': [],
        }
        for c in commands:
            if c.HasField('kill_unit'):
                tags = list(c.kill_unit.tag)
                call['kill'].append(tags)
                self.pending.append((self.loop + 2, 'kill', tags))
            else:
                cu = c.create_unit
                spec = (int(cu.owner), int(cu.unit_type), float(cu.pos.x), float(cu.pos.y))
                call['create'].append(spec)
                self.pending.append((self.loop + 1, 'create', spec))
        self.calls.append(call)

    def step(self, n: int) -> None:
        for _ in range(n):
            self.loop += 1
            due = [p for p in self.pending if p[0] <= self.loop]
            self.pending = [p for p in self.pending if p[0] > self.loop]
            for _, kind, arg in due:
                if kind == 'create':
                    owner, utype, x, y = arg
                    blocked = utype != 106 and any(
                        u[1] != 106 and math.hypot(u[2] - x, u[3] - y) < 0.5
                        for u in self.units.values()
                    )
                    self._add(owner, utype, x, y - 0.3135 if blocked else y)
            for _, kind, arg in due:
                if kind == 'kill':
                    for tag in arg:
                        self.units.pop(tag, None)

    def observe(self) -> list[Any]:
        out = []
        for pid in self.agent_ids:
            units = [
                _ns(tag=t, owner=u[0], unit_type=u[1], pos=_ns(x=u[2], y=u[3]), alliance=1)
                for t, u in self.units.items()
                if u[0] == pid
            ]
            out.append(_ns(observation=_ns(game_loop=self.loop, raw_data=_ns(units=units)),
                           player_result=[]))  # fmt: skip
        return out


# Coordinates are exact in float32 (Point2D uses float32)
SPAWNS = (Spawn(1, 48, 9.0, 16.0), Spawn(1, 48, 9.0, 15.25), Spawn(2, 48, 23.0, 16.0))


def _fake_backend(session: _FakeSession) -> Sc2Backend:
    backend = Sc2Backend()
    backend._session = cast(Any, session)  # noqa: SLF001
    return backend


def test_explicit_reset_two_phase_order() -> None:
    """Two-phase explicit reset kills old units with placeholders before respawning."""
    pytest.importorskip('s2clientprotocol')
    session = _FakeSession([(s.owner, s.unit_type, s.x, s.y) for s in SPAWNS])
    old = set(session.units)
    backend = _fake_backend(session)
    scenario = Scenario('new_maps/3m', players=session.players, spawns=SPAWNS)
    backend._reset_explicit(scenario)  # noqa: SLF001

    assert len(session.calls) == 2
    first, second = session.calls
    assert first['kill'] == [sorted(old)] and not first['alive'] - old
    holders = first['create']
    assert sorted(o for o, *_ in holders) == [1, 2] and all(t == 106 for _, t, _, _ in holders)
    hx, hy = holders[0][2:]
    assert 3 <= hx <= 29 and 3 <= hy <= 29
    assert min(math.hypot(hx - s.x, hy - s.y) for s in SPAWNS) > 10
    # By phase 2, old units are gone and only placeholder units remain
    assert not second['alive'] & old and len(second['alive']) == 2
    assert second['create'] == [(s.owner, s.unit_type, s.x, s.y) for s in SPAWNS]
    assert second['kill'] == [sorted(second['alive'])]

    tags = backend._tags  # noqa: SLF001
    assert not set(tags) & (old | second['alive'])
    for tag, s in zip(tags, SPAWNS, strict=True):
        assert session.units[tag] == [s.owner, s.unit_type, s.x, s.y]
    assert set(session.units) == set(tags)


def test_explicit_reset_without_old_units_is_single_phase() -> None:
    """Empty map reset spawns units directly in a single phase without placeholders."""
    pytest.importorskip('s2clientprotocol')
    session = _FakeSession([], playable=(0, 0, 0, 0))
    backend = _fake_backend(session)
    scenario = Scenario('SMAC_Maps/32x32_flat', players=session.players, spawns=SPAWNS)
    backend._reset_explicit(scenario)  # noqa: SLF001
    assert len(session.calls) == 1 and session.calls[0]['kill'] == []
    assert all(t != 106 for _, t, _, _ in session.calls[0]['create'])
    # When playable_area is empty, placeholder falls inside the full 32x32 map
    x, y = backend._placeholder_point(scenario)  # noqa: SLF001
    assert 1 <= x <= 31 and 1 <= y <= 31
