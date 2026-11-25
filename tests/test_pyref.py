"""Tests for the Python reference backend (pyref): contract conformance and unit tests for
commands, combat, and batching.

Static unit data and maps use extracted SC2 assets (`unismac.assets`). Terrain, pathing,
collision, kinematics, and targeting tests live in test_pyref_terrain.py. Fixtures are defined
inline because pytest imports tests in importlib mode.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems.damage import apply_hit
from unismac.contract import (
    LOOPS_PER_GAME_SECOND,
    NO_SLOT,
    AgentPlayer,
    Backend,
    CapabilityError,
    Command,
    CommandKind,
    ComputerPlayer,
    ContractError,
    Scenario,
    Spawn,
    StaticData,
    UnitTypeSpec,
    WorldState,
    conformance,
)

# ---------------------------------------------------------------- Fixtures

MARINE, ZEALOT, STALKER = 48, 73, 74
MAP_ID = 'SMAC_Maps/32x32_flat'  # passable region x, y in [2, 30)
STATIC = load_static_data()
MAPS = {m: load_map_info(m) for m in (MAP_ID, 'SMAC_Maps/3m', 'new_maps/3m')}
MARINE_SPEC = STATIC.get(MARINE)
STALKER_SPEC = STATIC.get(STALKER)

# Synthetic kinematics parameters (dist/loop^2, rad/loop) to exercise the accel branch.
SYNTHETIC_MEASURED = {'acceleration': 0.02, 'deceleration': 0.03, 'turn_rate': 0.5}


def with_measured(static: StaticData, measured: dict[str, float]) -> StaticData:
    """Attach synthetic measured kinematics parameters to all unit types."""
    units = {k: dataclasses.replace(s, measured=dict(measured)) for k, s in static.units.items()}
    return StaticData(units=units, source=f'{static.source} + synthetic measured')


STATIC_ACCEL = with_measured(STATIC, SYNTHETIC_MEASURED)

# Pathing x collision matrix plus full accel, 2-loop command delay, and SC2 alignment combos.
COMBOS: dict[str, dict[str, object]] = {
    'straight-none': {},
    'grid-none': {'pathing': 'grid'},
    'straight-hard': {'collision': 'hard'},
    'grid-hard': {'pathing': 'grid', 'collision': 'hard'},
    'full': {
        'pathing': 'grid',
        'collision': 'hard',
        'steering': 'separation',
        'kinematics': 'accel',
    },
    'delayed-2': {'commands': 'delayed', 'command_delay': 2},
    # Two-body collision alignment preset (see test_pyref_p2.py)
    'sc2-p2': {
        'collision': 'hard',
        'steering': 'sc2_avoid',
        'contact_slack': 0.0625,
        'idle_push_share': 0.9,
        'idle_return': True,
    },
    # Crowd steering rules
    'sc2-p3': {
        'collision': 'hard',
        'steering': 'sc2_avoid',
        'contact_slack': 0.0625,
        'idle_push_share': 0.9,
        'idle_return': True,
        'crowd_arrival': True,
        'avoid_side_check': True,
    },
    # Continuous navigation mesh pathing
    'navmesh': {'pathing': 'navmesh', 'collision': 'hard'},
    # Medivac healing (see test_pyref_heal.py)
    'heal': {'abilities': 'heal', 'spawn_energy': 50.0, 'heal_autocast': True},
    # Combat timing and command delay (see test_pyref_p5.py)
    'sc2-p5': {
        'commands': 'delayed',
        'command_delay': 1,
        'move_lag': 1,
        'attack': 'sc2',
        'damage': 'sc2',
        'auto_acquire': True,
        'acquire_start_loop': 2,
        'spawn_facing': 'random',
    },
}


def make(batch_size: int = 1, static: StaticData = STATIC, **config: object) -> PyrefBackend:
    return PyrefBackend(static, MAPS, batch_size, PyrefConfig(**config))  # type: ignore[arg-type]


def make_combo(name: str, batch_size: int = 1) -> PyrefBackend:
    static = STATIC_ACCEL if name == 'full' else STATIC
    return make(batch_size, static, **COMBOS[name])


def scenario(*spawns: Spawn, n_slots: int | None = None) -> Scenario:
    return Scenario(map_id=MAP_ID, spawns=spawns, n_slots=n_slots)


def duel(unit_type: int = MARINE, dx: float = 5.0) -> Scenario:
    """Two units of the same type on y=16 separated by dx."""
    return scenario(Spawn(1, unit_type, 10.0, 16.0), Spawn(2, unit_type, 10.0 + dx, 16.0))


def period(spec: UnitTypeSpec) -> int:
    """Firing period in pyref: ceil(cooldown * 16) loops (see attack.instant)."""
    return math.ceil(np.float32(spec.weapons[0].cooldown) * np.float32(LOOPS_PER_GAME_SECOND))


# ---------------------------------------------------------------- Contract conformance


@pytest.mark.parametrize('combo', COMBOS)
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('check', conformance.CHECKS, ids=lambda c: c.__name__)
def test_conformance(combo: str, batch_size: int, check) -> None:
    backend = make_combo(combo, batch_size)
    assert isinstance(backend, Backend)
    check(backend)


@pytest.mark.parametrize(
    ('config', 'pathing', 'collision'),
    [
        ({}, 'none', 'none'),
        ({'pathing': 'grid'}, 'grid', 'none'),
        ({'collision': 'hard'}, 'none', 'hard'),
        ({'pathing': 'grid', 'collision': 'hard', 'steering': 'separation'}, 'grid', 'hard'),
    ],
)
def test_capabilities(config: dict[str, object], pathing: str, collision: str) -> None:
    caps = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config)).capabilities  # type: ignore[arg-type]
    assert (caps.name, caps.max_batch, caps.pathing, caps.collision) == (
        'pyref',
        None,
        pathing,
        collision,
    )
    assert caps.deterministic
    assert not any(
        (caps.engine_ai, caps.map_trigger, caps.terrain_height, caps.abilities, caps.fog)
    )
    caps.require(pathing=pathing, collision=collision)


def test_unknown_implementation_name() -> None:
    with pytest.raises(ValueError, match='pathing'):
        make(pathing='astar')
    with pytest.raises(ValueError, match='collision_iterations'):
        PyrefConfig(collision_iterations=0)
    with pytest.raises(ValueError, match='delayed'):
        PyrefConfig(command_delay=2)
    with pytest.raises(ValueError, match='command_delay'):
        PyrefConfig(commands='delayed', command_delay=-1)


# ---------------------------------------------------------------- Command delay


def _move_x(backend: PyrefBackend, k_loops: int) -> list[float]:
    """Issue MOVE to slot 0, advance loop by loop, and return y after each loop."""
    state = backend.reset([duel(MARINE, dx=8.0)])
    cmd = Command.noop(1, state.n_slots).set(0, 0, CommandKind.MOVE, target_xy=(10.0, 26.0))
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(26.0, 16.0))
    out = [backend.step(cmd, 1)]
    out += [backend.step(Command.noop(1, state.n_slots), 1) for _ in range(k_loops - 1)]
    return [float(s.y[0, 0]) for s in out]


def test_delayed_commands_take_effect_after_k_loops() -> None:
    """k=2: commands take effect 2 loops after receipt, shifting the immediate trajectory."""
    ref = _move_x(make(), 30)
    got = _move_x(make(commands='delayed', command_delay=2), 30)
    assert got[:2] == [16.0, 16.0], 'should stay still for the first 2 loops'
    assert got[2:] == ref[:-2]
    assert _move_x(make(commands='delayed', command_delay=0), 30) == ref, (
        'k=0 should match immediate'
    )


def test_delayed_queue_order_and_reset() -> None:
    backend = make(commands='delayed', command_delay=2)
    state = backend.reset([duel(MARINE, dx=8.0)])
    up = Command.noop(1, 2).set(0, 0, CommandKind.MOVE, target_xy=(10.0, 26.0))
    down = Command.noop(1, 2).set(0, 0, CommandKind.MOVE, target_xy=(10.0, 6.0))
    s = [backend.step(up, 1), backend.step(down, 1)]
    s += [backend.step(Command.noop(1, 2), 1) for _ in range(3)]
    ys = [float(x.y[0, 0]) for x in s]
    assert ys[2] > 16.0 and ys[3] < ys[2] and ys[4] < ys[3], (
        'earlier command should take effect first, then later command overrides it'
    )
    assert s[2].order_y[0, 0] == 26.0 and s[3].order_y[0, 0] == 6.0
    # reset clears the queue: pre-reset commands must not fire in the new episode.
    backend.step(up, 1)
    state = backend.reset([duel(MARINE, dx=8.0)])
    end = backend.step(Command.noop(1, 2), 4)
    assert np.array_equal(end.y, state.y) and end.order_kind[0, 0] == CommandKind.NOOP


# ---------------------------------------------------------------- Damage resolution


def test_apply_hit_shield_then_armor() -> None:
    # Shields absorb damage first; armor only reduces spillover to health.
    assert apply_hit(np.float32(10), np.float32(50), 6, armor=1, min_damage=0.5) == (4, 50)
    assert apply_hit(np.float32(2), np.float32(10), 6, armor=1, min_damage=0.5) == (0, 7)
    # Minimum post-armor damage is 0.5.
    assert apply_hit(np.float32(0), np.float32(10), 1, armor=3, min_damage=0.5) == (0, 9.5)
    # Exact shield absorption deals no health damage and does not trigger min_damage.
    assert apply_hit(np.float32(6), np.float32(10), 6, armor=3, min_damage=0.5) == (0, 10)


def test_bonus_shield_and_armor_over_time() -> None:
    """Stalker vs Stalker: 13 + 5 (armored) = 18; depletes 80 shield first, then 18 - 1 armor."""
    backend = make()
    backend.reset([duel(STALKER, dx=7.0)])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    p = period(STALKER_SPEC)  # 30
    state = backend.step(cmd, 4 * p + 1)  # 5 shots: shield 80->62->44->26->8->0, hp 80-9
    assert (state.shield[0, 1], state.health[0, 1]) == (0, 71)
    state = backend.step(Command.noop(1, 2), p)  # 6th shot: 71 - 17
    assert (state.shield[0, 1], state.health[0, 1]) == (0, 54)
    assert state.health[0, 0] == 80 and state.shield[0, 0] == 80, 'idle unit should not retaliate'


def test_multiple_attacks_per_shot() -> None:
    """Zealot deals 2 x 8 per attack against 0-armor Marine: 16 damage per shot."""
    assert STATIC.get(ZEALOT).weapons[0].attacks == 2
    backend = make()
    backend.reset([scenario(Spawn(1, ZEALOT, 10.0, 16.0), Spawn(2, MARINE, 10.9, 16.0))])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    state = backend.step(cmd, 1)
    assert state.health[0, 1] == 45 - 16


# ---------------------------------------------------------------- Cooldown cadence


def test_cooldown_mutual_kill_timing() -> None:
    """Two Marines trading fire: shot k occurs on game_loop (k - 1) * 14; shot 8
    (ceil(45 / 6)) kills both simultaneously."""
    shots = math.ceil(MARINE_SPEC.health_max / MARINE_SPEC.weapons[0].damage)  # 8
    kill_loop = (shots - 1) * period(MARINE_SPEC)  # 98
    backend = make()
    backend.reset([duel(MARINE, dx=5.0)])
    cmd = Command.noop(1, 2)
    cmd.set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    cmd.set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    state = backend.step(cmd, 1)
    assert np.array_equal(state.health[0], [39, 39]), 'in-range units should fire on loop 1'
    cooldown = np.float32(MARINE_SPEC.weapons[0].cooldown) * np.float32(LOOPS_PER_GAME_SECOND)
    assert np.allclose(state.weapon_cooldown[0], cooldown)
    noop = Command.noop(1, 2)
    state = backend.step(noop, kill_loop - 1)
    assert state.game_loop[0] == kill_loop
    assert state.alive[0].all() and np.array_equal(state.health[0], [3, 3])
    state = backend.step(noop, 1)
    assert not state.alive[0].any(), 'both units should kill each other on the same loop'
    assert np.array_equal(state.health[0], [0, 0])
    assert np.all(state.order_kind[0] == CommandKind.NOOP)
    assert np.all(state.engaged_target_slot[0] == NO_SLOT)


def test_chase_then_fire_and_dead_target_clears_orders() -> None:
    """Out-of-range ATTACK_UNIT chases into range first; order clears to NOOP on target death."""
    backend = make()
    backend.reset([duel(MARINE, dx=8.0)])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    state = backend.step(cmd, 1)
    assert state.health[0, 1] == 45 and state.x[0, 0] > 10.0
    assert state.engaged_target_slot[0, 0] == 1
    for _ in range(200):
        if not state.alive[0, 1]:
            break
        state = backend.step(Command.noop(1, 2), 1)
    assert not state.alive[0, 1]
    edge = abs(state.x[0, 1] - state.x[0, 0]) - 2 * MARINE_SPEC.radius
    assert edge <= MARINE_SPEC.weapons[0].range
    assert state.order_kind[0, 0] == CommandKind.NOOP
    assert state.order_target_slot[0, 0] == NO_SLOT
    assert state.engaged_target_slot[0, 0] == NO_SLOT


# ---------------------------------------------------------------- Commands


def test_attack_move_engages_then_resumes() -> None:
    backend = make()
    backend.reset([scenario(Spawn(1, MARINE, 4.0, 16.0), Spawn(2, MARINE, 16.0, 18.0))])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_MOVE, target_xy=(28.0, 16.0))
    state = backend.step(cmd, 1)
    stopped_at = None
    for _ in range(400):
        if state.order_kind[0, 0] == CommandKind.NOOP:
            break
        prev = state
        state = backend.step(Command.noop(1, 2), 1)
        if state.engaged_target_slot[0, 0] == 1 and prev.engaged_target_slot[0, 0] == 1:
            assert state.x[0, 0] == prev.x[0, 0], 'should not move while engaged in range'
            stopped_at = stopped_at or float(state.x[0, 0])
    assert stopped_at is not None and stopped_at < 16.0, 'should stop in range to engage en route'
    assert not state.alive[0, 1], 'enemy en route should be killed'
    assert state.health[0, 0] == 45, 'idle enemy should not retaliate when auto_acquire=False'
    assert state.order_kind[0, 0] == CommandKind.NOOP, 'order should clear on arrival'
    assert np.hypot(state.x[0, 0] - 28.0, state.y[0, 0] - 16.0) <= 0.05


def test_auto_acquire_option() -> None:
    backend = make(auto_acquire=True)
    backend.reset([duel(MARINE, dx=5.0)])
    state = backend.step(Command.noop(1, 2), 1)
    assert np.array_equal(state.health[0], [39, 39]), (
        'idle units should auto-attack enemies in range'
    )
    assert np.array_equal(state.x[0], [10.0, 15.0]), 'should not move when already in range'


def test_stop_and_hold() -> None:
    backend = make()
    backend.reset([duel(MARINE, dx=5.0)])
    cmd = Command.noop(1, 2)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(10.0, 4.0))
    cmd.set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    backend.step(cmd, 4)
    cmd = Command.noop(1, 2)
    cmd.set(0, 0, CommandKind.HOLD)
    cmd.set(0, 1, CommandKind.STOP)
    before = backend.step(cmd, 1)
    after = backend.step(Command.noop(1, 2), 30)
    assert after.order_kind[0, 0] == CommandKind.HOLD
    assert after.order_kind[0, 1] == CommandKind.NOOP
    assert np.array_equal(after.x, before.x) and np.array_equal(after.y, before.y)
    assert np.array_equal(after.health, before.health), 'should not fire after STOP'


def test_rejects_invalid_commands() -> None:
    backend = make()
    backend.reset([scenario(Spawn(1, MARINE, 10.0, 16.0), Spawn(2, MARINE, 22.0, 16.0), n_slots=3)])
    with pytest.raises(CapabilityError):
        backend.step(Command.noop(1, 3).set(0, 0, CommandKind.ABILITY, ability=1), 1)
    for slot, kind, target in (
        (2, CommandKind.STOP, NO_SLOT),  # empty slot
        (0, CommandKind.ATTACK_UNIT, 0),  # self-attack
        (0, CommandKind.ATTACK_UNIT, 7),  # out of bounds
        (0, CommandKind.MOVE, NO_SLOT),  # missing target point
    ):
        with pytest.raises(ContractError):
            backend.step(Command.noop(1, 3).set(0, slot, kind, target_slot=target), 1)
    state = backend.step(Command.noop(1, 3), 1)
    assert state.game_loop[0] == 1, 'rejected commands must not advance simulation time'


def test_rejects_command_on_dead_unit() -> None:
    backend = make()
    backend.reset([duel(MARINE, dx=5.0)])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    state = backend.step(cmd, 200)
    assert not state.alive[0, 1]
    with pytest.raises(ContractError):
        backend.step(Command.noop(1, 2).set(0, 1, CommandKind.MOVE, target_xy=(1.0, 1.0)), 1)
    # ATTACK_UNIT on a dead target is silently ignored (matching SC2).
    state = backend.step(cmd, 1)
    assert state.order_kind[0, 0] == CommandKind.NOOP


# ---------------------------------------------------------------- reset


def test_rejects_unsupported_scenarios() -> None:
    backend = make()
    with pytest.raises(CapabilityError):
        backend.reset(
            [
                Scenario(
                    MAP_ID, players=(AgentPlayer(), ComputerPlayer('terran')), spawns=duel().spawns
                )
            ]
        )
    with pytest.raises(CapabilityError):
        backend.reset([Scenario(MAP_ID, spawn_mode='map_trigger', n_slots=4)])
    with pytest.raises(ContractError):
        backend.reset([Scenario('SMAC_Maps/nope', spawns=duel().spawns)])
    with pytest.raises(ContractError):
        backend.reset([scenario(Spawn(1, 9999, 10.0, 10.0))])  # unknown unit type
    with pytest.raises(ContractError):
        backend.reset([scenario(Spawn(1, MARINE, 40.0, 10.0))])  # outside playable area
    with pytest.raises(ContractError, match='reset'):
        backend.step(Command.noop(1, 2), 1)


def _mixed_command(envs: list[int]) -> Command:
    """Issue a different command per env in `envs` based on `env % 4`."""
    cmd = Command.noop(len(envs), 3)
    for i, env in enumerate(envs):
        match env % 4:
            case 0:
                cmd.set(i, 0, CommandKind.ATTACK_UNIT, target_slot=2)
                cmd.set(i, 1, CommandKind.ATTACK_UNIT, target_slot=2)
            case 1:
                cmd.set(i, 0, CommandKind.MOVE, target_xy=(4.0, 4.0))
            case 2:
                pass
            case 3:
                cmd.set(i, 1, CommandKind.ATTACK_MOVE, target_xy=(28.0, 18.0))
                cmd.set(i, 2, CommandKind.ATTACK_UNIT, target_slot=0)
    return cmd


def _play(backend: PyrefBackend, cmd: Command) -> list[WorldState]:
    """Issue `cmd`, then advance 40 * 8 loops with no-ops, returning states after each step."""
    n_envs, n_slots = cmd.kind.shape
    states = [backend.step(cmd, 8)]
    states += [backend.step(Command.noop(n_envs, n_slots), 8) for _ in range(40)]
    return states


@pytest.mark.parametrize('combo', COMBOS)
def test_envs_in_batch_are_independent(combo: str) -> None:
    """Envs with distinct commands and maps match standalone batch_size=1 runs bit-for-bit."""
    scenarios = [
        dataclasses.replace(conformance.SCENARIO, map_id=m)
        for m in (MAP_ID, 'SMAC_Maps/3m', MAP_ID, 'new_maps/3m')
    ]
    batch = make_combo(combo, 4)
    batch.reset(scenarios)
    batched = _play(batch, _mixed_command([0, 1, 2, 3]))
    assert not np.array_equal(batched[-1].x[0], batched[-1].x[1]), (
        'envs should produce distinct trajectories'
    )
    assert not batched[-1].alive[0, 2] and batched[-1].alive[2, 2]
    for env in range(4):
        single = make_combo(combo, 1)
        single.reset([scenarios[env]])
        alone = _play(single, _mixed_command([env]))
        for sb, sa in zip(batched, alone, strict=True):
            for name in WorldState.UNIT_FIELDS:
                np.testing.assert_array_equal(getattr(sb, name)[env], getattr(sa, name)[0], name)


def test_partial_reset() -> None:
    backend = make(3)
    fresh = backend.reset([conformance.SCENARIO] * 3)
    played = _play(backend, _mixed_command([0, 1, 2]))[-1]
    other = scenario(*conformance.SCENARIO.spawns[:2], Spawn(2, STALKER, 20.0, 20.0))
    state = backend.reset([other], env_ids=[1])
    assert np.array_equal(state.game_loop, [played.game_loop[0], 0, played.game_loop[2]])
    for name in WorldState.UNIT_FIELDS:
        for env in (0, 2):  # unreset envs remain unchanged
            np.testing.assert_array_equal(getattr(state, name)[env], getattr(played, name)[env])
    assert state.unit_type[1, 2] == STALKER and state.health[1, 2] == 80
    assert np.array_equal(state.x[1, :2], fresh.x[1, :2]) and state.alive[1].all()
    assert state.visible[1].all()
    # Partial reset must match the existing batch slot count.
    with pytest.raises(ContractError):
        backend.reset([duel()], env_ids=[0])
    # Reset env continues advancing normally without disturbing other envs.
    cmd = Command.noop(3, 3).set(1, 0, CommandKind.MOVE, target_xy=(10.0, 10.0))
    after = backend.step(cmd, 8)
    assert after.y[1, 0] < state.y[1, 0]
    assert np.array_equal(after.game_loop, state.game_loop + 8)
