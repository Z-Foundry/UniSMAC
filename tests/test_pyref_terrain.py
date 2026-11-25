"""Tests for pyref terrain, grid pathing, collision, separation, kinematics, and auto-acquire.

Real map assets come from `unismac.assets`; synthetic obstacle maps (U-trap, narrow slit, sealed
box) are constructed as `MapInfo` instances. Helpers stay local to this file so tests work under
pytest's importlib mode.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig, terrain
from unismac.backends.pyref.subsystems.kinematics import _steer_heading, rl_turn_step
from unismac.contract import (
    NO_SLOT,
    Command,
    CommandKind,
    MapInfo,
    Scenario,
    Spawn,
    StaticData,
    WorldState,
)

MARINE, STALKER, MEDIVAC = 48, 74, 54
STATIC = load_static_data()
MARINE_SPEC = STATIC.get(MARINE)
SPEED = np.float32(MARINE_SPEC.movement_speed) / np.float32(16)  # Marine tiles per loop
FLAT = 'SMAC_Maps/32x32_flat'
M3 = 'SMAC_Maps/3m'  # passable region x in [2, 30), y in [7, 25)
CORRIDOR = 'SMAC_Maps/corridor'


def ascii_map(map_id: str, art: str) -> MapInfo:
    """Build a MapInfo from ASCII art: '#' blocked, '.' passable. Top line has largest y
    to match world orientation (one row per y, one char per x); whole map is playable."""
    rows = [r.strip() for r in art.strip().splitlines()][::-1]
    pathing = np.array([[c == '.' for c in r] for r in rows], np.bool_)
    h, w = pathing.shape
    return MapInfo(
        map_id=map_id,
        sha256='0' * 64,
        width=w,
        height=h,
        playable=(0, 0, w, h),
        pathing=pathing,
        placement=pathing.copy(),
        terrain_height=np.zeros((h, w), np.float32),
    )


def blocked_rect(map_id: str, size: int, rects: list[tuple[int, int, int, int]]) -> MapInfo:
    """Create a size x size map with 1-cell border and blocked rects (inclusive)."""
    grid = np.ones((size, size), np.bool_)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
    for x0, y0, x1, y1 in rects:
        grid[y0 : y1 + 1, x0 : x1 + 1] = False
    return MapInfo(
        map_id=map_id,
        sha256='0' * 64,
        width=size,
        height=size,
        playable=(0, 0, size, size),
        pathing=grid,
        placement=grid.copy(),
        terrain_height=np.zeros((size, size), np.float32),
    )


# U-shaped obstacle opening toward -x: straight movement from the left enters the pocket.
U_TRAP = blocked_rect('test/u_trap', 32, [(18, 10, 18, 22), (12, 10, 18, 10), (12, 22, 18, 22)])
# Narrow slit: full wall along x=16 with a single 1-cell gap at y=20.
SLIT = blocked_rect('test/slit', 32, [(16, 1, 16, 19), (16, 21, 16, 30)])
# Sealed box: walls around x, y in [12, 20]; interior is unreachable from outside.
BOX = blocked_rect(
    'test/box', 32, [(12, 12, 20, 12), (12, 20, 20, 20), (12, 12, 12, 20), (20, 12, 20, 20)]
)
MAPS = {
    m.map_id: m
    for m in (load_map_info(FLAT), load_map_info(M3), load_map_info(CORRIDOR), U_TRAP, SLIT, BOX)
}


def make(batch_size: int = 1, static: StaticData = STATIC, **config: object) -> PyrefBackend:
    return PyrefBackend(static, MAPS, batch_size, PyrefConfig(**config))  # type: ignore[arg-type]


def with_measured(measured: dict[str, float]) -> StaticData:
    units = {k: dataclasses.replace(s, measured=dict(measured)) for k, s in STATIC.units.items()}
    return StaticData(units=units, source=f'{STATIC.source} + synthetic measured')


def move_all(state: WorldState, targets: dict[int, tuple[float, float]]) -> Command:
    cmd = Command.noop(state.batch_size, state.n_slots)
    for b in range(state.batch_size):
        for slot, xy in targets.items():
            cmd.set(b, slot, CommandKind.MOVE, target_xy=xy)
    return cmd


def run(backend: PyrefBackend, cmd: Command, n: int) -> list[WorldState]:
    """Issue cmd and step 1 loop at a time for n loops, returning all states."""
    b, u = cmd.kind.shape
    states = [backend.step(cmd, 1)]
    states += [backend.step(Command.noop(b, u), 1) for _ in range(n - 1)]
    return states


def assert_on_passable(info: MapInfo, states: list[WorldState], slots: list[int]) -> None:
    for s in states:
        for u in slots:
            ix, iy = math.floor(s.x[0, u]), math.floor(s.y[0, u])
            assert info.pathing[iy, ix], f'loop {s.game_loop[0]} slot {u} blocked {(ix, iy)}'


# ---------------------------------------------------------------- Terrain pure functions


def test_find_path_does_not_cut_corners() -> None:
    grid = ascii_map(
        't',
        """
        ...
        ...
        .#.
        """,
    ).pathing
    # (0, 0) -> (2, 0) must not cut the corner at (1, 0); must detour through row y=1.
    path = terrain.find_path(grid, (0, 0), (2, 0))
    assert path[0] == (0, 0) and path[-1] == (2, 0)
    assert (1, 0) not in path
    for (ax, ay), (bx, by) in zip(path, path[1:], strict=False):
        if ax != bx and ay != by:
            assert grid[ay, bx] and grid[by, ax], f'{(ax, ay)} -> {(bx, by)} cuts corner'
    assert path == ((0, 0), (0, 1), (1, 1), (2, 1), (2, 0))


def test_find_path_unreachable_goes_to_nearest_reachable() -> None:
    grid = BOX.pathing
    path = terrain.find_path(grid, (5, 16), (16, 16))  # sealed box interior unreachable
    assert path[-1] == (11, 16), 'should stop at nearest cell outside box'
    path = terrain.find_path(grid, (5, 16), (16, 12))  # goal itself is a wall
    assert path[-1] in ((11, 12), (16, 11))


def test_line_of_sight_through_grid_corner() -> None:
    grid = ascii_map(
        't',
        """
        ...
        #..
        ...
        """,
    ).pathing
    assert not grid[1, 0]
    assert not terrain.line_of_sight(grid, 0.5, 0.5, 0.5, 2.5), 'crosses blocked cell'
    assert terrain.line_of_sight(grid, 1.5, 0.5, 1.5, 2.5)
    # Passes exactly through grid vertex (1, 1): diagonal (0, 1) is blocked (matches A*).
    assert not terrain.line_of_sight(grid, 0.5, 0.5, 2.5, 2.5)
    # Passes through grid vertex (2, 1): both adjacent cells (1, 1) and (2, 0) are passable.
    assert terrain.line_of_sight(grid, 1.5, 0.5, 2.5, 1.5)
    assert terrain.line_of_sight(grid, 1.2, 1.5, 1.2, 1.5), 'identical endpoints'


def test_project_to_passable() -> None:
    grid = load_map_info(M3).pathing  # y < 7 is unpassable
    assert terrain.project_to_passable(grid, 10.3, 9.0) == (10.3, 9.0)
    x, y = terrain.project_to_passable(grid, 10.3, 5.5)
    assert x == 10.3 and 7.0 < y < 7.01


# ---------------------------------------------------------------- Global pathing


@pytest.mark.parametrize('collision', ['none', 'hard'])
def test_u_trap_detour(collision: str) -> None:
    """Straight path enters the U-pocket; grid pathing detours outside on passable cells."""
    sc = Scenario(map_id=U_TRAP.map_id, spawns=(Spawn(1, MARINE, 6.0, 16.0),))
    goal = (26.0, 16.0)
    backend = make(pathing='grid', collision=collision)
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: goal}), 400)
    assert_on_passable(U_TRAP, states, [0])
    done = [s.order_kind[0, 0] == CommandKind.NOOP for s in states]
    assert any(done), 'should arrive within 400 loops'
    arrive = done.index(True) + 1
    assert np.hypot(states[-1].x[0, 0] - goal[0], states[-1].y[0, 0] - goal[1]) <= 0.05
    assert arrive * SPEED > 22.0, 'detour should exceed straight-line distance 20'
    # Control: straight pathing + hard terrain collision gets stuck inside the pocket.
    stuck = make(collision='hard')
    stuck.reset([sc])
    end = run(stuck, move_all(state, {0: goal}), 400)[-1]
    assert end.order_kind[0, 0] == CommandKind.MOVE and end.x[0, 0] < 18.0


def test_narrow_slit() -> None:
    """1-cell wide gap: grid pathing routes through the slit in the wall."""
    sc = Scenario(map_id=SLIT.map_id, spawns=(Spawn(1, MARINE, 8.0, 8.0),))
    backend = make(pathing='grid', collision='hard')
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (24.0, 8.0)}), 600)
    assert_on_passable(SLIT, states, [0])
    crossing = [s for s in states if 16.0 <= s.x[0, 0] < 17.0]
    assert crossing and all(20.0 <= s.y[0, 0] < 21.0 for s in crossing), 'crosses at y=20'
    assert states[-1].order_kind[0, 0] == CommandKind.NOOP


@pytest.mark.parametrize('pathing', ['straight', 'grid'])
def test_unreachable_goal_stops_at_nearest_reachable(pathing: str) -> None:
    """In 3m, y < 7 is unpassable: grid pathing reaches the nearest passable cell and
    finishes the order; straight pathing + hard collision stops at the wall still moving."""
    sc = Scenario(map_id=M3, spawns=(Spawn(1, MARINE, 10.5, 16.0),))
    backend = make(pathing=pathing, collision='hard')
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (10.5, 4.0)}), 120)
    assert_on_passable(MAPS[M3], states, [0])
    end = states[-1]
    assert 7.0 <= end.y[0, 0] < 7.05 and abs(end.x[0, 0] - 10.5) < 1e-2
    expect = CommandKind.NOOP if pathing == 'grid' else CommandKind.MOVE
    assert end.order_kind[0, 0] == expect


def test_unreachable_sealed_box() -> None:
    sc = Scenario(map_id=BOX.map_id, spawns=(Spawn(1, MARINE, 5.0, 16.5),))
    backend = make(pathing='grid')
    state = backend.reset([sc])
    end = run(backend, move_all(state, {0: (16.5, 16.5)}), 200)[-1]
    assert end.order_kind[0, 0] == CommandKind.NOOP
    assert 11.9 < end.x[0, 0] < 12.0 and abs(end.y[0, 0] - 16.5) < 1e-4


def test_corridor_real_terrain() -> None:
    """Real SMAC_Maps/corridor: path from first to last passable cell on the map."""
    info = MAPS[CORRIDOR]
    free = np.argwhere(info.pathing)
    (sy, sx), (ey, ex) = free[0], free[-1]
    sc = Scenario(map_id=info.map_id, spawns=(Spawn(1, MARINE, sx + 0.5, sy + 0.5),))
    backend = make(pathing='grid', collision='hard')
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (ex + 0.5, ey + 0.5)}), 400)
    assert_on_passable(info, states, [0])
    assert states[-1].order_kind[0, 0] == CommandKind.NOOP


def test_air_units_fly_straight_over_walls() -> None:
    sc = Scenario(map_id=U_TRAP.map_id, spawns=(Spawn(1, MEDIVAC, 6.0, 16.0),))
    backend = make(pathing='grid', collision='hard')
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (26.0, 16.0)}), 200)
    assert all(abs(s.y[0, 0] - 16.0) < 1e-5 for s in states), 'air flies over walls'
    assert states[-1].order_kind[0, 0] == CommandKind.NOOP


def test_path_cache_key() -> None:
    sc = Scenario(map_id=U_TRAP.map_id, spawns=(Spawn(1, MARINE, 6.0, 16.0),))
    backend = make(pathing='grid')
    state = backend.reset([sc])
    backend.step(move_all(state, {0: (26.0, 16.0)}), 1)
    sim = backend._sim
    assert sim is not None and (U_TRAP.map_id, (6, 16), (26, 16)) in sim.path_cache


# ---------------------------------------------------------------- Collision and separation


def _min_gap(states: list[WorldState], u: int, t: int) -> float:
    return min(float(np.hypot(s.x[0, u] - s.x[0, t], s.y[0, u] - s.y[0, t])) for s in states)


@pytest.mark.parametrize('dy', [0.0, 0.2])
@pytest.mark.parametrize('steering', ['none', 'separation'])
def test_head_on_no_overlap(steering: str, dy: float) -> None:
    """Two Marines moving head-on never overlap under hard collision (tol 1e-4). With
    lateral offset they pass each other; exact symmetry (dy=0) has zero lateral push and
    deadlocks without avoidance."""
    sc = Scenario(
        map_id=FLAT, spawns=(Spawn(1, MARINE, 10.0, 16.0), Spawn(1, MARINE, 20.0, 16.0 + dy))
    )
    backend = make(pathing='grid', collision='hard', steering=steering)
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (20.0, 16.0), 1: (10.0, 16.0 + dy)}), 300)
    assert _min_gap(states, 0, 1) >= 2 * MARINE_SPEC.radius - 1e-4
    end = states[-1]
    if dy > 0:
        assert end.x[0, 0] > 15.0 > end.x[0, 1], 'should pass with lateral offset'
        assert np.all(end.order_kind[0] == CommandKind.NOOP)
    else:
        assert end.x[0, 0] < 15.0 < end.x[0, 1] and np.all(end.order_kind[0] == CommandKind.MOVE)


def test_overlapping_spawns_are_pushed_apart() -> None:
    sc = Scenario(
        map_id=FLAT,
        spawns=(
            Spawn(1, MARINE, 16.0, 16.0),
            Spawn(1, MARINE, 16.0, 16.0),
            Spawn(2, MARINE, 16.3, 16.0),
        ),
    )
    backend = make(collision='hard', collision_iterations=8)
    backend.reset([sc])
    s = backend.step(Command.noop(1, 3), 4)
    for u, t in ((0, 1), (0, 2), (1, 2)):
        d = np.hypot(s.x[0, u] - s.x[0, t], s.y[0, u] - s.y[0, t])
        assert d >= 2 * MARINE_SPEC.radius - 1e-3, f'{u}-{t} still overlapping: {d:.4f}'
    assert s.x[0, 0] < s.x[0, 1], 'coincident centers separate along +/-x by slot order'


def test_separation_spreads_moving_units() -> None:
    sc = Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 8.0, 16.0), Spawn(1, MARINE, 8.0, 16.8)))

    def final_gap(steering: str) -> float:
        backend = make(steering=steering)
        state = backend.reset([sc])
        end = run(backend, move_all(state, {0: (24.0, 16.0), 1: (24.0, 16.8)}), 40)[-1]
        return float(np.hypot(end.x[0, 0] - end.x[0, 1], end.y[0, 0] - end.y[0, 1]))

    assert final_gap('none') == pytest.approx(0.8, abs=1e-5)
    assert final_gap('separation') > 0.8 + 0.1


def test_hard_collision_blocks_terrain_with_straight_pathing() -> None:
    """Straight pathing into a wall: center stops on passable side, never enters wall."""
    sc = Scenario(map_id=SLIT.map_id, spawns=(Spawn(1, MARINE, 8.0, 8.0),))
    backend = make(collision='hard')
    state = backend.reset([sc])
    states = run(backend, move_all(state, {0: (24.0, 8.0)}), 200)
    assert_on_passable(SLIT, states, [0])
    assert 15.99 < states[-1].x[0, 0] < 16.0


def _crowd() -> Scenario:
    spawns = tuple(Spawn(1, MARINE, 6.0 + (i % 3) * 0.5, 14.0 + (i // 3) * 0.5) for i in range(6))
    return Scenario(map_id=U_TRAP.map_id, spawns=(*spawns, Spawn(2, STALKER, 26.0, 16.0)))


def _crowd_play(backend: PyrefBackend, n_envs: int) -> list[WorldState]:
    backend.reset([_crowd()] * n_envs)
    cmd = Command.noop(n_envs, 7)
    for b in range(n_envs):
        for u in range(6):
            if b % 2 == 0:
                cmd.set(b, u, CommandKind.MOVE, target_xy=(26.0, 14.0))
            else:
                cmd.set(b, u, CommandKind.ATTACK_UNIT, target_slot=6)
    states = [backend.step(cmd, 4)]
    states += [backend.step(Command.noop(n_envs, 7), 4) for _ in range(60)]
    return states


def test_crowd_deterministic_and_batch_independent() -> None:
    """With pathing, collision, separation, and accel enabled on 6 units around the U-trap,
    repeated runs match bitwise and batched envs match single-env runs bitwise."""
    static = with_measured({'acceleration': 0.02, 'deceleration': 0.03, 'turn_rate': 0.5})
    config = {
        'pathing': 'grid',
        'collision': 'hard',
        'steering': 'separation',
        'kinematics': 'accel',
    }
    a = _crowd_play(make(2, static, **config), 2)
    b = _crowd_play(make(2, static, **config), 2)
    for sa, sb in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            np.testing.assert_array_equal(getattr(sa, name), getattr(sb, name), name)
    assert not np.array_equal(a[-1].x[0], a[-1].x[1])
    for env in range(2):
        single = make(1, static, **config)
        single.reset([_crowd()])
        cmd = Command.noop(1, 7)
        for u in range(6):
            if env == 0:
                cmd.set(0, u, CommandKind.MOVE, target_xy=(26.0, 14.0))
            else:
                cmd.set(0, u, CommandKind.ATTACK_UNIT, target_slot=6)
        alone = [single.step(cmd, 4)] + [single.step(Command.noop(1, 7), 4) for _ in range(60)]
        for sb, sa in zip(a, alone, strict=True):
            for name in WorldState.UNIT_FIELDS:
                np.testing.assert_array_equal(getattr(sb, name)[env], getattr(sa, name)[0], name)
    assert_on_passable(U_TRAP, a, list(range(6)))
    for s in a[-1:]:
        for b in range(2):
            for u in range(6):
                for t in range(u + 1, 6):
                    d = np.hypot(s.x[b, u] - s.x[b, t], s.y[b, u] - s.y[b, t])
                    assert d >= 2 * MARINE_SPEC.radius - 0.05


# ---------------------------------------------------------------- Kinematics


ACCEL = {'acceleration': 0.02, 'deceleration': 0.03, 'turn_rate': 0.5}


def test_accel_ramp_and_stop_without_overshoot() -> None:
    backend = make(static=with_measured(ACCEL), kinematics='accel')
    state = backend.reset([Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 6.0, 16.0),))])
    states = run(backend, move_all(state, {0: (16.0, 16.0)}), 200)
    xs = np.array([6.0] + [float(s.x[0, 0]) for s in states])
    v = np.diff(xs)
    assert v[0] == pytest.approx(0.02, abs=1e-6), 'first loop ramps by acceleration'
    assert np.all(np.diff(v[:6]) == pytest.approx(0.02, abs=1e-6))
    assert v.max() == pytest.approx(float(SPEED), abs=1e-6), 'top speed unchanged'
    assert np.all(xs <= 16.0 + 1e-5), 'must not overshoot goal'
    assert np.all(np.diff(v[np.argmax(v) :]) <= 0.03 + 1e-6), 'decel <= deceleration'
    end = states[-1]
    assert end.order_kind[0, 0] == CommandKind.NOOP and abs(end.x[0, 0] - 16.0) <= 0.05


def test_turn_rate_limits_facing() -> None:
    backend = make(static=with_measured(ACCEL), kinematics='accel')
    state = backend.reset([Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 16.0, 16.0),))])
    assert state.facing[0, 0] == 0.0
    states = run(backend, move_all(state, {0: (6.0, 16.0)}), 12)  # 180-deg turn
    facings = [0.0] + [float(s.facing[0, 0]) for s in states]
    for a, b in zip(facings, facings[1:], strict=False):
        assert abs((b - a + math.pi) % (2 * math.pi) - math.pi) <= 0.5 + 1e-6
    assert abs(states[0].x[0, 0] - 16.0) < 1e-6, 'turns in place when facing away'
    assert abs(abs(facings[-1]) - math.pi) < 0.02, 'turn drift shifts final angle slightly'


def test_accel_without_measured_matches_constant_speed() -> None:
    """Without measured kinematics, 'accel' matches 'constant_speed' bitwise."""
    sc = Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 6.0, 9.0), Spawn(2, MARINE, 20.0, 20.0)))
    out = []
    for kin in ('constant_speed', 'accel'):
        backend = make(kinematics=kin, pathing='grid', collision='hard')
        state = backend.reset([sc])
        cmd = move_all(state, {0: (25.0, 21.0)})
        cmd.set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
        out.append(run(backend, cmd, 150))
    for sa, sb in zip(*out, strict=True):
        for name in WorldState.UNIT_FIELDS:
            np.testing.assert_array_equal(getattr(sa, name), getattr(sb, name), name)


# Measured kinematics in assets/measured.json: Medivac accel 9/1024, decel 37/4096
# tiles/loop^2, heading_blend 0.52; facing_rate 1.0923 rad/loop (merged by static loader).
MEDIVAC_SPEC = STATIC.get(MEDIVAC)
MED_A, MED_D = 0.0087890625, 0.009033203125
MED_VMAX = np.float32(MEDIVAC_SPEC.movement_speed) / np.float32(16)  # 0.15625


def _medivac(x: float = 6.0, y: float = 16.0) -> tuple[PyrefBackend, WorldState]:
    backend = make(kinematics='accel')
    return backend, backend.reset([Scenario(map_id=FLAT, spawns=(Spawn(1, MEDIVAC, x, y),))])


def test_measured_assets_loaded() -> None:
    m = MEDIVAC_SPEC.measured
    assert (m['acceleration'], m['deceleration'], m['heading_blend']) == (MED_A, MED_D, 0.52)
    assert 'turn_rate' not in m and 'acceleration' not in MARINE_SPEC.measured
    assert MARINE_SPEC.measured['facing_rate'] == pytest.approx(1.0923)


@pytest.mark.parametrize('dist', [1.0, 4.0, 12.0])
def test_medivac_straight_matches_sc2(dist: float) -> None:
    """Medivac straight-line trajectory: linear ramp by acceleration, linear brake by
    deceleration, exact landing without overshoot, reaching vmax 0.15625 over 12 tiles."""
    backend, state = _medivac()
    states = run(backend, move_all(state, {0: (6.0 + dist, 16.0)}), 140)
    xs = np.array([6.0] + [float(s.x[0, 0]) for s in states])
    v = np.diff(xs)
    assert v[0] == pytest.approx(MED_A, abs=1e-7)
    k = int(np.argmax(v))
    assert np.allclose(np.diff(v[: k + 1])[: min(k, 16)], MED_A, atol=1e-6), 'linear accel'
    tail = np.diff(v[k:])
    moving = tail[: np.count_nonzero(v[k:]) - 2]
    assert np.allclose(moving[moving < 0], -MED_D, atol=1e-6), 'linear decel'
    assert xs.max() <= 6.0 + dist and xs[-1] == 6.0 + dist, 'exact stop at goal'
    assert states[-1].order_kind[0, 0] == CommandKind.NOOP
    if dist == 12.0:
        assert v.max() == pytest.approx(float(MED_VMAX), abs=1e-7)


def test_medivac_stop_distance_matches_sc2() -> None:
    """Full-speed STOP: decelerates each loop, stopping after 17 loops and 1.274 tiles."""
    backend, state = _medivac()
    run(backend, move_all(state, {0: (28.0, 16.0)}), 40)
    x0 = float(backend.step(Command.noop(1, 1), 1).x[0, 0])
    stop = Command.noop(1, 1).set(0, 0, CommandKind.STOP)
    xs = [x0] + [float(s.x[0, 0]) for s in run(backend, stop, 30)]
    v = np.diff(xs)
    assert np.count_nonzero(v) == 17
    assert xs[-1] - x0 == pytest.approx(1.274, abs=1e-3)


def test_medivac_heading_blend() -> None:
    """90-deg turn closes 52% of angle error per loop at vmax; 180-deg reverses instantly."""
    backend, state = _medivac()
    run(backend, move_all(state, {0: (28.0, 16.0)}), 30)
    s0 = backend.step(Command.noop(1, 1), 1)
    x0, y0 = float(s0.x[0, 0]), float(s0.y[0, 0])
    turn = move_all(s0, {0: (x0, 28.0)})
    s1 = backend.step(turn, 1)
    heading = math.atan2(float(s1.y[0, 0]) - y0, float(s1.x[0, 0]) - x0)
    assert heading == pytest.approx(0.52 * math.pi / 2, abs=1e-5)
    assert math.hypot(s1.x[0, 0] - x0, s1.y[0, 0] - y0) == pytest.approx(float(MED_VMAX), abs=1e-6)
    states = run(backend, Command.noop(1, 1), 8)
    a, b = states[-2], states[-1]
    last = math.atan2(b.y[0, 0] - a.y[0, 0], b.x[0, 0] - a.x[0, 0])
    aim = math.atan2(28.0 - a.y[0, 0], x0 - a.x[0, 0])  # +x drift shifts aim > pi/2
    assert last == pytest.approx(aim, abs=0.01), 'aligned after ~6 loops'
    # 180-deg turn: full speed in the opposite direction on the next loop.
    backend, state = _medivac()
    run(backend, move_all(state, {0: (28.0, 16.0)}), 30)
    before = backend.step(Command.noop(1, 1), 1)
    after = backend.step(move_all(before, {0: (4.0, 16.0)}), 1)
    assert after.x[0, 0] - before.x[0, 0] == pytest.approx(-float(MED_VMAX), abs=1e-6)


def test_medivac_rl_turn_rate_caps_blend() -> None:
    """Medivac_RL (variant 1) has both heading_blend and turn_rate: turns by
    min(blend * err, turn_rate) signed by err; blend-only behavior is unchanged."""
    spec = STATIC.get(MEDIVAC, 1)
    blend, cap = spec.measured['heading_blend'], spec.measured['turn_rate']
    assert _steer_heading(0.0, math.pi / 2, cap, blend) == pytest.approx(cap)
    assert _steer_heading(0.0, -math.pi / 2, cap, blend) == pytest.approx(-cap)
    assert _steer_heading(0.0, 0.3, cap, blend) == pytest.approx(blend * 0.3)
    assert _steer_heading(0.0, 3.1415, cap, blend) == 3.1415, '180-deg turn still instant'
    assert _steer_heading(0.0, 1.0, math.inf, 0.52) == pytest.approx(0.52)
    # In backend: full speed east then turn north; first loop turns by turn_rate
    backend = make(kinematics='accel')
    state = backend.reset(
        [Scenario(map_id=FLAT, spawns=(Spawn(1, MEDIVAC, 6.0, 16.0, variant=1),))]
    )
    run(backend, move_all(state, {0: (28.0, 16.0)}), 40)
    prev = backend.step(Command.noop(1, 1), 1)
    cur = backend.step(move_all(prev, {0: (float(prev.x[0, 0]), 28.0)}), 1)
    heading = math.atan2(cur.y[0, 0] - prev.y[0, 0], cur.x[0, 0] - prev.x[0, 0])
    assert heading == pytest.approx(cap, abs=1e-5)


def test_medivac_rl_turn_curve() -> None:
    """rl_turn='curve': per-loop turn interpolates measured control points (83.41 -> 16.09,
    129.92 -> 30.32, 157.46 -> 80.43 deg), signed by error without overshoot; backend
    ~90-deg turn narrows monotonically per loop. Default 'cap' is unchanged."""
    assert PyrefConfig().rl_turn == 'cap'
    with pytest.raises(ValueError):
        PyrefConfig(rl_turn='bad')
    for err, turn in ((83.41, 16.09), (129.92, 30.32), (157.46, 80.43), (30.02, 12.41)):
        assert math.degrees(rl_turn_step(math.radians(err))) == pytest.approx(turn, abs=1e-6)
        assert math.degrees(rl_turn_step(-math.radians(err))) == pytest.approx(-turn, abs=1e-6)
    assert rl_turn_step(1e-4) == pytest.approx(1e-4 * 0.49 / 0.68, rel=1e-3)
    spec = STATIC.get(MEDIVAC, 1)
    blend, cap = spec.measured['heading_blend'], spec.measured['turn_rate']
    assert _steer_heading(0.0, 3.1415, cap, blend, True) == 3.1415, '180-deg instant'
    backend = make(kinematics='accel', rl_turn='curve')
    state = backend.reset(
        [Scenario(map_id=FLAT, spawns=(Spawn(1, MEDIVAC, 6.0, 16.0, variant=1),))]
    )
    run(backend, move_all(state, {0: (28.0, 16.0)}), 40)
    prev = backend.step(Command.noop(1, 1), 1)
    states = [prev, *run(backend, move_all(prev, {0: (float(prev.x[0, 0]), 28.0)}), 5)]
    hs = [
        math.degrees(math.atan2(b.y[0, 0] - a.y[0, 0], b.x[0, 0] - a.x[0, 0]))
        for a, b in zip(states, states[1:], strict=False)
    ]
    turns = [b - a for a, b in zip([0.0, *hs], hs, strict=False)]
    # Step 1 error ~92 deg (16-18 deg on curve), decreasing monotonically afterward
    assert 16.0 < turns[0] < 18.0, turns
    assert all(a > b > 12.0 for a, b in zip(turns, turns[1:], strict=False)), turns


def test_facing_rate_only_affects_facing() -> None:
    """Marine 180-deg turn: motion reverses immediately; facing turns in 3 loops."""
    backend = make()
    state = backend.reset([Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 16.0, 16.0),))])
    states = run(backend, move_all(state, {0: (6.0, 16.0)}), 4)
    assert float(states[0].x[0, 0]) == pytest.approx(16.0 - float(SPEED), abs=1e-6), 'flip'
    assert abs(float(states[0].facing[0, 0])) == pytest.approx(1.0923, abs=1e-5)
    assert abs(float(states[1].facing[0, 0])) == pytest.approx(2 * 1.0923, abs=1e-5)
    for s in states[2:]:
        assert abs(abs(float(s.facing[0, 0])) - math.pi) < 1e-6


def test_exact_landing_with_zero_arrival_radius() -> None:
    """Default arrival_radius=0 lands cleanly on diagonal goals and finishes the order."""
    assert PyrefConfig().arrival_radius == 0.0
    configs: list[dict[str, object]] = [
        {},
        {'pathing': 'grid', 'collision': 'hard'},
        {'kinematics': 'accel'},
    ]
    for config in configs:
        backend = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]
        state = backend.reset([Scenario(map_id=FLAT, spawns=(Spawn(1, MARINE, 6.0, 9.0),))])
        end = run(backend, move_all(state, {0: (25.3, 21.7)}), 200)[-1]
        assert (end.x[0, 0], end.y[0, 0]) == (np.float32(25.3), np.float32(21.7)), config
        assert end.order_kind[0, 0] == CommandKind.NOOP


# ---------------------------------------------------------------- Auto-acquire


def _duel(dx: float, a: int = MARINE, b: int = MARINE) -> Scenario:
    return Scenario(map_id=FLAT, spawns=(Spawn(1, a, 10.0, 16.0), Spawn(2, b, 10.0 + dx, 16.0)))


def test_auto_acquire_within_margin_chases() -> None:
    """Edge dist 5.3 (within range 5 + margin 0.5): idle units engage, close in, and fire."""
    backend = make(auto_acquire=True)
    backend.reset([_duel(5.3 + 0.75)])
    states = run(backend, Command.noop(1, 2), 30)
    assert states[0].engaged_target_slot[0, 0] == 1 and states[0].engaged_target_slot[0, 1] == 0
    assert states[0].x[0, 0] > 10.0, 'closes in when out of range'
    assert states[-1].health[0, 1] < 45 and states[-1].health[0, 0] < 45


def test_auto_acquire_outside_margin_stays_idle() -> None:
    backend = make(auto_acquire=True)
    start = backend.reset([_duel(5.6 + 0.75)])
    end = run(backend, Command.noop(1, 2), 30)[-1]
    assert np.all(end.engaged_target_slot[0] == NO_SLOT)
    assert np.array_equal(end.x, start.x) and np.all(end.health[0] == 45)


@pytest.mark.parametrize('auto', [False, True])
def test_retaliation(auto: bool) -> None:
    """Stalker (range 6) fires at idle Marine from outside acquire range: Marine retaliates
    and chases when auto_acquire=True, otherwise ignores it. Moving units do not."""
    dx = 6.0 + 0.375 + 0.625  # exact edge distance 6
    backend = make(auto_acquire=auto)
    backend.reset([_duel(dx, MARINE, STALKER)])
    cmd = Command.noop(1, 2).set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    states = run(backend, cmd, 60)
    end = states[-1]
    assert end.health[0, 0] < 45
    if auto:
        assert end.engaged_target_slot[0, 0] == 1
        assert end.x[0, 0] > 10.0 and end.health[0, 1] + end.shield[0, 1] < 160
    else:
        assert end.engaged_target_slot[0, 0] == NO_SLOT and end.x[0, 0] == 10.0
        assert end.health[0, 1] + end.shield[0, 1] == 160
    # Moving unit does not retaliate when hit.
    backend.reset([_duel(dx, MARINE, STALKER)])
    cmd = Command.noop(1, 2).set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(10.0, 26.0))
    states = run(backend, cmd, 40)
    assert all(s.engaged_target_slot[0, 0] == NO_SLOT for s in states)


def test_hold_auto_acquire_does_not_move() -> None:
    backend = make(auto_acquire=True)
    start = backend.reset([_duel(5.3 + 0.75)])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.HOLD).set(0, 1, CommandKind.HOLD)
    end = run(backend, cmd, 30)[-1]
    assert np.array_equal(end.x, start.x) and np.all(end.health[0] == 45), 'HOLD in-range'
