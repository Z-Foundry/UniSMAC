"""Tests for zgame navmesh geometry, pathing ``navmesh``, steering ``sc2_avoid`` and avoidance
options, splash, suicide, Barrier, and full v4/axo configs compared step-by-step with pyref.

Same tolerance policy as test_zgame.py: in-process x, y, order_x, order_y diff <= 1e-4, facing
angular diff <= 1e-3, all other fields bitwise-identical; fully bitwise-identical in a subprocess
with NumPy AVX-512 disabled.
"""

from __future__ import annotations

import dataclasses
import math
import platform
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

pytest.importorskip('zgame', reason="requires the optional 'zgame' extension")
from tests._support.simd import run_without_avx512
from tests._support.zgame_helpers import profile_subset_options as _profile_subset_options
from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref import navmesh as nm
from unismac.backends.zgame import ZgameBackend, capabilities_for
from unismac.backends.zgame.backend import _module
from unismac.contract import Command, CommandKind, MapInfo, Scenario, Spawn, WorldState

_zgame = _module()

MARINE, MARAUDER, MEDIVAC, ZEALOT, STALKER = 48, 51, 54, 73, 74
ALL_TYPES = (MARINE, MARAUDER, ZEALOT, STALKER, MEDIVAC)
STATIC = load_static_data()
FLAT_ID = 'SMAC_Maps/32x32_flat'
CORRIDOR_ID = 'SMAC_Maps/corridor'
FLAT = {FLAT_ID: load_map_info(FLAT_ID)}
CORRIDOR = {CORRIDOR_ID: load_map_info(CORRIDOR_ID)}
# In-process tolerances
POS_ATOL = 1e-4
FACING_ATOL = 1e-3
POS_FIELDS = ('x', 'y', 'order_x', 'order_y')

# ---------------------------------------------------------------- Navmesh geometry


def _random_blocks(seed: int) -> np.ndarray:
    """Place random rectangular obstacles on a 32x32 flat grid (adjacent, edge, 1-cell gaps)."""
    rng = np.random.default_rng(seed)
    g = np.ones((32, 32), dtype=bool)
    for _ in range(6):
        x, y = rng.integers(2, 26, 2)
        w, h = rng.integers(1, 6, 2)
        g[y : y + h, x : x + w] = False
    g[0:3, 0:2] = False  # obstacle touching map border
    return g


def _block_footprint(
    grid: np.ndarray, obstacles: tuple[tuple[str, float, float], ...]
) -> np.ndarray:
    """Apply square building footprints (pylon=2x2, barracks=3x3, cc=5x5) onto a copy of grid."""
    sizes = {'pylon': 2, 'barracks': 3, 'commandcenter': 5}
    out = np.array(grid, copy=True)
    h, w = out.shape
    for kind, cx, cy in obstacles:
        half = sizes[kind] / 2.0
        x0, x1 = max(0, int(round(cx - half))), min(w, int(round(cx + half)))
        y0, y1 = max(0, int(round(cy - half))), min(h, int(round(cy + half)))
        out[y0:y1, x0:x1] = False
    return out


def _nav_grids() -> dict[str, np.ndarray]:
    flat32 = load_map_info('SMAC_Maps/32x32_flat').pathing
    gap_obs = (
        ('pylon', 16.0, 15.0),
        ('pylon', 16.0, 18.0),
        ('commandcenter', 16.5, 11.5),
        ('commandcenter', 16.5, 6.5),
        ('pylon', 16.0, 3.0),
        ('commandcenter', 16.5, 21.5),
        ('barracks', 16.5, 25.5),
        ('barracks', 16.5, 28.5),
    )
    utrap_obs = (
        ('barracks', 15.5, 11.5),
        ('barracks', 15.5, 14.5),
        ('barracks', 15.5, 17.5),
        ('barracks', 15.5, 20.5),
        ('barracks', 18.5, 11.5),
        ('barracks', 21.5, 11.5),
        ('barracks', 18.5, 20.5),
        ('barracks', 21.5, 20.5),
    )
    return {
        'flat': load_map_info('SMAC_Maps/8m').pathing,
        'corridor': load_map_info('SMAC_Maps/corridor').pathing,
        'random': _random_blocks(3),
        'gap': _block_footprint(flat32, gap_obs),
        'utrap': _block_footprint(flat32, utrap_obs),
    }


NAV_GRIDS = _nav_grids()


def _nav_queries(rng: np.random.Generator, grid: np.ndarray, mesh: nm.NavMesh) -> np.ndarray:
    """Random (start, goal) pairs: inside/outside map, from nodes, into/along obstacles."""
    h, w = grid.shape
    k = 40
    q = rng.uniform(-0.5, [w + 0.5, h + 0.5, w + 0.5, h + 0.5], (k, 4))
    if mesh.nodes.shape[0]:
        i = rng.integers(0, mesh.nodes.shape[0], 12)
        q = np.vstack([q, np.column_stack([mesh.nodes[i], rng.uniform(0, [w, h], (12, 2))])])
    # Coordinates originate from float32 state
    return q.astype(np.float32).astype(np.float64)


def _same(a: object, b: object) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize('radius', [0.375, 0.625, 0.75, 1.0])
@pytest.mark.parametrize('name', sorted(NAV_GRIDS))
def test_navmesh_geometry_matches_pyref(name: str, radius: float) -> None:
    """Obstacle edges/corners, nodes, visibility graph, LOS checks, ray ends, and routes match
    pyref bitwise across corridor, obstacle maps, random blocks, and flat grids."""
    if not hasattr(_zgame, 'navmesh_debug'):
        pytest.skip('requires zgame built with the debug-hooks Cargo feature')
    grid = NAV_GRIDS[name]
    rho = nm.path_radius(radius)
    mesh = nm.build(grid, rho)
    edges, corners = nm.obstacle_edges(grid)
    rng = np.random.default_rng(int(radius * 8))
    q = _nav_queries(rng, grid, mesh)
    z = _zgame.navmesh_debug(grid, rho, q)
    assert _same(z['edges'], edges)
    assert _same(z['corners'], np.asarray(corners, dtype=np.float64).reshape(-1, 4))
    assert _same(z['nodes'], mesh.nodes)
    assert _same(z['adj'], mesh.adj)
    vis = [bool(nm.visible_from(mesh, r[0], r[1], r[None, 2:])[0]) for r in q]
    assert _same(z['visible'], vis)
    assert _same(z['ray'], [nm.ray_end(mesh, *r) for r in q])
    route = [[*wp, *end] for wp, end in (nm.route(mesh, *r) for r in q)]
    assert _same(z['route'], route)
    if name == 'corridor':
        # Route actually detours via nodes (not all directly visible)
        assert not all(vis)


# ---------------------------------------------------------------- Step-by-step fixtures


def skirmish(
    rng: np.random.Generator,
    maps: Mapping[str, MapInfo],
    n: int,
    types: tuple[int, ...],
    box: float | None = None,
) -> Scenario:
    """Spawn n random units per side on pathable cells (Player 1 left, Player 2 right).
    When box is given, pack units in a central box x box region (offset +-2 cells per side)
    to frequently trigger avoidance and pinned-unit rules."""
    ((map_id, info),) = maps.items()
    cells = np.argwhere(info.pathing)
    if box is not None:
        h, w = info.pathing.shape
        near = (np.abs(cells[:, 1] + 0.5 - w / 2) < box / 2 + 2) & (
            np.abs(cells[:, 0] + 0.5 - h / 2) < box / 2
        )
        cells = cells[near]
    split = 16 if box is None else info.pathing.shape[1] // 2
    spawns = []
    for owner in (1, 2):
        side = cells[cells[:, 1] < split] if owner == 1 else cells[cells[:, 1] >= split]
        for _ in range(n):
            iy, ix = side[rng.integers(len(side))]
            x, y = float(ix + rng.random()), float(iy + rng.random())
            spawns.append(Spawn(owner, int(rng.choice(types)), x, y))
    return Scenario(map_id=map_id, spawns=tuple(spawns))  # fmt: skip


def random_command(rng: np.random.Generator, world: WorldState, p_act: float = 0.3) -> Command:
    """Issue random commands (STOP, MOVE, ATTACK_UNIT, ATTACK_MOVE, HOLD) with prob p_act;
    targets may lie outside the playable area or reference allies/dead units."""
    b_size, n = world.alive.shape
    cmd = Command.noop(b_size, n)
    for b in range(b_size):
        for u in range(n):
            if not world.alive[b, u] or rng.random() > p_act:
                continue
            kind = CommandKind(int(rng.choice([1, 2, 3, 4, 5, 4, 3])))
            if kind == CommandKind.ATTACK_UNIT:
                t = int(rng.integers(n - 1))
                cmd.set(b, u, kind, target_slot=t + (t >= u))
            elif kind in (CommandKind.MOVE, CommandKind.ATTACK_MOVE):
                xy = (float(rng.uniform(-2.0, 34.0)), float(rng.uniform(-2.0, 34.0)))
                cmd.set(b, u, kind, target_xy=xy)
            else:
                cmd.set(b, u, kind)
    return cmd


def assert_close(a: WorldState, b: WorldState, *, exact: bool = False) -> None:
    """Bitwise-equal if exact; else POS_FIELDS <= POS_ATOL, facing <= FACING_ATOL, rest bitwise."""
    for f in dataclasses.fields(WorldState):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f.name
        if not exact and f.name in POS_FIELDS:
            np.testing.assert_allclose(x, y, rtol=0, atol=POS_ATOL, equal_nan=True, err_msg=f.name)
        elif not exact and f.name == 'facing':
            d = np.remainder(x.astype(np.float64) - y + math.pi, 2 * math.pi) - math.pi
            assert np.nanmax(np.abs(d), initial=0.0) <= FACING_ATOL, f.name
        elif x.dtype.kind == 'f':
            assert np.array_equal(x, y, equal_nan=True), f.name
        else:
            assert np.array_equal(x, y), f.name


# Avoidance option groups (PyrefConfig) and steering for v4 and axo
AVOID_V4: dict[str, Any] = {'steering': 'sc2_avoid', 'avoid_side_check': True}
AVOID_AXO: dict[str, Any] = AVOID_V4 | {
    'pin_slide': True,
    'avoid_share': 'speed',
    'avoid_share_power': 4,
    'pin_sidestep': True,
    'pin_sidestep_drift': True,
    'pin_tie_grid': True,
    'avoid_parallel_faster_only': True,
    'avoid_cross_cos': 0.8,
    'contact_orbit': True,
}
HARD: dict[str, Any] = {'collision': 'hard', 'contact_slack': 0.0625, 'pin_engaged': True}
# Non-default avoidance parameters (unused by v4/axo, but supported)
AVOID_PARAMS: dict[str, Any] = AVOID_AXO | {
    'avoid_range': 2.0,
    'avoid_margin': 0.1,
    'avoid_horizon': 8.0,
    'idle_nudge': 0.3,
    'avoid_side_gap': 0.5,
    'follow_gap': 0.2,
    'avoid_share_power': 2.0,
    'pin_gap_closed': 0.7,
    'orbit_hold': 4,
    'orbit_iters': 6,
    'orbit_max_turn': 3.0,
    'pin_sidestep_cone': 60.0,
    'pin_sidestep_speed': 0.5,
}


def _full(profile: str, **extra: Any) -> PyrefConfig:
    return PyrefConfig(**(_profile_subset_options(profile) | extra))


SKIRMISH = (MARINE, MARAUDER, ZEALOT, STALKER, MARINE)  # fewer Medivacs (air skips ground avoid)
# name -> (config_fn, maps, unit_types, n_per_side, box). Callable config for subprocess rebuild.
M3_CASES: dict[
    str,
    tuple[Callable[[], PyrefConfig], dict[str, MapInfo], tuple[int, ...], int, float | None],
] = {
    'navmesh': (lambda: PyrefConfig(pathing='navmesh'), CORRIDOR, ALL_TYPES, 4, None),
    'navmesh-hard': (
        lambda: PyrefConfig(pathing='navmesh', collision='hard', kinematics='accel'),
        CORRIDOR,
        ALL_TYPES,
        4,
        None,
    ),
    'avoid': (lambda: PyrefConfig(steering='sc2_avoid'), FLAT, SKIRMISH, 6, 6.0),
    'avoid-v4': (lambda: PyrefConfig(**AVOID_V4, **HARD), FLAT, SKIRMISH, 6, 6.0),
    'avoid-axo': (
        lambda: PyrefConfig(**AVOID_AXO, **HARD, kinematics='accel', mover_push_share='approach'),
        FLAT,
        SKIRMISH,
        6,
        6.0,
    ),
    'avoid-params': (lambda: PyrefConfig(**AVOID_PARAMS, **HARD), FLAT, SKIRMISH, 6, 6.0),
    'v4': (lambda: _full('v4'), FLAT, ALL_TYPES, 6, 6.0),
    'v4-corridor': (lambda: _full('v4'), CORRIDOR, ALL_TYPES, 6, 8.0),
    'axo': (lambda: _full('axo'), FLAT, ALL_TYPES, 6, 6.0),
    'axo-corridor': (lambda: _full('axo'), CORRIDOR, ALL_TYPES, 6, 8.0),
}


def _compare_m3(name: str, *, exact: bool, steps: int = 40, n_loops: int = 4) -> None:
    """M3_CASES[name]: 3-env random skirmish compared step-by-step under random commands."""
    make, maps, types, n, box = M3_CASES[name]
    config = make()
    rng = np.random.default_rng(sorted(M3_CASES).index(name))
    b_size = 3
    scenarios = [skirmish(rng, maps, n, types, box) for _ in range(b_size)]
    pyref = PyrefBackend(STATIC, maps, b_size, config)
    zg = ZgameBackend(STATIC, maps, b_size, config)
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    for _ in range(steps):
        assert_close(wp, wz, exact=exact)
        cmd = random_command(rng, wp)
        wp, wz = pyref.step(cmd, n_loops), zg.step(cmd, n_loops)
    assert_close(wp, wz, exact=exact)


@pytest.mark.parametrize('name', sorted(M3_CASES))
def test_matches_pyref_m3(name: str) -> None:
    """Compare navmesh, avoidance, and full-profile options step-by-step with pyref."""
    _compare_m3(name, exact=False)


# pin_tie_grid tie-breaking: a row of Player 2 Marines engaged/immobile (unpushable),
# approached head-on by a Player 1 Marine. Symmetric walls or HOLD pick right via
# _pin_tie_side; skewed walls use _grid_right.
TIE_WALLS = {
    'mirrored': ((16.0, 14.25), (16.0, 15.0), (16.0, 15.75)),
    'skewed': ((16.0, 14.25), (16.0, 15.0), (16.0, 15.75), (16.75, 15.75)),
}


@pytest.mark.parametrize('wall', sorted(TIE_WALLS))
@pytest.mark.parametrize('hold', [False, True])
def test_pin_tie_grid_matches_pyref(wall: str, hold: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """pin_tie_grid tie-breaking matches pyref bitwise (facing within FACING_ATOL), verifying
    that pyref actually executes _pin_tie_side (and _grid_right when skewed and not HOLD)."""
    from unismac.backends.pyref.subsystems import steering as pst

    hits = {'_pin_tie_side': 0, '_grid_right': 0}
    for fn_name in hits:
        fn = getattr(pst, fn_name)

        def counted(*a: object, _fn: Callable[..., bool] = fn, _k: str = fn_name) -> bool:
            hits[_k] += 1
            return _fn(*a)

        monkeypatch.setattr(pst, fn_name, counted)
    spots = TIE_WALLS[wall]
    spawns = [Spawn(2, MARINE, x, y) for x, y in spots]
    # Engaged target: durable Stalker; approaching Marine aligned with middle wall unit (y = 15)
    spawns += [Spawn(1, STALKER, 19.5, 15.0), Spawn(1, MARINE, 13.0, 15.0)]
    scenario = Scenario(map_id=FLAT_ID, spawns=tuple(spawns))
    config = PyrefConfig(**AVOID_V4, **HARD, pin_tie_grid=True)
    pyref = PyrefBackend(STATIC, FLAT, 1, config)
    zg = ZgameBackend(STATIC, FLAT, 1, config)
    wp, wz = pyref.reset([scenario], seeds=[0]), zg.reset([scenario], seeds=[0])
    n = len(spawns)
    cmd = Command.noop(1, n)
    for u in range(len(spots)):
        if hold:
            cmd.set(0, u, CommandKind.HOLD)
        else:
            cmd.set(0, u, CommandKind.ATTACK_UNIT, target_slot=n - 2)
    cmd.set(0, n - 2, CommandKind.HOLD)
    cmd.set(0, n - 1, CommandKind.MOVE, target_xy=(24.0, 15.0))
    for _ in range(40):
        wp, wz = pyref.step(cmd, 1), zg.step(cmd, 1)
        assert_close(wp, wz)
        assert np.array_equal(wp.x, wz.x) and np.array_equal(wp.y, wz.y)
        cmd = Command.noop(1, n)
    assert hits['_pin_tie_side'] > 0
    if wall == 'skewed' and not hold:
        assert hits['_grid_right'] > 0


# ---------------------------------------------------------------- Splash, suicide, Barrier

COLOSSUS, BANELING, SIEGE_SIEGED, HELLION, IMMORTAL, ZERGLING = 4, 9, 32, 53, 83, 105
BUNKER, SPINE = 24, 98
P5B_TYPES = (COLOSSUS, BANELING, BANELING, SIEGE_SIEGED, HELLION, IMMORTAL, MARINE, ZERGLING)
SC2_DAMAGE: dict[str, Any] = {'attack': 'sc2', 'damage': 'sc2'}
# name -> (config_fn, maps, unit_types, n_per_side, box); same layout as M3_CASES
P5B_CASES: dict[
    str,
    tuple[Callable[[], PyrefConfig], dict[str, MapInfo], tuple[int, ...], int, float | None],
] = {
    'sc2': (lambda: PyrefConfig(**SC2_DAMAGE), FLAT, P5B_TYPES, 6, 8.0),
    'sc2-commit': (
        lambda: PyrefConfig(**SC2_DAMAGE, attack_commit=True, slow_arrival=True),
        FLAT,
        P5B_TYPES,
        6,
        8.0,
    ),
    'v4': (lambda: _full('v4'), FLAT, P5B_TYPES, 6, 8.0),
    'axo': (lambda: _full('axo'), FLAT, P5B_TYPES, 6, 8.0),
    # Suicide unit vs structures (armor-ignoring suicide_damage_structure and trigger distance)
    'structure': (
        lambda: _full('v4'),
        FLAT,
        (BANELING, BANELING, BUNKER, SPINE, ZERGLING, MARINE),
        6,
        8.0,
    ),
}


@contextmanager
def _pyref_events() -> Iterator[dict[str, list[tuple]]]:
    """Wrap pyref damage.sc2 settlement helpers (without modifying pyref) to record hits
    (env, loop, attacker, target, shield_loss, hp_loss, lethal), fires (env, loop, attacker,
    target), and suicide bursts (env, loop, unit). _strike/_barrier_hit inherit the active
    attacker from _sc2_hit/_splash/_burst; nested _barrier_hit inside _strike is not double-
    counted. Also counts splash victims and Barrier absorptions to verify branch coverage."""
    from unismac.backends.pyref.subsystems import damage as dmg

    keys = ('hits', 'fires', 'bursts', 'burst_on', 'splash', 'absorb')
    rec: dict[str, list[tuple]] = {k: [] for k in keys}
    who: list[int] = []
    names = ('_hit', '_strike', '_barrier_hit', '_sc2_hit', '_splash', '_burst', '_fire')
    orig = {k: getattr(dmg, k) for k in names}

    def settle(sim, b, u, t, fn, *a):  # noqa: ANN001, ANN202
        w = sim.world
        s0, h0 = w.shield[b, t], w.health[b, t]
        fn(*a)
        s1, h1 = w.shield[b, t], w.health[b, t]
        lp = int(w.game_loop[b])
        rec['hits'].append((b, lp, u, t, float(s0 - s1), float(h0 - h1), bool(h0 > 0 and h1 <= 0)))

    def hit(sim, b, u, t, wi, n, min_damage):  # noqa: ANN001, ANN202
        settle(sim, b, u, t, orig['_hit'], sim, b, u, t, wi, n, min_damage)

    def strike(sim, b, t, *a):  # noqa: ANN001, ANN202
        who.append(-2)  # skip nested _barrier_hit recording
        try:
            settle(sim, b, who[-2], t, orig['_strike'], sim, b, t, *a)
        finally:
            who.pop()

    def barrier_hit(sim, b, t, *a):  # noqa: ANN001, ANN202
        left = float(sim.combat.barrier_left[b, t])
        if who[-1] == -2:
            orig['_barrier_hit'](sim, b, t, *a)
        else:
            settle(sim, b, who[-1], t, orig['_barrier_hit'], sim, b, t, *a)
        if float(sim.combat.barrier_left[b, t]) < left:
            rec['absorb'].append((b, t))

    def scoped(name: str):  # noqa: ANN202
        def f(sim, b, u, *a, **k):  # noqa: ANN001, ANN202
            who.append(u)
            try:
                h0 = sim.world.health[b].copy()
                orig[name](sim, b, u, *a, **k)
                if name == '_splash':
                    hit_ = (sim.world.health[b] < h0) & (np.arange(h0.size) != a[0])
                    rec['splash'].append((int(sim.world.unit_type[b, u]), int(hit_.sum())))
                if name == '_burst':
                    rec['bursts'].append((b, int(sim.world.game_loop[b]), u))
                    rec['burst_on'].append((bool(sim.params.structure[b, a[0]]),))
            finally:
                who.pop()

        return f

    def fire(sim, b, u, t, loop):  # noqa: ANN001, ANN202
        rec['fires'].append((b, loop, u, t))
        orig['_fire'](sim, b, u, t, loop)

    patched = {
        '_hit': hit,
        '_strike': strike,
        '_barrier_hit': barrier_hit,
        '_sc2_hit': scoped('_sc2_hit'),
        '_splash': scoped('_splash'),
        '_burst': scoped('_burst'),
        '_fire': fire,
    }
    for k, f in patched.items():
        setattr(dmg, k, f)
    try:
        yield rec
    finally:
        for k, f in orig.items():
            setattr(dmg, k, f)


def _check_events(ev: dict[str, np.ndarray], rec: dict[str, list[tuple]], b_size: int) -> None:
    """Verify zgame damage, death (with killer), and fire events match pyref per step.
    Suicide units have killer -1 unless dealt lethal damage by another unit on the same loop."""
    kind = ev['kind']
    for b in range(b_size):
        m = ev['env'] == b

        def col(k: int, *names: str, m: np.ndarray = m) -> list[tuple]:
            sel = m & (kind == k)
            cols = [ev[n][sel] for n in names]
            return [tuple(x.item() for x in row) for row in zip(*cols, strict=True)]

        dmg = col(2, 'game_loop', 'other', 'unit', 'f0', 'f1')
        want = [(lp, u, t, s, h) for bb, lp, u, t, s, h, _ in rec['hits'] if bb == b]
        assert dmg == want, f'env {b} damage events'
        killers: dict[tuple[int, int], int] = {}
        for bb, lp, u, t, _, _, killed in rec['hits']:
            if bb == b and killed:
                killers.setdefault((lp, t), u)
        for bb, lp, u in rec['bursts']:
            if bb == b:
                killers.setdefault((lp, u), -1)
        deaths = col(3, 'game_loop', 'unit', 'other')
        assert deaths == sorted((lp, t, u) for (lp, t), u in killers.items()), f'env {b} deaths'
        fired = col(1, 'game_loop', 'unit', 'other')
        assert fired == [(lp, u, t) for bb, lp, u, t in rec['fires'] if bb == b], f'env {b} fires'


def _compare_p5b(name: str, *, exact: bool, steps: int = 50, n_loops: int = 2) -> dict[str, int]:
    """P5B_CASES[name]: 3-env skirmish opening with ATTACK_MOVE then random commands (p=0.1);
    compares state and event stream each step and returns pyref splash, burst, structure-burst,
    and Barrier absorption counts."""
    make, maps, types, n, box = P5B_CASES[name]
    config = make()
    rng = np.random.default_rng(200 + sorted(P5B_CASES).index(name))
    b_size = 3
    scenarios = [skirmish(rng, maps, n, types, box) for _ in range(b_size)]
    pyref = PyrefBackend(STATIC, maps, b_size, config)
    zg = ZgameBackend(STATIC, maps, b_size, config)
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    cmd = Command.noop(b_size, 2 * n)
    for b in range(b_size):
        for u in range(2 * n):
            cmd.set(b, u, CommandKind.ATTACK_MOVE, target_xy=(10.0 if u >= n else 22.0, 16.0))
    counts: dict[str, int] = {'bursts': 0, 'on_structure': 0, 'absorb': 0}
    with _pyref_events() as rec:
        for _ in range(steps):
            assert_close(wp, wz, exact=exact)
            for v in rec.values():
                v.clear()
            wp, wz = pyref.step(cmd, n_loops), zg.step(cmd, n_loops)
            _check_events(zg.events(), rec, b_size)
            for ut, k in rec['splash']:
                counts[f'splash-{ut}'] = counts.get(f'splash-{ut}', 0) + k
            counts['bursts'] += len(rec['bursts'])
            counts['on_structure'] += sum(on for (on,) in rec['burst_on'])
            counts['absorb'] += len(rec['absorb'])
            cmd = random_command(rng, wp, 0.1)
    assert_close(wp, wz, exact=exact)
    return counts


# Required pyref branches per scenario (_compare_p5b counts > 0), covering radial splash
# (Baneling, Siege Tank), perp line (Colossus), forward line (Hellion), suicide, and Barrier.
P5B_COVER = {
    'sc2': (
        'bursts',
        'absorb',
        f'splash-{BANELING}',
        f'splash-{COLOSSUS}',
        f'splash-{SIEGE_SIEGED}',
    ),
    'sc2-commit': ('bursts', 'absorb', f'splash-{HELLION}', f'splash-{COLOSSUS}'),
    'v4': ('bursts', 'absorb', f'splash-{COLOSSUS}', f'splash-{SIEGE_SIEGED}'),
    'axo': ('bursts', f'splash-{BANELING}', f'splash-{SIEGE_SIEGED}'),
    'structure': ('on_structure', f'splash-{BANELING}'),
}


@pytest.mark.parametrize('name', sorted(P5B_CASES))
def test_splash_suicide_barrier_match_pyref(name: str) -> None:
    """Splash (radial, perp-line, forward-line), suicide (incl. structures), and Barrier match
    pyref state and events step-by-step; verifies all branches in P5B_COVER are hit."""
    counts = _compare_p5b(name, exact=False)
    assert all(counts.get(k, 0) > 0 for k in P5B_COVER[name]), counts


def test_suicide_constants_match_pyref() -> None:
    """Default suicide-vs-structure trigger distance in zgame matches pyref."""
    from unismac.backends.pyref import sim
    from unismac.backends.zgame import backend

    assert backend.SUICIDE_RANGE_STRUCTURE == sim.SUICIDE_RANGE_STRUCTURE


@pytest.mark.parity
@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_m3_bitwise_without_avx512() -> None:
    """With NumPy AVX-512 disabled (set in a subprocess before import), all M3_CASES and
    P5B_CASES match pyref bitwise."""
    body = (
        'for n in sorted(m["M3_CASES"]):\n'
        '    m["_compare_m3"](n, exact=True)\n'
        'for n in sorted(m["P5B_CASES"]):\n'
        '    m["_compare_p5b"](n, exact=True)\n'
    )
    run_without_avx512(__file__, body)


# ---------------------------------------------------------------- Thread & batch invariance


def _zg_trajectory(
    name: str, scenarios: list[Scenario], seeds: list[int], commands: list[Command], threads: int
) -> list[WorldState]:
    make, maps, *_ = M3_CASES[name]
    zg = ZgameBackend(STATIC, maps, len(scenarios), make(), threads=threads, min_parallel=1)
    out = [zg.reset(scenarios, seeds=seeds)]
    out += [zg.step(cmd, 4) for cmd in commands]
    return out


def _rows(x: Any, i: int) -> Any:
    """Slice env i from WorldState / Command while keeping the batch dimension."""
    return type(x)(**{f.name: getattr(x, f.name)[i : i + 1] for f in dataclasses.fields(x)})


@pytest.mark.parametrize('name', ['v4-corridor', 'axo'])
def test_r1_threads_and_batch_size_m3(name: str) -> None:
    """Full v4/axo profiles match bitwise across 1/2/4 threads and batched vs single-env
    stepping (navmesh caches are per-env; only read-only grids are shared)."""
    make, maps, types, n, box = M3_CASES[name]
    rng = np.random.default_rng(21)
    b_size = 5
    scenarios = [skirmish(rng, maps, n, types, box) for _ in range(b_size)]
    seeds = list(range(b_size))
    zg = ZgameBackend(STATIC, maps, b_size, make())
    world = zg.reset(scenarios, seeds=seeds)
    commands = []
    for _ in range(25):
        commands.append(random_command(rng, world))
        world = zg.step(commands[-1], 4)
    base = _zg_trajectory(name, scenarios, seeds, commands, 1)
    assert_close(base[-1], world, exact=True)
    for threads in (2, 4):
        traj = _zg_trajectory(name, scenarios, seeds, commands, threads)
        for a, b in zip(base, traj, strict=True):
            assert_close(a, b, exact=True)
    for i in range(b_size):
        single = [_rows(c, i) for c in commands]
        one = _zg_trajectory(name, [scenarios[i]], [seeds[i]], single, 1)
        for a, b in zip(base, one, strict=True):
            assert_close(_rows(a, i), b, exact=True)


def test_navmesh_capability() -> None:
    """pathing='navmesh' reports pathing='grid' capability (matching pyref)."""
    assert capabilities_for({'pathing': 'navmesh'}).pathing == 'grid'
    config = PyrefConfig(pathing='navmesh')
    zg = ZgameBackend(STATIC, CORRIDOR, 1, config)
    ref = PyrefBackend(STATIC, CORRIDOR, 1, config).capabilities
    assert (zg.capabilities.pathing, zg.capabilities.collision) == (ref.pathing, ref.collision)
