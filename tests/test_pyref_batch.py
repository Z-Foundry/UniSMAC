"""Bit-exact parity between pyref batched/vectorized routines and per-unit reference paths.

Compares vectorized vs per-unit implementations on randomized crowded states for targeting,
collision sweeps (batched rounds and exact pairwise fallback), sc2_avoid, and navmesh LOS.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig, navmesh
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import attack, collision, steering, targeting
from unismac.backends.pyref.subsystems.pathing import _mesh
from unismac.contract import NO_SLOT, CommandKind, Scenario
from unismac.scenarios import spawn_record

STATIC = load_static_data()


def _sim(name: str, batch: int, seed: int, **config: object) -> Sim:
    """Pack spawns from `name` into a small area with random orders, waypoints, and motion."""
    rec = spawn_record(name)
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    be = PyrefBackend(STATIC, {rec.map_id: load_map_info(rec.map_id)}, batch, PyrefConfig(**config))  # type: ignore[arg-type]
    be.reset([sc] * batch, seeds=list(range(batch)))
    sim = be._sim
    assert sim is not None
    rng = np.random.default_rng(seed)
    w, buf, m = sim.world, sim.buf, sim.motion
    shape = w.x.shape
    cx, cy = np.nanmean(w.x), np.nanmean(w.y)
    span = 0.6 * np.sqrt(shape[1])
    w.x[:] = (cx + rng.uniform(-span, span, shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-span, span, shape)).astype(np.float32)
    w.x[:, :2] = w.x[:, 2:4]  # coincident-center pairs
    w.y[:, :2] = w.y[:, 2:4]
    w.alive[rng.random(shape) < 0.1] = False
    moving = rng.random(shape) < 0.6
    ang = rng.uniform(-np.pi, np.pi, shape)
    buf.waypoint_x[:] = np.where(moving, w.x + 3 * np.cos(ang), np.nan).astype(np.float32)
    buf.waypoint_y[:] = np.where(moving, w.y + 3 * np.sin(ang), np.nan).astype(np.float32)
    m.speed[:] = np.where(rng.random(shape) < 0.5, 0, sim.params.speed).astype(np.float32)
    m.heading[:] = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    w.game_loop[:] = rng.integers(0, 6, batch)
    return sim


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('owners', [None, (2,)])
def test_targeting_matches_per_unit(seed: int, owners: tuple[int, ...] | None) -> None:
    sim = _sim('MMM2', 4, seed, auto_acquire=True, acquire_start_loop=2, auto_acquire_owners=owners)
    rng = np.random.default_rng(100 + seed)
    w, n = sim.world, sim.n_slots
    kinds = [CommandKind.NOOP, CommandKind.ATTACK_MOVE, CommandKind.HOLD, CommandKind.ATTACK_UNIT]
    w.order_kind[:] = rng.choice([int(k) for k in kinds], w.x.shape)
    w.order_target_slot[:] = rng.integers(0, n, w.x.shape)
    w.engaged_target_slot[:] = np.where(
        rng.random(w.x.shape) < 0.5, rng.integers(0, n, w.x.shape), NO_SLOT
    )
    sim.buf.fire_target[:] = np.where(
        rng.random(w.x.shape) < 0.3, rng.integers(0, n, w.x.shape), NO_SLOT
    )
    ref = copy.deepcopy(sim)
    targeting.basic(sim)
    rw, auto = ref.world, ref.config.auto_acquire
    want_t = np.full(w.x.shape, NO_SLOT)
    want_r = np.zeros(w.x.shape, bool)
    for b in range(ref.batch_size):  # per-unit reference loop before vectorization
        for u in range(n):
            t = NO_SLOT
            if rw.alive[b, u]:
                kind = int(rw.order_kind[b, u])
                if kind == CommandKind.ATTACK_UNIT:
                    t = int(rw.order_target_slot[b, u])
                elif kind == CommandKind.ATTACK_MOVE or (auto and kind == CommandKind.HOLD):
                    t = targeting._attack_move_target(ref, b, u)
                elif auto and kind == CommandKind.NOOP and targeting._acquires(ref, b, u):
                    t = targeting._idle_target(ref, b, u)
            want_t[b, u] = t
            want_r[b, u] = t != NO_SLOT and targeting.in_weapon_range(ref, b, u, t)
    np.testing.assert_array_equal(w.engaged_target_slot, want_t)
    np.testing.assert_array_equal(sim.buf.in_range, want_r)


def _pin_some(sim: Sim, seed: int) -> None:
    """Randomly set ~1/3 of units to HOLD or engaged (pin_engaged pins only stationary ones)."""
    rng = np.random.default_rng(400 + seed)
    w, shape = sim.world, sim.world.x.shape
    r = rng.random(shape)
    w.order_kind[r < 0.15] = CommandKind.HOLD
    w.engaged_target_slot[:] = np.where(
        (r >= 0.15) & (r < 0.35), rng.integers(0, sim.n_slots, shape), NO_SLOT
    )


def _sweep_both(sim: Sim) -> tuple[Sim, Sim]:
    k = collision._Consts(sim.config)
    if sim.config.pin_engaged:
        k.pin = collision.pinned_mask(sim)
    ground = sim.world.alive & ~sim.world.is_flying
    moving = ~np.isnan(sim.buf.waypoint_x)
    ref = copy.deepcopy(sim)
    collision._sweep(sim, ground, moving, k)
    for b in range(ref.batch_size):
        collision._sweep_exact(ref, b, np.flatnonzero(ground[b]).tolist(), moving[b], k)
    return sim, ref


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize(
    ('margin', 'small'), [(0.5, 48), (0.5, 0), (0.02, 0), (0.02, 10**6), (0.0, 0)]
)
@pytest.mark.parametrize('pin', [False, True])
def test_collision_sweep_matches_exact(
    seed: int, margin: float, small: int, pin: bool, monkeypatch
) -> None:
    """Default, forced-rounds, and tight-margin grow (`_sweep_grow`) paths all match pairwise
    sweep bit-for-bit, including `pin_engaged` immovable units."""
    monkeypatch.setattr(collision, 'SWEEP_MARGIN', margin)
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _sim(
        '27m_vs_30m', 3, seed, collision='hard', contact_slack=0.0625, idle_push_share=0.9,
        pin_engaged=pin,
    )  # fmt: skip
    if pin:
        _pin_some(sim, seed)
        assert collision.pinned_mask(sim).sum() > 20
    for _ in range(3):
        sim, ref = _sweep_both(sim)
        for name in ('x', 'y'):
            np.testing.assert_array_equal(getattr(sim.world, name), getattr(ref.world, name))
        for name in ('pushed_loop', 'pusher_radius'):
            np.testing.assert_array_equal(getattr(sim.motion, name), getattr(ref.motion, name))


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('side_check', [False, True])
@pytest.mark.parametrize('pin', [False, True])
def test_avoid_batch_matches_per_unit(seed: int, side_check: bool, pin: bool) -> None:
    sim = _sim(
        '27m_vs_30m', 3, seed, collision='hard', steering='sc2_avoid', contact_slack=0.0625,
        avoid_side_check=side_check, pin_engaged=pin,
    )  # fmt: skip
    if pin:
        _pin_some(sim, seed)
    sx, sy = steering._seek_all(sim)
    near = steering._avoid_candidates(sim, sx, sy)
    ox, oy = sx.copy(), sy.copy()
    rest = set(steering._avoid_batch(sim, sx, sy, near, ox, oy))
    units = list(zip(*np.nonzero(near.any(axis=2)), strict=True))
    assert len(units) > 50
    for b, u in units:
        b, u = int(b), int(u)
        if (b, u) in rest:
            continue
        vx, vy = steering._avoid_unit(
            sim, b, u, sx[b, u], sy[b, u], np.flatnonzero(near[b, u]).tolist()
        )
        assert np.float32(vx).tobytes() == ox[b, u].tobytes()
        assert np.float32(vy).tobytes() == oy[b, u].tobytes()


@pytest.mark.parametrize('seed', range(3))
def test_side_blocked_many_matches_per_unit(seed: int) -> None:
    sim = _sim('27m_vs_30m', 3, seed, steering='sc2_avoid', avoid_side_check=True)
    rng = np.random.default_rng(200 + seed)
    w = sim.world
    b, u = np.nonzero(w.alive & ~w.is_flying)
    ang = rng.uniform(-np.pi, np.pi, b.size)
    dx, dy = np.cos(ang), np.sin(ang)
    rb, lb = steering._side_blocked_many(sim, b, u, dx, dy)
    want_r = [
        steering._side_blocked(sim, int(i), int(j), x, y, right=True)
        for i, j, x, y in zip(b, u, dx, dy, strict=True)
    ]
    want_l = [
        steering._side_blocked(sim, int(i), int(j), x, y, right=False)
        for i, j, x, y in zip(b, u, dx, dy, strict=True)
    ]
    assert sum(want_r) + sum(want_l) > 5
    np.testing.assert_array_equal(rb, want_r)
    np.testing.assert_array_equal(lb, want_l)


@pytest.mark.parametrize('seed', range(3))
def test_face_many_matches_per_unit(seed: int) -> None:
    """Batched `_face_many` matches per-unit `_face` bit-for-bit on return value, heading, and
    turret (including turret_rate, infinite attack_arc, and self-coincident pairs)."""
    sim = _sim('MMM2', 4, seed, attack='sc2', damage='sc2')
    rng = np.random.default_rng(300 + seed)
    w, p, n = sim.world, sim.params, sim.n_slots
    shape = w.x.shape
    w.facing[:] = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    p.facing_rate[:] = rng.uniform(0, 0.5, shape).astype(np.float32)
    p.attack_arc[:] = np.where(rng.random(shape) < 0.2, np.inf, rng.uniform(0, 0.4, shape))
    p.turret_rate[:] = np.where(rng.random(shape) < 0.3, rng.uniform(0, 0.6, shape), np.nan)
    sim.combat.turret[:] = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    b, u = np.nonzero(w.alive)
    t = rng.integers(0, n, b.size)
    t[:5] = u[:5]  # self target: dx = dy = 0
    ref = copy.deepcopy(sim)
    got = attack._face_many(sim, b, u, t)
    want = [attack._face(ref, int(i), int(j), int(k)) for i, j, k in zip(b, u, t, strict=True)]
    np.testing.assert_array_equal(got, want)
    assert sim.motion.heading.tobytes() == ref.motion.heading.tobytes()
    assert sim.combat.turret.tobytes() == ref.combat.turret.tobytes()


@pytest.mark.parametrize('rho', [0.4375, 0.8125])
def test_visible_many_matches_visible_from(rho: float) -> None:
    info = load_map_info('SMAC_Maps/corridor')
    sim = _sim('corridor', 1, 0)
    mesh = _mesh(sim, info, rho)
    rng = np.random.default_rng(0)
    h, wd = info.pathing.shape
    x, y, gx, gy = (rng.uniform(0, s, 400) for s in (wd, h, wd, h))
    got = navmesh.visible_many(mesh, x, y, gx, gy)
    want = [
        bool(navmesh.visible_from(mesh, x[k], y[k], np.array([[gx[k], gy[k]]]))[0])
        for k in range(400)
    ]
    assert got.tolist() == want
    assert 0 < got.sum() < 400


@pytest.mark.parametrize('rho', [0.3125, 0.8125])
@pytest.mark.parametrize('scen', ['corridor', '3m'])
def test_ray_end_and_route_many_match_scalar(rho: float, scen: str) -> None:
    """`ray_end_many` / `route_many` match scalar `ray_end` / `route` bit-for-bit (including
    out-of-bounds and obstacle goals; fresh meshes ensure caches do not affect results)."""
    info = load_map_info(f'SMAC_Maps/{scen}')
    sim = _sim(scen, 1, 0)
    mesh = _mesh(sim, info, rho)
    rng = np.random.default_rng(1)
    h, wd = info.pathing.shape
    k = 120
    x, y = rng.uniform(0, wd, k), rng.uniform(0, h, k)
    gx, gy = x + rng.uniform(-10, 10, k), y + rng.uniform(-10, 10, k)
    gx[:20] = -3.0  # outside map bounds
    got = navmesh.ray_end_many(mesh, x, y, gx, gy)
    want = [navmesh.ray_end(mesh, *map(float, p)) for p in zip(x, y, gx, gy, strict=True)]
    np.testing.assert_array_equal(got, np.array(want))
    fresh = [dataclasses.replace(mesh, cost_cache={}, ray_cache={}) for _ in range(2)]
    many = navmesh.route_many(fresh[0], x, y, gx, gy)
    one = [navmesh.route(fresh[1], *map(float, p)) for p in zip(x, y, gx, gy, strict=True)]
    assert many == one
    assert any(end != (float(a), float(b)) for (_, end), a, b in zip(one, gx, gy, strict=True))


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('chain', [True, False])
def test_crowd_arrival_many_matches_per_env(seed: int, chain: bool) -> None:
    """Crowd arrival: batched `_crowd_arrival_many` matches per-env `_crowd_arrival` bit-for-bit
    (exercising shared goals, contact, stopped allies, and blocked-count thresholds)."""
    sim = _sim('27m_vs_30m', 6, seed, crowd_arrival=True, crowd_chain=chain, contact_slack=0.0625)
    rng = np.random.default_rng(300 + seed)
    w, m = sim.world, sim.motion
    shape = w.x.shape
    goal = rng.integers(0, 3, shape)  # 3 shared goals with small jitter
    cx, cy = float(np.nanmean(w.x)), float(np.nanmean(w.y))
    w.order_x[:] = (cx + 2.0 * goal + rng.uniform(-0.3, 0.3, shape)).astype(np.float32)
    w.order_y[:] = (cy + rng.uniform(-0.3, 0.3, shape)).astype(np.float32)
    kinds = [CommandKind.MOVE] * 6 + [CommandKind.NOOP, CommandKind.ATTACK_MOVE]
    w.order_kind[:] = rng.choice([int(k) for k in kinds], shape)
    m.crowd_goal_x[:] = np.where(rng.random(shape) < 0.7, w.order_x, np.nan)
    m.crowd_goal_y[:] = w.order_y
    m.crowd_dist[:] = np.where(
        rng.random(shape) < 0.8,
        np.hypot(w.order_x - w.x, w.order_y - w.y) + rng.uniform(-0.1, 0.2, shape),
        np.nan,
    ).astype(np.float32)
    m.crowd_blocked[:] = rng.integers(0, sim.config.crowd_loops + 1, shape)
    ground = w.alive & ~w.is_flying
    exact = copy.deepcopy(sim)
    envs = np.flatnonzero((ground & (w.order_kind == CommandKind.MOVE)).any(axis=1))
    for b in envs.tolist():
        collision._crowd_arrival(exact, b, np.flatnonzero(ground[b]).tolist())
    collision._crowd_arrival_many(sim, envs, ground)
    for name in ('order_kind', 'order_target_slot', 'order_x', 'order_y'):
        np.testing.assert_array_equal(getattr(sim.world, name), getattr(exact.world, name))
    for name in ('crowd_goal_x', 'crowd_goal_y', 'crowd_dist', 'crowd_blocked'):
        np.testing.assert_array_equal(getattr(sim.motion, name), getattr(exact.motion, name))
    cleared = (exact.world.order_kind == CommandKind.NOOP) & (w.order_kind == CommandKind.NOOP)
    assert envs.size > 0 and cleared.any()
