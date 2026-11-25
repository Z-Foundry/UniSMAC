"""Tests for pyref `mover_push_share='axis'` (per-axis world-coordinate push splitting):

* Option validation; `'half'`, `'speed'`, and `'approach'` never construct `_Axis`.
* Batched sweep `collision._sweep` (chunked and widened-candidate rescan) matches pairwise
  `collision._sweep_exact` bit-for-bit on random crowded states and full episodes (batch 1/8
  with mid-episode resets).
* Shares: split per axis by approach velocity, clamp to `[1 - cap, cap]`, weight by `v^p`, and
  combine by squared line-of-centers components.
* Behavior: invariant under 90-degree rotation; when a leading unit angles slightly toward a
  trailing unit, the trailer yields less than under `'approach'`.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

from .test_pyref_mover_push_approach import (
    BASE,
    MAP_ID,
    MAPS,
    MARINE,
    STATIC,
    _episode,
    _exact_sweep,
    _pair,
    _pin_some,
    _sim,
)

AXIS: dict[str, Any] = {'mover_push_share': 'axis', 'mover_push_power': 8.0}


def test_axis_options_validated() -> None:
    cfg = PyrefConfig(**BASE, **AXIS, mover_push_cap=0.9)
    assert cfg.mover_push_share == 'axis'
    assert PyrefConfig().mover_push_share == 'half'
    with pytest.raises(ValueError, match='mover_push_share'):
        dataclasses.replace(cfg, mover_push_share='axes')


# ---------------------------------------------------------------- Random crowded states


def _sweep_both(sim: Sim) -> tuple[Sim, Sim]:
    k = collision._Consts(sim.config)
    if sim.config.pin_engaged:
        k.pin = collision.pinned_mask(sim)
    k.axis = collision._Axis(sim)
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
@pytest.mark.parametrize(('cap', 'power'), [(0.8, 8.0), (1.0, 1.0)])
@pytest.mark.parametrize('pin', [False, True])
def test_axis_sweep_matches_exact(
    seed: int, margin: float, small: int, cap: float, power: float, pin: bool, monkeypatch
) -> None:
    """Default, forced-chunked, and tiny-margin rescan paths match pairwise sweep bit-for-bit."""
    monkeypatch.setattr(collision, 'SWEEP_MARGIN', margin)
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _sim(
        3, seed, collision='hard', contact_slack=0.0625, idle_push_share=0.9, pin_engaged=pin,
        mover_push_share='axis', mover_push_cap=cap, mover_push_power=power,
    )  # fmt: skip
    if pin:
        _pin_some(sim, seed)
    moved = 0
    for _ in range(3):
        before = sim.world.x.copy()
        sim, ref = _sweep_both(sim)
        moved += int((sim.world.x != before).sum())
        for name in ('x', 'y'):
            np.testing.assert_array_equal(getattr(sim.world, name), getattr(ref.world, name))
        for name in ('pushed_loop', 'pusher_radius'):
            np.testing.assert_array_equal(getattr(sim.motion, name), getattr(ref.motion, name))
    assert moved > 50


# ---------------------------------------------------------------- Push shares


def _part(cap: float, vu: float, vt: float, n: float, wu: float, wt: float) -> float:
    sg = 1.0 if n >= 0 else -1.0
    au, at = max(sg * vu, 0.0), max(-sg * vt, 0.0)
    s = au / (au + at) if au + at > 0 else 0.5
    s = min(max(s, 1 - cap), cap)
    return s * wu / (s * wu + (1 - s) * wt) if s * wu + (1 - s) * wt > 0 else s


@pytest.mark.parametrize(('cap', 'power'), [(0.8, 8.0), (1.0, 1.0)])
def test_axis_split_shares(cap: float, power: float) -> None:
    """Single push (u at (10, 10), t at distance 1 along angle theta): x and y axes each split via
    `_part`, giving u share `n_x^2 * s_x + n_y^2 * s_y` along the line of centers."""
    cfg = PyrefConfig(**BASE, mover_push_share='axis', mover_push_cap=cap, mover_push_power=power)
    sim = _pair(cfg)
    w, buf = sim.world, sim.buf
    overlap = 2 * 0.625 - 0.0625 - 1.0
    cases = [
        # (theta deg, v_u, v_t)
        (0, (0.18, 0.0), (0.18, 0.0)),  # u chases t: u yields on x-axis
        (0, (0.0, 0.18), (0.0, -0.18)),  # perpendicular: half each
        (30, (-0.18, 0.0), (-0.18, 0.0)),  # staggered follow (t behind/above): t yields x, half y
        (60, (0.0, 0.18), (0.05, 0.0)),  # u moves toward t's side: u yields on y-axis
        (135, (0.1, 0.1), (-0.15, 0.02)),  # unequal speeds: weighted by v^p
        (90, (0.0, 0.0), (0.0, 0.0)),  # both stationary
        (-45, (0.12, -0.12), (0.12, -0.12)),  # equal velocity along diagonal contact
    ]
    for th, vu, vt in cases:
        n = (math.cos(math.radians(th)), math.sin(math.radians(th)))
        w.x[0, :2] = (10.0, 10.0 + n[0])
        w.y[0, :2] = (10.0, 10.0 + n[1])
        buf.waypoint_x[0, :2] = (12.0, 8.0)
        buf.waypoint_y[0, :2] = 10.0
        buf.vel_x[0, :2] = (vu[0], vt[0])
        buf.vel_y[0, :2] = (vu[1], vt[1])
        k = collision._Consts(cfg)
        k.axis = collision._Axis(sim)
        dx = float(np.float32(w.x[0, 1] - w.x[0, 0]))
        dy = float(np.float32(w.y[0, 1] - w.y[0, 0]))
        d = math.hypot(dx, dy)
        nx, ny = dx / d, dy / d
        wu = math.hypot(float(np.float32(vu[0])), float(np.float32(vu[1]))) ** power
        wt = math.hypot(float(np.float32(vt[0])), float(np.float32(vt[1]))) ** power
        f = [float(np.float32(v)) for v in (*vu, *vt)]
        sx = _part(cap, f[0], f[2], nx, wu, wt)
        sy = _part(cap, f[1], f[3], ny, wu, wt)
        want = (nx * nx * sx + ny * ny * sy) / (nx * nx + ny * ny)
        ov = 2 * 0.625 - 0.0625 - d
        assert ov == pytest.approx(overlap, abs=1e-5)
        r = collision._push(sim, 0, 0, 1, True, True, k)
        assert r is not None
        tot = abs(nx) + abs(ny)
        assert r[2] == pytest.approx(ov * want * tot, abs=1e-6), (th, vu, vt)
        assert r[3] == pytest.approx(ov * (1 - want) * tot, abs=1e-6), (th, vu, vt)


def test_axis_split_scalar_matches_array() -> None:
    """Scalar and array paths match bit-for-bit (random velocities/angles, zero speed, axis n)."""
    cfg = PyrefConfig(**BASE, **AXIS)
    sim = _pair(cfg)
    rng = np.random.default_rng(3)
    shape = sim.buf.vel_x.shape
    sim.buf.vel_x[:] = rng.normal(0, 0.15, shape).astype(np.float32)
    sim.buf.vel_y[:] = rng.normal(0, 0.15, shape).astype(np.float32)
    sim.buf.vel_x[0, 0] = sim.buf.vel_y[0, 0] = 0
    ax = collision._Axis(sim)
    m = 64
    b = np.zeros(m, np.int64)
    u = rng.integers(0, 2, m)
    t = 1 - u
    ang = rng.uniform(-np.pi, np.pi, m)
    ang[:8] = np.arange(8) * np.pi / 4
    nx, ny = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    su, st = collision._axis_split(ax, b, u, t, nx, ny)
    for i in range(m):
        a, c = collision._axis_split(ax, int(b[i]), int(u[i]), int(t[i]), nx[i], ny[i])
        assert a.tobytes() == su[i].tobytes() and c.tobytes() == st[i].tobytes()


# ---------------------------------------------------------------- Episodes (with resets)


@pytest.mark.parametrize('batch', [1, 8])
@pytest.mark.parametrize('cap', [False, True])
def test_axis_episode_matches_exact(batch: int, cap: bool, monkeypatch) -> None:
    config = {**BASE, **AXIS, 'mover_speed_cap': cap}
    pairs = [0]
    split = collision._axis_split

    def counted(ax, b, u, t, nx, ny):
        su, st = split(ax, b, u, t, nx, ny)
        pairs[0] += int(np.sum(su != st))
        return su, st

    monkeypatch.setattr(collision, '_axis_split', counted)
    got = _episode(batch, config)
    monkeypatch.setattr(collision, '_sweep', _exact_sweep)
    want = _episode(batch, config)
    assert pairs[0] > 5 * batch
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (k, name)


def test_other_shares_skip_axis(monkeypatch) -> None:
    """`'half'`, `'speed'`, and `'approach'` skip `_Axis`; `'axis'` differs from `'approach'`."""

    def boom(*_a, **_k):
        raise AssertionError("_Axis should not be called unless mover_push_share='axis'")

    monkeypatch.setattr(collision, '_Axis', boom)
    for share in ('half', 'speed', 'approach'):
        _episode(1, {**BASE, 'mover_push_share': share})
    monkeypatch.undo()
    a = _episode(2, {**BASE, 'mover_push_share': 'approach'})
    c = _episode(2, {**BASE, **AXIS})
    assert any(x.x.tobytes() != y.x.tobytes() for x, y in zip(a, c, strict=True))


# ---------------------------------------------------------------- Behavior


def _follow(rot: float, **extra: Any) -> tuple[float, float]:
    """Two staggered Marines (trailer B behind leader F with 0.75-distance lateral offset, nearly
    touching) rotated by `rot` degrees: B walks straight while F angles 3 deg toward B's side,
    both moving 12 tiles with `steering='none'` for 40 loops. Returns total yield (yb, yf) along
    the separating axis relative to free-motion steps."""
    opts: dict[str, Any] = {**BASE, 'steering': 'none', **extra}
    cfg = PyrefConfig(**opts)
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    d = 2 * 0.375 - 0.0625 + 1 / 1024
    lat = 0.75 * d
    h = np.array([math.cos(math.radians(180 + rot)), math.sin(math.radians(180 + rot))])
    side = np.array([-h[1], h[0]])  # left of h
    f0 = np.array([20.0, 16.0])
    b0 = f0 - h * math.sqrt(d * d - lat * lat) - side * lat
    df = h * math.cos(math.radians(3)) - side * math.sin(math.radians(3))
    sc = Scenario(MAP_ID, spawns=(Spawn(1, MARINE, *f0), Spawn(1, MARINE, *b0)))
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=tuple(f0 + 12 * df))
    cmd.set(0, 1, CommandKind.MOVE, target_xy=tuple(b0 + 12 * h))
    st = be.step(cmd, 1)
    x = [np.array([st.x[0, i], st.y[0, i]], np.float64) for i in (0, 1)]
    sp = [float(be._sim.params.speed[0, i]) for i in (0, 1)]
    yf = yb = 0.0
    for _ in range(40):
        st = be.step(Command.noop(1, n), 1)
        y = [np.array([st.x[0, i], st.y[0, i]], np.float64) for i in (0, 1)]
        nv = (x[0] - x[1]) / np.linalg.norm(x[0] - x[1])  # B -> F
        free_f, free_b = df * sp[0], h * sp[1]  # unobstructed full-speed step
        yf += float((y[0] - x[0] - free_f) @ nv)
        yb += float(-(y[1] - x[1] - free_b) @ nv)
        x = y
    return yb, yf


def test_axis_follow_rotation() -> None:
    """Staggered follow with leader angling toward trailer: `'approach'` makes trailer yield near
    the 0.8 cap, `'axis'` gives a smaller share invariant under 90-deg rotation, and a 30-deg
    rotation increases trailer yield."""
    ap = _follow(0, mover_push_share='approach')
    a0 = _follow(0, **AXIS)
    a90 = _follow(90, **AXIS)
    a30 = _follow(30, **AXIS)

    def share(r):
        return r[0] / (r[0] + r[1])

    assert share(ap) > 0.75
    assert 0.3 < share(a0) < 0.65
    assert share(a90) == pytest.approx(share(a0), abs=0.02)
    assert share(a30) > share(a0) + 0.05


def test_axis_converging_stalkers() -> None:
    """Equal-speed converging Stalkers (line of centers along y): `'axis'` makes converging B
    yield like `'approach'`, reducing A's downward drift compared to `'half'`."""
    from .test_pyref_mover_push_approach import _converge

    half_a, _ = _converge()
    ax_a, ax_b = _converge(**AXIS)
    assert -0.5 * abs(half_a) < ax_a <= 0
    assert ax_b > -0.5
