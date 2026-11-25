"""Tests for pyref `mover_push_share='approach_same_half'`.

Two movers with equal desired speeds (relative diff <= `collision.SAME_SPEED_RTOL`) split
50/50; unequal-speed pairs fall back to `'approach'`.

* Option validation; `'half'`, `'speed'`, and `'axis'` skip `_Approach`, and `'approach'`
  leaves `_Approach.speed` as None.
* Shares: equal-speed pairs split 50/50 while unequal-speed pairs match `'approach'`
  bit-for-bit (including zero speed and boundary cases); scalar matches array bit-for-bit.
* Batched sweep `collision._sweep` matches pairwise `collision._sweep_exact` bit-for-bit on
  random crowded states and full episodes (batch 1/8 with mid-episode resets).
* Episodes: mixed-speed pushes match `'approach'`; equal-speed pushes differ.
* Behavior: equal-speed converging pair splits 50/50 like `'half'`.
"""

from __future__ import annotations

import copy
import dataclasses
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
    STALKER,
    STATIC,
    ZEALOT,
    _converge,
    _episode,
    _exact_sweep,
    _pair,
    _pin_some,
    _sim,
)

SH: dict[str, Any] = {'mover_push_share': 'approach_same_half'}


def test_same_half_options_validated() -> None:
    cfg = PyrefConfig(**BASE, **SH, mover_push_cap=0.9)
    assert cfg.mover_push_share == 'approach_same_half'
    assert PyrefConfig().mover_push_share == 'half'
    assert collision.SAME_SPEED_RTOL == 0.02
    with pytest.raises(ValueError, match='mover_push_share'):
        dataclasses.replace(cfg, mover_push_share='approach_same')


# ---------------------------------------------------------------- Push shares


def _random_vel(sim: Sim, seed: int) -> np.ndarray:
    """Random desired velocities: half near 0.18 +/- 1% (equal-speed), others 0.12..0.4, some 0;
    returns per-unit speeds as float64."""
    rng = np.random.default_rng(seed)
    shape = sim.buf.vel_x.shape
    r = rng.random(shape)
    v = np.where(
        r < 0.5, 0.18 * (1 + rng.uniform(-0.01, 0.01, shape)), rng.uniform(0.12, 0.4, shape)
    )
    v[r > 0.95] = 0.0
    a = rng.uniform(-np.pi, np.pi, shape)
    sim.buf.vel_x[:] = (v * np.cos(a)).astype(np.float32)
    sim.buf.vel_y[:] = (v * np.sin(a)).astype(np.float32)
    return np.hypot(sim.buf.vel_x.astype(np.float64), sim.buf.vel_y.astype(np.float64))


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cap', [0.8, 1.0])
def test_same_half_split_vs_approach(seed: int, cap: float) -> None:
    """Equal-speed pairs (rel diff <= 2%) split 50/50; others match `'approach'`; scalar = array."""
    sim = _sim(2, seed, collision='hard', contact_slack=0.0625, **SH, mover_push_cap=cap)
    sp = _random_vel(sim, 50 + seed)
    # Boundary pairs: (0, 1) has 1.9% rel diff (equal); (0, 2) has 2.1% rel diff (unequal)
    sim.buf.vel_x[0, :3] = (0.2, -0.2 * 0.981, -0.2 * 0.979)
    sim.buf.vel_y[0, :3] = 0.0
    sp[0, :3] = np.abs(sim.buf.vel_x[0, :3].astype(np.float64))
    sh = collision._Approach(sim)
    assert sh.speed is not None
    ap = copy.copy(sh)
    ap.speed = None
    rng = np.random.default_rng(seed)
    m = 400
    b = rng.integers(0, 2, m)
    u = rng.integers(0, sim.n_slots, m)
    t = (u + rng.integers(1, sim.n_slots, m)) % sim.n_slots
    b[:2], u[:2], t[:2] = 0, 0, (1, 2)
    ang = rng.uniform(-np.pi, np.pi, m)
    ang[:2] = 0.0
    nx, ny = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    su, st = collision._approach_split(sh, b, u, t, nx, ny)
    au, at = collision._approach_split(ap, b, u, t, nx, ny)
    a, c = sp[b, u], sp[b, t]
    same = np.abs(a - c) <= collision.SAME_SPEED_RTOL * np.maximum(a, c)
    assert same[0] and not same[1] and 50 < same.sum() < m - 50
    np.testing.assert_array_equal(su[same], np.float32(0.5))
    np.testing.assert_array_equal(st[same], np.float32(0.5))
    assert su[~same].tobytes() == au[~same].tobytes() and st[~same].tobytes() == at[~same].tobytes()
    assert (au[same] != np.float32(0.5)).sum() > 20  # most equal-speed pairs != 0.5 in approach
    for i in range(m):
        x, y = collision._approach_split(sh, int(b[i]), int(u[i]), int(t[i]), nx[i], ny[i])
        assert x.tobytes() == su[i].tobytes() and y.tobytes() == st[i].tobytes()


def test_same_half_push_shares() -> None:
    """Single push (u left, t right, n = +x): equal-speed u chasing t splits 50/50 (unlike
    `'approach'`); unequal speeds match `'approach'`."""
    cfg = PyrefConfig(**BASE, **SH)
    sim = _pair(cfg)
    w, buf = sim.world, sim.buf
    overlap = 2 * 0.625 - 0.0625 - 1.0
    cases = [
        # (v_u, v_t, u's share)
        ((0.18, 0.0), (0.18, 0.0), 0.5),  # equal-speed chase: 50/50
        ((0.0, 0.18), (-0.18, 0.0), 0.5),  # equal-speed, only t approaches u: 50/50
        ((0.1, 0.0), (-0.3, 0.0), 0.25),  # unequal speeds: split by approach component
        ((0.0, 0.18), (-0.12, 0.0), 0.2),  # unequal speeds, only t approaches u: 1 - cap
        ((0.0, 0.0), (0.0, 0.0), 0.5),
    ]
    for vu, vt, want in cases:
        w.x[0, :2] = (10.0, 11.0)
        w.y[0, :2] = 10.0
        buf.waypoint_x[0, :2] = (12.0, 8.0)
        buf.waypoint_y[0, :2] = 10.0
        buf.vel_x[0, :2] = (vu[0], vt[0])
        buf.vel_y[0, :2] = (vu[1], vt[1])
        k = collision._Consts(cfg)
        k.approach = collision._Approach(sim)
        r = collision._push(sim, 0, 0, 1, True, True, k)
        assert r is not None
        assert r[2] == pytest.approx(overlap * want, abs=1e-6), (vu, vt)
        assert r[3] == pytest.approx(overlap * (1 - want), abs=1e-6), (vu, vt)


# ---------------------------------------------------------------- Random crowded states


def _sweep_both(sim: Sim) -> tuple[Sim, Sim]:
    k = collision._Consts(sim.config)
    if sim.config.pin_engaged:
        k.pin = collision.pinned_mask(sim)
    k.approach = collision._Approach(sim)
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
def test_same_half_sweep_matches_exact(
    seed: int, margin: float, small: int, pin: bool, monkeypatch
) -> None:
    """Default, forced-chunked, and tiny-margin rescan paths all match pairwise sweep bit-for-bit
    (`_sim` includes zero, equal, and varied speeds)."""
    monkeypatch.setattr(collision, 'SWEEP_MARGIN', margin)
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _sim(
        3, seed, collision='hard', contact_slack=0.0625, idle_push_share=0.9, pin_engaged=pin, **SH
    )
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


# ---------------------------------------------------------------- Episodes (with resets)


@pytest.mark.parametrize('batch', [1, 8])
@pytest.mark.parametrize('cap', [False, True])
def test_same_half_episode_matches_exact(batch: int, cap: bool, monkeypatch) -> None:
    config = {**BASE, **SH, 'mover_speed_cap': cap}
    pairs = [0, 0]
    split = collision._approach_split

    def counted(ap, b, u, t, nx, ny):
        su, st = split(ap, b, u, t, nx, ny)
        pairs[0] += int(np.sum(su != st))
        pairs[1] += int(np.sum(su == np.float32(0.5)))
        return su, st

    monkeypatch.setattr(collision, '_approach_split', counted)
    got = _episode(batch, config)
    monkeypatch.setattr(collision, '_sweep', _exact_sweep)
    want = _episode(batch, config)
    assert pairs[0] > 5 * batch and pairs[1] > 5 * batch
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (k, name)


def test_other_shares_skip_speed(monkeypatch) -> None:
    """`'half'`, `'speed'`, and `'axis'` skip `_Approach`; `'approach'` leaves `_Approach.speed`
    as None."""

    def boom(*_a, **_k):
        raise AssertionError('_Approach should not be called')

    monkeypatch.setattr(collision, '_Approach', boom)
    for share in ('half', 'speed', 'axis'):
        _episode(1, {**BASE, 'mover_push_share': share})
    monkeypatch.undo()
    seen: list[Any] = []
    init = collision._Approach.__init__

    def spy(self, sim):
        init(self, sim)
        seen.append(self.speed)

    monkeypatch.setattr(collision._Approach, '__init__', spy)
    _episode(1, {**BASE, 'mover_push_share': 'approach'})
    assert seen and all(s is None for s in seen)


# ---------------------------------------------------------------- Episode comparison


def _cross(share: str, units: tuple[int, int], steps: int = 6) -> list[WorldState]:
    """Two units cross diagonally (a up-right, b down-right, colliding midway) 8 tiles away with
    `steering='none'`, 8 loops per step."""
    kw: dict[str, Any] = {**BASE, 'steering': 'none', 'mover_push_share': share}
    cfg = PyrefConfig(**kw, mover_speed_cap=True)
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    sc = Scenario(MAP_ID, spawns=(Spawn(1, units[0], 10.0, 15.0), Spawn(1, units[1], 10.0, 16.4)))
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(18.0, 18.0))
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(18.0, 13.0))
    out = [be.step(cmd, 8)]
    for _ in range(steps - 1):
        out.append(be.step(Command.noop(1, n), 8))
    return out


def test_same_half_equals_approach_when_speeds_differ() -> None:
    """Stalker/Zealot crossing (31% speed gap) matches `'approach'` bit-for-bit and differs from
    `'half'`; two crossing Stalkers differ from `'approach'`."""
    sh = _cross('approach_same_half', (STALKER, ZEALOT))
    ap = _cross('approach', (STALKER, ZEALOT))
    hf = _cross('half', (STALKER, ZEALOT))
    for a, b in zip(sh, ap, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name
    assert any(a.y.tobytes() != b.y.tobytes() for a, b in zip(sh, hf, strict=True))
    s2 = _cross('approach_same_half', (STALKER, STALKER))
    a2 = _cross('approach', (STALKER, STALKER))
    assert any(a.y.tobytes() != b.y.tobytes() for a, b in zip(s2, a2, strict=True))


# ---------------------------------------------------------------- Behavior


def test_same_half_converging_stalkers() -> None:
    """Equal-speed converging Stalkers split 50/50, matching `'half'` bit-for-bit, whereas
    `'approach'` gives straight-moving A much less downward drift."""
    half_a, half_b = _converge()
    sh_a, sh_b = _converge(**SH)
    ap_a, _ = _converge(mover_push_share='approach')
    assert (sh_a, sh_b) == (half_a, half_b)
    assert ap_a > 0.5 * half_a
