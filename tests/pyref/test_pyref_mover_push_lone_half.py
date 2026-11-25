"""Tests for pyref `mover_push_share='approach_lone_half'`.

Equal-speed mover pairs (within `SAME_SPEED_RTOL`) split 50/50 only when isolated: neither
unit touches a third ground unit of different desired speed (`overlap > -collision.CHAIN_EPS`)
before collision resolution; otherwise falls back to `'approach'`.

* Option validation; `'approach'` and `'approach_same_half'` skip touch detection
  (`_Approach.odd` is None and `_touching` is never called).
* `_Approach.odd` matches a scalar per-unit reference bit-for-bit (random crowds with dead,
  flying, and zero-speed units).
* Shares: non-chain pairs match `'approach_same_half'`, chain pairs match `'approach'`; scalar
  and array paths match bit-for-bit; single-push three-body configurations.
* Batched sweep `collision._sweep` matches pairwise `collision._sweep_exact` bit-for-bit on
  random crowded states and full episodes (batch 1/8 with mid-episode resets).
* Episodes: mixed-speed crossings match `'approach'`; lone equal-speed crossings and
  same-speed trios match `'approach_same_half'`.
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
    _episode,
    _exact_sweep,
    _pin_some,
    _sim,
)
from .test_pyref_mover_push_same_half import _cross

LO: dict[str, Any] = {'mover_push_share': 'approach_lone_half'}


def test_lone_half_options_validated() -> None:
    cfg = PyrefConfig(**BASE, **LO, mover_push_cap=0.9)
    assert cfg.mover_push_share == 'approach_lone_half'
    assert PyrefConfig().mover_push_share == 'half'
    assert collision.CHAIN_EPS == 0.01
    with pytest.raises(ValueError, match='mover_push_share'):
        dataclasses.replace(cfg, mover_push_share='approach_lone')


# ---------------------------------------------------------------- Contact predicate


def _lone_vel(sim: Sim, seed: int) -> None:
    """Random desired velocities: most near 0.18 +/- 1% (equal-speed), some 0.12..0.4, some 0."""
    rng = np.random.default_rng(seed)
    shape = sim.buf.vel_x.shape
    r = rng.random(shape)
    v = np.where(
        r < 0.8, 0.18 * (1 + rng.uniform(-0.01, 0.01, shape)), rng.uniform(0.12, 0.4, shape)
    )
    v[r > 0.97] = 0.0
    a = rng.uniform(-np.pi, np.pi, shape)
    sim.buf.vel_x[:] = (v * np.cos(a)).astype(np.float32)
    sim.buf.vel_y[:] = (v * np.sin(a)).astype(np.float32)


def _lone_sim(batch: int, seed: int, **extra: Any) -> Sim:
    sim = _sim(batch, seed, collision='hard', contact_slack=0.0625, **LO, **extra)
    _lone_vel(sim, 70 + seed)
    w = sim.world
    w.is_flying[0, 3] = True  # air units are ignored
    return sim


def _odd_scalar(sim: Sim, b: int, u: int) -> bool:
    w, buf = sim.world, sim.buf
    if not (w.alive[b, u] and not w.is_flying[b, u]):
        return False
    vu = float(np.hypot(np.float64(buf.vel_x[b, u]), np.float64(buf.vel_y[b, u])))
    for k in range(w.x.shape[1]):
        if k == u or not w.alive[b, k] or w.is_flying[b, k]:
            continue
        d = float(np.hypot(np.float64(w.x[b, k]) - w.x[b, u], np.float64(w.y[b, k]) - w.y[b, u]))
        ov = float(w.radius[b, u]) + float(w.radius[b, k]) - float(sim.config.contact_slack) - d
        vk = float(np.hypot(np.float64(buf.vel_x[b, k]), np.float64(buf.vel_y[b, k])))
        same = abs(vu - vk) <= collision.SAME_SPEED_RTOL * max(vu, vk)
        if ov > -collision.CHAIN_EPS and not same:
            return True
    return False


@pytest.mark.parametrize('seed', range(3))
def test_lone_half_odd_matches_scalar(seed: int) -> None:
    sim = _lone_sim(2, seed)
    odd = collision._Approach(sim).odd
    assert odd is not None and odd.shape == sim.world.x.shape
    want = np.array(
        [[_odd_scalar(sim, b, u) for u in range(sim.n_slots)] for b in range(2)], np.bool_
    )
    np.testing.assert_array_equal(odd, want)
    ground = sim.world.alive & ~sim.world.is_flying
    assert 10 < odd.sum() < ground.sum() - 10


# ---------------------------------------------------------------- Push shares


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cap', [0.8, 1.0])
def test_lone_half_split_vs_same_half(seed: int, cap: float) -> None:
    """Pairs outside a chain (neither odd) match `'approach_same_half'`, pairs in a chain match
    `'approach'`, and scalar and array paths match bit-for-bit."""
    sim = _lone_sim(2, seed, mover_push_cap=cap)
    lo = collision._Approach(sim)
    assert lo.speed is not None and lo.odd is not None
    sh = copy.copy(lo)
    sh.odd = None
    ap = copy.copy(lo)
    ap.speed, ap.odd = None, None
    rng = np.random.default_rng(seed)
    m = 2000
    b = rng.integers(0, 2, m)
    u = rng.integers(0, sim.n_slots, m)
    t = (u + rng.integers(1, sim.n_slots, m)) % sim.n_slots
    ang = rng.uniform(-np.pi, np.pi, m)
    nx, ny = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    su, st = collision._approach_split(lo, b, u, t, nx, ny)
    hu, ht = collision._approach_split(sh, b, u, t, nx, ny)
    au, at = collision._approach_split(ap, b, u, t, nx, ny)
    chain = lo.odd[b, u] | lo.odd[b, t]
    a, c = lo.speed[b, u], lo.speed[b, t]
    same = np.abs(a - c) <= collision.SAME_SPEED_RTOL * np.maximum(a, c)
    for x, h, y in ((su, hu, au), (st, ht, at)):
        assert x[~chain].tobytes() == h[~chain].tobytes()
        assert x[chain].tobytes() == y[chain].tobytes()
    # Both cases have plenty of equal-speed pairs, and most chained pairs are not 50/50
    assert (same & ~chain).sum() > 30 and (same & chain).sum() > 100
    assert (su[same & chain] != np.float32(0.5)).sum() > 50
    for i in range(0, m, 5):
        x, y = collision._approach_split(lo, int(b[i]), int(u[i]), int(t[i]), nx[i], ny[i])
        assert x.tobytes() == su[i].tobytes() and y.tobytes() == st[i].tobytes()


def _trio(cfg: PyrefConfig) -> Sim:
    sc = Scenario(
        MAP_ID,
        spawns=(
            Spawn(1, STALKER, 10.0, 10.0),
            Spawn(1, STALKER, 11.0, 10.0),
            Spawn(1, ZEALOT, 10.0, 12.0),
        ),
    )
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    return be._sim


@pytest.mark.parametrize('cap', [0.8, 1.0])
def test_lone_half_push_shares(cap: float) -> None:
    """Single push: equal-speed Stalker u chases t (u left, t right, n = +x) with Zealot k above
    u. If k touches u at a different speed (or idle), falls back to `'approach'` (u yields cap);
    if k is separated or equal speed, splits 50/50; unequal u/t speeds use `'approach'`."""
    cfg = PyrefConfig(**BASE, **LO, mover_push_cap=cap)
    sim = _trio(cfg)
    w, buf = sim.world, sim.buf
    overlap = 2 * 0.625 - 0.0625 - 1.0
    touch = 0.625 + 0.5 - 0.0625  # u-k contact distance
    cases = [
        # (u-k center dist, v_t, v_k, k moving, u's share)
        (touch - 0.004, 0.18, (0.0, 0.12), True, cap),  # unequal-speed k overlaps u: 3-body chain
        (touch + 0.006, 0.18, (0.0, 0.12), True, cap),  # within CHAIN_EPS: still touching
        (touch + 0.02, 0.18, (0.0, 0.12), True, 0.5),  # separated by 0.02: isolated pair
        (touch - 0.004, 0.18, (0.0, 0.18), True, 0.5),  # equal-speed k: homogeneous trio
        (touch - 0.004, 0.18, (0.0, 0.0), False, cap),  # idle k (speed 0)
        (touch + 0.02, 0.12, (0.0, 0.0), False, cap),  # unequal u/t speeds: use 'approach'
    ]
    for dk, vt, vk, mk, want in cases:
        w.x[0, :3] = (10.0, 11.0, 10.0)
        w.y[0, :3] = (10.0, 10.0, 10.0 + dk)
        buf.waypoint_x[0, :3] = (12.0, 13.0, 10.0 if mk else np.nan)
        buf.waypoint_y[0, :3] = (10.0, 10.0, 14.0 if mk else np.nan)
        buf.vel_x[0, :3] = (0.18, vt, vk[0])
        buf.vel_y[0, :3] = (0.0, 0.0, vk[1])
        k = collision._Consts(cfg)
        k.approach = collision._Approach(sim)
        r = collision._push(sim, 0, 0, 1, True, True, k)
        assert r is not None
        assert r[2] == pytest.approx(overlap * want, abs=1e-6), (dk, vt, vk)
        assert r[3] == pytest.approx(overlap * (1 - want), abs=1e-6), (dk, vt, vk)


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
def test_lone_half_sweep_matches_exact(
    seed: int, margin: float, small: int, pin: bool, monkeypatch
) -> None:
    """Default, forced-chunked, and tiny-margin rescan paths match pairwise sweep bit-for-bit."""
    monkeypatch.setattr(collision, 'SWEEP_MARGIN', margin)
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _lone_sim(3, seed, idle_push_share=0.9, pin_engaged=pin)
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
def test_lone_half_episode_matches_exact(batch: int, cap: bool, monkeypatch) -> None:
    """Full episodes (mixed crowds): batched sweep matches pairwise sweep bit-for-bit, and every
    split matches `'approach_same_half'` off-chain or `'approach'` on-chain, with both seen."""
    config = {**BASE, **LO, 'mover_speed_cap': cap}
    seen = [0, 0, 0]
    split = collision._approach_split

    def checked(ap, b, u, t, nx, ny):
        su, st = split(ap, b, u, t, nx, ny)
        assert ap.odd is not None
        sh = copy.copy(ap)
        sh.odd = None
        ac = copy.copy(ap)
        ac.speed, ac.odd = None, None
        hu, _ = split(sh, b, u, t, nx, ny)
        au, _ = split(ac, b, u, t, nx, ny)
        chain = ap.odd[b, u] | ap.odd[b, t]
        np.testing.assert_array_equal(su, np.where(chain, au, hu))
        seen[0] += int(np.sum(chain & (hu != au)))  # chained pairs fall back to 'approach'
        seen[1] += int(np.sum(~chain & (hu != au)))  # isolated equal-speed pairs split 50/50
        seen[2] += int(np.sum(su != st))
        return su, st

    monkeypatch.setattr(collision, '_approach_split', checked)
    got = _episode(batch, config)
    monkeypatch.setattr(collision, '_sweep', _exact_sweep)
    want = _episode(batch, config)
    assert seen[0] > 0 and seen[1] > 0 and seen[2] > 5 * batch
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (k, name)


def test_other_shares_skip_touching(monkeypatch) -> None:
    """`'approach'` and `'approach_same_half'` skip touch detection (`_touching` not called,
    `_Approach.odd` is None)."""

    def boom(*_a, **_k):
        raise AssertionError('_touching should not be called')

    monkeypatch.setattr(collision, '_touching', boom)
    seen: list[Any] = []
    init = collision._Approach.__init__

    def spy(self, sim):
        init(self, sim)
        seen.append(self.odd)

    monkeypatch.setattr(collision._Approach, '__init__', spy)
    for share in ('approach', 'approach_same_half'):
        _episode(1, {**BASE, 'mover_push_share': share})
    assert seen and all(s is None for s in seen)


# ---------------------------------------------------------------- Episode comparisons


def _same_states(a: list[WorldState], b: list[WorldState]) -> bool:
    return all(
        getattr(x, n).tobytes() == getattr(y, n).tobytes()
        for x, y in zip(a, b, strict=True)
        for n in WorldState.UNIT_FIELDS
    )


def test_lone_half_pairs_crossing() -> None:
    """Stalker/Zealot crossing matches `'approach'`; lone Stalker/Stalker crossing matches
    `'approach_same_half'` and differs from `'approach'`."""
    sz = (STALKER, ZEALOT)
    assert _same_states(_cross('approach_lone_half', sz), _cross('approach', sz))
    ss = (STALKER, STALKER)
    lo = _cross('approach_lone_half', ss)
    assert _same_states(lo, _cross('approach_same_half', ss))
    assert not _same_states(lo, _cross('approach', ss))


def _converge3(third: int, below: float = 1.2, **extra: Any) -> list[WorldState]:
    """Three units move to the same target: middle Stalker A walks +x, upper Stalker B converges
    downward, and a third unit (`third`) at offset `below` converges upward. Runs 40 loops with
    `steering='none'`."""
    options: dict[str, Any] = {**BASE, 'steering': 'none', 'mover_speed_cap': True, **extra}
    cfg = PyrefConfig(**options)
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    y = 16.0
    sc = Scenario(
        MAP_ID,
        spawns=(
            Spawn(1, STALKER, 8.0, y),
            Spawn(1, STALKER, 8.0, y + 1.2),
            Spawn(1, third, 8.0, y - below),
        ),
    )
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    for slot in (0, 1, 2):
        cmd.set(0, slot, CommandKind.MOVE, target_xy=(24.0, y))
    out = [be.step(cmd, 1)]
    for _ in range(39):
        out.append(be.step(Command.noop(1, n), 1))
    return out


def test_lone_half_chains() -> None:
    """Three converging Stalkers match `'approach_same_half'` and differ from `'approach'`;
    replacing the bottom unit with a slower Zealot (making A-B chained) differs from both."""
    lo = _converge3(STALKER, **LO)
    assert _same_states(lo, _converge3(STALKER, mover_push_share='approach_same_half'))
    assert not _same_states(lo, _converge3(STALKER, mover_push_share='approach'))
    lz = _converge3(ZEALOT, 1.1, **LO)
    assert not _same_states(lz, _converge3(ZEALOT, 1.1, mover_push_share='approach_same_half'))
    assert not _same_states(lz, _converge3(ZEALOT, 1.1, mover_push_share='approach'))
