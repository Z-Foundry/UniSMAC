"""Tests for pyref `mover_push_share='approach'` (mover-vs-mover split by approach speed):

* Option validation (including `mover_push_cap` bounds); `'half'` and `'speed'` skip the new
  path so `mover_push_cap` has no effect.
* Batched sweep `collision._sweep` (chunked and widened-candidate rescan) matches pairwise
  `collision._sweep_exact` bit-for-bit on random crowded states and full episodes (batch 1/8
  with mid-episode resets).
* Shares: positive parts of `a_u = v_u . n` and `a_t = -v_t . n` split proportionally, clamped
  to `[1 - cap, cap]`, defaulting to 0.5/0.5 when both <= 0.
* Behavior: when two equal-speed movers converge side by side, the converging unit yields more
  (unlike `'half'`).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.scenarios import spawn_record

STATIC = load_static_data()
MAP_ID = 'SMAC_Maps/32x32_flat'
MAPS = {MAP_ID: load_map_info(MAP_ID)}
MARINE, ZEALOT, STALKER, ZERGLING = 48, 73, 74, 105
BASE: dict[str, Any] = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'crowd_arrival': True,
    'avoid_side_check': True,
}


def test_approach_options_validated() -> None:
    cfg = PyrefConfig(**BASE, mover_push_share='approach', mover_push_cap=0.9)
    assert cfg.mover_push_cap == 0.9
    assert PyrefConfig().mover_push_cap == 0.8
    bad_values: list[tuple[str, Any]] = [
        ('mover_push_share', 'approaching'),
        ('mover_push_cap', 0.49),
        ('mover_push_cap', 1.01),
        ('mover_push_cap', float('nan')),
    ]
    for name, bad in bad_values:
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(cfg, **{name: bad})


# ---------------------------------------------------------------- Random crowded states


def _sim(batch: int, seed: int, **config: Any) -> Sim:
    """Crowd `27m_vs_30m` spawns into a small area with random waypoints and desired velocities
    (zero, equal, and varied speeds, random headings), plus coincident-center pairs."""
    rec = spawn_record('27m_vs_30m')
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    be = PyrefBackend(STATIC, {rec.map_id: load_map_info(rec.map_id)}, batch, PyrefConfig(**config))
    be.reset([sc] * batch, seeds=list(range(batch)))
    sim = be._sim
    assert sim is not None
    rng = np.random.default_rng(seed)
    w, buf = sim.world, sim.buf
    shape = w.x.shape
    cx, cy = np.nanmean(w.x), np.nanmean(w.y)
    span = 0.6 * np.sqrt(shape[1])
    w.x[:] = (cx + rng.uniform(-span, span, shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-span, span, shape)).astype(np.float32)
    w.x[:, :2] = w.x[:, 2:4]
    w.y[:, :2] = w.y[:, 2:4]
    w.alive[rng.random(shape) < 0.1] = False
    moving = rng.random(shape) < 0.7
    ang = rng.uniform(-np.pi, np.pi, shape)
    buf.waypoint_x[:] = np.where(moving, w.x + 3 * np.cos(ang), np.nan).astype(np.float32)
    buf.waypoint_y[:] = np.where(moving, w.y + 3 * np.sin(ang), np.nan).astype(np.float32)
    r = rng.random(shape)
    v = np.where(r < 0.15, 0.0, np.where(r < 0.5, sim.params.speed, rng.uniform(0.02, 0.4, shape)))
    va = rng.uniform(-np.pi, np.pi, shape)
    buf.vel_x[:] = np.where(moving, v * np.cos(va), 0).astype(np.float32)
    buf.vel_y[:] = np.where(moving, v * np.sin(va), 0).astype(np.float32)
    w.game_loop[:] = rng.integers(0, 6, batch)
    return sim


def _pin_some(sim: Sim, seed: int) -> None:
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
@pytest.mark.parametrize('cap', [0.8, 1.0])
@pytest.mark.parametrize('pin', [False, True])
def test_approach_sweep_matches_exact(
    seed: int, margin: float, small: int, cap: float, pin: bool, monkeypatch
) -> None:
    """Default, forced-chunked, and tiny-margin rescan paths match pairwise sweep bit-for-bit."""
    monkeypatch.setattr(collision, 'SWEEP_MARGIN', margin)
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _sim(
        3, seed, collision='hard', contact_slack=0.0625, idle_push_share=0.9, pin_engaged=pin,
        mover_push_share='approach', mover_push_cap=cap,
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


def _pair(cfg: PyrefConfig, unit: int = STALKER) -> Sim:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    sc = Scenario(MAP_ID, spawns=(Spawn(1, unit, 10.0, 10.0), Spawn(1, unit, 11.0, 10.0)))
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    return be._sim


@pytest.mark.parametrize('cap', [0.8, 1.0])
def test_approach_split_shares(cap: float) -> None:
    """Single push (u left, t right, n = +x): split proportionally by positive parts of
    `a_u = v_u,x` and `a_t = -v_t,x`, clamped to `[1 - cap, cap]`; 0.5/0.5 when both <= 0."""
    cfg = PyrefConfig(**BASE, mover_push_share='approach', mover_push_cap=cap)
    sim = _pair(cfg)
    w, buf = sim.world, sim.buf
    overlap = 2 * 0.625 - 0.0625 - 1.0
    cases = [
        # (v_u, v_t, u's share)
        ((0.18, 0.0), (0.18, 0.0), cap),  # u chases t: u yields (clamped to cap)
        ((0.1, 0.0), (-0.3, 0.0), 0.25),
        ((0.15, 0.05), (-0.15, 0.05), 0.5),
        ((-0.1, 0.0), (0.1, 0.0), 0.5),  # separating
        ((0.0, 0.18), (0.0, -0.18), 0.5),  # perpendicular
        ((0.0, 0.0), (0.0, 0.0), 0.5),
        ((0.0, 0.18), (-0.18, 0.0), 1 - cap),  # only t approaches u
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


# ---------------------------------------------------------------- Episodes (with resets)


def _crowd(k: int) -> tuple[Scenario, list[tuple[int, CommandKind, object]]]:
    """Crowd variant `k`: 5 Zerglings moving +x, 4 Marines +y, and 2 Zealots -x across the center,
    plus one HOLD Zealot and one idle Marine nearby."""
    rng = np.random.default_rng(k)
    cx, cy = 16.0 + rng.uniform(-1, 1), 16.0 + rng.uniform(-1, 1)
    spawns: list[Spawn] = []
    cmds: list[tuple[int, CommandKind, object]] = []
    for i in range(5):
        y = cy + (i - 2) * 0.8 + rng.uniform(-0.1, 0.1)
        spawns.append(Spawn(1, ZERGLING, cx - 5.0 + rng.uniform(-0.3, 0.3), y))
        cmds.append((len(spawns) - 1, CommandKind.MOVE, (cx + 5.0, y)))
    for i in range(4):
        x = cx + (i - 1.5) * 0.85 + rng.uniform(-0.1, 0.1)
        spawns.append(Spawn(1 + i % 2, MARINE, x, cy - 3.8 + rng.uniform(-0.3, 0.3)))
        cmds.append((len(spawns) - 1, CommandKind.MOVE, (x, cy + 4.0)))
    for i in range(2):
        y = cy + (i - 0.5) * 1.1
        spawns.append(Spawn(2, ZEALOT, cx + 4.0, y))
        cmds.append((len(spawns) - 1, CommandKind.MOVE, (cx - 4.0, y)))
    spawns.append(Spawn(1, ZEALOT, cx + 0.6, cy + 1.4))
    cmds.append((len(spawns) - 1, CommandKind.HOLD, None))
    spawns.append(Spawn(1, MARINE, cx - 0.8, cy - 0.9))
    return Scenario(MAP_ID, spawns=tuple(spawns)), cmds


def _command(batch: int, n: int, plans: dict[int, list]) -> Command:
    cmd = Command.noop(batch, n)
    for env, plan in plans.items():
        for slot, kind, tgt in plan:
            if tgt is None:
                cmd.set(env, slot, kind)
            else:
                cmd.set(env, slot, kind, target_xy=tgt)
    return cmd


def _episode(batch: int, config: dict[str, Any]) -> list[WorldState]:
    """Run `batch` envs for 12 steps (8 loops each), resetting half to new crowds at step 5."""
    be = PyrefBackend(STATIC, MAPS, batch, PyrefConfig(**config))
    first = [_crowd(k) for k in range(batch)]
    be.reset([s for s, _ in first], seeds=list(range(batch)))
    assert be._sim is not None
    n = be._sim.n_slots
    plans = {e: first[e][1] for e in range(batch)}
    states = []
    for step in range(12):
        if step == 5:
            ids = list(range(0, batch, 2))
            again = {e: _crowd(100 + e) for e in ids}
            be.reset([again[e][0] for e in ids], env_ids=ids, seeds=[7 + e for e in ids])
            plans = {e: again[e][1] for e in ids}
        states.append(be.step(_command(batch, n, plans), 8))
        plans = {}
    return states


def _exact_sweep(sim, ground_m, moving_m, k) -> None:
    for b in range(sim.batch_size):
        collision._sweep_exact(sim, b, np.flatnonzero(ground_m[b]).tolist(), moving_m[b], k)


@pytest.mark.parametrize('batch', [1, 8])
@pytest.mark.parametrize('cap', [False, True])
def test_approach_episode_matches_exact(batch: int, cap: bool, monkeypatch) -> None:
    config = {**BASE, 'mover_push_share': 'approach', 'mover_speed_cap': cap}
    pairs = [0]
    split = collision._approach_split

    def counted(ap, b, u, t, nx, ny):
        su, st = split(ap, b, u, t, nx, ny)
        pairs[0] += int(np.sum(su != st))
        return su, st

    monkeypatch.setattr(collision, '_approach_split', counted)
    got = _episode(batch, config)
    monkeypatch.setattr(collision, '_sweep', _exact_sweep)
    want = _episode(batch, config)
    assert pairs[0] > 5 * batch
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (k, name)


def test_other_shares_ignore_cap(monkeypatch) -> None:
    """`'half'` and `'speed'` skip `_Approach`: non-default `mover_push_cap` is bit-identical to
    default, whereas `'approach'` differs."""

    def boom(*_a, **_k):
        raise AssertionError("_Approach should not be called unless mover_push_share='approach'")

    monkeypatch.setattr(collision, '_Approach', boom)
    for share in ('half', 'speed'):
        cfg = {**BASE, 'mover_push_share': share}
        a = _episode(2, cfg)
        b = _episode(2, {**cfg, 'mover_push_cap': 0.55})
        for x, y in zip(a, b, strict=True):
            for name in WorldState.UNIT_FIELDS:
                assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), (share, name)
    monkeypatch.undo()
    c = _episode(2, {**BASE, 'mover_push_share': 'approach'})
    assert any(x.x.tobytes() != y.x.tobytes() for x, y in zip(a, c, strict=True))


# ---------------------------------------------------------------- Behavior


def _converge(**extra: Any) -> tuple[float, float]:
    """Two side-by-side Stalkers (center distance 0.0125 above contact) move to the same target:
    lower A walks straight +x while upper B converges diagonally downward. Runs 40 loops with
    `steering='none'` and returns y displacements of A and B."""
    options: dict[str, Any] = {**BASE, 'steering': 'none', **extra}
    cfg = PyrefConfig(**options)
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    y_a, y_b = 16.0, 16.0 + 1.2
    sc = Scenario(MAP_ID, spawns=(Spawn(1, STALKER, 8.0, y_a), Spawn(1, STALKER, 8.0, y_b)))
    be.reset([sc], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    for slot in (0, 1):
        cmd.set(0, slot, CommandKind.MOVE, target_xy=(24.0, y_a))
    st = be.step(cmd, 1)
    for _ in range(39):
        st = be.step(Command.noop(1, n), 1)
    return float(st.y[0, 0] - y_a), float(st.y[0, 1] - y_b)


def test_approach_converging_mover_yields() -> None:
    """Equal-speed converging pair: `'half'` (and `'speed'`) pushes straight-moving A downward;
    `'approach'` makes converging B yield more so A drifts far less (though A also approaches B
    slightly once displaced); `cap=1.0` keeps A strictly straight."""
    half_a, half_b = _converge()
    speed_a, _ = _converge(mover_push_share='speed', mover_push_power=32.0)
    ap_a, ap_b = _converge(mover_push_share='approach')
    one_a, one_b = _converge(mover_push_share='approach', mover_push_cap=1.0)
    assert half_a < -0.1
    assert speed_a == pytest.approx(half_a, abs=1e-6)
    assert -0.5 * abs(half_a) < ap_a < 0
    assert ap_b > half_b  # B drops less because B yields
    assert one_a == 0.0
    assert one_b > ap_b
