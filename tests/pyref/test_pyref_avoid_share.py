"""Tests for pyref `avoid_share` (speed-weighted avoidance sharing between moving units):

* Option validation; when off ('full'), the new path is skipped and `avoid_share_power` has
  no effect;
* Turn-angle gain: slower unit uses 2 * v_u^p / (v_u^p + v_t^p), while equal-speed and faster
  units use exactly 1;
* Per-branch exponents (`avoid_share_power_headon` / `avoid_share_power_cross`): default to
  `avoid_share_power` (bit-identical when omitted), and each branch uses only its own p;
* Batch `steering._avoid_batch` matches per-unit `steering._avoid_unit` bit-for-bit on random
  crowded mixed-speed states and full episodes (batch size 1 and 8, with mid-episode resets).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import steering
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.scenarios import spawn_record

STATIC = load_static_data()
MAP_ID = 'SMAC_Maps/32x32_flat'
MAPS = {MAP_ID: load_map_info(MAP_ID)}
MARINE, ZEALOT, ZERGLING = 48, 73, 105
BASE: dict[str, Any] = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'crowd_arrival': True,
    'avoid_side_check': True,
}


def test_avoid_share_options_validated() -> None:
    cfg = PyrefConfig(**BASE, avoid_share='speed', avoid_share_power=4.0)
    bad_values: list[tuple[str, Any]] = [
        ('avoid_share', 'slow'),
        ('avoid_share_power', 0.0),
        ('avoid_share_power', -1.0),
        ('avoid_share_power', float('inf')),
        ('avoid_share_power', float('nan')),
    ]
    for name in ('avoid_share_power_headon', 'avoid_share_power_cross'):
        assert getattr(cfg, name) is None
        two: dict[str, Any] = {name: 2.0}
        assert getattr(dataclasses.replace(cfg, **two), name) == 2.0
        bad_values += [(name, 0.0), (name, -1.0), (name, float('inf')), (name, float('nan'))]
    for name, bad in bad_values:
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(cfg, **{name: bad})


def test_share_gain() -> None:
    """Slower unit gets min(1, 2*w_u/(w_u+w_t)) with w=v^p; equal, faster, or both-zero get 1."""
    be = PyrefBackend(
        STATIC, MAPS, 1, PyrefConfig(**BASE, avoid_share='speed', avoid_share_power=4.0)
    )
    be.reset([Scenario(MAP_ID, spawns=(Spawn(1, MARINE, 10.0, 10.0),))], seeds=[0])
    sim = be._sim
    assert sim is not None
    vu = np.array([0.14, 0.14, 0.18, 0.0, 0.0, 0.1], np.float32)
    vt = np.array([0.18, 0.14, 0.14, 0.0, 0.1, 0.2], np.float32)
    g = steering._share_gain(sim, vu, vt)
    a, b = np.float64(vu) ** 4, np.float64(vt) ** 4
    want = np.where(a + b > 0, 2 * a / np.where(a + b > 0, a + b, 1), 1)
    np.testing.assert_array_equal(g, np.minimum(want, 1.0))
    assert g[1] == 1.0 and g[2] == 1.0 and g[3] == 1.0 and g[4] == 0.0
    r = (0.14 / 0.18) ** 4
    assert g[0] == pytest.approx(2 * r / (1 + r), rel=1e-6)
    assert 0.0 < g[5] < g[0] < 1.0
    for i in range(len(vu)):
        assert steering._share_gain(sim, vu[i], vt[i]).tobytes() == g[i].tobytes()
    # Explicit p: bit-identical when matching avoid_share_power; p=2 is closer to 1 than p=4
    assert steering._share_gain(sim, vu, vt, 4.0).tobytes() == g.tobytes()
    g2 = steering._share_gain(sim, vu, vt, 2.0)
    r2 = (0.14 / 0.18) ** 2
    assert g2[0] == pytest.approx(2 * r2 / (1 + r2), rel=1e-6)
    assert g[0] < g2[0] < 1.0


def test_share_powers() -> None:
    """Branch exponents fall back to avoid_share_power when None, else use their own value."""
    cases = [
        ({'avoid_share_power': 4.0}, (4.0, 4.0)),
        ({'avoid_share_power': 4.0, 'avoid_share_power_headon': 2.0}, (2.0, 4.0)),
        ({'avoid_share_power_cross': 4.0}, (1.0, 4.0)),
        ({'avoid_share_power_headon': 2.0, 'avoid_share_power_cross': 6.0}, (2.0, 6.0)),
    ]
    for extra, want in cases:
        opts: dict[str, Any] = {**BASE, 'avoid_share': 'speed', 'avoid_share_power': 1.0, **extra}
        cfg = PyrefConfig(**opts)
        be = PyrefBackend(STATIC, MAPS, 1, cfg)
        be.reset([Scenario(MAP_ID, spawns=(Spawn(1, MARINE, 10.0, 10.0),))], seeds=[0])
        assert be._sim is not None
        assert steering._share_powers(be._sim) == want


# ---------------------------------------------------------------- Random crowded-state parity


def _sim(batch: int, seed: int, **config: Any) -> Sim:
    """Pack 27m_vs_30m spawns into a tight region with random waypoints, speeds, and headings
    (as in test_pyref_batch), sampling varied max speeds so slower units get gain < 1."""
    rec = spawn_record('27m_vs_30m')
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    be = PyrefBackend(STATIC, {rec.map_id: load_map_info(rec.map_id)}, batch, PyrefConfig(**config))
    be.reset([sc] * batch, seeds=list(range(batch)))
    sim = be._sim
    assert sim is not None
    rng = np.random.default_rng(seed)
    w, buf, m = sim.world, sim.buf, sim.motion
    shape = w.x.shape
    choices = np.array([0.0, 0.1171875, 0.140625, 0.1845703125, 0.2578125], np.float32)
    sim.params.speed[:] = choices[rng.integers(0, len(choices), shape)]
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


def _pin_some(sim: Sim, seed: int) -> None:
    rng = np.random.default_rng(400 + seed)
    w, shape = sim.world, sim.world.x.shape
    r = rng.random(shape)
    w.order_kind[r < 0.15] = CommandKind.HOLD
    w.engaged_target_slot[:] = np.where(
        (r >= 0.15) & (r < 0.35), rng.integers(0, sim.n_slots, shape), NO_SLOT
    )


# (avoid_share_power, avoid_share_power_headon, avoid_share_power_cross)
POWERS = [(1.0, None, None), (4.0, None, None), (4.0, 2.0, None), (1.0, 2.0, 4.0)]


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('powers', POWERS)
@pytest.mark.parametrize('side_check', [False, True])
@pytest.mark.parametrize('pin', [False, True])
def test_speed_batch_matches_per_unit(
    seed: int, powers: tuple, side_check: bool, pin: bool, monkeypatch
) -> None:
    sim = _sim(
        6, seed, collision='hard', steering='sc2_avoid', contact_slack=0.0625,
        avoid_side_check=side_check, pin_engaged=pin, avoid_share='speed',
        avoid_share_power=powers[0], avoid_share_power_headon=powers[1],
        avoid_share_power_cross=powers[2],
    )  # fmt: skip
    if pin:
        _pin_some(sim, seed)
    turned = [0]
    share_turn = steering._share_turn

    def counted(sx, sy, ax, ay, spd, g):
        turned[0] += int(np.size(g))
        return share_turn(sx, sy, ax, ay, spd, g)

    monkeypatch.setattr(steering, '_share_turn', counted)
    sx, sy = steering._seek_all(sim)
    near = steering._avoid_candidates(sim, sx, sy)
    ox, oy = sx.copy(), sy.copy()
    rest = set(steering._avoid_batch(sim, sx, sy, near, ox, oy))
    assert turned[0] >= 8
    units = list(zip(*np.nonzero(near.any(axis=2)), strict=True))
    assert len(units) > 100
    for b, u in units:
        b, u = int(b), int(u)
        if (b, u) in rest:
            continue
        vx, vy = steering._avoid_unit(
            sim, b, u, sx[b, u], sy[b, u], np.flatnonzero(near[b, u]).tolist()
        )
        assert np.float32(vx).tobytes() == ox[b, u].tobytes()
        assert np.float32(vy).tobytes() == oy[b, u].tobytes()


# ---------------------------------------------------------------- Full-episode parity (with resets)


def _crowd(k: int) -> tuple[Scenario, list[tuple[int, CommandKind, object]]]:
    """Variant k crowd: 5 Zerglings moving +x, 4 Marines +y, 2 Zealots -x across center, plus
    one HOLD Zealot and one idle Marine nearby. Returns scenario and initial commands."""
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
    """Run `batch` envs for 12 steps (8 loops each), partially resetting half after step 5."""
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


def _all_per_unit(sim, seek_x, seek_y, near, out_x, out_y) -> list[tuple[int, int]]:
    """Stub for `_avoid_batch` that delegates every unit with candidates to `_avoid_unit`."""
    return [(int(b), int(u)) for b, u in zip(*np.nonzero(near.any(axis=2)), strict=True)]


@pytest.mark.parametrize('batch', [1, 8])
@pytest.mark.parametrize('pin', [False, True])
@pytest.mark.parametrize('headon', [None, 2.0])
def test_speed_episode_matches_per_unit(
    batch: int, pin: bool, headon: float | None, monkeypatch
) -> None:
    config = {**BASE, 'pin_engaged': pin, 'avoid_share': 'speed', 'avoid_share_power': 4.0,
              'avoid_share_power_headon': headon}  # fmt: skip
    turned = [0]
    share_turn = steering._share_turn

    def counted(sx, sy, ax, ay, spd, g):
        turned[0] += int(np.size(g))
        return share_turn(sx, sy, ax, ay, spd, g)

    monkeypatch.setattr(steering, '_share_turn', counted)
    got = _episode(batch, config)
    assert turned[0] > 5 * batch
    monkeypatch.setattr(steering, '_avoid_batch', _all_per_unit)
    want = _episode(batch, config)
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (k, name)


def test_full_ignores_share_power(monkeypatch) -> None:
    """avoid_share='full' skips share logic: non-default powers match default bit-for-bit,
    while 'speed' produces different trajectories."""

    def boom(*_a, **_k):
        raise AssertionError("_share_gain must not be called when avoid_share='full'")

    monkeypatch.setattr(steering, '_share_gain', boom)
    a = _episode(4, BASE)
    b = _episode(4, {**BASE, 'avoid_share_power': 5.0, 'avoid_share_power_headon': 2.0,
                     'avoid_share_power_cross': 3.0})  # fmt: skip
    for x, y in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), name
    monkeypatch.undo()
    c = _episode(4, {**BASE, 'avoid_share': 'speed'})
    assert any(x.x.tobytes() != y.x.tobytes() for x, y in zip(a, c, strict=True))


def test_branch_powers_default_to_share_power() -> None:
    """Omitting both branch powers matches setting both to avoid_share_power bit-for-bit."""
    base = {**BASE, 'pin_engaged': True, 'avoid_share': 'speed', 'avoid_share_power': 4.0}
    a = _episode(4, base)
    b = _episode(4, {**base, 'avoid_share_power_headon': 4.0, 'avoid_share_power_cross': 4.0})
    for x, y in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), name
