"""Tests for pyref `pin_sidestep` and `pin_sidestep_drift`: lateral sidestep around pinned blockers.

`pin_sidestep` makes ground units executing MOVE step sideways when an immovable unit is directly
in front of them:
* Option validation; disabled mode skips the new path completely (pin_sidestep_cone and
  pin_sidestep_speed have no effect).
* Batch implementation `steering._sidestep` matches the per-unit reference bitwise on random
  crowded states.
* Behavior: a MOVE Marine directly behind a HOLD Marine sidesteps first without advancing, moving
  away from the HOLD unit; when disabled, it slides around the outer edge and advances immediately.

`pin_sidestep_drift` preserves sidestep side using previous-loop displacement:
* Option validation and bitwise parity between the batch implementation and reference.
* With three HOLD Marines in a row and a MOVE Marine aimed at the middle, `pin_sidestep` alone
  oscillates back and forth without passing, whereas adding `pin_sidestep_drift` keeps moving to
  one side and clears the row.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision, steering
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.scenarios import spawn_record

STATIC = load_static_data()
MAP_ID = 'SMAC_Maps/32x32_flat'
MAPS = {MAP_ID: load_map_info(MAP_ID)}
MARINE = 48
BASE: dict[str, Any] = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'crowd_arrival': True,
    'avoid_side_check': True,
    'pin_engaged': True,
    'pin_slide': True,
}


def test_sidestep_options_validated() -> None:
    with pytest.raises(ValueError, match='pin_sidestep'):
        PyrefConfig(**(BASE | {'pin_engaged': False, 'pin_sidestep': True}))
    with pytest.raises(ValueError, match='pin_sidestep'):
        PyrefConfig(pin_engaged=True, pin_sidestep=True)  # steering='none'
    cfg = PyrefConfig(**BASE, pin_sidestep=True)
    for name, bad in (
        ('pin_sidestep_cone', 0.0),
        ('pin_sidestep_cone', 91.0),
        ('pin_sidestep_speed', 0.0),
        ('pin_sidestep_speed', 1.5),
    ):
        kw: dict[str, Any] = {name: bad}
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(cfg, **kw)
    with pytest.raises(ValueError, match='pin_sidestep_drift'):
        PyrefConfig(**BASE, pin_sidestep_drift=True)
    assert PyrefConfig(**BASE, pin_sidestep=True, pin_sidestep_drift=True).pin_sidestep_drift


# ---------------------------------------------------------------- Batch vs reference on crowds


def _sim(batch: int, seed: int, **config: Any) -> Sim:
    """Pack 27m_vs_30m spawns into a tight cluster with random waypoints, orders, and
    engaged targets (same setup as test_pyref_contact_orbit)."""
    rec = spawn_record('27m_vs_30m')
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    maps = {rec.map_id: load_map_info(rec.map_id)}
    be = PyrefBackend(STATIC, maps, batch, PyrefConfig(**config))
    be.reset([sc] * batch, seeds=list(range(batch)))
    sim = be._sim
    assert sim is not None
    rng = np.random.default_rng(seed)
    w, buf = sim.world, sim.buf
    shape = w.x.shape
    cx, cy = np.nanmean(w.x), np.nanmean(w.y)
    span = 0.4 * np.sqrt(shape[1])
    w.x[:] = (cx + rng.uniform(-span, span, shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-span, span, shape)).astype(np.float32)
    w.x[:, :2] = w.x[:, 2:4]  # overlapping-center pairs
    w.y[:, :2] = w.y[:, 2:4]
    w.alive[rng.random(shape) < 0.1] = False
    moving = rng.random(shape) < 0.55
    ang = rng.uniform(-np.pi, np.pi, shape)
    reach = np.where(rng.random(shape) < 0.1, 0.05, 3.0)  # some reach waypoint on last step
    buf.waypoint_x[:] = np.where(moving, w.x + reach * np.cos(ang), np.nan).astype(np.float32)
    buf.waypoint_y[:] = np.where(moving, w.y + reach * np.sin(ang), np.nan).astype(np.float32)
    r = rng.random(shape)
    w.order_kind[:] = CommandKind.MOVE
    w.order_kind[r < 0.25] = CommandKind.HOLD
    w.order_kind[(r >= 0.25) & (r < 0.45)] = CommandKind.ATTACK_UNIT
    w.order_kind[(r >= 0.45) & (r < 0.55)] = CommandKind.ATTACK_MOVE
    w.order_target_slot[:] = np.where(
        (r >= 0.25) & (r < 0.45), rng.integers(0, sim.n_slots, shape), NO_SLOT
    )
    w.engaged_target_slot[:] = np.where(
        rng.random(shape) < 0.3, rng.integers(0, sim.n_slots, shape), NO_SLOT
    )
    return sim


def _sidestep_ref(sim: Sim, seek_x, seek_y, out_x, out_y) -> None:
    """Per-unit, per-blocker reference implementation (Python float arithmetic)."""
    w, cfg = sim.world, sim.config
    pin = collision.pinned_mask(sim)
    cos_cone = float(np.cos(np.radians(cfg.pin_sidestep_cone)))
    vx0, vy0 = out_x.copy(), out_y.copy()
    prev = (sim.motion.drift_px.copy(), sim.motion.drift_py.copy())
    for b, u in zip(*np.nonzero((seek_x != 0) | (seek_y != 0)), strict=True):
        if w.is_flying[b, u] or not w.alive[b, u] or int(w.order_kind[b, u]) != CommandKind.MOVE:
            continue
        spd = float(np.hypot(seek_x[b, u], seek_y[b, u]))
        dx, dy = float(seek_x[b, u]) / spd, float(seek_y[b, u]) / spd
        own = (int(w.engaged_target_slot[b, u]), int(w.order_target_slot[b, u]))
        lat = None
        for t in np.flatnonzero(pin[b]).tolist():
            if t == u or t in own:
                continue
            rx, ry = float(w.x[b, t] - w.x[b, u]), float(w.y[b, t] - w.y[b, u])
            dist = float(np.hypot(rx, ry))
            contact = float(w.radius[b, u] + w.radius[b, t]) - cfg.contact_slack
            ahead = rx * dx + ry * dy
            if dist <= 0 or dist > contact + steering.SIDESTEP_RANGE or ahead <= 0:
                continue
            if ahead / dist < cos_cone:
                continue
            lat = dx * ry - dy * rx
            break
        if lat is None:
            continue
        left = dx * float(vy0[b, u]) - dy * float(vx0[b, u]) > 1e-9
        if abs(lat) > steering.SIDESTEP_TIE:
            left = lat < 0
        if cfg.pin_sidestep_drift:
            mvx = float(w.x[b, u]) - float(prev[0][b, u])
            mvy = float(w.y[b, u]) - float(prev[1][b, u])
            drift = dx * mvy - dy * mvx
            if abs(drift) > steering.SIDESTEP_DRIFT_TIE:  # False when NaN (no prev loop)
                left = drift > 0
        nx, ny = (-dy, dx) if left else (dy, -dx)
        out_x[b, u] = np.float32(nx * spd * float(cfg.pin_sidestep_speed))
        out_y[b, u] = np.float32(ny * spd * float(cfg.pin_sidestep_speed))


@pytest.mark.parametrize('drift', [False, True])
@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize(('cone', 'speed'), [(45.0, 0.75), (60.0, 1.0), (90.0, 0.3)])
def test_sidestep_batch_matches_per_unit(seed: int, cone: float, speed: float, drift: bool) -> None:
    sim = _sim(6, seed, **BASE, pin_sidestep=True, pin_sidestep_cone=cone, pin_sidestep_speed=speed,
               pin_sidestep_drift=drift)  # fmt: skip
    sx, sy = steering._seek_all(sim)
    rng = np.random.default_rng(100 + seed)
    m = sim.motion  # prev-loop positions: random steps, some NaN, some along seek dir
    ang = rng.uniform(-np.pi, np.pi, m.drift_px.shape)
    ang = np.where(rng.random(ang.shape) < 0.15, np.arctan2(sy, sx), ang)
    step = np.where(rng.random(ang.shape) < 0.15, np.nan, rng.uniform(0.0, 0.14, ang.shape))
    m.drift_px[:] = (sim.world.x - step * np.cos(ang)).astype(np.float32)
    m.drift_py[:] = (sim.world.y - step * np.sin(ang)).astype(np.float32)
    ox = (sx + rng.normal(0, 0.03, sx.shape)).astype(np.float32)  # proxy for sc2_avoid
    oy = (sy + rng.normal(0, 0.03, sy.shape)).astype(np.float32)
    bx, by, rx, ry = ox.copy(), oy.copy(), ox.copy(), oy.copy()
    _sidestep_ref(sim, sx, sy, rx, ry)
    steering._sidestep(sim, sx, sy, bx, by)
    assert bx.tobytes() == rx.tobytes()
    assert by.tobytes() == ry.tobytes()
    changed = (bx != ox) | (by != oy)
    assert changed.sum() >= 5


# ---------------------------------------------------------------- Disabled skips new code


def _line(dy: float, config: dict[str, Any], steps: int = 2) -> list[WorldState]:
    """Player 1 Marine H (slot 0) HOLDs at (16, 16 + dy); Marine M (slot 1) starts directly
    behind it (center dist 0.69) and MOVEs to (20, 16); a distant player 2 Marine idles.
    Returns per-loop states."""
    spawns = (
        Spawn(1, MARINE, 16.0, 16.0 + dy),
        Spawn(1, MARINE, 16.0 - 0.69, 16.0),
        Spawn(2, MARINE, 28.0, 28.0),
    )
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    cmd = Command.noop(1, be._sim.n_slots)
    cmd.set(0, 0, CommandKind.HOLD)
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(20.0, 16.0))
    out = [be.step(cmd, 1)]
    for _ in range(8 * steps - 1):
        out.append(be.step(Command.noop(1, be._sim.n_slots), 1))
    be.close()
    return out


def test_sidestep_off_skips_new_code(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError('_sidestep must not be called when pin_sidestep=False')

    monkeypatch.setattr(steering, '_sidestep', boom)
    a = _line(0.0, BASE)
    b = _line(0.0, {**BASE, 'pin_sidestep_cone': 80.0, 'pin_sidestep_speed': 0.2})
    for x, y in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), name


# ---------------------------------------------------------------- Behavior


def _path(states: list[WorldState], slot: int) -> np.ndarray:
    return np.array([(float(s.x[0, slot]), float(s.y[0, slot])) for s in states])


@pytest.mark.parametrize('dy', [0.0, 0.2])
def test_sidestep_moves_sideways_behind_hold(dy: float) -> None:
    on = _path(_line(dy, {**BASE, 'pin_sidestep': True}), 1)
    off = _path(_line(dy, BASE), 1)
    hold = _path(_line(dy, {**BASE, 'pin_sidestep': True}), 0)
    assert np.abs(hold - hold[0]).max() < 1e-6  # HOLD unit is not pushed
    # Enabled: while in contact within 45 deg cone, stalls forward and sidesteps at ~0.75 vmax
    step = np.diff(np.vstack([[16.0 - 0.69, 16.0], on]), axis=0)
    first = step[:3]
    assert np.abs(first[:, 0]).max() < 0.01
    assert np.all(np.abs(first[:, 1]) > 0.09)
    if dy > 0:
        assert np.all(first[:, 1] < 0)  # H offset toward +y: sidestep toward -y
    # Disabled: slides around outer edge, already advancing in the first 3 loops
    assert off[2, 0] - (16.0 - 0.69) > 0.05
    # After clearing the blocker, resumes toward waypoint and advances past x = 16
    assert on[-1, 0] > 16.0
    assert math.isfinite(float(on[-1, 1]))


# ---------------------------------------------------------------- pin_sidestep_drift: row


def _row(wy: float, config: dict[str, Any], loops: int = 60) -> np.ndarray:
    """Three player 1 Marines HOLD at x = 16, y = 15.3 / 16 / 16.7 (spacing 0.7); slot 3
    MOVEs from (14.95, wy) to (20, wy). Returns slot 3 positions per loop."""
    spawns = (*(Spawn(1, MARINE, 16.0, y) for y in (15.3, 16.0, 16.7)), Spawn(1, MARINE, 14.95, wy),
              Spawn(2, MARINE, 28.0, 28.0))  # fmt: skip
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    cmd = Command.noop(1, be._sim.n_slots)
    for s in range(3):
        cmd.set(0, s, CommandKind.HOLD)
    cmd.set(0, 3, CommandKind.MOVE, target_xy=(20.0, wy))
    out = []
    for k in range(loops):
        st = be.step(cmd if k == 0 else Command.noop(1, be._sim.n_slots), 1)
        out.append((float(st.x[0, 3]), float(st.y[0, 3])))
    be.close()
    return np.array(out)


def _flips(p: np.ndarray) -> int:
    dy = np.diff(p[:, 1])
    s = np.sign(dy[np.abs(dy) > 0.03])
    return int(np.sum(s[1:] != s[:-1]))


@pytest.mark.parametrize('wy', [16.0, 16.2, 16.35])
def test_sidestep_drift_keeps_side_along_row(wy: float) -> None:
    clear = 16.0 + 0.75 + 0.3  # cleared: x past row + two radii + 0.3
    ps = _row(wy, {**BASE, 'pin_sidestep': True})
    assert ps[:, 0].max() < clear  # pin_sidestep alone flips side when blocker t changes
    assert _flips(ps) >= 20
    on = _row(wy, {**BASE, 'pin_sidestep': True, 'pin_sidestep_drift': True})
    passed = int(np.argmax(on[:, 0] > clear))
    assert on[passed, 0] > clear and passed < 40
    behind = int(np.argmax(on[:, 0] > 15.6))  # loops still sidestepping behind the row
    assert behind > 5 and _flips(on[:behind]) == 0  # commits to one side
    assert abs(on[passed, 1] - 16.0) > 1.0  # cleared past one end of the row
