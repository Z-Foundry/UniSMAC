"""Tests for pyref `pin_sidestep_lat`: lateral distance filter for `pin_sidestep` blockers.

A blocker triggers sidestepping only when its lateral distance from unit u's seek line is strictly
less than `pin_sidestep_lat`.
* Option validation; `None` matches a threshold larger than maximum contact range bitwise.
* Batch `steering._sidestep` matches the per-unit reference (`_ref`) bitwise on crowded states.
* Behavior: a MOVE Marine behind an obliquely offset (lateral 0.5) HOLD Marine sidesteps without
  advancing under `pin_sidestep` alone, but advances immediately with `pin_sidestep_lat=0.42`. At
  lateral offset 0.2 both configs sidestep identically at first.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from unismac.backends.pyref import PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision, steering
from unismac.contract import CommandKind, WorldState

from .test_pyref_pin_sidestep import BASE, _line, _path, _sim

ON: dict[str, Any] = {**BASE, 'pin_sidestep': True}


def test_sidestep_lat_validated() -> None:
    for bad in (0.0, -0.4, math.nan, math.inf):
        with pytest.raises(ValueError, match='pin_sidestep_lat'):
            PyrefConfig(**ON, pin_sidestep_lat=bad)
    cfg = PyrefConfig(**ON, pin_sidestep_lat=0.42)
    assert cfg.pin_sidestep_lat == 0.42
    assert PyrefConfig().pin_sidestep_lat is None
    assert dataclasses.replace(cfg, pin_sidestep_lat=None).pin_sidestep_lat is None


def _ref(sim: Sim, seek_x, seek_y, out_x, out_y) -> None:
    """Per-unit, per-blocker reference (Python float), extending
    test_pyref_pin_sidestep._sidestep_ref with the lateral filter (restricted to pairs
    involving a Marauder when `pin_sidestep_lat_scope='marauder'`)."""
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
            cross = dx * ry - dy * rx
            pair = cfg.pin_sidestep_lat_scope == 'all' or steering.MARAUDER in (
                int(w.unit_type[b, u]),
                int(w.unit_type[b, t]),
            )  # pin_sidestep_lat_scope (see test_pyref_pin_sidestep_lat_scope.py)
            if cfg.pin_sidestep_lat is not None and pair and abs(cross) >= cfg.pin_sidestep_lat:
                continue
            lat = cross
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
            if abs(drift) > steering.SIDESTEP_DRIFT_TIE:
                left = drift > 0
        nx, ny = (-dy, dx) if left else (dy, -dx)
        out_x[b, u] = np.float32(nx * spd * float(cfg.pin_sidestep_speed))
        out_y[b, u] = np.float32(ny * spd * float(cfg.pin_sidestep_speed))


def _run(sim: Sim, seed: int, fn) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = steering._seek_all(sim)
    rng = np.random.default_rng(100 + seed)
    m = sim.motion
    ang = rng.uniform(-np.pi, np.pi, m.drift_px.shape)
    step = np.where(rng.random(ang.shape) < 0.15, np.nan, rng.uniform(0.0, 0.14, ang.shape))
    m.drift_px[:] = (sim.world.x - step * np.cos(ang)).astype(np.float32)
    m.drift_py[:] = (sim.world.y - step * np.sin(ang)).astype(np.float32)
    ox = (sx + rng.normal(0, 0.03, sx.shape)).astype(np.float32)
    oy = (sy + rng.normal(0, 0.03, sy.shape)).astype(np.float32)
    fn(sim, sx, sy, ox, oy)
    return ox, oy


@pytest.mark.parametrize('drift', [False, True])
@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('lat', [0.2, 0.42])
def test_sidestep_lat_batch_matches_per_unit(seed: int, lat: float, drift: bool) -> None:
    kw: dict[str, Any] = {**ON, 'pin_sidestep_drift': drift}
    bx, by = _run(_sim(6, seed, **kw, pin_sidestep_lat=lat), seed, steering._sidestep)
    rx, ry = _run(_sim(6, seed, **kw, pin_sidestep_lat=lat), seed, _ref)
    assert bx.tobytes() == rx.tobytes()
    assert by.tobytes() == ry.tobytes()
    # Lateral filter removes some sidesteps (more at 0.2, fewer at 0.42)
    nx, ny = _run(_sim(6, seed, **kw), seed, steering._sidestep)
    assert ((nx != bx) | (ny != by)).sum() >= (3 if lat < 0.3 else 1)


@pytest.mark.parametrize('seed', range(4))
def test_sidestep_lat_wide_equals_none(seed: int) -> None:
    """Lateral distance <= center dist <= contact + 0.35, so 9.0 always passes and matches
    None bitwise."""
    a = _run(_sim(6, seed, **ON, pin_sidestep_lat=9.0), seed, steering._sidestep)
    b = _run(_sim(6, seed, **ON), seed, steering._sidestep)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


def test_sidestep_lat_none_same_states() -> None:
    a = _line(0.5, ON)
    b = _line(0.5, {**ON, 'pin_sidestep_lat': None})
    for x, y in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), name


# ---------------------------------------------------------------- Behavior


def test_sidestep_lat_moves_forward_when_offset() -> None:
    start = 16.0 - 0.69
    # Lateral 0.5 (~36 deg): sidestep alone stalls; lat=0.42 advances immediately
    ps = _path(_line(0.5, ON), 1)
    on = _path(_line(0.5, {**ON, 'pin_sidestep_lat': 0.42}), 1)
    assert abs(ps[2, 0] - start) < 0.01
    assert on[2, 0] - start > 0.1
    # Lateral 0.2: both sidestep for 3 loops; once lateral > 0.42, lat advances sooner
    a = _path(_line(0.2, ON), 1)
    b = _path(_line(0.2, {**ON, 'pin_sidestep_lat': 0.42}), 1)
    assert a[:3].tobytes() == b[:3].tobytes()
    assert abs(a[2, 0] - start) < 0.01
    assert b[-1, 0] > a[-1, 0]
