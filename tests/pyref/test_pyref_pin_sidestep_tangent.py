"""Tests for pyref `pin_sidestep_tangent`: tangent-direction sidestep around the nearest blocker.

When multiple pinned blockers fall within the sidestep cone, `pin_sidestep_tangent` picks the
nearest by center distance and directs the sidestep along its tangent (perpendicular to the u-to-t
center line on the chosen side) at the same speed.
* Option validation; disabled (default) matches omitting the option bitwise.
* Batch `steering._sidestep` matches the per-unit reference (`_ref`) bitwise on crowded states.
* Behavior: behind an obliquely offset (lateral 0.5, within 60 deg cone) HOLD Marine, sidestep
  alone does not advance whereas tangent mode advances while going around; directly behind
  (lateral 0), the first loop matches. With two HOLD Marines where the lower-slot one is farther
  at 45 deg and the higher-slot one is closer ahead (+y), `pin_sidestep` alone avoids the
  lower-slot unit (+y) while tangent mode avoids the nearest one (-y).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision, steering
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

from .test_pyref_pin_sidestep import BASE, MAP_ID, MAPS, MARINE, STATIC, _line, _path, _sim

ON: dict[str, Any] = {**BASE, 'pin_sidestep': True}
TAN: dict[str, Any] = {**ON, 'pin_sidestep_tangent': True}


def test_tangent_validated() -> None:
    assert not PyrefConfig().pin_sidestep_tangent
    with pytest.raises(ValueError, match='pin_sidestep_tangent.*pin_sidestep'):
        PyrefConfig(**BASE, pin_sidestep_tangent=True)
    assert PyrefConfig(**TAN).pin_sidestep_tangent


def _ref(sim: Sim, seek_x, seek_y, out_x, out_y) -> None:
    """Per-unit, per-blocker reference (Python float), extending
    test_pyref_pin_sidestep._sidestep_ref: with `pin_sidestep_tangent`, picks the nearest
    blocker by center distance (tie-broken by slot index) and steps along its tangent."""
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
        best = None  # (center_dist, rx, ry)
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
            if best is None or (cfg.pin_sidestep_tangent and dist < best[0]):
                best = (dist, rx, ry)
            if not cfg.pin_sidestep_tangent:
                break
        if best is None:
            continue
        dist, rx, ry = best
        lat = dx * ry - dy * rx
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
        if cfg.pin_sidestep_tangent:
            cx, cy = rx / dist, ry / dist
            nx, ny = (-cy, cx) if left else (cy, -cx)
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
@pytest.mark.parametrize('tangent', [False, True])
def test_tangent_batch_matches_per_unit(seed: int, tangent: bool, drift: bool) -> None:
    kw: dict[str, Any] = {**ON, 'pin_sidestep_drift': drift, 'pin_sidestep_tangent': tangent}
    bx, by = _run(_sim(6, seed, **kw), seed, steering._sidestep)
    rx, ry = _run(_sim(6, seed, **kw), seed, _ref)
    assert bx.tobytes() == rx.tobytes()
    assert by.tobytes() == ry.tobytes()
    if tangent:  # tangent mode changes sidestep direction for some units
        nx, ny = _run(_sim(6, seed, **ON, pin_sidestep_drift=drift), seed, steering._sidestep)
        assert ((nx != bx) | (ny != by)).sum() >= 3


def test_tangent_off_same_states() -> None:
    a = _line(0.2, ON)
    b = _line(0.2, {**ON, 'pin_sidestep_tangent': False})
    for x, y in zip(a, b, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), name


# ---------------------------------------------------------------- Behavior


def test_tangent_moves_forward_when_oblique() -> None:
    start = 16.0 - 0.69
    cone = {'pin_sidestep_cone': 60.0}
    # Lateral 0.5 (dist ~0.85, ~36 deg): sidestep alone stalls; tangent advances around
    ps = _path(_line(0.5, {**ON, **cone}), 1)
    tan = _path(_line(0.5, {**TAN, **cone}), 1)
    assert abs(ps[2, 0] - start) < 0.01
    assert tan[2, 0] - start > 0.1
    assert np.all(np.diff(tan[:3, 1]) < 0)  # moves away from H toward -y
    # Directly behind: loop 0 differs only by rounding; once offset, tangent advances more
    a = _path(_line(0.0, ON), 1)
    b = _path(_line(0.0, TAN), 1)
    assert np.abs(a[0] - b[0]).max() < 1e-4
    assert b[2, 0] > a[2, 0] + 0.02


def _two(config: dict[str, Any], loops: int = 4) -> np.ndarray:
    """Player 1 Marine A (slot 0) HOLDs at 45 deg ahead of M (dist 0.97, lat -0.68), B
    (slot 1) HOLDs right ahead of M offset +0.06 in y; M (slot 2) at (15.3, 16) MOVEs to
    (20, 16). Returns M positions."""
    spawns = (
        Spawn(1, MARINE, 15.3 + 0.69, 16.0 - 0.69),
        Spawn(1, MARINE, 15.3 + 0.75, 16.06),
        Spawn(1, MARINE, 15.3, 16.0),
        Spawn(2, MARINE, 28.0, 28.0),
    )
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    cmd = Command.noop(1, be._sim.n_slots)
    cmd.set(0, 0, CommandKind.HOLD)
    cmd.set(0, 1, CommandKind.HOLD)
    cmd.set(0, 2, CommandKind.MOVE, target_xy=(20.0, 16.0))
    out = []
    for k in range(loops):
        st = be.step(cmd if k == 0 else Command.noop(1, be._sim.n_slots), 1)
        out.append((float(st.x[0, 2]), float(st.y[0, 2])))
    be.close()
    return np.array(out)


def test_tangent_picks_nearest_blocker() -> None:
    cone = {'pin_sidestep_cone': 46.0}  # slight margin since A sits right at 45 deg
    ps = _two({**ON, **cone})
    tan = _two({**TAN, **cone})
    assert ps[1, 1] > 16.0 + 0.05  # lower-slot A at -y: moves +y (toward B)
    assert tan[1, 1] < 16.0 - 0.05  # nearest B at +y: moves -y
