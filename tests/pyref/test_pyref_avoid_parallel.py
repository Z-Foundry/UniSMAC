"""Tests for pyref `avoid_parallel_faster_only` (in `steering='sc2_avoid'` predicted
closest-approach avoidance, only steer around a parallel mover if unit u has higher max speed):

* Disabled by default;
* Batch `steering._avoid_batch` matches per-unit `steering._avoid_unit` bit-for-bit on random
  crowded states when enabled, and enabling it changes a subset of units;
* Two Marines walking side-by-side in the same direction walk straight when enabled;
* Same-speed Marine trailing a decelerating parallel Marine with lateral offset (tau > 0, where
  `avoid_closing_only` still steers): does not steer when enabled; higher max speed unchanged;
* Faster Zergling overtaking a Marine (lateral offset 0.25): bit-identical until passing, and
  the faster unit still steers around;
* Head-on offset and 45/90/135 deg crossing pairs are bit-identical to default when enabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.assets import load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import steering
from unismac.contract import Scenario, Spawn, WorldState

from .test_pyref_avoid_closing import BASE, MAP_ID, MAPS, MARINE, _pair, _run, _side_by_side
from .test_pyref_avoid_share import ZERGLING, _pin_some, _sim

STATIC = load_static_data()
OPT = {'avoid_parallel_faster_only': True}


def test_default_off() -> None:
    assert PyrefConfig().avoid_parallel_faster_only is False


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('share', ['full', 'speed'])
@pytest.mark.parametrize('pin', [False, True])
def test_batch_matches_per_unit(seed: int, share: str, pin: bool) -> None:
    cfg: dict[str, Any] = {
        'collision': 'hard', 'steering': 'sc2_avoid', 'contact_slack': 0.0625,
        'avoid_side_check': True, 'pin_engaged': pin, 'avoid_share': share,
        'avoid_share_power': 4.0,
    }  # fmt: skip
    outs = []
    for on in (False, True):
        sim = _sim(6, seed, **cfg, avoid_parallel_faster_only=on)
        if pin:
            _pin_some(sim, seed)
        sx, sy = steering._seek_all(sim)
        near = steering._avoid_candidates(sim, sx, sy)
        ox, oy = sx.copy(), sy.copy()
        rest = set(steering._avoid_batch(sim, sx, sy, near, ox, oy))
        outs.append((ox, oy))
        if not on:
            continue
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
    (x0, y0), (x1, y1) = outs
    assert ((x0 != x1) | (y0 != y1)).any()  # enabling must change at least some units


def test_side_by_side_neighbour() -> None:
    d0 = _side_by_side()
    d1 = _side_by_side(**OPT)
    assert d0[0, 1] < -0.1  # default: A steers away from B on loop 0
    assert np.abs(d1[:, 1]).max() < 1e-3  # enabled: A walks straight
    assert d1[:, 0].min() > 0.1


def _behind(u_speed: float | None, **opts: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Marines u (10, 10) and t (10.4, 10.6): edge dist 0.034, t ahead-left by 0.6, both
    heading +x; t moves at half max speed (decelerating) while u seeks +x at max speed.
    Predicted closest point is at tau ~ 5.7 loops with dist 0.6 < contact + avoid_margin.
    Overrides u's max speed if `u_speed` is given. Returns (seek velocity, avoided velocity)."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**BASE, **opts))
    spawns = (Spawn(1, MARINE, 10.0, 10.0), Spawn(1, MARINE, 10.4, 10.6))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    sim = be._sim
    assert sim is not None
    if u_speed is not None:
        sim.params.speed[0, 0] = np.float32(u_speed)
    buf, m = sim.buf, sim.motion
    buf.waypoint_x[0, :2] = (18.0, 18.4)
    buf.waypoint_y[0, :2] = (10.0, 10.6)
    m.speed[0, 1] = np.float32(0.5 * float(sim.params.speed[0, 1]))
    m.heading[0, 1] = 0.0
    spd = np.float32(sim.params.speed[0, 0])
    vx, vy = steering._avoid_unit(sim, 0, 0, spd, np.float32(0.0), [1])
    return (float(spd), 0.0), (float(vx), float(vy))


def test_same_speed_behind_slowing_neighbour() -> None:
    seek, v0 = _behind(None)
    _, v_co = _behind(None, avoid_closing_only=True)
    _, v1 = _behind(None, **OPT)
    assert v0[1] < -0.02  # default: steers away from t (-y)
    assert v_co == v0  # avoid_closing_only: still closing (tau > 0), unchanged
    assert v1 == seek  # enabled: equal max speed, does not steer


def test_faster_unit_unchanged() -> None:
    # u has higher max speed than t (Zergling speed 0.2578): unchanged when enabled
    _, v0 = _behind(0.2578125)
    _, v1 = _behind(0.2578125, **OPT)
    assert v0[1] < -0.02
    assert v1 == v0


def _overtake(**opts: Any) -> list[WorldState]:
    """Zergling 1.5 behind Marine with lateral offset 0.25; both MOVE +10 along +x."""
    spawns = (Spawn(1, MARINE, 10.0, 10.0), Spawn(1, ZERGLING, 8.5, 10.25))
    return _run(spawns, [(20.0, 10.0), (18.5, 10.25)], 80, **opts)


def test_overtake_unchanged_until_passing() -> None:
    a, b = _overtake(), _overtake(**OPT)
    passed = next(k for k, s in enumerate(a) if s.x[0, 1] > s.x[0, 0])
    assert 5 < passed < 60
    for k in range(passed + 1):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a[k], name).tobytes() == getattr(b[k], name).tobytes(), (k, name)
    for s in (a, b):  # faster unit steers around with max lateral deviation > 0.3
        dev = max(abs(float(x.y[0, 1]) - 10.25) for x in s)
        assert dev > 0.3


def _same(a: list[WorldState], b: list[WorldState]) -> None:
    for k, (x, y) in enumerate(zip(a, b, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), (k, name)


@pytest.mark.parametrize('angle', [45.0, 90.0, 135.0])
def test_crossing_pairs_unchanged(angle: float) -> None:
    _same(_pair(0.0, angle), _pair(0.0, angle, **OPT))


@pytest.mark.parametrize('dy', [0.0, 0.25, 0.5])
def test_head_on_pairs_unchanged(dy: float) -> None:
    """Two Marines moving head-on (A along +x, B along -x) with lateral offset dy."""
    spawns = (Spawn(1, MARINE, 10.0, 16.0), Spawn(1, MARINE, 22.0, 16.0 + dy))
    goals = [(22.0, 16.0), (10.0, 16.0 + dy)]
    a, b = _run(spawns, goals, 120), _run(spawns, goals, 120, **OPT)
    assert abs(float(a[-1].x[0, 0]) - 22.0) < 0.5
    _same(a, b)
