"""Tests for pyref `avoid_closing_only` (skip predicted closest-approach avoidance in
`steering='sc2_avoid'` when a moving neighbor is not closing in):

* Disabled by default;
* Batch `steering._avoid_batch` matches per-unit `steering._avoid_unit` bit-for-bit on random
  crowded states when enabled;
* Two Marines walking side-by-side in the same direction with one slightly ahead: by default,
  the trailing unit's predicted closest point is the current distance (not closing, tau = 0),
  which is below contact + avoid_margin and causes a sideways turn; when enabled, it walks
  straight;
* Head-on offset and 45/90/135 deg crossing pairs are bit-identical to default when enabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import steering
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

from .test_pyref_avoid_share import _pin_some, _sim

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
}


def test_default_off() -> None:
    assert PyrefConfig().avoid_closing_only is False


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('share', ['full', 'speed'])
@pytest.mark.parametrize('pin', [False, True])
def test_batch_matches_per_unit(seed: int, share: str, pin: bool) -> None:
    sim = _sim(
        6, seed, collision='hard', steering='sc2_avoid', contact_slack=0.0625,
        avoid_side_check=True, pin_engaged=pin, avoid_share=share, avoid_share_power=4.0,
        avoid_closing_only=True,
    )  # fmt: skip
    if pin:
        _pin_some(sim, seed)
    sx, sy = steering._seek_all(sim)
    near = steering._avoid_candidates(sim, sx, sy)
    ox, oy = sx.copy(), sy.copy()
    rest = set(steering._avoid_batch(sim, sx, sy, near, ox, oy))
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


def _run(spawns: tuple, goals: list[tuple[float, float]], loops: int, **opts: Any) -> list:
    """Return initial post-reset state plus each loop's state (MOVE issued on loop 0)."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**BASE, **opts))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    for s, g in enumerate(goals):
        cmd.set(0, s, CommandKind.MOVE, target_xy=g)
    states = [be._sim.world.copy(), be.step(cmd, 1)]
    states += [be.step(Command.noop(1, n), 1) for _ in range(loops - 1)]
    return states


def _side_by_side(**opts: Any) -> np.ndarray:
    """Marines A (10, 10) and B (10.02, 10.6875) at contact distance with B slightly ahead;
    both MOVE +8 along +x. Both start at speed 0, so projected relative velocity along max
    speed toward waypoints is 0 (not closing, tau = 0). Returns A's per-loop deltas [loop, 2]."""
    spawns = (Spawn(1, MARINE, 10.0, 10.0), Spawn(1, MARINE, 10.02, 10.6875))
    states = _run(spawns, [(18.0, 10.0), (18.02, 10.6875)], 20, **opts)
    xy = np.array([[s.x[0, 0], s.y[0, 0]] for s in states], dtype=float)
    return np.diff(xy, axis=0)


def test_side_by_side_neighbour() -> None:
    d0 = _side_by_side()
    d1 = _side_by_side(avoid_closing_only=True)
    # Default: on loop 0, A steers away from B (-y) with lateral delta > forward delta
    assert d0[0, 1] < -0.1
    assert 0.0 < d0[0, 0] < abs(d0[0, 1])
    # Enabled: A walks straight
    assert np.abs(d1[:, 1]).max() < 1e-3
    assert d1[:, 0].min() > 0.1


def _pair(dy: float, angle: float, **opts: Any) -> list[WorldState]:
    """Two Marines moving head-on (angle=180) or crossing at `angle` deg through (16, 16),
    offset laterally by dy."""
    c, half = np.array([16.0, 16.0]), np.radians(angle / 2)
    da, db = np.array([np.cos(half), -np.sin(half)]), np.array([-np.cos(half), -np.sin(half)])
    off = np.array([-da[1], da[0]]) * dy / 2
    a0, b0 = c - 6.0 * da + off, c - 6.0 * db - off
    a1, b1 = c + 6.0 * da + off, c + 6.0 * db - off
    spawns = (Spawn(1, MARINE, *a0), Spawn(1, MARINE, *b0))
    goals = [(float(a1[0]), float(a1[1])), (float(b1[0]), float(b1[1]))]
    return _run(spawns, goals, 120, **opts)


@pytest.mark.parametrize(
    ('dy', 'angle'), [(0.0, 180.0), (0.25, 180.0), (0.5, 180.0), (0.0, 45.0), (0.0, 90.0),
                      (0.0, 135.0)],
)  # fmt: skip
def test_closing_pairs_unchanged(dy: float, angle: float) -> None:
    a, b = _pair(dy, angle), _pair(dy, angle, avoid_closing_only=True)
    moved = np.hypot(a[-1].x[0] - a[0].x[0], a[-1].y[0] - a[0].y[0])
    assert (moved > 5.0).all()
    for k, (x, y) in enumerate(zip(a, b, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), (k, name)
