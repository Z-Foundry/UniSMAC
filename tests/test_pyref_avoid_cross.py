"""Tests for pyref `avoid_cross_cos` (upper cosine threshold for the crossing branch in
`steering='sc2_avoid'`):

* Option validation; defaults to 0.9 (matching PARALLEL_COS for bit-identical baseline);
* Batch `steering._avoid_batch` matches per-unit `steering._avoid_unit` bit-for-bit on random
  crowded states;
* 45/90/135 deg crossings (cosine <= 0.71) at 0.8 match default bit-for-bit;
* Two Zealots converging at ~30 deg onto the same target: by default the trailing unit enters
  the crossing branch on contact and slides backward around the other's circle; at 0.8 it is
  handled as parallel offset and does not step backward.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import steering
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V5

from .test_pyref_avoid_share import _pin_some, _sim

STATIC = load_static_data()
MAP_ID = 'SMAC_Maps/32x32_flat'
MAPS = {MAP_ID: load_map_info(MAP_ID)}
ZEALOT = 73
BASE: dict[str, Any] = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'crowd_arrival': True,
    'avoid_side_check': True,
}
REPLAY: dict[str, Any] = {**_V5, 'commands': 'delayed', 'command_delay': 1, 'move_lag': 1}


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.avoid_cross_cos == 0.9 == steering.PARALLEL_COS
    for ok in (0.0, 0.8, 0.9):
        assert dataclasses.replace(cfg, avoid_cross_cos=ok).avoid_cross_cos == ok
    for bad in (-0.1, 0.95, 1.0, float('nan')):
        with pytest.raises(ValueError, match='avoid_cross_cos'):
            dataclasses.replace(cfg, avoid_cross_cos=bad)


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('share', ['full', 'speed'])
@pytest.mark.parametrize('pin', [False, True])
def test_batch_matches_per_unit(seed: int, share: str, pin: bool) -> None:
    sim = _sim(
        6, seed, collision='hard', steering='sc2_avoid', contact_slack=0.0625,
        avoid_side_check=True, pin_engaged=pin, avoid_share=share, avoid_share_power=4.0,
        avoid_cross_cos=0.8,
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


def _crossing(angle: float, cross_cos: float) -> list[WorldState]:
    """Two Zealots crossing at `angle` deg through (16, 16), each MOVEing to the opposite side."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**BASE, avoid_cross_cos=cross_cos))
    c, half = np.array([16.0, 16.0]), np.radians(angle / 2)
    da, db = np.array([np.cos(half), -np.sin(half)]), np.array([np.cos(half), np.sin(half)])
    a0, b0 = c - 6.0 * da, c - 6.0 * db
    be.reset([Scenario(MAP_ID, spawns=(Spawn(1, ZEALOT, *a0), Spawn(1, ZEALOT, *b0)))], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(float(c[0] + 6 * da[0]), float(c[1] + 6 * da[1])))
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(float(c[0] + 6 * db[0]), float(c[1] + 6 * db[1])))
    states = [be.step(cmd, 1)]
    states += [be.step(Command.noop(1, n), 1) for _ in range(100)]
    return states


@pytest.mark.parametrize('angle', [45.0, 90.0, 135.0])
def test_p2_crossings_unchanged(angle: float) -> None:
    a, b = _crossing(angle, 0.9), _crossing(angle, 0.8)
    for k, (x, y) in enumerate(zip(a, b, strict=True)):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(x, name).tobytes() == getattr(y, name).tobytes(), (k, name)


def _converge(cross_cos: float) -> np.ndarray:
    """Allied Zealots A (12, 16.8) and B (12.25, 15.6) both ATTACK_UNIT a HOLD enemy Zealot
    at (16.6, 16.1): B is ahead-right of A with headings ~30 deg apart. Returns per-loop dx."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**REPLAY, avoid_cross_cos=cross_cos))
    spawns = (Spawn(1, ZEALOT, 12.0, 16.8), Spawn(1, ZEALOT, 12.25, 15.6),
              Spawn(2, ZEALOT, 16.6, 16.1))  # fmt: skip
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    n = be._sim.n_slots
    cmd = Command.noop(1, n)
    for s in (0, 1):
        cmd.set(0, s, CommandKind.ATTACK_UNIT, target_slot=2)
    cmd.set(0, 2, CommandKind.HOLD)
    xs = [be.step(cmd, 1).x[0, :2].copy()]
    xs += [be.step(Command.noop(1, n), 1).x[0, :2].copy() for _ in range(40)]
    return np.diff(np.array(xs, dtype=float), axis=0)


def test_converging_zealots_do_not_step_back() -> None:
    dx0, dx8 = _converge(0.9), _converge(0.8)
    # Default: A takes the crossing branch on contact with B and steps backward (dx < 0)
    assert (dx0[:, 0] < -0.02).sum() >= 2
    assert (dx0[:, 1] >= -0.02).all()
    # 0.8: neither steps backward and A still advances to the target
    assert (dx8 >= -0.02).all()
    assert dx8[:, 0].sum() > 3.0
