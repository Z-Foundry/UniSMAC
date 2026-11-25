"""Tests for pyref `mover_speed_cap`.

Under `collision='hard'`, a moving unit's total single-loop displacement (kinematics plus
collision push) is capped at `max(max_speed, kinematic_step_len)`.

Scenario: two vertically offset Stalkers spawn overlapping and both MOVE to the same distant
point. Without the cap, collision pushes on top of a full-speed step make the leading unit
exceed its max speed per loop; with the cap enabled, neither unit exceeds `VMAX`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V4, _V5, _V6

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
STALKER = 74
VMAX = STATIC.get(STALKER).movement_speed / 16
BASE = PyrefConfig(commands='delayed', command_delay=1, move_lag=1, **_V5)  # type: ignore[arg-type]
ON = dataclasses.replace(BASE, mover_speed_cap=True)
PAIR = (Spawn(1, STALKER, 23.0, 16.0), Spawn(1, STALKER, 22.7, 15.0))
ALONE = (Spawn(1, STALKER, 23.0, 16.0),)
GOAL = (5.0, 15.4)


def _run(cfg: PyrefConfig, spawns: tuple[Spawn, ...], loops: int = 120) -> list[WorldState]:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        rows = [be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])]
        cmd = Command.noop(1, len(spawns))
        for u in range(len(spawns)):
            cmd.set(0, u, CommandKind.MOVE, target_xy=GOAL)
        rows.append(be.step(cmd, 1))
        for _ in range(loops):
            rows.append(be.step(Command.noop(1, len(spawns)), 1))
    finally:
        be.close()
    return rows


def _steps(rows: list[WorldState], u: int) -> np.ndarray:
    x = np.array([float(r.x[0, u]) for r in rows])
    y = np.array([float(r.y[0, u]) for r in rows])
    return np.hypot(np.diff(x), np.diff(y))


def test_defaults() -> None:
    assert PyrefConfig().mover_speed_cap is False
    for opts in (_V4, _V5, _V6):
        assert 'mover_speed_cap' not in opts


def test_pushed_mover_capped() -> None:
    off, on = _run(BASE, PAIR), _run(ON, PAIR)
    assert int((_steps(off, 1) > VMAX + 3e-3).sum()) >= 10
    for u in (0, 1):
        assert _steps(on, u).max() <= VMAX + 1e-5  # float32 coordinate rounding
    # Both units arrive (idle) and finish without exceeding contact_slack overlap
    r = float(STATIC.get(STALKER).radius)
    last = on[-1]
    assert [int(last.order_kind[0, u]) for u in (0, 1)] == [CommandKind.NOOP] * 2
    d = float(np.hypot(last.x[0, 0] - last.x[0, 1], last.y[0, 0] - last.y[0, 1]))
    assert d >= 2 * r - ON.contact_slack - 1e-4


def test_unpushed_unchanged() -> None:
    """An unpushed lone mover is bit-identical with or without the cap."""
    off, on = _run(BASE, ALONE), _run(ON, ALONE)
    for a, b in zip(off, on, strict=True):
        for name in WorldState.UNIT_FIELDS:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name
