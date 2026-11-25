"""Tests for pyref `pin_lag`: one-loop lag before engaged or HOLD units become immovable.

`pin_engaged` treats ground units with an engaged target or HOLD and no waypoint in the current
loop as immovable. `pin_lag` additionally requires the condition to have held during collision
resolution on the previous loop (`motion.pinned_prev`), so a unit can still be pushed on the first
loop it engages or stops in range, and becomes immovable on subsequent loops.

Scenario matches test_pyref_pin_after_fire: player 2 Marine X in front with Marine Y directly
behind it both ATTACK_MOVE toward +x on loop 0; player 1 Marine D uses HOLD ahead. X enters
range and stops first while Y pushes into X. Disabled by default (bitwise-identical behavior).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import collision
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V4, _V5, _V6

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE = 48
RM = STATIC.get(MARINE).radius
X, Y, D = 0, 1, 2
CONTACT = 2 * RM - 0.0625  # contact_slack in aligned config
BASE = PyrefConfig(commands='delayed', command_delay=1, move_lag=1, **_V5)  # type: ignore[arg-type]
ON = dataclasses.replace(BASE, pin_lag=True)


def _scenario() -> Scenario:
    return Scenario(
        FLAT,
        spawns=(
            Spawn(2, MARINE, 10.0, 16.0),
            Spawn(2, MARINE, 10.0 - CONTACT, 16.0),
            Spawn(1, MARINE, 10.0 + 2 * RM + 7.0, 16.0),
        ),
    )


def _run(cfg: PyrefConfig, loops: int = 40) -> list[WorldState]:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        rows = [be.reset([_scenario()], seeds=[0])]
        for loop in range(loops):
            cmd = Command.noop(1, 3)
            if loop == 0:
                cmd.set(0, X, CommandKind.ATTACK_MOVE, target_xy=(26.0, 16.0))
                cmd.set(0, Y, CommandKind.ATTACK_MOVE, target_xy=(26.0, 16.0))
                cmd.set(0, D, CommandKind.HOLD)
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def _step(rows: list[WorldState], k: int, u: int) -> float:
    a, b = rows[k - 1], rows[k]
    return float(np.hypot(b.x[0, u] - a.x[0, u], b.y[0, u] - a.y[0, u]))


def _stop(rows: list[WorldState], u: int) -> int:
    """First loop after u engages where u does not move."""
    eng = next(k for k, s in enumerate(rows) if int(s.engaged_target_slot[0, u]) != NO_SLOT)
    return next(k for k in range(eng, len(rows)) if _step(rows, k, u) < 1e-6)


def test_defaults_and_validation() -> None:
    assert PyrefConfig().pin_lag is False
    with pytest.raises(ValueError, match='pin_lag.*pin_engaged'):
        PyrefConfig(pin_lag=True)
    for opts in (_V4, _V5, _V6):
        assert 'pin_lag' not in opts


def test_pinned_mask_needs_previous_loop() -> None:
    """Not pinned on the first loop the condition holds; becomes pinned on the next loop
    after collision resolution records it, and resets if the condition breaks for a loop."""
    be = PyrefBackend(STATIC, MAPS, 1, ON)
    try:
        be.reset([_scenario()], seeds=[0])
        sim = be._sim  # noqa: SLF001 - test mutates state directly
        assert sim is not None
        w = sim.world
        sim.buf.waypoint_x[:] = np.nan
        w.engaged_target_slot[0, X] = D
        w.order_kind[0, Y] = CommandKind.HOLD
        assert not collision.pinned_mask(sim)[0, [X, Y]].any()
        collision.hard(sim)
        assert collision.pinned_mask(sim)[0, [X, Y]].all()
        w.engaged_target_slot[0, X] = NO_SLOT
        collision.hard(sim)
        w.engaged_target_slot[0, X] = D
        assert not bool(collision.pinned_mask(sim)[0, X])
        assert bool(collision.pinned_mask(sim)[0, Y])
    finally:
        be.close()


def test_pushed_on_first_still_loop() -> None:
    """X is pushed once along +x on its first stationary loop in range, then stays still;
    when disabled, X stays still immediately. Both configs match bitwise before X stops."""
    off, on = _run(BASE), _run(ON)
    s_off = _stop(off, X)
    for f in WorldState.UNIT_FIELDS:
        for k in range(s_off):
            a, b = getattr(off[k], f), getattr(on[k], f)
            assert np.array_equal(a, b, equal_nan=True), (f, k)
    assert all(_step(off, k, X) < 1e-6 for k in range(s_off, len(off)))
    assert _step(on, s_off, X) > 0.05
    assert float(on[s_off].x[0, X] - on[s_off - 1].x[0, X]) > 0.05  # along center line (+x)
    assert all(_step(on, k, X) < 1e-6 for k in range(s_off + 1, len(on)))
    assert int(on[-1].engaged_target_slot[0, X]) == D
