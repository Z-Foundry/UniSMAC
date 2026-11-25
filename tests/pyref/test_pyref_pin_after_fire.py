"""Tests for pyref `pin_after_fire`: engaged units remain pushable until their first shot.

`pin_engaged` treats ground units with an engaged target and no waypoint in the current loop as
immovable. With `pin_after_fire`, only units that have already fired at their current engaged
target (`combat.last_fire_target`) are pinned; units that have engaged but not yet fired their
first shot can still be pushed aside by moving allies behind them.

Scenario: player 2 Marine X is in front with Marine Y directly behind it, both given ATTACK_MOVE
toward +x on loop 0; player 1 Marine D uses HOLD ahead (no auto-acquire, does not retaliate). X
enters weapon range and stops first while Y keeps moving and pushes into X. Disabled by default
with bitwise-identical behavior.
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
ON = dataclasses.replace(BASE, pin_after_fire=True)


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


def _first_fire(rows: list[WorldState], u: int) -> int:
    cd = [float(s.weapon_cooldown[0, u]) for s in rows]
    return next(k for k in range(1, len(cd)) if cd[k] > cd[k - 1] + 0.5)


def _stop(rows: list[WorldState], u: int) -> int:
    """First loop after u engages where u does not move."""
    eng = next(k for k, s in enumerate(rows) if int(s.engaged_target_slot[0, u]) != NO_SLOT)
    return next(k for k in range(eng, len(rows)) if _step(rows, k, u) < 1e-6)


def _step(rows: list[WorldState], k: int, u: int) -> float:
    a, b = rows[k - 1], rows[k]
    return float(np.hypot(b.x[0, u] - a.x[0, u], b.y[0, u] - a.y[0, u]))


def test_defaults_and_validation() -> None:
    assert PyrefConfig().pin_after_fire is False
    with pytest.raises(ValueError, match='pin_after_fire.*pin_engaged'):
        PyrefConfig(pin_after_fire=True)
    for opts in (_V4, _V5, _V6):
        assert 'pin_after_fire' not in opts


@pytest.mark.parametrize('after', [False, True])
def test_pinned_mask(after: bool) -> None:
    """Engaged unit with no waypoint: when enabled, pinned only if last_fire_target
    matches the engaged target; HOLD is always pinned."""
    be = PyrefBackend(STATIC, MAPS, 1, dataclasses.replace(BASE, pin_after_fire=after))
    try:
        be.reset([_scenario()], seeds=[0])
        sim = be._sim  # noqa: SLF001 - test mutates state directly
        assert sim is not None
        w = sim.world
        sim.buf.waypoint_x[:] = np.nan
        w.engaged_target_slot[0, X] = D
        w.order_kind[0, Y] = CommandKind.HOLD
        assert bool(collision.pinned_mask(sim)[0, X]) is not after
        assert bool(collision.pinned_mask(sim)[0, Y])
        sim.combat.last_fire_target[0, X] = D
        assert bool(collision.pinned_mask(sim)[0, X])
        sim.combat.last_fire_target[0, X] = Y  # last shot was at a different target
        assert bool(collision.pinned_mask(sim)[0, X]) is not after
    finally:
        be.close()


def test_pushed_until_first_fire() -> None:
    """After X stops in range, Y pushes X along +x until its first shot, then X stays
    still; when disabled, X stays still upon stopping. Both match bitwise beforehand."""
    off, on = _run(BASE), _run(ON)
    s_off = _stop(off, X)
    for f in WorldState.UNIT_FIELDS:
        for k in range(s_off):
            a, b = getattr(off[k], f), getattr(on[k], f)
            assert np.array_equal(a, b, equal_nan=True), (f, k)
    fire_off, fire_on = _first_fire(off, X), _first_fire(on, X)
    assert s_off < fire_off
    assert all(_step(off, k, X) < 1e-6 for k in range(s_off, len(off)))
    pushed = [k for k in range(s_off, fire_on + 1) if _step(on, k, X) > 0.05]
    assert pushed, 'expected X to be pushed before first fire'
    for k in pushed:
        assert float(on[k].x[0, X] - on[k - 1].x[0, X]) > 0.05  # along center line (+x)
    assert all(_step(on, k, X) < 1e-6 for k in range(fire_on + 1, len(on)))
    assert int(on[-1].engaged_target_slot[0, X]) == D
