"""Tests for pyref `pin_tie_grid`: coordinate-grid tie-breaking around pinned blockers.

* Option validation; disabled mode never invokes `steering._pin_tie_side` or `_pin_group`.
* `_pin_aim`: in an 8m-style formation (a0 stops to fire, a2 ahead, a3 to one side, a5/a6
  diagonally ahead on both sides, a4 behind a0 moving forward) with equal asymmetric extents on
  both sides, disabled defaults to the right side; enabled chooses the side from a0's lateral
  coordinate, flipping every 1/8 tile and repeating every 1/4 tile, preserving the world direction
  when travel reverses, and remaining invariant to small a4 lateral offsets (+/-8/4096 tiles).
  Removing a3 (symmetric) or making a0 HOLD falls back to right. For a 3-unit row, an offset of
  1/4096 tile picks the smaller-extent side while exact symmetry goes right.
* Full-loop stepping: a4's first lateral excursion > 0.05 tiles matches `_pin_aim` both with
  `pin_engaged` alone and with `pin_sidestep` + `pin_sidestep_drift`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision, steering
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn

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
PSD = {'pin_sidestep': True, 'pin_sidestep_drift': True}
Q = 2820 / 4096  # lateral offset of a3 relative to a0 in 8m formation
# Formation offsets (forward, left) relative to a0: a0, a2, a3, a5, a6 (pinned), a4 (MOVE)
FRONT = ((0.0, 0.0), (0.69, 0.0), (0.0, Q), (0.75, -0.75), (0.75, 0.75))
A4 = (-0.69, 0.0)


def test_tie_grid_options_validated() -> None:
    with pytest.raises(ValueError, match='pin_tie_grid'):
        PyrefConfig(**(BASE | {'pin_engaged': False, 'pin_tie_grid': True}))
    with pytest.raises(ValueError, match='pin_tie_grid'):
        PyrefConfig(pin_engaged=True, pin_tie_grid=True)  # steering='none'
    assert PyrefConfig(**BASE, pin_tie_grid=True).pin_tie_grid


def _spawns(y0: float, sgn: float, front: tuple, off4: float) -> tuple[list[Spawn], int]:
    """Place a0 at (16, y0) with travel dir sgn * +x, front units at `front`, and a4 behind
    a0 offset laterally by off4; returns (spawns, a4_slot). Last spawn is a player 2 Marine
    4 tiles ahead of a0 (a0's ATTACK_UNIT target)."""
    pos = (*front, (A4[0], A4[1] + off4))
    own = [Spawn(1, MARINE, 16.0 + sgn * f, y0 + sgn * lt) for f, lt in pos]
    return [*own, Spawn(2, MARINE, 16.0 + sgn * 4.0, y0)], len(own) - 1


def _backend(
    y0: float, config: dict[str, Any], sgn: float = 1.0, drop_a3: bool = False,
    front: tuple | None = None, off4: float = 0.0,
):  # fmt: skip
    if front is None:
        front = tuple(p for i, p in enumerate(FRONT) if not (drop_a3 and i == 2))
    spawns, w = _spawns(y0, sgn, front, off4)
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))
    be.reset([Scenario(MAP_ID, spawns=tuple(spawns))], seeds=[0])
    assert be._sim is not None
    return be, w, len(spawns) - 1


def _pin_state(sim: Sim, w: int, enemy: int, hold_a0: bool) -> None:
    """Set a0 engaged (or HOLD) and remaining front units to HOLD with no waypoints."""
    wd, buf = sim.world, sim.buf
    for s in range(w):
        wd.order_kind[0, s] = CommandKind.HOLD
        buf.waypoint_x[0, s] = np.nan
        buf.waypoint_y[0, s] = np.nan
    if not hold_a0:
        wd.order_kind[0, 0] = CommandKind.ATTACK_UNIT
        wd.order_target_slot[0, 0] = enemy
        wd.engaged_target_slot[0, 0] = enemy
    wd.order_kind[0, w] = CommandKind.MOVE
    wd.engaged_target_slot[0, w] = NO_SLOT


def _aim_side(
    y0: float, config: dict[str, Any], sgn: float = 1.0, drop_a3: bool = False,
    hold_a0: bool = False, front: tuple | None = None, off4: float = 0.0, t: int = 0,
) -> str:  # fmt: skip
    """Return '+' or '-' for world-y side of a4's `_pin_aim` point around t (default a0)."""
    be, w, enemy = _backend(y0, config, sgn, drop_a3, front, off4)
    sim = be._sim
    assert sim is not None
    _pin_state(sim, w, enemy, hold_a0)
    pin = collision.pinned_mask(sim)[0]
    assert pin[:w].all() and not pin[w]
    own = (NO_SLOT, NO_SLOT)
    wd = sim.world
    lat = sgn * float(wd.y[0, t] - wd.y[0, w])
    right = lat >= 0 if abs(lat) >= steering.TIE_EPS else True
    py = steering._pin_aim(sim, 0, w, t, sgn, 0.0, right, pin, own)[1]
    uy = float(sim.world.y[0, w])
    be.close()
    return '+' if py > uy else '-'


def test_tie_grid_off_skips_new_code(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError('_pin_tie_side must not be called when pin_tie_grid=False')

    monkeypatch.setattr(steering, '_pin_tie_side', boom)
    monkeypatch.setattr(steering, '_pin_group', boom)
    for y0 in (16.0, 16.125):
        assert _aim_side(y0, BASE) == '-'  # equal extents default to right (-y for +x)
    _first_side(16.0, BASE)
    _first_side(16.0, BASE | PSD)


@pytest.mark.parametrize('sgn', [1.0, -1.0])
def test_tie_grid_follows_lateral_coordinate(sgn: float) -> None:
    on = BASE | {'pin_tie_grid': True}
    sides = [_aim_side(16.0 + k / 16, on, sgn) for k in range(9)]
    # Even floor(8 * y0) goes +y, odd goes -y (independent of +x vs -x travel)
    assert sides == ['+', '+', '-', '-', '+', '+', '-', '-', '+']
    assert _aim_side(13.0, on, sgn) == '+'
    assert _aim_side(13.125, on, sgn) == '-'
    # Small a4 lateral offset (< TIE_EPS) has no effect: extents compare relative to a0
    for off4 in (8 / 4096, -8 / 4096, 40 / 4096, -40 / 4096):
        assert _aim_side(16.0, on, sgn, off4=off4) == '+'
        assert _aim_side(16.125, on, sgn, off4=off4) == '-'
    # Switching t to a3 in the same row (equidistant tie) still uses a0 coordinate
    assert _aim_side(16.0, on, sgn, t=2) == '+'
    assert _aim_side(16.125, on, sgn, t=2) == '-'


@pytest.mark.parametrize('y0', [16.0, 16.125])
def test_tie_grid_symmetric_or_hold_goes_right(y0: float) -> None:
    on = BASE | {'pin_tie_grid': True}
    assert _aim_side(y0, on, drop_a3=True) == '-'
    assert _aim_side(y0, on, hold_a0=True) == '-'
    assert _aim_side(y0, on, sgn=-1.0, drop_a3=True) == '+'  # right of -x travel is +y


def test_tie_grid_row_goes_to_smaller_extent() -> None:
    """Three-unit row (a0 plus one on each side): when the top unit is 1/4096 tile closer,
    both enabled and disabled pick +y (smaller extent); exact symmetry goes right (-y)."""
    on = BASE | {'pin_tie_grid': True}
    q = 2868 / 4096
    for y0 in (16.0, 16.125):
        near_top = ((0.0, 0.0), (0.0, -q), (0.0, q - 1 / 4096))
        assert _aim_side(y0, on, front=near_top) == '+'
        assert _aim_side(y0, BASE, front=near_top) == '+'
        near_bottom = ((0.0, 0.0), (0.0, -q + 1 / 4096), (0.0, q))
        assert _aim_side(y0, on, front=near_bottom) == '-'
        assert _aim_side(y0, on, front=((0.0, 0.0), (0.0, -q), (0.0, q))) == '-'


# ---------------------------------------------------------------- Full-loop stepping


def _first_side(y0: float, config: dict[str, Any], loops: int = 24) -> str:
    """a0 ATTACK_UNITs an enemy Marine 4 tiles ahead, a2/a3/a5/a6 HOLD, a4 MOVEs 4 tiles
    ahead of a0; returns '+'/'-' for a4's first lateral move > 0.05 tiles ('0' if none)."""
    be, w, enemy = _backend(y0, config)
    sim = be._sim
    assert sim is not None
    cmd = Command.noop(1, sim.n_slots)
    cmd.set(0, 0, CommandKind.ATTACK_UNIT, target_slot=enemy)
    for s in range(1, w):
        cmd.set(0, s, CommandKind.HOLD)
    cmd.set(0, w, CommandKind.MOVE, target_xy=(20.0, y0))
    side = '0'
    for k in range(loops):
        st = be.step(cmd if k == 0 else Command.noop(1, sim.n_slots), 1)
        assert int(st.order_kind[0, 0]) == CommandKind.ATTACK_UNIT
        dy = float(st.y[0, w]) - y0
        if abs(dy) > 0.05:
            side = '+' if dy > 0 else '-'
            break
    be.close()
    return side


@pytest.mark.parametrize('extra', [{}, PSD])
def test_tie_grid_stepping(extra: dict[str, Any]) -> None:
    off = BASE | extra
    on = off | {'pin_tie_grid': True}
    assert _first_side(16.0, off) == '-'
    assert _first_side(16.0, on) == '+'
    assert _first_side(16.125, on) == '-'
    assert _first_side(16.25, on) == '+'
