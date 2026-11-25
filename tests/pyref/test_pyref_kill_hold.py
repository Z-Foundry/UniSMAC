"""Tests for pyref `kill_hold`: hold position by clearing waypoints while awaiting rescan."""

from __future__ import annotations

import numpy as np

from unismac.backends.pyref.subsystems import pathing
from unismac.contract import NO_SLOT, CommandKind

from .test_pyref_batch import _sim


def test_kill_hold_clears_waypoint_until_rescan() -> None:
    """With `kill_hold`, idle/ATTACK_MOVE units waiting to rescan clear waypoints; others stay."""
    sim = _sim('8m', 2, 0, kill_hold=True)
    w, c, buf = sim.world, sim.combat, sim.buf
    w.alive[:] = True
    w.order_kind[:] = CommandKind.ATTACK_MOVE
    w.order_x[:], w.order_y[:] = w.x + 5, w.y
    w.engaged_target_slot[:] = NO_SLOT
    c.scan_loop[:] = -1
    c.scan_loop[0, 0] = int(w.game_loop[0]) + 2  # waiting to rescan
    c.scan_loop[0, 1] = int(w.game_loop[0])  # scans on this loop, no wait
    w.order_kind[0, 2] = CommandKind.MOVE
    c.scan_loop[0, 2] = int(w.game_loop[0]) + 2
    pathing.straight(sim)
    assert np.isnan(buf.waypoint_x[0, 0])
    assert not np.isnan(buf.waypoint_x[0, 1])
    assert not np.isnan(buf.waypoint_x[0, 2])
    assert not np.isnan(buf.waypoint_x[1]).any()
