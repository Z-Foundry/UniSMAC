"""Tests for pyref `retarget_after_move`: periodic retargeting evaluated after movement.

Periodic scan-frame retarget decisions (outside deferral, due-frame, or forced scans) run in
`targeting.late_scan` using post-move positions for the current loop; deferred, due, and forced
scans remain in `targeting.sc2`.

X (player 2 Marine) idles at (16, 16) engaged with distant A (player 1 Marine, outside weapon and
acquire range in +x); B (player 1 Marine) is in +y with a smaller center distance than A. Scan
frames are loops 2, 10, 18, ... (acquire_start_loop=2, period=8). Disabled by default with
bitwise-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE = 48
RM = STATIC.get(MARINE).radius
ACQ = STATIC.get(MARINE).weapons[0].range + 0.5
X, A, B = 0, 1, 2
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2, 'retarget': True, 'retarget_acquire': True,
        'retarget_lock': 'none', 'scan_after_move': True}  # fmt: skip


def _sim(**config: object) -> tuple[PyrefBackend, Sim]:
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    spawns = (Spawn(2, MARINE, 16.0, 16.0), Spawn(1, MARINE, 16.0 + 2 * RM + 8.0, 16.0),
              Spawn(1, MARINE, 16.0, 16.0 + 2 * RM + ACQ + 0.1))  # fmt: skip
    be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates state directly
    assert sim is not None
    sim.world.engaged_target_slot[0, X] = A
    return be, sim


def _put_b(sim: Sim, edge: float) -> None:
    sim.world.y[0, B] = np.float32(16.0 + 2 * RM + edge)


def _eng(sim: Sim) -> int:
    return int(sim.world.engaged_target_slot[0, X])


def test_defaults_and_validation() -> None:
    assert PyrefConfig().retarget_after_move is False
    with pytest.raises(ValueError, match='retarget_after_move.*retarget.*scan_after_move'):
        PyrefConfig(**{**BASE, 'scan_after_move': False, 'retarget_after_move': True})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='retarget_after_move.*retarget.*scan_after_move'):
        PyrefConfig(targeting='sc2', scan_after_move=True, retarget_after_move=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'retarget_after_move' not in opts


@pytest.mark.parametrize('after', [False, True])
def test_reads_position_after_move(after: bool) -> None:
    """Scan frame 10: B is 0.1 outside acquire range during sc2 and moves 0.1 inside.
    When enabled, late_scan switches to B; when disabled, keeps A using pre-move pos."""
    be, sim = _sim(retarget_after_move=after)
    try:
        sim.world.game_loop[:] = 10
        targeting.sc2(sim)
        assert _eng(sim) == A
        assert bool(sim.buf.late_retarget[0, X]) == after
        _put_b(sim, ACQ - 0.1)
        targeting.late_scan(sim)
        assert _eng(sim) == (B if after else A)
        assert not sim.buf.late_retarget.any()
    finally:
        be.close()


@pytest.mark.parametrize('after', [False, True])
def test_moved_out_before_decision(after: bool) -> None:
    """Converse: B is 0.1 inside acquire range during sc2 and moves 0.1 outside.
    When disabled, sc2 switches immediately; when enabled, late_scan keeps A."""
    be, sim = _sim(retarget_after_move=after)
    try:
        sim.world.game_loop[:] = 10
        _put_b(sim, ACQ - 0.1)
        targeting.sc2(sim)
        assert _eng(sim) == (A if after else B)
        _put_b(sim, ACQ + 0.1)
        targeting.late_scan(sim)
        assert _eng(sim) == (A if after else B)
    finally:
        be.close()


def test_busy_deferral_stays_in_sc2() -> None:
    """Scan frame inside fire deferral (last fired loop 5, retarget_busy=8) still sets
    rt_due=13 in sc2; same-loop fire and non-scan frames are not queued for late_scan."""
    be, sim = _sim(retarget_after_move=True)
    try:
        c = sim.combat
        sim.world.game_loop[:] = 10
        c.last_fire_loop[0, X] = 5
        targeting.sc2(sim)
        assert not sim.buf.late_retarget[0, X]
        assert int(c.rt_due[0, X]) == 13
        c.rt_due[0, X] = -1
        c.last_fire_loop[0, X] = -1
        targeting.sc2(sim)
        assert sim.buf.late_retarget[0, X]
        c.last_fire_loop[0, X] = 10
        _put_b(sim, ACQ - 0.1)
        targeting.late_scan(sim)
        assert _eng(sim) == A
        c.last_fire_loop[0, X] = -1
        sim.world.game_loop[:] = 11
        _put_b(sim, ACQ + 0.1)
        targeting.sc2(sim)
        assert not sim.buf.late_retarget.any()
    finally:
        be.close()


def _episode(**config: object) -> list[WorldState]:
    """B MOVEs from (16, 16 + 2*RM + 6.8) toward X while X chases A for 40 loops."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    spawns = (Spawn(2, MARINE, 16.0, 16.0), Spawn(1, MARINE, 16.0 + 2 * RM + 8.0, 16.0),
              Spawn(1, MARINE, 16.0, 16.0 + 2 * RM + 6.8))  # fmt: skip
    try:
        rows = [be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])]
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.engaged_target_slot[0, X] = A
        for loop in range(40):
            cmd = Command.noop(1, 3)
            if loop == 0:
                cmd.set(0, A, CommandKind.HOLD)
                cmd.set(0, B, CommandKind.MOVE, target_xy=(16.0, 16.0))
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def test_switch_row_in_episode() -> None:
    """In a full episode, switching to B occurs on a scan frame: when enabled, uses
    post-move positions (loop 10); when disabled, uses previous-row positions (loop 18)."""
    first = {}
    for after in (False, True):
        rows = _episode(retarget_after_move=after)
        r = next(i for i, s in enumerate(rows) if int(s.engaged_target_slot[0, X]) == B)
        loop = int(rows[r - 1].game_loop[0])
        assert (loop - 2) % 8 == 0  # row r reflects loop r - 1; switches on scan frame

        def edge(s: WorldState) -> float:
            return float(np.hypot(s.x[0, B] - s.x[0, X], s.y[0, B] - s.y[0, X])) - 2 * RM

        assert edge(rows[r] if after else rows[r - 1]) <= ACQ
        first[after] = loop
    assert first == {True: 10, False: 18}
