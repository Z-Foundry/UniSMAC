"""Tests for pyref `scan_after_move`: periodic auto-acquire evaluated after movement.

Idle and ATTACK_MOVE units without an engaged target perform periodic scan-frame acquisition in
`targeting.late_scan` using post-move positions for the current loop rather than previous-loop
positions. Forced scans from command activation or post-kill rescanning remain in `targeting.sc2`.

X (player 2 Marine) idles at (16, 16) with A (player 1 Marine) along +x; periodic scan frames are
loops 2, 10, 18, ... (acquire_start_loop=2, period=8), with loop L results appearing in row L + 1.
Disabled by default with bitwise-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE = 48
RM = STATIC.get(MARINE).radius
ACQ = STATIC.get(MARINE).weapons[0].range + 0.5
X, A = 0, 1
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2}  # fmt: skip


def _scenario(edge: float) -> Scenario:
    return Scenario(
        FLAT, spawns=(Spawn(2, MARINE, 16.0, 16.0), Spawn(1, MARINE, 16.0 + 2 * RM + edge, 16.0))
    )


def _sim(edge: float, **config: object) -> tuple[PyrefBackend, Sim]:
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([_scenario(edge)], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates state directly
    assert sim is not None
    return be, sim


def _put(sim: Sim, edge: float) -> None:
    sim.world.x[0, A] = np.float32(16.0 + 2 * RM + edge)


def test_defaults_and_validation() -> None:
    assert PyrefConfig().scan_after_move is False
    with pytest.raises(ValueError, match="scan_after_move.*targeting='sc2'"):
        PyrefConfig(scan_after_move=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'scan_after_move' not in opts


@pytest.mark.parametrize('after', [False, True])
def test_reads_position_after_move(after: bool) -> None:
    """Scan frame 10: A is 0.1 outside acquire range during sc2 and moves 0.1 inside.
    When enabled, sc2 skips periodic scan and late_scan engages A; when disabled, sc2
    uses pre-move positions and does not engage."""
    be, sim = _sim(ACQ - 0.1, scan_after_move=after)
    try:
        w = sim.world
        w.game_loop[:] = 10
        targeting.sc2(sim)
        assert int(w.engaged_target_slot[0, X]) == (NO_SLOT if after else A)
        w.engaged_target_slot[0, X] = NO_SLOT
        _put(sim, ACQ + 0.1)
        targeting.sc2(sim)
        assert int(w.engaged_target_slot[0, X]) == NO_SLOT
        _put(sim, ACQ - 0.1)
        if after:
            targeting.late_scan(sim)
            assert int(w.engaged_target_slot[0, X]) == A
            assert not np.isnan(sim.combat.leash_x[0, X])
    finally:
        be.close()


@pytest.mark.parametrize(('loop', 'scan_loop', 'want'), [(11, -1, NO_SLOT), (10, 14, NO_SLOT),
                                                         (10, -1, A)])  # fmt: skip
def test_late_scan_only_on_periodic_frames(loop: int, scan_loop: int, want: int) -> None:
    """late_scan acquires only on periodic scan frames when not waiting on scan_loop."""
    be, sim = _sim(ACQ - 0.1, scan_after_move=True)
    try:
        sim.world.game_loop[:] = loop
        sim.combat.scan_loop[0, X] = scan_loop
        targeting.late_scan(sim)
        assert int(sim.world.engaged_target_slot[0, X]) == want
    finally:
        be.close()


def test_forced_scan_still_in_sc2() -> None:
    """Forced scans (scan_loop due, e.g. order start or post-kill) still run in sc2."""
    be, sim = _sim(ACQ - 0.1, scan_after_move=True)
    try:
        sim.world.game_loop[:] = 13
        sim.combat.scan_loop[0, X] = 13
        targeting.sc2(sim)
        assert int(sim.world.engaged_target_slot[0, X]) == A
        assert int(sim.combat.scan_loop[0, X]) == -1
    finally:
        be.close()


def _run(edges: list[float], after: bool, seeds: list[int] | None = None) -> list[WorldState]:
    """A MOVEs toward X from edge distance edges[b] per env for 40 loops."""
    cfg = PyrefConfig(**{**BASE, 'scan_after_move': after})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(edges), cfg)
    try:
        rows = [be.reset([_scenario(e) for e in edges], seeds=seeds or list(range(len(edges))))]
        for loop in range(40):
            cmd = Command.noop(len(edges), 2)
            if loop == 0:
                for b in range(len(edges)):
                    cmd.set(b, A, CommandKind.MOVE, target_xy=(10.0, 16.0))
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def _edge(s: WorldState, b: int) -> float:
    d = np.hypot(s.x[b, X] - s.x[b, A], s.y[b, X] - s.y[b, A])
    return float(d - s.radius[b, X] - s.radius[b, A])


def _first_engage(rows: list[WorldState], b: int) -> int:
    return next(k for k, s in enumerate(rows) if int(s.engaged_target_slot[b, X]) == A)


@pytest.mark.parametrize('start', [7.0, 7.6])
def test_episode_first_engage(start: float) -> None:
    """A walks into X's acquire range: when enabled, engages on the first scan row
    (row L + 1, L == 2 mod 8) where row L + 1 is in range; else requires prior row."""
    for after in (False, True):
        rows = _run([start], after)
        scan_rows = [k for k in range(1, len(rows)) if (k - 1 - 2) % 8 == 0]
        ok = [k for k in scan_rows if _edge(rows[k - (0 if after else 1)], 0) <= ACQ]
        assert _first_engage(rows, 0) == ok[0]


def test_batch_matches_single() -> None:
    """Batched run (varying distances/seeds) matches single-env runs bitwise; at least
    one env engages one scan period earlier than when disabled."""
    edges = [6.2, 7.0, 7.6]
    every = _run(edges, True)
    off = _run(edges, False)
    earlier = 0
    for b, e in enumerate(edges):
        one = _run([e], True, seeds=[b])
        for s_all, s_one in zip(every, one, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True), f
        earlier += _first_engage(off, b) - _first_engage(every, b) == 8
    assert earlier >= 1
