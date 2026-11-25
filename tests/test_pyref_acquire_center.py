"""Tests for pyref `acquire_center` center-to-center target acquisition.

When idle or ATTACK_MOVE units scan for targets (periodic scans, order activation,
post-kill rescans, and `scan_after_move` `late_scan`), armed enemies are prioritized first
and then selected by nearest center-to-center distance rather than edge-to-edge distance;
HOLD continues to pick in-range targets by edge-to-edge distance.

Setup: X (Player 2 Stalker) at (16, 16), Player 1 Zealot Z at center dist 5.0 (edge 3.875),
and Ultralisk U (radius 1) at center dist 5.2 (edge 3.575). U is closer edge-to-edge, Z is
closer center-to-center, and both are within X's range 6. Scan loops are 2, 10, 18...
(acquire_start_loop=2, period 8). Disabled by default with bit-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
ZEALOT, STALKER, ULTRALISK = 73, 74, 109
X, Z, U = 0, 1, 2
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2}  # fmt: skip


def _scenario(dz: float = 5.0, du: float = 5.2) -> Scenario:
    return Scenario(
        FLAT,
        spawns=(
            Spawn(2, STALKER, 16.0, 16.0),
            Spawn(1, ZEALOT, 16.0 - dz, 16.0),
            Spawn(1, ULTRALISK, 16.0, 16.0 - du),
        ),
    )


def _sim(kind: CommandKind, loop: int, **config: object) -> tuple[PyrefBackend, Sim]:
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**{**BASE, **config}))  # type: ignore[arg-type]
    be.reset([_scenario()], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates internal state directly
    assert sim is not None
    sim.world.order_kind[0, X] = kind
    if kind == CommandKind.ATTACK_MOVE:
        sim.world.order_x[0, X], sim.world.order_y[0, X] = np.float32(16), np.float32(28)
    sim.world.game_loop[:] = loop
    return be, sim


def test_defaults_and_validation() -> None:
    assert PyrefConfig().acquire_center is False
    with pytest.raises(ValueError, match="acquire_center.*targeting='sc2'"):
        PyrefConfig(acquire_center=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'acquire_center' not in opts


@pytest.mark.parametrize('kind', [CommandKind.NOOP, CommandKind.ATTACK_MOVE, CommandKind.HOLD])
@pytest.mark.parametrize('center', [False, True])
def test_periodic_scan(kind: CommandKind, center: bool) -> None:
    """Scan loop 10: picks center-nearest Z when on, edge-nearest U when off; HOLD always U."""
    be, sim = _sim(kind, 10, acquire_center=center)
    try:
        targeting.sc2(sim)
        want = Z if center and kind != CommandKind.HOLD else U
        assert int(sim.world.engaged_target_slot[0, X]) == want
    finally:
        be.close()


@pytest.mark.parametrize('center', [False, True])
def test_kill_rescan(center: bool) -> None:
    """Post-kill rescan expiring on non-scan loop 13 (combat.scan_loop = 13) also uses center."""
    be, sim = _sim(CommandKind.ATTACK_MOVE, 13, acquire_center=center)
    try:
        sim.combat.scan_loop[0, X] = 13
        targeting.sc2(sim)
        assert int(sim.world.engaged_target_slot[0, X]) == (Z if center else U)
        assert int(sim.combat.scan_loop[0, X]) == -1
    finally:
        be.close()


@pytest.mark.parametrize('center', [False, True])
def test_late_scan(center: bool) -> None:
    """With `scan_after_move`, periodic scan runs in late_scan with the same target choice."""
    be, sim = _sim(CommandKind.NOOP, 10, acquire_center=center, scan_after_move=True)
    try:
        targeting.sc2(sim)
        assert int(sim.world.engaged_target_slot[0, X]) == NO_SLOT
        targeting.late_scan(sim)
        assert int(sim.world.engaged_target_slot[0, X]) == (Z if center else U)
    finally:
        be.close()


def _run(geo: list[tuple[float, float]], seeds: list[int], **config: object) -> list:
    """Run 24 loops with per-env (Z center dist, U center dist), X idle, Player 1 on HOLD."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(geo), cfg)
    st = be.reset([_scenario(*g) for g in geo], seeds=seeds)
    states = [st]
    for loop in range(24):
        cmd = Command.noop(len(geo), 3)
        if loop == 0:
            for b in range(len(geo)):
                cmd.set(b, Z, CommandKind.HOLD)
                cmd.set(b, U, CommandKind.HOLD)
        states.append(be.step(cmd, 1))
    be.close()
    return states


def test_batch_matches_single_episode() -> None:
    """Batch run (varied distances and seeds) matches single-env runs bit-for-bit."""
    geo, seeds = [(5.0, 5.2), (5.4, 5.2), (4.0, 6.5)], [0, 1, 2]
    every = _run(geo, seeds, acquire_center=True)
    first = [
        next(int(s.engaged_target_slot[b, X]) for s in every if s.engaged_target_slot[b, X] >= 0)
        for b in range(3)
    ]
    assert first == [Z, U, Z]
    for b, (g, seed) in enumerate(zip(geo, seeds, strict=True)):
        one = _run([g], [seed], acquire_center=True)
        for s_all, s_one in zip(every, one, strict=True):
            for f in ('x', 'y', 'weapon_cooldown', 'engaged_target_slot', 'health', 'shield'):
                a, o = getattr(s_all, f)[b], getattr(s_one, f)[0]
                assert np.array_equal(a, o, equal_nan=True)
