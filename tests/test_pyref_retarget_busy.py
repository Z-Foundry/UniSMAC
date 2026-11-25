"""Tests for pyref `retarget` deferral options.

* `retarget_busy_lock`: per-unit deferral window derived from measured `follow_lock - 1`
  (12 for Marine), falling back to `retarget_busy` for units without `follow_lock`.
* `retarget_defer='switch'`: defers only units that already decided to switch on the scan frame;
  non-switching units wait for the next scan frame.

F (player 2 Marine) executes ATTACK_MOVE with engaged target A out of range and B closer in range;
scan frames are loops 2, 10, 18, ... (acquire_start_loop=2, period=8). Both disabled by default
with bitwise-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, CommandKind, Scenario, Spawn

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, ULTRALISK = 48, 109
F, A, B = 0, 1, 2
RNG = STATIC.get(MARINE).weapons[0].range
RR = STATIC.get(MARINE).radius + STATIC.get(ULTRALISK).radius
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2, 'retarget': True}  # fmt: skip
FIRE = 5  # game_loop of F's last shot


def _sim(**config: object) -> Sim:
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    sc = Scenario(
        FLAT,
        spawns=(
            Spawn(2, MARINE, 16.0, 16.0),
            Spawn(1, ULTRALISK, 16.0 + RR + RNG + 0.5, 16.0),  # A: 0.5 out of range
            Spawn(1, ULTRALISK, 16.0, 16.0 - RR - RNG + 0.5),  # B: in range, closer
        ),
    )
    be.reset([sc], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates state directly
    assert sim is not None
    w = sim.world
    w.order_kind[0, F] = CommandKind.ATTACK_MOVE
    w.order_x[0, F], w.order_y[0, F] = np.float32(28), np.float32(16)
    w.engaged_target_slot[0, F] = A
    sim.combat.last_fire_loop[0, F] = FIRE
    return sim


def _at(sim: Sim, loop: int) -> int:
    sim.world.game_loop[:] = loop
    targeting.sc2(sim)
    return int(sim.world.engaged_target_slot[0, F])


def _switch_loop(sim: Sim, loops: range) -> int | None:
    for loop in loops:
        if _at(sim, loop) == B:
            return loop
    return None


def _a_in_range(sim: Sim, inside: bool) -> None:
    sim.world.x[0, A] = np.float32(16.0 + RR + RNG + (-0.2 if inside else 0.5))


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert not cfg.retarget_busy_lock
    assert cfg.retarget_defer == 'scan'
    with pytest.raises(ValueError, match='retarget_busy_lock.*retarget'):
        PyrefConfig(retarget_busy_lock=True)
    with pytest.raises(ValueError, match='retarget_defer'):
        PyrefConfig(retarget=True, retarget_defer='later')
    with pytest.raises(ValueError, match='retarget_defer.*retarget'):
        PyrefConfig(retarget_defer='switch')


@pytest.mark.parametrize(
    ('options', 'want'),
    [
        ({}, FIRE + 8),
        ({'retarget_busy_lock': True}, FIRE + 12),  # Marine: follow_lock 13 - 1
        ({'retarget_defer': 'switch'}, FIRE + 8),
        ({'retarget_busy_lock': True, 'retarget_defer': 'switch'}, FIRE + 12),
    ],
)
def test_deferred_switch_loop(options: dict, want: int) -> None:
    """Scan frame 10 inside fire deferral (loop 5): switch defers to fire + busy."""
    sim = _sim(**options)
    assert _switch_loop(sim, range(10, 20)) == want


@pytest.mark.parametrize(('lock', 'want'), [(1, 10), (-1, FIRE + 8), (9, FIRE + 8)])
def test_busy_follows_measured_lock(lock: int, want: int) -> None:
    """follow_lock=1 (Marauder) switches on scan frame without deferring; 9 (Stalker,
    Zealot) defers to fire + 8; missing follow_lock (-1) uses retarget_busy."""
    sim = _sim(retarget_busy_lock=True)
    sim.params.follow_lock[0, F] = lock
    assert _switch_loop(sim, range(10, 20)) == want


@pytest.mark.parametrize(('defer', 'want'), [('scan', FIRE + 8), ('switch', 18)])
def test_defer_only_decided_switches(defer: str, want: int) -> None:
    """A is in range on scan frame 10 (no switch) then leaves range: 'scan' switches on
    due frame 13 while 'switch' waits until the next scan frame 18."""
    sim = _sim(retarget_defer=defer)
    _a_in_range(sim, True)
    assert _at(sim, 10) == A
    assert int(sim.combat.rt_due[0, F]) == (FIRE + 8 if defer == 'scan' else -1)
    _a_in_range(sim, False)
    assert _switch_loop(sim, range(11, 20)) == want


def test_defer_switch_rechecks_at_due() -> None:
    """'switch': records rt_due when a switch is warranted on the scan frame; if A
    returns to range by the due frame, keeps A and clears rt_due."""
    sim = _sim(retarget_defer='switch')
    assert _at(sim, 10) == A
    assert int(sim.combat.rt_due[0, F]) == FIRE + 8
    _a_in_range(sim, True)
    assert _at(sim, FIRE + 8) == A
    assert int(sim.combat.rt_due[0, F]) == -1
    assert sim.world.engaged_target_slot[0, F] != NO_SLOT
