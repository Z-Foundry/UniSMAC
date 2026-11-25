"""Tests for pyref `retarget_defer_memory`.

When `retarget_defer='switch'` defers a retarget, remember the target chosen on the scan loop.
When deferral expires, if no candidate is currently in acquire range and the current target is
outside acquire range, switch to the remembered target; if it died, drop the current target.

Setup: F (idle Player 2 Marine) engages A (Ultralisk on +x, 0.6 outside range and outside
acquire range) while B (Hellion on -y, 0.3 outside range, inside acquire range, closer center)
is present; last fire is loop 5, scan loop 10 falls in the busy window (retarget_busy=8), and
deferral expires on loop 13. Disabled by default with bit-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.backends.pyref import PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.contract import NO_SLOT, CommandKind

from .test_pyref_retarget_acquire import HELLION, MARINE, RNG, ULTRALISK, A, B, F, _at, _r, _sim

OPTS = {'retarget_acquire': True, 'retarget_defer': 'switch'}


def _deferred(memory: bool, ea: float = 0.6, eb: float = 0.3) -> Sim:
    sim = _sim(ULTRALISK, HELLION, ea, eb, CommandKind.NOOP, retarget_defer_memory=memory, **OPTS)
    sim.combat.last_fire_loop[0, F] = 5
    assert [_at(sim, t) for t in (10, 11, 12)] == [A, A, A]
    assert int(sim.combat.rt_due[0, F]) == 13
    assert int(sim.combat.rt_alt[0, F]) == (B if memory else NO_SLOT)
    return sim


def _move(sim: Sim, slot: int, unit: int, e: float) -> None:
    """Place `slot` at edge distance `range + e` from F along its original axis."""
    w = sim.world
    d = np.float32(_r(MARINE) + _r(unit) + RNG + e)
    if slot == A:
        w.x[0, A] = w.x[0, F] + d
    else:
        w.y[0, B] = w.y[0, F] - d


def test_defaults_and_validation() -> None:
    assert PyrefConfig().retarget_defer_memory is False
    with pytest.raises(ValueError, match="retarget_defer_memory.*retarget_defer='switch'"):
        PyrefConfig(targeting='sc2', retarget=True, retarget_defer_memory=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'retarget_defer_memory' not in opts


@pytest.mark.parametrize('memory', [False, True])
def test_candidate_still_there(memory: bool) -> None:
    """When B is still a candidate at expiry, both modes switch to B."""
    sim = _deferred(memory)
    assert _at(sim, 13) == B


@pytest.mark.parametrize('memory', [False, True])
def test_remembered_target_left_acquire_range(memory: bool) -> None:
    """B backs off to range + 1.6 (outside acquire range and farther than A): with no current
    candidate at expiry, switch to B when memory is on, keep A when off."""
    sim = _deferred(memory)
    _move(sim, B, HELLION, 1.6)
    assert _at(sim, 13) == (B if memory else A)
    assert int(sim.combat.rt_alt[0, F]) == NO_SLOT
    assert _at(sim, 14) == (B if memory else A)


def test_current_target_back_in_acquire_range() -> None:
    """At expiry, A is back in acquire range (range + 0.3) and B is outside: keep A."""
    sim = _deferred(True)
    _move(sim, A, ULTRALISK, 0.3)
    _move(sim, B, HELLION, 1.6)
    assert _at(sim, 13) == A


@pytest.mark.parametrize('memory', [False, True])
def test_remembered_target_dead(memory: bool) -> None:
    """If B dies during deferral: drop A (NO_SLOT) when memory is on, keep A when off."""
    sim = _deferred(memory)
    sim.world.alive[0, B] = False
    assert _at(sim, 13) == (NO_SLOT if memory else A)
    assert int(sim.combat.rt_lock[0, F]) != B


def test_no_memory_without_defer() -> None:
    """Scan loop outside busy window (last fire loop 0): switches immediately without rt_alt."""
    sim = _sim(ULTRALISK, HELLION, 0.6, 0.3, CommandKind.NOOP, retarget_defer_memory=True, **OPTS)
    assert _at(sim, 10) == B
    assert int(sim.combat.rt_alt[0, F]) == NO_SLOT
