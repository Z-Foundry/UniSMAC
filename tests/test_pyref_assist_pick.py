"""Tests for pyref `hit_assist_pick` attacker selection when an ally is hit.

Controls which attacker a responder picks on `hit_assist`: 'first' (default) picks the
first-resolved hit and keeps it; 'nearest' picks the attacker at smallest center distance,
switches to a closer attacker until the responder fires, and with `scan_after_move` defers
to a periodic scan target if the assist occurs on a scan loop with an enemy in scan range.

Setup: V (Player 2 Marine, slot 0) at (16, 16); attackers S1 (slot 1) at (23, 16) and S2
(slot 2) at (16, 9), both Player 1 Stalkers; responder H (slot 3, Player 2 Marine) at
(14.5, 16), center dist 8.5 to S1 and 7.16 to S2 (S2 is closer); optional N (slot 4,
Player 1 Marine) at (14.5, 21) within H's acquire range.
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
MARINE, STALKER = 48, 74
V, S1, S2, H, N = 0, 1, 2, 3, 4
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2, 'hit_assist': True}  # fmt: skip


def _sim(with_n: bool = False, **config: object) -> tuple[PyrefBackend, Sim]:
    spawns = [Spawn(2, MARINE, 16.0, 16.0), Spawn(1, STALKER, 23.0, 16.0),
              Spawn(1, STALKER, 16.0, 9.0), Spawn(2, MARINE, 14.5, 16.0)]  # fmt: skip
    spawns.append(Spawn(1, MARINE, 14.5, 21.0) if with_n else Spawn(1, MARINE, 3.0, 3.0))
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([Scenario(FLAT, spawns=tuple(spawns))], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates internal state directly
    assert sim is not None
    sim.world.game_loop[:] = 11  # non-scan loop (acquire_start_loop=2, period 8)
    return be, sim


def test_defaults_and_validation() -> None:
    assert PyrefConfig().hit_assist_pick == 'first'
    with pytest.raises(ValueError, match='hit_assist_pick'):
        PyrefConfig(targeting='sc2', hit_assist=True, hit_assist_pick='last')
    with pytest.raises(ValueError, match='hit_assist_pick.*hit_assist'):
        PyrefConfig(targeting='sc2', hit_assist_pick='nearest')
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'hit_assist_pick' not in opts


@pytest.mark.parametrize(('pick', 'order', 'want'), [
    ('first', (S1, S2), S1), ('first', (S2, S1), S2),
    ('nearest', (S1, S2), S2), ('nearest', (S2, S1), S2),
])  # fmt: skip
def test_same_loop_hits(pick: str, order: tuple[int, int], want: int) -> None:
    """Two hits on the same loop: 'first' keeps the first resolved; 'nearest' picks S2."""
    be, sim = _sim(hit_assist_pick=pick)
    try:
        for a in order:
            targeting.on_hit(sim, 0, a, V)
        assert int(sim.world.engaged_target_slot[0, H]) == want
    finally:
        be.close()


@pytest.mark.parametrize(('fired', 'want'), [(False, S2), (True, S1)])
def test_later_nearer_hit(fired: bool, want: int) -> None:
    """S1 hits first and H assists; 2 loops later closer S2 hits: switch unless H already fired."""
    be, sim = _sim(hit_assist_pick='nearest')
    try:
        targeting.on_hit(sim, 0, S1, V)
        assert int(sim.combat.assist_target[0, H]) == S1
        assert int(sim.combat.assist_loop[0, H]) == 11
        if fired:
            sim.combat.last_fire_loop[0, H] = 12
        sim.world.game_loop[:] = 13
        targeting.on_hit(sim, 0, S2, V)
        assert int(sim.world.engaged_target_slot[0, H]) == want
    finally:
        be.close()


def test_no_switch_from_non_assist_target() -> None:
    """Do not switch away from a scan-acquired target or to a farther attacker."""
    be, sim = _sim(hit_assist_pick='nearest')
    try:
        sim.world.engaged_target_slot[0, H] = S1
        targeting.on_hit(sim, 0, S2, V)
        assert int(sim.world.engaged_target_slot[0, H]) == S1
        sim.world.engaged_target_slot[0, H] = NO_SLOT
        targeting.on_hit(sim, 0, S2, V)
        targeting.on_hit(sim, 0, S1, V)
        assert int(sim.world.engaged_target_slot[0, H]) == S2
    finally:
        be.close()


@pytest.mark.parametrize(('pick', 'loop', 'want'), [
    ('first', 10, S1), ('nearest', 10, N), ('nearest', 11, S1),
])  # fmt: skip
def test_late_scan_after_assist(pick: str, loop: int, want: int) -> None:
    """scan_after_move: on scan loop 10 with N in acquire range, 'nearest' picks N;
    'first' and non-scan loops keep the assist target S1."""
    be, sim = _sim(with_n=True, scan_after_move=True, hit_assist_pick=pick)
    try:
        sim.world.game_loop[:] = loop
        targeting.on_hit(sim, 0, S1, V)
        assert int(sim.world.engaged_target_slot[0, H]) == S1
        targeting.late_scan(sim)
        assert int(sim.world.engaged_target_slot[0, H]) == want
    finally:
        be.close()


def _episode(pick: str) -> list[WorldState]:
    """S1 ATTACK_UNITs V on loop 0 while S2 HOLDs far away, for 60 loops."""
    cfg = PyrefConfig(**{**BASE, 'hit_assist_pick': pick})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    spawns = (Spawn(2, MARINE, 16.0, 16.0), Spawn(1, STALKER, 23.0, 16.0),
              Spawn(1, STALKER, 28.0, 28.0), Spawn(2, MARINE, 14.5, 16.0),
              Spawn(1, MARINE, 3.0, 3.0))  # fmt: skip
    try:
        rows = [be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])]
        for loop in range(60):
            cmd = Command.noop(1, len(spawns))
            if loop == 0:
                cmd.set(0, S1, CommandKind.ATTACK_UNIT, target_slot=V)
                cmd.set(0, S2, CommandKind.HOLD)
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def test_single_attacker_same_as_first() -> None:
    """With one attacker, 'nearest' matches 'first' bit-for-bit and H engages S1 on hit."""
    a, b = _episode('first'), _episode('nearest')
    for s_a, s_b in zip(a, b, strict=True):
        for f in WorldState.UNIT_FIELDS:
            assert np.array_equal(getattr(s_a, f), getattr(s_b, f), equal_nan=True), f
    assert any(int(s.engaged_target_slot[0, H]) == S1 for s in b)
