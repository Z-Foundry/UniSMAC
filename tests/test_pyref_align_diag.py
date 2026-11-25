"""Tests for pyref SC2 alignment options:

- chase_lock: auto-engaged units do not chase out-of-range targets during the post-fire lock
  window (e.g., 12 loops for Marine);
- stop_keeps_heal: STOP does not cancel an active heal order;
- heal_counts_armed: actively healing units count as armed for target prioritization;
- amove_needs_weapon: unarmed units ignore ATTACK_MOVE.

All options default to off, preserving baseline behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

MARINE, MEDIVAC = 48, 54
HEAL = 386
FLAT = 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {FLAT: load_map_info(FLAT)}


def make(**config: object) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]


def start(backend: PyrefBackend, *spawns: Spawn, life: dict[int, float] | None = None) -> int:
    backend.reset([Scenario(map_id=FLAT, spawns=spawns)])
    sim = backend._sim  # noqa: SLF001 - test sets unit health directly
    assert sim is not None
    for slot, hp in (life or {}).items():
        sim.world.health[0, slot] = hp
    return len(spawns)


def test_config_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.chase_lock == ()
    assert not (cfg.stop_keeps_heal or cfg.heal_counts_armed or cfg.amove_needs_weapon)
    lists = PyrefConfig(chase_lock=[[48, 12], [73, 8]])  # type: ignore[arg-type]
    assert lists.chase_lock == ((48, 12), (73, 8))
    with pytest.raises(ValueError, match='chase_lock'):
        PyrefConfig(chase_lock=((48, -1),))


def _chase(lock: tuple[tuple[int, int], ...]) -> tuple[int, list[float]]:
    """Player 2 Marine ATTACK_MOVEs to engage; Player 1 Marine flees on the first firing loop.
    Returns (firing state index k, Player 2 x positions per state)."""
    backend = make(targeting='sc2', chase_lock=lock)
    n = start(backend, Spawn(1, MARINE, 10.0, 16.0), Spawn(2, MARINE, 17.0, 16.0))
    amove = Command.noop(1, n).set(0, 1, CommandKind.ATTACK_MOVE, target_xy=(4.0, 16.0))
    flee = Command.noop(1, n).set(0, 0, CommandKind.MOVE, target_xy=(1.0, 16.0))
    states: list[WorldState] = [backend.step(amove, 1)]
    k = -1
    for i in range(1, 80):
        before = states[-2].weapon_cooldown[0, 1] if i > 1 else 0
        if states[-1].weapon_cooldown[0, 1] > before and k < 0:
            k = len(states) - 1
        states.append(backend.step(flee if len(states) - 1 == k else Command.noop(1, n), 1))
    assert k > 0, 'unit should have fired'
    return k, [float(s.x[0, 1]) for s in states]


def test_chase_lock_waits_after_fire() -> None:
    """Marine waits loops 0..11 after firing without chasing, then starts chasing on loop 12."""
    k, x = _chase(((MARINE, 12),))
    moved = [abs(x[i + 1] - x[i]) > 1e-6 for i in range(len(x) - 1)]
    assert not any(moved[k : k + 12])
    assert moved[k + 12]
    k0, x0 = _chase(())
    moved0 = [abs(x0[i + 1] - x0[i]) > 1e-6 for i in range(len(x0) - 1)]
    assert k0 == k and any(moved0[k : k + 12]), 'default should chase immediately'


def test_chase_lock_ignores_other_types() -> None:
    k, x = _chase(((73, 8),))  # lock Zealot only
    k0, x0 = _chase(())
    assert (k, x) == (k0, x0)


def _heal_then_stop(keep: bool) -> tuple[list[int], np.ndarray]:
    backend = make(abilities='heal', stop_keeps_heal=keep)
    n = start(backend, Spawn(1, MEDIVAC, 16.0, 16.0), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    heal = Command.noop(1, n).set(0, 0, CommandKind.ABILITY, target_slot=1, ability=HEAL)
    stop = Command.noop(1, n).set(0, 0, CommandKind.STOP)
    states = [backend.step(heal, 1)] + [backend.step(Command.noop(1, n), 1) for _ in range(5)]
    states += [backend.step(stop if i % 8 == 0 else Command.noop(1, n), 1) for i in range(24)]
    kinds = [int(s.order_kind[0, 0]) for s in states[6:]]
    return kinds, np.array([float(s.health[0, 1]) for s in states[6:]])


def test_stop_keeps_heal() -> None:
    kinds, hp = _heal_then_stop(True)
    assert all(k == CommandKind.ABILITY for k in kinds)
    assert np.all(np.diff(hp) > 0), 'should heal on every loop'
    kinds0, hp0 = _heal_then_stop(False)
    assert kinds0[0] == CommandKind.NOOP and hp0[-1] == hp0[0], 'default STOP should cancel heal'


def _first_target(armed: bool) -> int:
    """Idle Player 2 Marine auto-acquires: healing Medivac edge 2.4 vs Player 1 Marine edge 4.3."""
    backend = make(abilities='heal', targeting='sc2', auto_acquire=True, heal_counts_armed=armed)
    n = start(
        backend,
        Spawn(2, MARINE, 10.0, 16.0),
        Spawn(1, MEDIVAC, 13.5, 16.0),
        Spawn(1, MARINE, 15.0, 16.0),
        life={2: 10.0},
    )
    heal = Command.noop(1, n).set(0, 1, CommandKind.ABILITY, target_slot=2, ability=HEAL)
    states = [backend.step(heal, 1)] + [backend.step(Command.noop(1, n), 1) for _ in range(12)]
    assert int(states[-1].order_kind[0, 1]) == CommandKind.ABILITY
    return next(
        int(s.engaged_target_slot[0, 0]) for s in states if s.engaged_target_slot[0, 0] >= 0
    )


def test_heal_counts_armed() -> None:
    assert _first_target(False) == 2, 'default should prioritize armed Marine'
    assert _first_target(True) == 1, 'healing Medivac counts as armed and is closer'


@pytest.mark.parametrize('needs', [False, True])
def test_amove_needs_weapon(needs: bool) -> None:
    backend = make(amove_needs_weapon=needs)
    n = start(backend, Spawn(2, MEDIVAC, 16.0, 16.0), Spawn(2, MARINE, 16.0, 12.0))
    cmd = Command.noop(1, n)
    for u in range(n):
        cmd = cmd.set(0, u, CommandKind.ATTACK_MOVE, target_xy=(4.0, 16.0))
    states = [backend.step(cmd, 1)] + [backend.step(Command.noop(1, n), 1) for _ in range(10)]
    med_moved = float(states[-1].x[0, 0]) != 16.0
    assert med_moved != needs
    kind = CommandKind.NOOP if needs else CommandKind.ATTACK_MOVE
    assert int(states[-1].order_kind[0, 0]) == kind
    assert float(states[-1].x[0, 1]) < 16.0, 'Marine should move normally'
