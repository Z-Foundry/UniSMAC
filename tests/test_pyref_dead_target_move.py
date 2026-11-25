"""Tests for pyref `dead_target_move`.

When an ATTACK_UNIT command activates after its target died during command delay: by default
the command is ignored; when enabled, it converts into a MOVE toward the target's death
position (matching SC2 behavior).

Setup: Player 1 Stalkers A, B and Player 2 Zealots E1, E2. On loop 0, A ATTACK_UNITs E1 and
B ATTACK_UNITs E2 (command_delay=1, active next loop); E1 is killed before activation (health
set to 0 and cleaned up via `damage._cleanup`, leaving the death position recorded).
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import commands, damage
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V4, _V5, _V6

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
ZEALOT, STALKER = 73, 74
A, B, E1, E2 = 0, 1, 2, 3
BASE = PyrefConfig(commands='delayed', command_delay=1, move_lag=1, **_V5)  # type: ignore[arg-type]
ON = dataclasses.replace(BASE, dead_target_move=True)
SPAWNS = (
    Spawn(1, STALKER, 10.0, 16.0),
    Spawn(1, STALKER, 10.0, 19.0),
    Spawn(2, ZEALOT, 14.0, 15.0),
    Spawn(2, ZEALOT, 14.5, 19.0),
)


def _cmd() -> Command:
    cmd = Command.noop(1, len(SPAWNS))
    cmd.set(0, A, CommandKind.ATTACK_UNIT, target_slot=E1)
    cmd.set(0, B, CommandKind.ATTACK_UNIT, target_slot=E2)
    return cmd


def _kill(be: PyrefBackend, u: int) -> tuple[float, float]:
    sim = be._sim  # noqa: SLF001 - test mutates internal state directly
    assert sim is not None
    sim.world.health[0, u] = 0.0
    sim.world.shield[0, u] = 0.0
    damage._cleanup(sim, 0)  # noqa: SLF001
    assert not sim.world.alive[0, u]
    return float(sim.world.x[0, u]), float(sim.world.y[0, u])


def _run(cfg: PyrefConfig, loops: int = 60) -> tuple[list[WorldState], tuple[float, float]]:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        rows = [be.reset([Scenario(FLAT, spawns=SPAWNS)], seeds=[0])]
        rows.append(be.step(_cmd(), 1))  # command queued, activates next loop
        dead_at = _kill(be, E1)
        for _ in range(loops):
            rows.append(be.step(Command.noop(1, len(SPAWNS)), 1))
    finally:
        be.close()
    return rows, dead_at


def test_defaults() -> None:
    assert PyrefConfig().dead_target_move is False
    for opts in (_V4, _V5, _V6):
        assert 'dead_target_move' not in opts


def test_off_ignores_command() -> None:
    rows, _ = _run(BASE, 4)
    s = rows[2]  # after command activation loop
    assert int(s.order_kind[0, A]) == CommandKind.NOOP
    assert int(s.order_kind[0, B]) == CommandKind.ATTACK_UNIT
    assert int(s.order_target_slot[0, B]) == E2


def test_on_moves_to_dead_position() -> None:
    rows, (dx, dy) = _run(ON)
    s = rows[2]
    assert int(s.order_kind[0, A]) == CommandKind.MOVE
    assert int(s.order_target_slot[0, A]) == NO_SLOT
    assert np.float32(s.order_x[0, A]) == np.float32(dx)
    assert np.float32(s.order_y[0, A]) == np.float32(dy)
    # Living target behaves unchanged
    assert int(s.order_kind[0, B]) == CommandKind.ATTACK_UNIT
    assert int(s.order_target_slot[0, B]) == E2
    # Moves toward the death position and becomes idle upon arrival
    d = [float(np.hypot(r.x[0, A] - dx, r.y[0, A] - dy)) for r in rows[1:]]
    assert d[-1] < d[0] - 1.0
    arrived = next(k for k, r in enumerate(rows) if k > 2 and int(r.order_kind[0, A]) != 2)
    assert int(rows[arrived].order_kind[0, A]) == CommandKind.NOOP
    assert d[arrived - 1] < 0.05


def test_on_off_same_before_command() -> None:
    """Bit-identical before command activation; B (living target) matches under both configs."""
    off, _ = _run(BASE, 4)
    on, _ = _run(ON, 4)
    for name in WorldState.UNIT_FIELDS:
        assert getattr(off[1], name).tobytes() == getattr(on[1], name).tobytes(), name
    for k in (2, 3):
        assert int(off[k].order_kind[0, B]) == int(on[k].order_kind[0, B])
        assert int(off[k].order_target_slot[0, B]) == int(on[k].order_target_slot[0, B])


def test_scalar_matches_batch() -> None:
    """Per-unit `commands._apply_all` matches batched `commands._apply_rows` bit-for-bit."""
    be = PyrefBackend(STATIC, MAPS, 1, ON)
    try:
        be.reset([Scenario(FLAT, spawns=SPAWNS)], seeds=[0])
        be.step(Command.noop(1, len(SPAWNS)), 1)
        _kill(be, E1)
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        ref = copy.deepcopy(sim)
        commands._apply_rows(sim, _cmd(), np.ones(1, np.bool_))  # noqa: SLF001
        commands._apply_all(ref, _cmd(), 0)  # noqa: SLF001
    finally:
        be.close()
    assert int(sim.world.order_kind[0, A]) == CommandKind.MOVE
    for part in ('world', 'motion', 'combat', 'ability'):
        a, b = getattr(sim, part), getattr(ref, part)
        for name, v in vars(a).items():
            if isinstance(v, np.ndarray):
                assert v.tobytes() == getattr(b, name).tobytes(), (part, name)
