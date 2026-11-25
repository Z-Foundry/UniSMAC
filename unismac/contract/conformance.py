"""Contract conformance checks that every backend must pass.

Usage (pytest)::

    from unismac.contract import conformance

    @pytest.mark.parametrize('check', conformance.CHECKS, ids=lambda c: c.__name__)
    def test_conformance(backend, check):
        check(backend)

Checks depend only on the contract, using a fixed scenario of 3 Marines (`unit_type=48`) on
`32x32_flat`: slots 0 and 1 belong to Player 1, slot 2 belongs to Player 2, separated by ~8
tiles (within sight range 9, outside weapon range 5). SC2 rejects `ATTACK_UNIT` against
invisible enemies, so the target must start in sight. Backends must have static data for Marine.

Each check accepts a `scenario` argument to swap maps or players while keeping unit placement.
Checks assume both players are `AgentPlayer` (uncommanded units remain idle): SC2 `SMAC_Maps`
only allow 1 user slot where built-in AI moves Player 2, so SC2 tests run on a 2-user map.
Target points stay within x in [2, 29], y in [7, 24] (the playable area of `new_maps/3m`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .backend import Backend
from .types import Command, CommandKind, ContractError, Scenario, Spawn

MARINE = 48
MAP_ID = 'SMAC_Maps/32x32_flat'
SPAWN_TOLERANCE = 1.0  # engine may nudge spawned units to nearest valid placement

SCENARIO = Scenario(
    map_id=MAP_ID,
    spawns=(
        Spawn(owner=1, unit_type=MARINE, x=10.0, y=16.0),
        Spawn(owner=1, unit_type=MARINE, x=10.0, y=18.0),
        Spawn(owner=2, unit_type=MARINE, x=18.0, y=16.0),
    ),
)


def _reset(backend: Backend, scenario: Scenario):
    state = backend.reset([scenario] * backend.batch_size, seeds=[0] * backend.batch_size)
    state.validate()
    return state


def check_reset_layout(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """After reset, slots, owners, types, and positions match the scenario."""
    state = _reset(backend, scenario)
    n = scenario.slot_count()
    assert state.batch_size == backend.batch_size
    assert state.n_slots == n
    for b in range(state.batch_size):
        for slot, spawn in enumerate(scenario.spawns):
            assert state.alive[b, slot], f'env {b} slot {slot} should be alive'
            assert state.owner[b, slot] == spawn.owner
            assert state.unit_type[b, slot] == spawn.unit_type
            assert state.variant[b, slot] == spawn.variant
            dist = np.hypot(state.x[b, slot] - spawn.x, state.y[b, slot] - spawn.y)
            assert dist <= SPAWN_TOLERANCE, f'slot {slot} spawn offset {dist:.2f}'
            assert state.health[b, slot] == state.health_max[b, slot] > 0
            assert state.order_kind[b, slot] == CommandKind.NOOP


def check_step_advances_loops(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """step(n_loops) advances exactly n_loops game loops and returns an independent state."""
    state = _reset(backend, scenario)
    before = state.game_loop.copy()
    noop = Command.noop(state.batch_size, state.n_slots)
    after = backend.step(noop, 8)
    after.validate()
    assert np.array_equal(after.game_loop - before, np.full_like(before, 8))
    assert np.array_equal(state.game_loop, before), 'backend must not mutate returned state'
    after2 = backend.step(noop, 1)
    assert np.array_equal(after2.game_loop - after.game_loop, np.ones_like(before))


def check_rejects_malformed_command(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """Commands with invalid shape or values raise ContractError."""
    state = _reset(backend, scenario)
    wrong_shape = Command.noop(state.batch_size, state.n_slots + 1)
    bad_kind = Command.noop(state.batch_size, state.n_slots)
    bad_kind.kind[0, 0] = 99
    for cmd in (wrong_shape, bad_kind):
        try:
            backend.step(cmd, 1)
        except ContractError:
            continue
        raise AssertionError('backend must reject malformed commands')


def check_move(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """MOVE advances the unit to the target point while other units stay idle."""
    state = _reset(backend, scenario)
    target = (10.0, 10.0)
    cmd = Command.noop(state.batch_size, state.n_slots)
    for b in range(state.batch_size):
        cmd.set(b, 0, CommandKind.MOVE, target_xy=target)
    start = state
    mid = backend.step(cmd, 16)
    noop = Command.noop(state.batch_size, state.n_slots)
    end = mid
    for _ in range(8):
        end = backend.step(noop, 8)
    for b in range(state.batch_size):
        d0 = np.hypot(start.x[b, 0] - target[0], start.y[b, 0] - target[1])
        d1 = np.hypot(mid.x[b, 0] - target[0], mid.y[b, 0] - target[1])
        d2 = np.hypot(end.x[b, 0] - target[0], end.y[b, 0] - target[1])
        assert d1 < d0 - 0.5, f'unit should approach target after 16 loops: {d0:.2f} -> {d1:.2f}'
        assert d2 < 1.0, f'unit should reach target after 80 loops, remaining {d2:.2f}'
        assert abs(end.x[b, 0] - start.x[b, 0]) < 0.5, 'straight move drifted laterally'
        for slot in (1, 2):
            moved = np.hypot(end.x[b, slot] - start.x[b, slot], end.y[b, slot] - start.y[b, slot])
            assert moved < 0.5, f'uncommanded slot {slot} moved by {moved:.2f}'


def check_attack_kills_and_dead_stay_dead(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """Two units focus-fire a target; dead units stay dead with alive=False and health=0."""
    state = _reset(backend, scenario)
    cmd = Command.noop(state.batch_size, state.n_slots)
    for b in range(state.batch_size):
        cmd.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=2)
        cmd.set(b, 1, CommandKind.ATTACK_UNIT, target_slot=2)
    state = backend.step(cmd, 8)
    noop = Command.noop(state.batch_size, state.n_slots)
    for _ in range(60):
        if not state.alive[:, 2].any():
            break
        assert np.all(state.order_kind[state.alive[:, 0], 0] != CommandKind.MOVE)
        state = backend.step(noop, 8)
    assert not state.alive[:, 2].any(), 'target should be killed within 480 loops'
    assert np.all(state.health[:, 2] == 0)
    assert state.alive[:, 0].all() and state.alive[:, 1].all(), 'attackers should survive'
    for _ in range(3):
        state = backend.step(noop, 8)
        assert not state.alive[:, 2].any(), 'dead unit must not revive'
    assert np.all(state.unit_type[:, 2] == MARINE), 'dead slot should retain unit_type'


def check_reset_restores(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """Subsequent reset restores the initial layout (dead units respawn)."""
    first = _reset(backend, scenario)
    cmd = Command.noop(first.batch_size, first.n_slots)
    for b in range(first.batch_size):
        cmd.set(b, 0, CommandKind.MOVE, target_xy=(10.0, 12.0))
    for _ in range(4):
        backend.step(cmd, 8)
    again = _reset(backend, scenario)
    assert np.array_equal(again.alive, first.alive)
    assert np.array_equal(again.owner, first.owner)
    assert np.array_equal(again.unit_type, first.unit_type)
    assert np.allclose(again.x, first.x, atol=SPAWN_TOLERANCE)
    assert np.allclose(again.y, first.y, atol=SPAWN_TOLERANCE)


def check_determinism(backend: Backend, scenario: Scenario = SCENARIO) -> None:
    """Deterministic backends produce bit-identical states for the same seed and commands."""
    if not backend.capabilities.deterministic:
        return

    def run():
        state = _reset(backend, scenario)
        cmd = Command.noop(state.batch_size, state.n_slots)
        for b in range(state.batch_size):
            cmd.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=2)
            cmd.set(b, 1, CommandKind.MOVE, target_xy=(16.0, 20.0))
        states = [backend.step(cmd, 8)]
        noop = Command.noop(state.batch_size, state.n_slots)
        states += [backend.step(noop, 8) for _ in range(10)]
        return states

    a, b = run(), run()
    for sa, sb in zip(a, b, strict=True):
        for name in ('game_loop', 'alive', 'x', 'y', 'health', 'weapon_cooldown'):
            np.testing.assert_array_equal(getattr(sa, name), getattr(sb, name), err_msg=name)


CHECKS: tuple[Callable[..., None], ...] = (
    check_reset_layout,
    check_step_advances_loops,
    check_rejects_malformed_command,
    check_move,
    check_attack_kills_and_dead_stay_dead,
    check_reset_restores,
    check_determinism,
)
