"""Tests for pyref `attack_unit_needs_weapon`, `slow_arrival`, `retarget_armed_only`, and
`hit_slow_owner1_only`.

`attack_unit_needs_weapon`: units without a weapon able to hit the target (e.g., Marauder
targeting air Medivac) ignore ATTACK_UNIT and keep their prior order (idle, MOVE, or ground
ATTACK_UNIT).

`slow_arrival`: hit-slow only caps movement speed, so the final step to a waypoint moves by
min(remaining_dist, slow_factor * speed) and finishes MOVE on arrival; by default scaling the
whole step causes the last few loops to cover only half the remaining distance each loop.

`retarget_armed_only`: when retargeting from an armed target, only switch to an armed enemy,
not to an unarmed Medivac.

`hit_slow_owner1_only`: hit-slow under `damage='sc2'` applies only to player 1 attackers.

All four flags are disabled by default, leaving default behavior bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, MARAUDER, MEDIVAC = 48, 51, 54


def _backend(spawns: list[Spawn], **config: object) -> PyrefBackend:
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]
    be.reset([Scenario(FLAT, spawns=tuple(spawns))], seeds=[0])
    return be


def _steps(be: PyrefBackend, n: int, first: Command, loops: int) -> list[WorldState]:
    out = [be.step(first, 1)]
    out += [be.step(Command.noop(1, n), 1) for _ in range(loops - 1)]
    return out


# ---------------------------------------------------------------- attack_unit_needs_weapon

# Player 1: Marauder (0), Marine (1); Player 2: Medivac (2, air), Marine (3, ground), >6 away
UNITS = [
    Spawn(1, MARAUDER, 10.0, 16.0),
    Spawn(1, MARINE, 10.0, 18.0),
    Spawn(2, MEDIVAC, 20.0, 16.0),
    Spawn(2, MARINE, 20.0, 18.0),
]


def _attack(n: int, unit: int, target: int) -> Command:
    return Command.noop(1, n).set(0, unit, CommandKind.ATTACK_UNIT, target_slot=target)


@pytest.mark.parametrize('on', [False, True])
def test_marauder_attack_medivac_idle(on: bool) -> None:
    """Idle Marauder ordered to attack Medivac: chases by default; ignores and stays put if on."""
    be = _backend(UNITS, attack_unit_needs_weapon=on)
    states = _steps(be, 4, _attack(4, 0, 2), 8)
    last = states[-1]
    if on:
        assert int(last.order_kind[0, 0]) == CommandKind.NOOP
        assert int(last.order_target_slot[0, 0]) == NO_SLOT
        assert last.x[0, 0] == pytest.approx(10.0)
    else:
        assert int(last.order_kind[0, 0]) == CommandKind.ATTACK_UNIT
        assert int(last.order_target_slot[0, 0]) == 2
        assert last.x[0, 0] > 10.5
    # Never fires: Medivac stays at full HP in both modes
    assert last.health[0, 2] == STATIC.get(MEDIVAC).health_max


def test_rejected_attack_keeps_previous_orders() -> None:
    """When enabled, rejected ATTACK_UNIT preserves prior MOVE and ground ATTACK_UNIT orders."""
    be = _backend(UNITS, attack_unit_needs_weapon=True)
    mv = Command.noop(1, 4).set(0, 0, CommandKind.MOVE, target_xy=(5.0, 16.0))
    be.step(mv, 1)
    s = be.step(_attack(4, 0, 2), 1)
    assert int(s.order_kind[0, 0]) == CommandKind.MOVE
    assert s.order_x[0, 0] == pytest.approx(5.0)
    be.step(_attack(4, 0, 3), 1)  # ground target accepted normally
    s = be.step(_attack(4, 0, 2), 1)
    assert int(s.order_kind[0, 0]) == CommandKind.ATTACK_UNIT
    assert int(s.order_target_slot[0, 0]) == 3


def test_armed_attackers_unchanged() -> None:
    """Valid attackers (Marine vs Medivac, Marauder vs Marine) match default bit-for-bit."""
    runs = []
    for on in (False, True):
        be = _backend(UNITS, attack_unit_needs_weapon=on)
        cmd = _attack(4, 1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=3)
        runs.append(_steps(be, 4, cmd, 40))
    for a, b in zip(*runs, strict=True):
        for f in ('x', 'y', 'health', 'weapon_cooldown', 'order_kind', 'order_target_slot'):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f


# ---------------------------------------------------------------- slow_arrival


def _slowed_move(on: bool, dist: float, loops: int) -> list[WorldState]:
    """Player 1 Marauder permanently slowed (factor 0.5) moving `dist` along +x."""
    be = _backend([Spawn(1, MARAUDER, 10.0, 16.0), Spawn(2, MARINE, 28.0, 28.0)], slow_arrival=on)
    sim = be._sim  # noqa: SLF001 - direct slow state setup for test
    assert sim is not None
    sim.combat.slow_from[0, 0] = 0
    sim.combat.slow_to[0, 0] = 10**6
    sim.combat.slow_factor[0, 0] = 0.5
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.MOVE, target_xy=(10.0 + dist, 16.0))
    return _steps(be, 2, cmd, loops)


def test_slowed_move_arrives() -> None:
    """0.25-tile MOVE: slowed cap is 0.5 * 2.25 / 16 ~= 0.0703 per loop, arriving and clearing
    MOVE on loop 4; default halves the remaining distance each step and still has not arrived
    after 16 loops."""
    cap = 0.5 * STATIC.get(MARAUDER).movement_speed / 16
    on = _slowed_move(True, 0.25, 16)
    n_on = next(i for i, s in enumerate(on) if int(s.order_kind[0, 0]) == CommandKind.NOOP)
    assert n_on + 1 == int(np.ceil(0.25 / cap))
    assert on[n_on].x[0, 0] == pytest.approx(10.25, abs=1e-6)
    off = _slowed_move(False, 0.25, 16)
    assert all(int(s.order_kind[0, 0]) == CommandKind.MOVE for s in off)
    # While remaining distance >= unslowed step (non-final steps), both take full slowed step
    speed = 2 * cap
    same = sum(0.25 - i * cap >= speed for i in range(n_on))
    assert same >= 1
    for a, b in zip(on[:same], off[:same], strict=True):
        assert a.x[0, 0] == b.x[0, 0]
    assert off[same].x[0, 0] < on[same].x[0, 0]


def test_slow_arrival_off_when_not_slowed() -> None:
    """Unslowed units are unaffected and match default bit-for-bit."""
    runs = []
    for on in (False, True):
        be = _backend(
            [Spawn(1, MARAUDER, 10.0, 16.0), Spawn(2, MARINE, 28.0, 28.0)], slow_arrival=on
        )
        cmd = Command.noop(1, 2).set(0, 0, CommandKind.MOVE, target_xy=(10.25, 16.0))
        runs.append(_steps(be, 2, cmd, 8))
    for a, b in zip(*runs, strict=True):
        assert np.array_equal(a.x, b.x) and np.array_equal(a.order_kind, b.order_kind)


# ---------------------------------------------------------------- retarget_armed_only

RT = {'targeting': 'sc2', 'auto_acquire': True, 'acquire_start_loop': 2, 'retarget': True}


def _retarget_from(cur_type: int, near_type: int, on: bool) -> tuple[int, int, int]:
    """Player 2 Marine (0) on ATTACK_MOVE engaged on out-of-range player 1 `cur_type` (1, edge
    ~7) with nearer `near_type` (2, edge ~2) in range; scans once, returns (new, cur, near)."""
    be = _backend(
        [
            Spawn(2, MARINE, 10.0, 16.0),
            Spawn(1, cur_type, 18.0, 16.0),
            Spawn(1, near_type, 13.0, 16.0),
        ],
        retarget_armed_only=on,
        **RT,
    )
    sim = be._sim  # noqa: SLF001 - direct engagement setup for test
    assert sim is not None
    w = sim.world
    w.order_kind[0, 0] = CommandKind.ATTACK_MOVE
    w.order_x[0, 0], w.order_y[0, 0] = np.float32(20), np.float32(16)
    w.engaged_target_slot[0, 0] = 1
    w.game_loop[0] = 10  # scan loop: (10 - 2) % 8 == 0
    targeting.sc2(sim)
    return int(w.engaged_target_slot[0, 0]), 1, 2


@pytest.mark.parametrize('on', [False, True])
def test_retarget_armed_only_keeps_armed_target(on: bool) -> None:
    """Armed target Marine with closer unarmed Medivac in range: switches if off, keeps if on."""
    got, cur, near = _retarget_from(MARINE, MEDIVAC, on)
    assert got == (cur if on else near)


@pytest.mark.parametrize('on', [False, True])
@pytest.mark.parametrize(('cur_type', 'near_type'), [(MARINE, MARINE), (MEDIVAC, MARINE)])
def test_retarget_armed_only_other_switches_unchanged(
    cur_type: int, near_type: int, on: bool
) -> None:
    """Switching to an armed enemy is unaffected regardless of whether current target is armed."""
    got, _, near = _retarget_from(cur_type, near_type, on)
    assert got == near


def test_retarget_armed_only_needs_retarget() -> None:
    with pytest.raises(ValueError, match='retarget_armed_only'):
        PyrefConfig(retarget_armed_only=True)


# ---------------------------------------------------------------- hit_slow_owner1_only


def _marauder_hit(owner: int, on: bool) -> tuple[PyrefBackend, list[WorldState]]:
    """`owner`'s Marauder (0) attacks enemy Marine (1) at edge ~1.4 under `damage='sc2'`; after 20
    loops the Marine moves away along +x (no turn) for 2 loops."""
    be = _backend(
        [Spawn(owner, MARAUDER, 10.0, 16.0), Spawn(3 - owner, MARINE, 12.5, 16.0)],
        damage='sc2',
        hit_slow_owner1_only=on,
    )
    states = _steps(be, 2, _attack(2, 0, 1), 20)
    mv = Command.noop(1, 2).set(0, 1, CommandKind.MOVE, target_xy=(28.0, 16.0))
    states += _steps(be, 2, mv, 2)
    return be, states


def test_hit_slow_owner2_off() -> None:
    """Player 2 Marauder hitting player 1 Marine: slows by half by default; no slow when on."""
    assert STATIC.get(MARAUDER).measured['hit_slow_loops'] > 0
    full = STATIC.get(MARINE).movement_speed / 16
    for on in (False, True):
        be, states = _marauder_hit(2, on)
        assert states[19].health[0, 1] < STATIC.get(MARINE).health_max  # hit landed
        sim = be._sim  # noqa: SLF001 - inspect slow state
        assert sim is not None
        c = sim.combat
        step = states[-1].x[0, 1] - states[-2].x[0, 1]
        if on:
            assert int(c.slow_from[0, 1]) == -1
            assert step == pytest.approx(full, rel=1e-4)
        else:
            assert int(c.slow_from[0, 1]) >= 0
            assert step == pytest.approx(0.5 * full, rel=1e-4)


def test_hit_slow_owner1_unchanged() -> None:
    """Player 1 Marauder still applies slow, matching default bit-for-bit."""
    runs = [_marauder_hit(1, on)[1] for on in (False, True)]
    assert runs[0][-1].x[0, 1] - runs[0][-2].x[0, 1] < STATIC.get(MARINE).movement_speed / 16
    for a, b in zip(*runs, strict=True):
        for f in ('x', 'y', 'health', 'weapon_cooldown', 'order_kind'):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
