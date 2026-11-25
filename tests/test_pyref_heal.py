"""Tests for pyref Medivac healing (`abilities='heal'`), aligned with SC2 behavior.

Fixtures are defined inline because pytest imports tests in importlib mode. All units belong
to Player 1 unless noted, and initial damage is set directly via `world.health`.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import abilities as ab
from unismac.contract import (
    NO_SLOT,
    CapabilityError,
    Command,
    CommandKind,
    ContractError,
    Scenario,
    Spawn,
    Variant,
    WorldState,
)

MARINE, MARAUDER, MEDIVAC = 48, 51, 54
HEAL = 386
STATIC = load_static_data()
FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
# Config matching SC2 heal behavior
SC2_HEAL: dict[str, object] = {
    'abilities': 'heal',
    'kinematics': 'accel',
    'spawn_energy': 50.0,
}
TICK = 2302 / 4096
COST = 759 / 4096
REGEN = 144 / 4096


def make(**config: object) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]


def start(backend: PyrefBackend, *spawns: Spawn, life: dict[int, float] | None = None) -> int:
    """Reset with `spawns`, set initial HP from `life`, and return slot count."""
    backend.reset([Scenario(map_id=FLAT, spawns=spawns)])
    sim = backend._sim  # noqa: SLF001 - test sets unit health directly
    assert sim is not None
    for slot, hp in (life or {}).items():
        sim.world.health[0, slot] = hp
    return len(spawns)


def set_energy(backend: PyrefBackend, slot: int, value: float) -> None:
    sim = backend._sim  # noqa: SLF001 - test sets unit energy directly
    assert sim is not None
    sim.world.energy[0, slot] = value


def heal_cmd(n: int, target: int = 1, caster: int = 0) -> Command:
    return Command.noop(1, n).set(0, caster, CommandKind.ABILITY, target_slot=target, ability=HEAL)


def run(
    backend: PyrefBackend, n: int, loops: int, first: Command | None = None
) -> list[WorldState]:
    """Issue `first` (default NOOP) on loop 0, advance `loops` loops, return per-loop states."""
    out = [backend.step(first if first is not None else Command.noop(1, n), 1)]
    out += [backend.step(Command.noop(1, n), 1) for _ in range(loops - 1)]
    return out


def series(states: list[WorldState], field: str, slot: int) -> np.ndarray:
    return np.array([getattr(s, field)[0, slot] for s in states], np.float64)


def medivac(x: float = 16.0, y: float = 16.0, variant: Variant = Variant.STD) -> Spawn:
    return Spawn(1, MEDIVAC, x, y, variant=variant)


# ---------------------------------------------------------------- Defaults and capabilities


def test_default_rejects_ability_and_keeps_energy() -> None:
    backend = make()
    assert not backend.capabilities.abilities
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    with pytest.raises(CapabilityError, match='abilities'):
        backend.step(heal_cmd(n), 1)
    states = run(backend, n, 20)
    assert np.all(series(states, 'energy', 0) == STATIC.get(MEDIVAC).energy_max)
    assert np.all(series(states, 'health', 1) == 10.0)


def test_capabilities_and_validation() -> None:
    backend = make(abilities='heal')
    caps = backend.capabilities
    assert caps.abilities
    caps.require(abilities=True)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0))
    with pytest.raises(CapabilityError, match='386'):
        backend.step(Command.noop(1, n).set(0, 0, CommandKind.ABILITY, target_slot=1, ability=1), 1)
    for target in (NO_SLOT, 0, 9):  # missing target / self / out of bounds
        with pytest.raises(ContractError):
            backend.step(heal_cmd(n, target=target), 1)
    with pytest.raises(ValueError, match='spawn_energy'):
        PyrefConfig(spawn_energy=-1.0)


def test_spawn_energy_and_regen() -> None:
    backend = make(abilities='heal', spawn_energy=50.0)
    n = start(backend, medivac(), Spawn(1, MARINE, 20.0, 16.0))
    states = run(backend, n, 16)
    energy = series(states, 'energy', 0)
    assert energy[-1] == pytest.approx(50.0 + 16 * REGEN, abs=1e-4)
    assert np.all(series(states, 'energy', 1) == 0)  # Marine has no energy
    # Energy cap
    backend = make(abilities='heal')
    n = start(backend, medivac())
    assert series(run(backend, n, 4), 'energy', 0)[-1] == STATIC.get(MEDIVAC).energy_max


# ---------------------------------------------------------------- Heal amount and energy


@pytest.mark.parametrize(('target', 'variant'), [(MARINE, Variant.STD), (MARAUDER, Variant.RL)])
def test_heal_amount_and_energy_cost(target: int, variant: Variant) -> None:
    """Channeling begins on the loop after the order with 2 ticks, then 1 tick per loop."""
    backend = make(**SC2_HEAL)
    n = start(
        backend,
        medivac(variant=variant),
        Spawn(1, target, 17.0, 16.0, variant=variant),
        life={1: 10.0},
    )
    states = run(backend, n, 12, heal_cmd(n))
    hp, en = series(states, 'health', 1), series(states, 'energy', 0)
    assert states[0].order_kind[0, 0] == CommandKind.ABILITY
    assert states[0].order_ability[0, 0] == HEAL
    assert states[0].order_target_slot[0, 0] == 1
    assert hp[0] == 10.0
    assert hp[1] == pytest.approx(10.0 + 2 * TICK, abs=1e-5)
    assert np.diff(hp[1:]) == pytest.approx(TICK, abs=1e-5)
    assert en[1] - en[0] == pytest.approx(REGEN - 2 * COST, abs=1e-5)
    assert np.diff(en[1:]) == pytest.approx(REGEN - COST, abs=1e-5)
    assert np.all(series(states, 'x', 0) == 16.0)  # does not move when already in range


def test_out_of_energy_ends_order_and_low_energy_command_ignored() -> None:
    backend = make(**SC2_HEAL)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    set_energy(backend, 0, 1.07)
    states = run(backend, n, 6, heal_cmd(n))
    hp = series(states, 'health', 1)
    assert hp[1] == pytest.approx(10.0 + TICK, abs=1e-5), (
        'energy < 1 before second tick should stop healing'
    )
    assert np.all(hp[2:] == hp[1])
    assert states[2].order_kind[0, 0] == CommandKind.NOOP
    # Order with energy < 1 is ignored (NotEnoughEnergy in SC2)
    set_energy(backend, 0, 0.5)
    assert backend.step(heal_cmd(n), 1).order_kind[0, 0] == CommandKind.NOOP


@pytest.mark.parametrize(
    ('spawns', 'life', 'target'),
    [
        ((Spawn(1, MARINE, 17.0, 16.0),), {}, 1),  # full HP (TooMuchLife in SC2)
        ((medivac(17.0),), {1: 50.0}, 1),  # mechanical unit
        ((Spawn(2, MARINE, 17.0, 16.0),), {1: 10.0}, 1),  # enemy unit
    ],
)
def test_invalid_target_ignored(spawns: tuple[Spawn, ...], life: dict[int, float], target: int):
    backend = make(abilities='heal')
    n = start(backend, medivac(), *spawns, life=life)
    hold = Command.noop(1, n).set(0, 0, CommandKind.HOLD)
    backend.step(hold, 1)
    state = backend.step(heal_cmd(n, target), 1)
    assert state.order_kind[0, 0] == CommandKind.HOLD, 'ignored command must keep current order'


def test_non_caster_ignored() -> None:
    backend = make(abilities='heal')
    n = start(backend, Spawn(1, MARINE, 16.0, 16.0), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    state = run(backend, n, 4, heal_cmd(n))[-1]
    assert state.order_kind[0, 0] == CommandKind.NOOP
    assert state.health[0, 1] == 10.0


# ---------------------------------------------------------------- Approach, full HP, death, leash


def test_approach_from_out_of_range() -> None:
    """Edge dist 10.875: approaches, starts braking, channels at range 4 and coasts to a stop
    (starts healing on loop 53 after order, stops at edge dist 2.9255)."""
    backend = make(**SC2_HEAL)
    n = start(backend, medivac(8.0), Spawn(1, MARINE, 20.0, 16.0), life={1: 10.0})
    states = run(backend, n, 80, heal_cmd(n))
    hp, x = series(states, 'health', 1), series(states, 'x', 0)
    first = int(np.argmax(hp > 10.0))
    assert first == 53
    edge = 20.0 - x - (0.75 + 0.375)
    assert edge[first - 1] <= ab.HEAL_RANGE < edge[first - 2]
    assert x[-1] == x[-2]
    assert 20.0 - x[-1] - 1.125 == pytest.approx(2.9255, abs=1e-3)
    assert np.all(np.diff(hp[first:]) > 0)


def test_approach_constant_speed() -> None:
    """Units without accel parameters (Medivac_RL) approach at constant speed and stop in range."""
    backend = make(abilities='heal')
    n = start(
        backend,
        medivac(8.0, variant=Variant.RL),
        Spawn(1, MARINE, 20.0, 16.0, variant=Variant.RL),
        life={1: 10.0},
    )
    states = run(backend, n, 60, heal_cmd(n))
    x = series(states, 'x', 0)
    edge = 20.0 - x[-1] - 1.125
    assert 0 < edge <= ab.HEAL_RANGE
    assert series(states, 'health', 1)[-1] > 10.0


def test_full_hp_ends_order_after_three_loops() -> None:
    backend = make(**SC2_HEAL)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 44.0})
    states = run(backend, n, 8, heal_cmd(n))
    hp, en = series(states, 'health', 1), series(states, 'energy', 0)
    full = int(np.argmax(hp == 45.0))
    assert full == 1
    kinds = [int(s.order_kind[0, 0]) for s in states]
    assert kinds[: full + 3] == [CommandKind.ABILITY] * (full + 3)
    assert kinds[full + 3] == CommandKind.NOOP
    assert states[full + 3].order_ability[0, 0] == 0
    assert np.diff(en[full:]) == pytest.approx(REGEN, abs=1e-5), (
        'should stop spending energy once target is at full HP'
    )


def test_target_death_clears_order() -> None:
    """If the heal target dies, Medivac clears its order on the death loop without moving."""
    backend = make(abilities='heal')
    n = start(
        backend,
        medivac(),
        Spawn(1, MARINE, 17.0, 16.0),
        Spawn(2, MARAUDER, 18.0, 16.0),
        life={1: 3.0},
    )
    cmd = heal_cmd(n).set(0, 2, CommandKind.ATTACK_UNIT, target_slot=1)
    states = run(backend, n, 40, cmd)
    dead = next(i for i, s in enumerate(states) if not s.alive[0, 1])
    s = states[dead]
    assert (s.order_kind[0, 0], s.order_ability[0, 0], s.order_target_slot[0, 0]) == (
        CommandKind.NOOP,
        0,
        NO_SLOT,
    )
    assert np.all(series(states, 'x', 0) == 16.0)
    assert np.all(series(states[dead:], 'order_kind', 0) == CommandKind.NOOP)


def test_target_leaves_breaks_and_resumes() -> None:
    """Target leaving past leash range 5 breaks channeling; catching up restarts with 2 ticks."""
    backend = make(**SC2_HEAL)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    states = run(backend, n, 4, heal_cmd(n))
    states += run(
        backend, n, 120, Command.noop(1, n).set(0, 1, CommandKind.MOVE, target_xy=(30, 16))
    )
    hp, x0, x1 = (series(states, f, s) for f, s in (('health', 1), ('x', 0), ('x', 1)))
    gain = np.diff(hp)
    edge = x1 - x0 - 1.125
    stop = next(i for i in range(4, len(gain)) if gain[i] == 0)
    assert edge[stop] > ab.HEAL_LEASH >= edge[stop - 1]
    resume = next(i for i in range(stop, len(gain)) if gain[i] > 0)
    assert gain[resume] == pytest.approx(2 * TICK, abs=1e-5)
    assert x0[-1] > 20.0, 'Medivac should chase the moving target'
    assert states[-1].order_kind[0, 0] == CommandKind.ABILITY


def test_new_command_cancels_heal() -> None:
    backend = make(abilities='heal')
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    run(backend, n, 3, heal_cmd(n))
    states = run(backend, n, 3, Command.noop(1, n).set(0, 0, CommandKind.MOVE, target_xy=(16, 26)))
    assert states[0].order_kind[0, 0] == CommandKind.MOVE
    hp = series(states, 'health', 1)
    assert np.all(hp == hp[0])


# ---------------------------------------------------------------- Autocast heal


def test_autocast_off_by_default() -> None:
    backend = make(abilities='heal')
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    states = run(backend, n, 30)
    assert np.all(series(states, 'health', 1) == 10.0)


def test_autocast_idle_std_only() -> None:
    """Standard Medivac autocasts when idle (scan loop 5, heal loop 7); Medivac_RL does not."""
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
    states = run(backend, n, 12)
    kinds = series(states, 'order_kind', 0)
    first_order = int(np.argmax(kinds == CommandKind.ABILITY))
    assert first_order == 5
    hp = series(states, 'health', 1)
    assert int(np.argmax(hp > 10.0)) == 7
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(
        backend,
        medivac(variant=Variant.RL),
        Spawn(1, MARINE, 17.0, 16.0, variant=Variant.RL),
        life={1: 10.0},
    )
    assert np.all(series(run(backend, n, 30), 'health', 1) == 10.0)


def test_autocast_range_energy_and_pick() -> None:
    def first_heal(x: float, energy: float | None = None, n_loops: int = 60) -> int | None:
        backend = make(**SC2_HEAL, heal_autocast=True)
        n = start(backend, medivac(), Spawn(1, MARINE, x, 16.0), life={1: 10.0})
        if energy is not None:
            set_energy(backend, 0, energy)
        hp = series(run(backend, n, n_loops), 'health', 1)
        return int(np.argmax(hp > 10.0)) if hp[-1] > 10.0 else None

    assert first_heal(23.125) is not None  # edge dist 6.0: flies over
    assert first_heal(23.625) is None  # edge dist 6.5: out of autocast range
    # Energy 4 needs ~28 loops to regen to 5, so no heal within first 20 loops
    assert first_heal(17.0, energy=4.0, n_loops=20) is None
    assert first_heal(17.0, energy=5.0, n_loops=20) == 7
    # Two candidates: picks lower absolute life (even if farther)
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(
        backend,
        medivac(),
        Spawn(1, MARINE, 19.125, 16.0),
        Spawn(1, MARINE, 16.0, 17.625),
        life={1: 10.0, 2: 40.0},
    )
    assert run(backend, n, 8)[-1].order_target_slot[0, 0] == 1


@pytest.mark.parametrize(
    ('marine', 'marauder', 'pick', 'expect'),
    [
        (30.0, 60.0, 'life', 1),  # default: lower absolute life (Marine)
        (30.0, 60.0, 'fraction', 2),  # 0.667 vs 0.48: Marauder
        (20.0, 60.0, 'fraction', 1),  # 0.444 vs 0.48: Marine
        (22.5, 62.5, 'fraction', 2),  # tied at 0.5: closer Marauder
    ],
)
def test_autocast_heal_pick(marine: float, marauder: float, pick: str, expect: int) -> None:
    backend = make(**SC2_HEAL, heal_autocast=True, heal_pick=pick)
    # Edge distances: Marine 2.0 (right), Marauder 0.5 (left)
    n = start(
        backend,
        medivac(),
        Spawn(1, MARINE, 16.0 + 0.75 + 0.375 + 2.0, 16.0),
        Spawn(1, MARAUDER, 16.0 - 0.75 - 0.5625 - 0.5, 16.0),
        life={1: marine, 2: marauder},
    )
    assert run(backend, n, 8)[-1].order_target_slot[0, 0] == expect


def test_heal_pick_validation() -> None:
    with pytest.raises(ValueError, match='heal_pick'):
        PyrefConfig(heal_pick='ratio')


def test_autocast_not_while_moving_and_hold_returns_to_hold() -> None:
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(backend, medivac(8.0), Spawn(1, MARINE, 16.0, 17.5), life={1: 10.0})
    states = run(
        backend, n, 100, Command.noop(1, n).set(0, 0, CommandKind.MOVE, target_xy=(26, 16))
    )
    assert np.all(series(states, 'health', 1) == 10.0)
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 40.0})
    states = run(backend, n, 30, Command.noop(1, n).set(0, 0, CommandKind.HOLD))
    kinds = [int(k) for k in series(states, 'order_kind', 0)]
    assert CommandKind.ABILITY in kinds
    assert kinds[-1] == CommandKind.HOLD
    assert states[-1].health[0, 1] == 45.0
    # Under HOLD, only heal targets within range without moving
    backend = make(**SC2_HEAL, heal_autocast=True)
    n = start(backend, medivac(), Spawn(1, MARINE, 22.625, 16.0), life={1: 10.0})
    states = run(backend, n, 40, Command.noop(1, n).set(0, 0, CommandKind.HOLD))
    assert np.all(series(states, 'health', 1) == 10.0)
    assert np.all(series(states, 'x', 0) == 16.0)


# ---------------------------------------------------------------- heal_on_arrival / heal_facing


def _move_after_heal(dx: float, **opts: object) -> tuple[list[WorldState], int]:
    """After autocast heal starts, MOVE Medivac to (16 + dx, 16) on loop 20; returns states
    and arrival frame (first 1-based frame where x equals goal, or -1 if dx == 0)."""
    backend = make(**SC2_HEAL, heal_autocast=True, **opts)
    n = start(backend, medivac(), Spawn(1, MARAUDER, 17.5, 16.0), life={1: 10.0})
    states = run(backend, n, 20)
    move = Command.noop(1, n).set(0, 0, CommandKind.MOVE, target_xy=(16.0 + dx, 16.0))
    states += run(backend, n, 30, move)
    x = series(states, 'x', 0)
    arrive = int(np.argmax(x == np.float32(16.0 + dx))) + 1 if dx else -1
    return states, arrive


def _first(states: list[WorldState], start: int, cond: np.ndarray) -> int:
    """First 1-based frame >= `start` where `cond` is True."""
    return int(np.argmax(cond[start - 1 :])) + start


def test_heal_on_arrival_rescans_on_arrival() -> None:
    """Non-zero MOVE resumes healing on arrival frame without waiting for scan phase;
    zero-displacement MOVE behaves identically whether on or off."""
    for opts, same_frame in (({}, False), ({'heal_on_arrival': True}, True)):
        states, arrive = _move_after_heal(0.3, **opts)
        ab = series(states, 'order_ability', 0)
        assert ab[arrive - 2] != HEAL  # still moving on frame before arrival
        assert (ab[arrive - 1] == HEAL) == same_frame
    a, _ = _move_after_heal(0.0)
    b, _ = _move_after_heal(0.0, heal_on_arrival=True)
    for f in ('order_kind', 'energy', 'health'):
        assert np.array_equal(series(a, f, 0 if f != 'health' else 1),
                              series(b, f, 0 if f != 'health' else 1))  # fmt: skip


def test_heal_facing_start_follows_facing() -> None:
    """`heal_facing`: heals on arrival frame if facing target, 3 frames later if facing away
    (180 deg); when off, fixed 2-frame delay. Manual heal orders are unaffected."""
    for dx, lag_face in ((0.3, 0), (-0.3, 3)):
        for opts, lag in (({'heal_facing': True}, lag_face), ({}, 2)):
            states, arrive = _move_after_heal(dx, heal_on_arrival=True, **opts)
            hp = series(states, 'health', 1)
            gain = np.r_[False, np.diff(hp) > 0]
            assert _first(states, arrive, gain) - arrive == lag
    out = []
    for opts in ({}, {'heal_facing': True}):
        backend = make(**SC2_HEAL, **opts)
        n = start(backend, medivac(), Spawn(1, MARINE, 17.0, 16.0), life={1: 10.0})
        sim = backend._sim  # noqa: SLF001 - test sets unit facing directly
        assert sim is not None
        sim.world.facing[0, 0] = np.float32(np.pi)  # facing away from injured ally
        out.append(series(run(backend, n, 12, heal_cmd(n)), 'health', 1))
    assert np.array_equal(out[0], out[1])
