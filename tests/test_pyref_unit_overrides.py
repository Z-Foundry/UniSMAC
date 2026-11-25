"""Tests for pyref `Scenario.unit_overrides`: units matching (owner, unit_type, variant) spawn at
overridden health_max / shield_max, and regen/heal/shield recovery cap at those values. Empty
overrides are bitwise-identical."""

from __future__ import annotations

import dataclasses

import numpy as np

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, UnitOverride, Variant

MARINE, ZEALOT, MEDIVAC, ZERGLING = 48, 73, 54, 105
HEAL = 386
FLAT = 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {FLAT: load_map_info(FLAT)}


def make(**config: object) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]


def scenario(*spawns: Spawn, overrides: tuple[UnitOverride, ...] = ()) -> Scenario:
    return Scenario(map_id=FLAT, spawns=spawns, unit_overrides=overrides)


def test_capability_declared() -> None:
    assert make().capabilities.unit_overrides
    assert make(abilities='heal', visibility='sc2', targeting='sc2').capabilities.unit_overrides


def test_spawn_uses_overridden_caps() -> None:
    """Matches (owner, unit_type, variant) exactly: player 2 Marine and player 1 Marine_RL
    are unchanged; None fields fall back to static data."""
    s = make().reset(
        [
            scenario(
                Spawn(1, MARINE, 10.0, 16.0),
                Spawn(2, MARINE, 20.0, 16.0),
                Spawn(1, MARINE, 10.0, 18.0, variant=Variant.RL),
                Spawn(2, ZEALOT, 20.0, 18.0),
                overrides=(
                    UnitOverride(1, MARINE, health_max=55.0),
                    UnitOverride(2, ZEALOT, shield_max=0.0),
                ),
            )
        ]
    )
    assert s.health_max[0].tolist() == [55.0, 45.0, 45.0, 100.0]
    assert s.health[0].tolist() == s.health_max[0].tolist(), 'spawns at full health'
    assert s.shield_max[0, 3] == 0.0 and s.shield[0, 3] == 0.0


def test_empty_overrides_bitwise_unchanged() -> None:
    def run(sc: Scenario) -> list:
        backend = make(attack='sc2', damage='sc2', targeting='sc2')
        out = [backend.reset([sc])]
        cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
        out += [backend.step(cmd if k == 0 else Command.noop(1, 2), 1) for k in range(60)]
        return out

    base = scenario(Spawn(1, MARINE, 12.0, 16.0), Spawn(2, ZEALOT, 15.0, 16.0))
    a = run(base)
    b = run(dataclasses.replace(base, unit_overrides=()))
    for x, y in zip(a, b, strict=True):
        for f in dataclasses.fields(x):
            assert np.array_equal(getattr(x, f.name), getattr(y, f.name), equal_nan=True), f.name


def test_heal_caps_at_overridden_max() -> None:
    backend = make(abilities='heal')
    backend.reset(
        [
            scenario(
                Spawn(1, MEDIVAC, 16.0, 16.0),
                Spawn(1, MARINE, 17.0, 16.0),
                overrides=(UnitOverride(1, MARINE, health_max=55.0),),
            )
        ]
    )
    sim = backend._sim  # noqa: SLF001 - test sets health directly
    assert sim is not None
    sim.world.health[0, 1] = 44.0
    heal = Command.noop(1, 2).set(0, 0, CommandKind.ABILITY, target_slot=1, ability=HEAL)
    states = [backend.step(heal, 1)] + [backend.step(Command.noop(1, 2), 1) for _ in range(40)]
    hp = [float(s.health[0, 1]) for s in states]
    assert max(hp) == 55.0, 'heals to overridden max beyond static 45'
    assert int(states[-1].order_kind[0, 0]) == CommandKind.NOOP, 'stops at full health'


def test_regen_caps_at_overridden_max() -> None:
    """Zergling health regen and Zealot shield regen both cap at overridden max values."""
    backend = make(damage='sc2')
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(1, ZEALOT, 20.0, 16.0),
                overrides=(
                    UnitOverride(1, ZERGLING, health_max=36.0),
                    UnitOverride(1, ZEALOT, shield_max=60.0),
                ),
            )
        ]
    )
    sim = backend._sim  # noqa: SLF001
    assert sim is not None
    sim.world.health[0, 0] = 34.9
    sim.world.shield[0, 1] = 49.0
    states = [backend.step(Command.noop(1, 2), 1) for _ in range(200)]
    assert float(states[-1].health[0, 0]) == 36.0
    assert float(states[-1].shield[0, 1]) == 60.0
