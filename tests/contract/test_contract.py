"""Unit tests for contract types (independent of any backend)."""

from __future__ import annotations

import numpy as np
import pytest

from unismac.contract import (
    AgentPlayer,
    Capabilities,
    CapabilityError,
    Command,
    CommandKind,
    ComputerPlayer,
    ContractError,
    Scenario,
    Spawn,
    UnitOverride,
    WorldState,
)


def test_world_state_empty_is_valid() -> None:
    state = WorldState.empty(batch_size=3, n_slots=5, n_players=2)
    state.validate()
    assert (state.batch_size, state.n_slots, state.n_players) == (3, 5, 2)
    assert np.all(state.order_target_slot == -1)
    assert np.isnan(state.order_x).all()


def test_world_state_validate_rejects_bad_dtype() -> None:
    state = WorldState.empty(2, 4)
    state.x = state.x.astype(np.float64)
    with pytest.raises(ContractError, match='x'):
        state.validate()


def test_world_state_copy_is_deep() -> None:
    state = WorldState.empty(1, 2)
    clone = state.copy()
    clone.x[0, 0] = 5
    assert state.x[0, 0] == 0


def test_command_set_and_validate() -> None:
    cmd = Command.noop(2, 3).set(1, 2, CommandKind.MOVE, target_xy=(4.0, 5.0))
    cmd.validate(2, 3)
    assert cmd.kind[1, 2] == CommandKind.MOVE
    assert (cmd.target_x[1, 2], cmd.target_y[1, 2]) == (4.0, 5.0)
    cmd.kind[0, 0] = 42
    with pytest.raises(ContractError):
        cmd.validate(2, 3)


def test_scenario_validation() -> None:
    ok = Scenario('SMAC_Maps/32x32_flat', spawns=(Spawn(1, 48, 1, 1), Spawn(2, 48, 2, 2)))
    ok.validate()
    assert ok.slot_count() == 2
    with pytest.raises(ContractError):
        Scenario('m', spawns=(Spawn(3, 48, 0, 0),)).validate()
    with pytest.raises(ContractError):
        Scenario('m', spawn_mode='map_trigger').validate()
    Scenario(
        'SMAC_Maps/3m',
        players=(AgentPlayer('terran'), ComputerPlayer('terran')),
        spawn_mode='map_trigger',
        n_slots=6,
    ).validate()


def test_scenario_unit_overrides_validation() -> None:
    spawns = (Spawn(1, 48, 1, 1), Spawn(2, 48, 2, 2))
    ok = Scenario('m', spawns=spawns, unit_overrides=(UnitOverride(1, 48, health_max=55.0),))
    ok.validate()
    Scenario(
        'm',
        spawn_mode='map_trigger',
        n_slots=6,
        unit_overrides=(UnitOverride(2, 73, shield_max=0.0),),
    ).validate()
    bad = [
        UnitOverride(3, 48, health_max=55.0),  # owner out of range
        UnitOverride(1, 48),  # no fields set
        UnitOverride(1, 48, health_max=0.0),
        UnitOverride(1, 48, shield_max=-1.0),
    ]
    for o in bad:
        with pytest.raises(ContractError):
            Scenario('m', spawns=spawns, unit_overrides=(o,)).validate()
    dup = (UnitOverride(1, 48, health_max=55.0), UnitOverride(1, 48, shield_max=10.0))
    with pytest.raises(ContractError, match='(?i)duplicate'):
        Scenario('m', spawns=spawns, unit_overrides=dup).validate()
    # Different variants are not duplicates
    Scenario(
        'm',
        spawns=spawns,
        unit_overrides=(
            UnitOverride(1, 48, health_max=55.0),
            UnitOverride(1, 48, 1, health_max=50.0),
        ),
    ).validate()


def test_capabilities_require() -> None:
    caps = Capabilities(
        name='t',
        max_batch=None,
        engine_ai=False,
        map_trigger=False,
        pathing='grid',
        collision='none',
        terrain_height=False,
        abilities=False,
        fog=False,
        deterministic=True,
    )
    caps.require(deterministic=True, pathing='grid')
    with pytest.raises(CapabilityError):
        caps.require(engine_ai=True)
    with pytest.raises(CapabilityError):
        caps.require(collision='hard')
    with pytest.raises(KeyError):
        caps.require(nope=True)
