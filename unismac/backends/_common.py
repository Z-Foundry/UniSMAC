"""Shared helper functions used by both `pyref` and `zgame` simulation backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, TypeAlias

from unismac.contract import (
    AgentPlayer,
    Capabilities,
    CapabilityError,
    ContractError,
    MapInfo,
    Scenario,
    StaticData,
)

PassiveOwners: TypeAlias = tuple[tuple[int, ...] | None, tuple[int, ...]]


def collect_attributes(static_data: StaticData) -> tuple[str, ...]:
    """Collect and sort all unit/weapon attribute names for 31-bit bitmask indexing."""
    names: set[str] = set()
    for spec in static_data.units.values():
        names.update(spec.attributes)
        for weapon in spec.weapons:
            names.update(name for name, _ in weapon.bonus)
    if len(names) > 31:
        raise ContractError(f'Attribute count {len(names)} exceeds 31-bit mask capacity')
    return tuple(sorted(names))


def capabilities_for_options(
    base: Capabilities,
    *,
    pathing: str,
    collision: str,
    abilities: str,
    visibility: str,
    projectile_3d: bool,
    grid_pathing: tuple[str, ...] = ('grid', 'navmesh'),
) -> Capabilities:
    """Derive effective `Capabilities` from backend option values."""
    pathing_cap: Literal['none', 'grid'] = 'grid' if pathing in grid_pathing else 'none'
    collision_cap: Literal['none', 'hard'] = 'hard' if collision in ('hard', 'native') else 'none'
    fog = visibility == 'sc2'
    return replace(
        base,
        pathing=pathing_cap,
        collision=collision_cap,
        abilities=abilities == 'heal',
        fog=fog,
        terrain_height=fog or projectile_3d,
    )


def resolve_passive_owners(map_id: str, n_players: int) -> PassiveOwners | None:
    """Resolve `(auto_acquire_owners, retreat_owners)` from `assets/map_abilities.json`.

    Returns `None` when `map_id` has no entry, `(None, ())` when the entry is empty,
    or `(active_owners, passive_owners)` otherwise.
    """
    from unismac.assets import load_passive_owners

    passive = load_passive_owners(map_id)
    if passive is None:
        return None
    if not passive:
        return None, ()
    active = tuple(p for p in range(1, n_players + 1) if p not in passive)
    return active, tuple(passive)


def validate_agent_scenario(
    scenario: Scenario, maps: Mapping[str, MapInfo], backend_name: str
) -> MapInfo:
    """Validate scenario contract requirements for agent-only backends and return its `MapInfo`."""
    scenario.validate()
    if scenario.spawn_mode == 'map_trigger':
        raise CapabilityError(f'Backend {backend_name} does not support spawn_mode=map_trigger')
    for i, player in enumerate(scenario.players):
        if not isinstance(player, AgentPlayer):
            raise CapabilityError(
                f'Backend {backend_name} does not support {type(player).__name__} (players[{i}])'
            )
    try:
        return maps[scenario.map_id]
    except KeyError:
        raise ContractError(f'Unknown map {scenario.map_id!r}') from None


__all__ = [
    'PassiveOwners',
    'capabilities_for_options',
    'collect_attributes',
    'resolve_passive_owners',
    'validate_agent_scenario',
]
