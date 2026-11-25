"""`PyrefBackend`: Python reference backend implementing the unismac `Backend` interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from unismac.assets import with_stalker_shield_regen
from unismac.contract import (
    NO_SLOT,
    Capabilities,
    Command,
    CommandKind,
    ContractError,
    MapInfo,
    Scenario,
    StaticData,
    WorldState,
)

from .._common import (
    capabilities_for_options,
    collect_attributes,
    resolve_passive_owners,
    validate_agent_scenario,
)
from . import pipeline
from .config import PyrefConfig
from .sim import Sim, reset_env
from .subsystems.commands import check_command

# Capabilities for the default config (straight movement, no collision); actual
# capabilities are derived from the active config via `capabilities_for`.
CAPABILITIES = Capabilities(
    name='pyref',
    max_batch=None,
    engine_ai=False,
    map_trigger=False,
    pathing='none',
    collision='none',
    terrain_height=False,
    abilities=False,
    fog=False,
    deterministic=True,
    unit_overrides=True,  # applies Scenario.unit_overrides to health_max / shield_max on reset
    mirror_spawn=True,
)


def _charge_owners(map_id: str) -> tuple[int, ...]:
    """Return player owners whose Zealots have Charge on this map (assets/map_abilities.json)."""
    from unismac.assets import load_charge_owners

    return load_charge_owners(map_id)


def _passive_config(config: PyrefConfig, scenario: Scenario) -> PyrefConfig:
    """Apply map Passive behavior to auto_acquire_owners and retreat_owners when
    `config.passive_from_map` is enabled (from assets/map_abilities.json; unchanged if absent).
    """
    if not config.passive_from_map:
        return config
    resolved = resolve_passive_owners(scenario.map_id, len(scenario.players))
    if resolved is None:
        return config
    acquire, retreat = resolved
    return replace(config, auto_acquire_owners=acquire, retreat_owners=retreat)


def _check_scenario(
    scenario: Scenario, maps: Mapping[str, MapInfo], static_data: StaticData
) -> MapInfo:
    """Validate scenario contract and spawn bounds for `PyrefBackend`."""
    info = validate_agent_scenario(scenario, maps, 'pyref')
    x0, y0, x1, y1 = info.playable
    for i, s in enumerate(scenario.spawns):
        static_data.get(s.unit_type, s.variant)
        if not (x0 <= s.x <= x1 and y0 <= s.y <= y1):
            raise ContractError(
                f'spawns[{i}] position ({s.x}, {s.y}) outside playable area {info.playable}'
            )
    return info


def capabilities_for(config: PyrefConfig) -> Capabilities:
    """Derive backend capabilities from config: pathing='grid'/'navmesh' -> 'grid',
    collision='hard'/'native' -> 'hard', abilities='heal' -> abilities=True (MedivacHeal 386
    only), visibility='sc2' -> fog=True and terrain_height=True (height-layered vision).
    """
    return capabilities_for_options(
        CAPABILITIES,
        pathing=config.pathing,
        collision=config.collision,
        abilities=config.abilities,
        visibility=config.visibility,
        projectile_3d=config.projectile_3d,
    )


class PyrefBackend:
    """Python reference backend.

    * Rejects `ComputerPlayer` and `map_trigger` scenarios with `CapabilityError`.
    * All scenarios in a single `reset()` must share slot and player counts; partial resets
      must match the existing batch shape.
    * `seeds` are stored per environment and used by counter-based RNG streams.
    """

    def __init__(
        self,
        static_data: StaticData,
        maps: Mapping[str, MapInfo],
        batch_size: int,
        config: PyrefConfig | None = None,
    ) -> None:
        if batch_size < 1:
            raise ContractError(f'batch_size must be >= 1, got {batch_size}')
        for key, info in maps.items():
            info.validate()
            if key != info.map_id:
                raise ContractError(
                    f'maps key {key!r} does not match MapInfo.map_id {info.map_id!r}'
                )
        self._maps = dict(maps)
        self._batch_size = batch_size
        self._config = config or PyrefConfig()
        self._static = (
            with_stalker_shield_regen(static_data)
            if self._config.stalker_shield_regen
            else static_data
        )
        self._stages = pipeline.resolve(self._config)
        self._capabilities = capabilities_for(self._config)
        # Mapping from attribute name to bitmask index, fixed by static_data for this instance.
        self.attributes = collect_attributes(static_data)
        self._sim: Sim | None = None

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def static_data(self) -> StaticData:
        return self._static

    @property
    def config(self) -> PyrefConfig:
        return self._config

    def reset(
        self,
        scenarios: Sequence[Scenario],
        *,
        env_ids: Sequence[int] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> WorldState:
        """Reset environments listed in `env_ids` (default all); `scenarios` and `seeds` align
        1-to-1 with `env_ids`.
        """
        ids = list(range(self._batch_size)) if env_ids is None else [int(i) for i in env_ids]
        if len(set(ids)) != len(ids) or not all(0 <= i < self._batch_size for i in ids):
            raise ContractError(
                f'env_ids={ids} has duplicates or values outside [0, {self._batch_size})'
            )
        if len(scenarios) != len(ids):
            raise ContractError(
                f'scenarios length {len(scenarios)} does not match env count {len(ids)}'
            )
        seed_list = [None] * len(ids) if seeds is None else list(seeds)
        if len(seed_list) != len(ids):
            raise ContractError(
                f'seeds length {len(seed_list)} does not match env count {len(ids)}'
            )

        # Validate all scenarios before mutating state so failures leave no half-reset batch.
        infos = [_check_scenario(s, self._maps, self._static) for s in scenarios]
        shapes = {(s.slot_count(), len(s.players)) for s in scenarios}
        if len(shapes) != 1:
            raise ContractError(
                f'All scenarios in one reset() must share (slots, players), got {shapes}'
            )
        ((n_slots, n_players),) = shapes
        # passive_from_map: config is batch-wide, so all maps in one reset must resolve alike.
        configs = {_passive_config(self._config, s) for s in scenarios}
        if len(configs) != 1:
            raise ContractError(
                'passive_from_map: scenarios in reset() resolve to different '
                'auto-acquire/retreat owners'
            )
        (config,) = configs
        sim = self._sim
        if sim is None or (
            env_ids is None and (sim.n_slots, sim.world.n_players) != (n_slots, n_players)
        ):
            sim = Sim.empty(config, self._batch_size, n_slots, n_players, len(self.attributes))
        elif (sim.n_slots, sim.world.n_players) != (n_slots, n_players):
            raise ContractError(
                f'Partial reset (slots, players) {(n_slots, n_players)} does not match '
                f'existing batch {(sim.n_slots, sim.world.n_players)}'
            )
        elif env_ids is None:
            sim.config = config
        elif sim.config != config:
            raise ContractError(
                'passive_from_map: partial reset resolves to different auto-acquire/retreat '
                'owners than existing batch'
            )
        for b, scenario, info, seed in zip(ids, scenarios, infos, seed_list, strict=True):
            owners = _charge_owners(scenario.map_id) if self._config.charge else ()
            reset_env(sim, b, scenario, info, self._static, self.attributes, seed or 0, owners)
        self._stages[pipeline.STAGES.index('visibility')](sim)
        self._sim = sim
        return self._output(sim)

    def step(self, command: Command, n_loops: int) -> WorldState:
        """Pass `command` to the command subsystem on the first loop, then advance `n_loops`."""
        sim = self._sim
        if sim is None:
            raise ContractError('Must call reset() before step()')
        if not isinstance(n_loops, int | np.integer) or n_loops < 1:
            raise ContractError(f'n_loops must be an integer >= 1, got {n_loops!r}')
        check_command(sim, command)
        for name in ('kind', 'target_slot', 'target_x', 'target_y', 'ability'):
            getattr(sim.pending, name)[:] = getattr(command, name)
        for _ in range(n_loops):
            pipeline.run_loop(sim, self._stages)
        return self._output(sim)

    def _output(self, sim: Sim) -> WorldState:
        """Return a copied `WorldState`. When `report_engaged_as_attack_unit` is enabled, idle
        or ATTACK_MOVE units auto-engaging a target are reported as ATTACK_UNIT on that target
        (copy only; internal simulation orders are untouched).
        """
        out = sim.world.copy()
        if self._config.report_engaged_as_attack_unit:
            kind = out.order_kind
            auto = (kind == CommandKind.NOOP) | (kind == CommandKind.ATTACK_MOVE)
            m = out.alive & auto & (out.engaged_target_slot != NO_SLOT)
            kind[m] = CommandKind.ATTACK_UNIT
            out.order_target_slot[m] = out.engaged_target_slot[m]
            out.order_x[m] = np.nan
            out.order_y[m] = np.nan
        return out

    def close(self) -> None:
        self._sim = None
