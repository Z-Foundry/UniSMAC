"""`ZgameBackend`: Rust simulation backend implementing the UniSMAC `Backend` contract.

This module wraps the optional ``zgame._zgame`` native extension (distributed via
the ``zfoundry-zgame`` package) and validates cross-package contract and engine schema
compatibility on import.

* Configuration fields mirror `PyrefConfig`: callers may pass either a
  `PyrefConfig` dataclass instance (only non-default fields are forwarded) or a
  mapping of non-default options. Unimplemented non-default options raise
  `CapabilityError`.
* Effective capabilities track configuration: ``pathing='navmesh'`` enables
  ``pathing='grid'``; ``collision='hard'`` or ``'native'`` enables
  ``collision='hard'``; ``abilities='heal'`` enables ``abilities=True``
  (Medivac Heal 386); ``visibility='sc2'`` enables ``fog=True`` and
  ``terrain_height=True``; ``projectile_3d=True`` enables
  ``terrain_height=True``.
* Map passability grids (`MapInfo.pathing`) and terrain height grids
  (`MapInfo.terrain_height`) are passed to the native engine at construction time.
* Native exceptions are mapped onto contract exceptions (`ContractError`,
  `CapabilityError`, and `ValueError`).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from unismac.assets import load_charge_owners, with_stalker_shield_regen
from unismac.contract import (
    CONTRACT_VERSION,
    NO_SLOT,
    AgentPlayer,
    Capabilities,
    CapabilityError,
    Command,
    CommandKind,
    ContractError,
    MapInfo,
    Scenario,
    StaticData,
    WorldState,
)

from .._common import (
    PassiveOwners as _PassiveOwners,
)
from .._common import (
    capabilities_for_options,
    collect_attributes,
    resolve_passive_owners,
    validate_agent_scenario,
)

ENGINE_SCHEMA_VERSION = 1

CAPABILITIES = Capabilities(
    name='zgame',
    max_batch=None,
    engine_ai=False,
    map_trigger=False,
    pathing='none',
    collision='none',
    terrain_height=False,
    abilities=False,
    fog=False,
    deterministic=True,
    unit_overrides=True,
    mirror_spawn=True,
)

_MASK64 = (1 << 64) - 1


def _base_owners(options: Mapping[str, Any]) -> _PassiveOwners:
    """Extract configured or default `(auto_acquire_owners, retreat_owners)`."""
    acq = options.get('auto_acquire_owners')
    ret = options.get('retreat_owners', ())
    try:
        acq_t = None if acq is None else tuple(int(p) for p in acq)
        ret_t = tuple(int(p) for p in ret)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'auto_acquire_owners / retreat_owners must be sequences of player ids: {exc}'
        ) from None
    return acq_t, ret_t


def _passive_owners(
    base_owners: _PassiveOwners,
    map_id: str,
    n_players: int,
) -> _PassiveOwners:
    """Resolve `(auto_acquire_owners, retreat_owners)` from `assets/map_abilities.json`."""
    resolved = resolve_passive_owners(map_id, n_players)
    return base_owners if resolved is None else resolved


def _options_with_owners(
    base_options: Mapping[str, Any],
    base_owners: _PassiveOwners,
    owners: _PassiveOwners,
) -> dict[str, Any]:
    """Return engine options with map-derived owner rules applied when they differ."""
    opts = dict(base_options)
    if owners != base_owners:
        opts['auto_acquire_owners'], opts['retreat_owners'] = owners
    return opts


def _module() -> Any:
    try:
        from zgame import _zgame
    except ImportError as exc:
        raise ImportError(
            "The optional 'zgame' Rust extension is not installed. "
            'Install it with `pip install zfoundry-zgame` or `pip install "unismac[zgame]"`.'
        ) from exc
    got_contract = getattr(_zgame, 'CONTRACT_VERSION', None)
    if got_contract != CONTRACT_VERSION:
        raise ImportError(
            f'Incompatible zgame contract version: expected {CONTRACT_VERSION!r}, '
            f'got {got_contract!r}. '
            'Please install compatible versions of `unismac` and `zfoundry-zgame`.'
        )
    got_schema = getattr(_zgame, 'ENGINE_SCHEMA_VERSION', None)
    if got_schema != ENGINE_SCHEMA_VERSION:
        raise ImportError(
            f'Incompatible zgame engine schema version: expected {ENGINE_SCHEMA_VERSION!r}, '
            f'got {got_schema!r}. '
            'Please install compatible versions of `unismac` and `zfoundry-zgame`.'
        )
    return _zgame


@functools.cache
def build_info() -> dict[str, Any]:
    """Return build metadata for the loaded `zgame._zgame` extension.

    The returned dictionary contains JSON-serializable keys: ``version``,
    ``module``, ``sha256``, ``commit``, ``dirty``, ``built``, and ``source``
    (``'module'`` when embedded metadata is provided by ``_zgame.build_info()``,
    otherwise ``None``).
    """
    z = _module()
    so = Path(z.__file__)
    info: dict[str, Any] = {
        'version': str(z.__version__),
        'module': so.name,
        'sha256': hashlib.sha256(so.read_bytes()).hexdigest(),
        'commit': None,
        'dirty': None,
        'built': None,
        'source': None,
    }
    builtin = getattr(z, 'build_info', None)
    data: dict[str, Any] = dict(builtin()) if callable(builtin) else {}
    if data:
        info.update(
            commit=None if data.get('commit') is None else str(data['commit']),
            dirty=data.get('dirty'),
            built=data.get('built'),
            source='module',
        )
    return info


@contextmanager
def _errors() -> Iterator[None]:
    """Translate native engine exceptions into contract exceptions."""
    z = _module()
    try:
        yield
    except z.ContractViolation as exc:
        raise ContractError(str(exc)) from None
    except z.Unsupported as exc:
        raise CapabilityError(str(exc)) from None
    except z.ConfigError as exc:
        raise ValueError(str(exc)) from None


def config_options(config: object | Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract non-default engine options from a config dataclass or mapping."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        default = type(config)()
        return {
            f.name: getattr(config, f.name)
            for f in dataclasses.fields(config)
            if getattr(config, f.name) != getattr(default, f.name)
        }
    raise TypeError(
        f'config must be a PyrefConfig-like dataclass instance or mapping, got {type(config)}'
    )


def capabilities_for(options: Mapping[str, Any]) -> Capabilities:
    """Derive effective `Capabilities` from non-default backend options."""
    return capabilities_for_options(
        CAPABILITIES,
        pathing=str(options.get('pathing', 'straight')),
        collision=str(options.get('collision', 'none')),
        abilities=str(options.get('abilities', 'none')),
        visibility=str(options.get('visibility', 'full')),
        projectile_3d=bool(options.get('projectile_3d', False)),
        grid_pathing=('navmesh',),
    )


_SPLASH_RINGS = 3
SUICIDE_RANGE_STRUCTURE = 0.17


class _WeaponDict(TypedDict):
    target: str
    damage: float
    attacks: int
    range: float
    cooldown: float
    bonus: list[tuple[int, float]]


class _SuicideDict(TypedDict):
    damage: float
    trigger_range: float
    bonus: list[tuple[int, float]]
    damage_structure: float
    range_structure: float


class _CombatRequired(TypedDict):
    attack_arc: float
    cooldown_jitter_min: float
    cooldown_jitter_max: float
    projectile_speed: float
    projectile_offset: float
    projectile_instant_edge: float
    hit_slow_factor: float
    shield_regen_rate: float
    health_regen_rate: float
    turret_rate: float
    min_range: float
    charge_speed: float
    charge_range_small: float
    charge_range_large: float
    charge_small_radius: float
    charge_bonus: float
    attack_windup: int
    projectile_delay: int
    attack_spacing: int
    hit_slow_loops: int
    shield_regen_delay: int
    follow_lock: int
    charge_max_loops: int
    charge_cooldown: int
    projectile_to_center: bool
    splash_rings: list[tuple[float, float]]
    splash_friendly: bool
    barrier_absorb: float
    barrier_duration: int
    barrier_cooldown: int


class _CombatDict(_CombatRequired, total=False):
    projectile_speed_air: float
    projectile_delay_air: int
    projectile_offset_air: float
    splash_perp: tuple[float, float]
    splash_forward: tuple[float, float]
    suicide: _SuicideDict


class _UnitRequired(_CombatRequired):
    unit_type: int
    variant: int
    radius: float
    health_max: float
    shield_max: float
    energy_max: float
    armor: float
    movement_speed: float
    sight_range: float
    is_flying: bool
    attr_mask: int
    facing_rate: float
    acceleration: float
    deceleration: float
    turn_rate: float
    heading_blend: float
    shield_regen_fallback: bool
    health_regen_fallback: bool
    biological: bool
    structure: bool
    weapons: list[_WeaponDict]


class _UnitDict(_UnitRequired, total=False):
    projectile_speed_air: float
    projectile_delay_air: int
    projectile_offset_air: float
    splash_perp: tuple[float, float]
    splash_forward: tuple[float, float]
    suicide: _SuicideDict
    range_radius: float


def _combat_dict(measured: Mapping[str, Any], attributes: tuple[str, ...]) -> _CombatDict:
    """Serialize combat, splash, suicide, and barrier parameters for the native engine."""
    m = measured
    rings = sorted(
        (m[f'splash_radial_radius_{k}'], m.get(f'splash_radial_fraction_{k}', 1.0))
        for k in range(1, _SPLASH_RINGS + 1)
        if f'splash_radial_radius_{k}' in m
    )
    out: _CombatDict = {
        'attack_arc': float(m.get('attack_arc', math.inf)),
        'cooldown_jitter_min': float(m.get('cooldown_jitter_min', 0.0)),
        'cooldown_jitter_max': float(m.get('cooldown_jitter_max', 0.0)),
        'projectile_speed': float(m.get('projectile_speed', math.nan)),
        'projectile_offset': float(m.get('projectile_offset', 0.0)),
        'projectile_instant_edge': float(m.get('projectile_instant_edge', math.nan)),
        'hit_slow_factor': float(m.get('hit_slow_factor', 1.0)),
        'shield_regen_rate': float(m.get('shield_regen_rate', 0.0)),
        'health_regen_rate': float(m.get('health_regen_rate', 0.0)),
        'turret_rate': float(m.get('turret_rate', math.nan)),
        'min_range': float(m.get('min_range', 0.0)),
        'charge_speed': float(m.get('charge_speed', math.nan)),
        'charge_range_small': float(m.get('charge_range_small', 0.0)),
        'charge_range_large': float(m.get('charge_range_large', 0.0)),
        'charge_small_radius': float(m.get('charge_small_radius', 0.0)),
        'charge_bonus': float(m.get('charge_bonus', 0.0)),
        'attack_windup': int(m.get('attack_windup', 0)),
        'projectile_delay': int(m.get('projectile_delay', 0)),
        'attack_spacing': int(m.get('attack_spacing', 0)),
        'hit_slow_loops': int(m.get('hit_slow_loops', 0)),
        'shield_regen_delay': int(m.get('shield_regen_delay', 0)),
        'follow_lock': int(m.get('follow_lock', -1)),
        'charge_max_loops': int(m.get('charge_max_loops', 0)),
        'charge_cooldown': int(m.get('charge_cooldown', 0)),
        'projectile_to_center': bool(m.get('projectile_to_center', 0)),
        'splash_rings': [(float(r), float(f)) for r, f in rings],
        'splash_friendly': bool(m.get('splash_friendly', 0)),
        'barrier_absorb': float(m.get('barrier_absorb', 0.0)),
        'barrier_duration': int(m.get('barrier_duration', 0)),
        'barrier_cooldown': int(m.get('barrier_cooldown', 0)),
    }
    if 'projectile_speed_air' in m:
        out['projectile_speed_air'] = float(m['projectile_speed_air'])
    if 'projectile_delay_air' in m:
        out['projectile_delay_air'] = int(m['projectile_delay_air'])
    if 'projectile_offset_air' in m:
        out['projectile_offset_air'] = float(m['projectile_offset_air'])
    if 'splash_perp_half_length' in m:
        out['splash_perp'] = (
            float(m['splash_perp_half_length']),
            float(m['splash_perp_half_width']),
        )
    if 'splash_forward_length' in m:
        out['splash_forward'] = (
            float(m['splash_forward_length']),
            float(m['splash_forward_half_width']),
        )
    if 'suicide_damage' in m:
        bonus: list[tuple[int, float]] = []
        if 'light' in attributes:
            bonus = [(attributes.index('light'), float(m.get('suicide_bonus_light', 0.0)))]
        suicide: _SuicideDict = {
            'damage': float(m['suicide_damage']),
            'trigger_range': float(m.get('suicide_trigger_range', 0.0)),
            'bonus': bonus,
            'damage_structure': float(m.get('suicide_damage_structure', m['suicide_damage'])),
            'range_structure': float(
                m.get('suicide_trigger_range_structure', SUICIDE_RANGE_STRUCTURE)
            ),
        }
        out['suicide'] = suicide
    return out


def _unit_dicts(static_data: StaticData, attributes: tuple[str, ...]) -> list[_UnitDict]:
    """Serialize unit static specifications into native engine input dictionaries."""
    units: list[_UnitDict] = []
    for spec in static_data.units.values():
        mask = 0
        for a in spec.attributes:
            mask |= 1 << attributes.index(a)
        weapons: list[_WeaponDict] = [
            {
                'target': w.target,
                'damage': float(w.damage),
                'attacks': int(w.attacks),
                'range': float(w.range),
                'cooldown': float(w.cooldown),
                'bonus': [(attributes.index(name), float(extra)) for name, extra in w.bonus],
            }
            for w in spec.weapons
        ]
        combat = _combat_dict(spec.measured, attributes)
        unit_dict: _UnitDict = {
            'unit_type': int(spec.unit_type),
            'variant': int(spec.variant),
            'radius': float(spec.radius),
            'health_max': float(spec.health_max),
            'shield_max': float(spec.shield_max),
            'energy_max': float(spec.energy_max),
            'armor': float(spec.armor),
            'movement_speed': float(spec.movement_speed),
            'sight_range': float(spec.sight_range),
            'is_flying': bool(spec.is_flying),
            'attr_mask': mask,
            'facing_rate': float(spec.measured.get('facing_rate', math.inf)),
            'acceleration': float(spec.measured.get('acceleration', math.inf)),
            'deceleration': float(spec.measured.get('deceleration', math.inf)),
            'turn_rate': float(spec.measured.get('turn_rate', math.inf)),
            'heading_blend': float(spec.measured.get('heading_blend', math.nan)),
            **combat,
            'shield_regen_fallback': spec.shield_max > 0
            and 'shield_regen_rate' not in spec.measured,
            'health_regen_fallback': spec.race == 'zerg'
            and 'health_regen_rate' not in spec.measured,
            'biological': 'biological' in spec.attributes,
            'structure': 'structure' in spec.attributes,
            'weapons': weapons,
        }
        if 'range_radius' in spec.measured:
            unit_dict['range_radius'] = float(spec.measured['range_radius'])
        units.append(unit_dict)
    return units


def _check_scenario(scenario: Scenario, maps: Mapping[str, MapInfo]) -> MapInfo:
    """Validate scenario contract requirements for `ZgameBackend`."""
    return validate_agent_scenario(scenario, maps, 'zgame')


class ZgameBackend:
    """Batched deterministic Rust simulation backend backed by ``zgame._zgame``.

    * Rejects `ComputerPlayer` and ``spawn_mode='map_trigger'`` with
      `CapabilityError`.
    * All scenarios passed to a single `reset()` must share the same slot count
      and player count; partial resets must match the active batch shape.
    * ``threads`` sets the engine's private Rayon thread pool size; simulation
      results are bitwise invariant to thread count and batch size.
    * `events()` returns the columnar event stream from the most recent step.
    """

    def __init__(
        self,
        static_data: StaticData,
        maps: Mapping[str, MapInfo],
        batch_size: int,
        config: object | Mapping[str, Any] | None = None,
        *,
        threads: int = 1,
        events: bool = True,
        min_parallel: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ContractError(f'batch_size must be >= 1, got {batch_size}')
        if threads < 1:
            raise ValueError(f'threads must be >= 1, got {threads}')
        for key, info in maps.items():
            info.validate()
            if key != info.map_id:
                raise ContractError(
                    f'maps key {key!r} does not match MapInfo.map_id {info.map_id!r}'
                )
        self._maps = dict(maps)
        self._map_index = {key: i for i, key in enumerate(self._maps)}
        self._batch_size = batch_size
        self._config = config
        self._options = config_options(config)
        if self._options.pop('stalker_shield_regen', False):
            static_data = with_stalker_shield_regen(static_data)
        raw_passive = self._options.pop('passive_from_map', False)
        if not isinstance(raw_passive, bool):
            raise ValueError(f'passive_from_map must be bool, got {raw_passive!r}')
        self._passive_from_map = raw_passive
        self._static = static_data
        self._capabilities = capabilities_for(self._options)
        self.attributes = collect_attributes(static_data)
        self._threads = threads
        self._events = events
        self._min_parallel = min_parallel
        self._base_owners: _PassiveOwners = (None, ())
        self._engine_owners: _PassiveOwners = (None, ())
        engine_options: dict[str, Any] = self._options
        if self._passive_from_map:
            self._base_owners = _base_owners(self._options)
            if len(self._maps) == 1:
                only_map = next(iter(self._maps))
                self._engine_owners = _passive_owners(self._base_owners, only_map, 2)
            else:
                self._engine_owners = self._base_owners
            engine_options = _options_with_owners(
                self._options, self._base_owners, self._engine_owners
            )
        z = _module()
        self._engine = self._build_engine(engine_options)
        self._fields: list[tuple[str, str]] = list(z.STATE_FIELDS)

    def _build_engine(self, options: dict[str, Any]) -> Any:
        z = _module()
        with _errors():
            return z.Engine(
                _unit_dicts(self._static, self.attributes),
                len(self.attributes),
                options,
                self._batch_size,
                threads=self._threads,
                events=self._events,
                min_parallel=self._min_parallel,
                maps=[np.ascontiguousarray(info.pathing, np.bool_) for info in self._maps.values()],
                heights=[
                    np.ascontiguousarray(info.terrain_height, np.float32)
                    for info in self._maps.values()
                ],
            )

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
    def config(self) -> object | Mapping[str, Any] | None:
        return self._config

    @property
    def threads(self) -> int:
        return int(self._engine.threads)

    @property
    def impls(self) -> dict[str, str]:
        """Active subsystem implementation names in the native engine."""
        return dict(self._engine.impls)

    def reset(
        self,
        scenarios: Sequence[Scenario],
        *,
        env_ids: Sequence[int] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> WorldState:
        """Reset all environments (or `env_ids`); `scenarios` and `seeds` align 1-to-1."""
        ids = list(range(self._batch_size)) if env_ids is None else [int(i) for i in env_ids]
        if len(set(ids)) != len(ids) or not all(0 <= i < self._batch_size for i in ids):
            raise ContractError(
                f'env_ids={ids} has duplicates or values outside [0, {self._batch_size})'
            )
        if len(scenarios) != len(ids):
            raise ContractError(
                f'len(scenarios)={len(scenarios)} does not match reset count {len(ids)}'
            )
        seed_list = [None] * len(ids) if seeds is None else list(seeds)
        if len(seed_list) != len(ids):
            raise ContractError(
                f'len(seeds)={len(seed_list)} does not match reset count {len(ids)}'
            )
        infos = [_check_scenario(s, self._maps) for s in scenarios]
        shapes = {(s.slot_count(), len(s.players)) for s in scenarios}
        if len(shapes) != 1:
            raise ContractError(
                f'All scenarios in one reset must have identical (n_slots, n_players), got {shapes}'
            )
        ((n_slots, n_players),) = shapes
        rebuild_owners: _PassiveOwners | None = None
        if self._passive_from_map:
            resolved = {_passive_owners(self._base_owners, s.map_id, n_players) for s in scenarios}
            if len(resolved) != 1:
                raise ContractError(
                    'passive_from_map: maps in one reset resolved to different '
                    'acquire/retreat owners'
                )
            (owners,) = resolved
            cur_shape = self._engine.shape
            if env_ids is not None and cur_shape is not None:
                if cur_shape != (n_slots, n_players):
                    raise ContractError(
                        f'Partial reset shape {(n_slots, n_players)} does not match active batch '
                        f'shape {cur_shape}'
                    )
                if owners != self._engine_owners:
                    raise ContractError(
                        'passive_from_map: partial reset resolved to different owners '
                        'than active batch'
                    )
            elif owners != self._engine_owners:
                rebuild_owners = owners
        charge_on = bool(self._options.get('charge', False))
        envs = [
            {
                'seed': int(seed or 0) & _MASK64,
                'playable': tuple(float(v) for v in info.playable),
                'agent': [isinstance(p, AgentPlayer) for p in s.players],
                'spawns': [
                    (int(sp.owner), int(sp.unit_type), int(sp.variant), float(sp.x), float(sp.y))
                    for sp in s.spawns
                ],
                'overrides': [
                    (int(o.owner), int(o.unit_type), int(o.variant), o.health_max, o.shield_max)
                    for o in s.unit_overrides
                ],
                'map': self._map_index[s.map_id],
                'mirror': s.mirror,
                'charge_owners': list(load_charge_owners(s.map_id)) if charge_on else [],
            }
            for s, info, seed in zip(scenarios, infos, seed_list, strict=True)
        ]
        if rebuild_owners is not None:
            new_engine = self._build_engine(
                _options_with_owners(self._options, self._base_owners, rebuild_owners)
            )
            try:
                with _errors():
                    new_engine.reset(n_slots, n_players, None if env_ids is None else ids, envs)
            except Exception:
                new_engine.close()
                raise
            old_engine = self._engine
            self._engine = new_engine
            self._engine_owners = rebuild_owners
            old_engine.close()
        else:
            with _errors():
                self._engine.reset(n_slots, n_players, None if env_ids is None else ids, envs)
        return self._output()

    def step(self, command: Command, n_loops: int) -> WorldState:
        """Apply `command` on the first loop and advance `n_loops` simulation loops."""
        self._advance(command, n_loops)
        return self._output()

    def step_rows(
        self, command: Command, n_loops: int, world: WorldState, rows: np.ndarray
    ) -> None:
        """Advance simulation and write updated rows directly into `world` (in-place fast path)."""
        idx = np.arange(self._batch_size)[rows].tolist()
        self._advance(command, n_loops)
        out = {name: getattr(world, name) for name, _ in self._fields}
        report = bool(self._options.get('report_engaged_as_attack_unit', False))
        with _errors():
            self._engine.write_rows(out, idx, report)

    def _advance(self, command: Command, n_loops: int) -> None:
        shape = self._engine.shape
        if shape is None:
            raise ContractError('Must call reset() before step()')
        if not isinstance(n_loops, int | np.integer) or n_loops < 1:
            raise ContractError(f'n_loops must be an integer >= 1, got {n_loops!r}')
        command.validate(self._batch_size, shape[0])
        c = np.ascontiguousarray
        with _errors():
            self._engine.step(
                c(command.kind),
                c(command.target_slot),
                c(command.target_x),
                c(command.target_y),
                c(command.ability),
                int(n_loops),
            )

    def _output(self) -> WorldState:
        """Materialize a fresh full-batch `WorldState` snapshot."""
        n_slots, n_players = self._engine.shape
        b = self._batch_size
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in self._fields:
            if name == 'game_loop':
                shape: tuple[int, ...] = (b,)
            elif name == 'visible':
                shape = (b, n_slots, n_players)
            else:
                shape = (b, n_slots)
            arrays[name] = np.empty(shape, np.dtype(dtype))
        with _errors():
            self._engine.write_state(arrays)
        if self._options.get('report_engaged_as_attack_unit', False):
            kind = arrays['order_kind']
            engaged = arrays['engaged_target_slot']
            auto = (kind == CommandKind.NOOP) | (kind == CommandKind.ATTACK_MOVE)
            m = arrays['alive'] & auto & (engaged != NO_SLOT)
            kind[m] = CommandKind.ATTACK_UNIT
            arrays['order_target_slot'][m] = engaged[m]
            arrays['order_x'][m] = np.nan
            arrays['order_y'][m] = np.nan
        return WorldState(**arrays)

    def events(self) -> dict[str, np.ndarray]:
        """Return columnar events emitted during the last step or reset."""
        return self._engine.events()

    def close(self) -> None:
        self._engine.close()


__all__ = [
    'CAPABILITIES',
    'ENGINE_SCHEMA_VERSION',
    'ZgameBackend',
    'build_info',
    'capabilities_for',
    'collect_attributes',
    'config_options',
]
