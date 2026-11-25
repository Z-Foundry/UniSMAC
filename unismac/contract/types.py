"""Data types for the backend contract.

Conventions (contract version in `CONTRACT_VERSION`; breaking changes allowed during 0.x):

* Per-unit state is stored as field-wise numpy arrays of shape `[B, U]`, where `B` is the batch
  size and `U` is the number of unit slots per episode.
* Slots are assigned at `reset` and remain fixed within an episode. Under `explicit` spawning,
  slot index equals the index in `Scenario.spawns`; under `map_trigger`, slots are ordered by
  `(owner, engine_order)`. Empty slots have `owner=0` and `alive=False`.
* Upon death, `alive=False` and `health=0`, remaining fields retain their last alive values,
  and units never revive.
* Coordinates match SC2 world coordinates (in map tiles, y-axis upward); time is measured in
  game loops.
* `unit_type` is the standard SC2 UnitTypeId; custom SMAC map units (`*_RL`) are distinguished
  by `variant` rather than map-private type IDs.
* For units not controlled by the caller (e.g., `ComputerPlayer` units), backends may lack order
  and cooldown info and fill sentinels: `weapon_cooldown=NaN`, `order_kind=NOOP`,
  `order_ability=0`, `order_target_slot=engaged_target_slot=NO_SLOT`, `order_x/y=NaN`.
* Initial unit facing is backend-defined (random in SC2); `Spawn` omits facing. With mirror
  spawning (`Scenario.mirror`), Player 2's facings mirror Player 1's via `mirror_facing`.
* Command latency from issue to effect is backend-specific (1 loop in single-process SC2, 2 loops
  in dual-process lockstep), documented per backend for cross-backend alignment.

Command semantics:

* `ATTACK_UNIT` targeting a dead or empty slot is ignored (unit keeps its current order);
  targeting self or an out-of-range slot raises `ContractError`. Backends may reject attacking an
  invisible target (SC2 returns NotSupported); this is engine behavior, not a contract error.
* NaN target coordinates for `MOVE` / `ATTACK_MOVE` raise `ContractError`. Out-of-bounds targets
  are clamped or handled by the engine; callers should not rely on specific clamping behavior.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Literal, TypeAlias

import numpy as np

CONTRACT_VERSION = '0.2'

# Sentinel value for no target / no slot.
NO_SLOT = -1

# SC2 stats (speed, cooldown, etc.) use normal-speed game seconds (1 game second = 16 loops;
# verified on SC2 4.10: Marine moves 2.25 / 16 per loop and cooldown is ~0.861 * 16).
LOOPS_PER_GAME_SECOND = 16


class ContractError(ValueError):
    """Raised when data violates the contract (shape, dtype, or value range)."""


class Variant(enum.IntEnum):
    """Variant of a standard unit type."""

    STD = 0
    RL = 1  # *_RL custom unit in SMAC maps


class CommandKind(enum.IntEnum):
    """Unit command kind. `NOOP` issues no new command and keeps the unit's current order."""

    NOOP = 0
    STOP = 1
    MOVE = 2  # target point (target_x, target_y)
    ATTACK_UNIT = 3  # target slot (target_slot)
    ATTACK_MOVE = 4  # target point (target_x, target_y)
    HOLD = 5
    ABILITY = 6  # ability id; target resolved in order: target_slot, target_xy, or untargeted


# ---------------------------------------------------------------- Players & scenarios


@dataclass(frozen=True)
class AgentPlayer:
    """Player controlled by commands from the caller."""

    race: Literal['terran', 'zerg', 'protoss', 'random'] = 'random'


@dataclass(frozen=True)
class ComputerPlayer:
    """Engine built-in AI player. Supported only by backends with `engine_ai` capability."""

    race: Literal['terran', 'zerg', 'protoss', 'random']
    difficulty: Literal[
        'very_easy',
        'easy',
        'medium',
        'medium_hard',
        'hard',
        'harder',
        'very_hard',
        'cheat_vision',
        'cheat_money',
        'cheat_insane',
    ] = 'very_hard'


Player: TypeAlias = AgentPlayer | ComputerPlayer


@dataclass(frozen=True)
class Spawn:
    """Unit to spawn; its index in `Scenario.spawns` is its slot index."""

    owner: int  # 1-based player ID
    unit_type: int
    x: float
    y: float
    variant: int = Variant.STD


@dataclass(frozen=True)
class UnitOverride:
    """Per-player unit stat override relative to `StaticData` (e.g., map upgrades).

    On reset, matching `(unit_type, variant)` units for `owner` spawn at full overridden caps.
    Applies only at spawn time, not as runtime mutations. `None` fields inherit `StaticData`.
    """

    owner: int  # 1-based player ID
    unit_type: int
    variant: int = Variant.STD
    health_max: float | None = None
    shield_max: float | None = None


def mirror_facing(theta: float, axis: Literal['x', 'y']) -> np.float32:
    """Mirror spawn facing: `'x'` (across a vertical line) -> pi - theta, `'y'` -> -theta.

    Computed in float64, wrapped into [-pi, pi), and cast to float32 (matching zgame
    `Mirror::facing`).
    """
    t = float(theta)
    m = math.pi - t if axis == 'x' else -t
    if m >= math.pi:
        m -= 2.0 * math.pi
    elif m < -math.pi:
        m += 2.0 * math.pi
    return np.float32(m)


@dataclass(frozen=True, repr=False)
class Scenario:
    """Explicit starting specification for an episode.

    `map_id` is a map path relative to `$SC2PATH/Maps` without extension (e.g.,
    `'SMAC_Maps/32x32_flat'`); bare names are avoided because identical names exist in different
    subdirectories.

    `mirror` (contract 0.2): mirror spawning across a vertical (`'x'`) or horizontal (`'y'`)
    axis. When set, `spawns` must contain `n` Player 1 units followed by `n` Player 2 units with
    matching `(unit_type, variant)` at index `j` and `n + j`, with mirrored positions supplied by
    the caller; the backend mirrors Player 2's spawn facing (plus displacement direction and
    turret facing) via `mirror_facing`. Requires backend `mirror_spawn` capability.
    """

    map_id: str
    players: tuple[Player, ...] = (AgentPlayer(), AgentPlayer())
    spawns: tuple[Spawn, ...] = ()
    spawn_mode: Literal['explicit', 'map_trigger'] = 'explicit'
    n_slots: int | None = None  # slot count U; defaults to len(spawns)
    # Map-upgrade stat overrides; when non-empty, backend must declare `unit_overrides`
    # (applying overrides or verifying them after reset and raising ContractError on mismatch).
    unit_overrides: tuple[UnitOverride, ...] = ()
    mirror: Literal['x', 'y'] | None = None

    def __repr__(self) -> str:
        """Dataclass-like repr that omits `mirror` when None (preserving contract 0.1 repr)."""
        parts = [
            f'{f.name}={getattr(self, f.name)!r}'
            for f in fields(self)
            if f.name != 'mirror' or self.mirror is not None
        ]
        return f'{type(self).__qualname__}({", ".join(parts)})'

    def slot_count(self) -> int:
        n = len(self.spawns) if self.n_slots is None else self.n_slots
        if n < len(self.spawns):
            raise ContractError(f'n_slots={n} is smaller than len(spawns)={len(self.spawns)}')
        if n <= 0:
            raise ContractError('Scenario requires at least 1 slot')
        return n

    def validate(self) -> None:
        if not self.players:
            raise ContractError('players must not be empty')
        if self.spawn_mode == 'map_trigger' and self.spawns:
            raise ContractError('map_trigger scenario spawns units via map; spawns must be empty')
        if self.spawn_mode == 'map_trigger' and self.n_slots is None:
            raise ContractError('map_trigger scenario must specify n_slots')
        for i, s in enumerate(self.spawns):
            if not 1 <= s.owner <= len(self.players):
                raise ContractError(f'spawns[{i}].owner={s.owner} out of player range')
        seen: set[tuple[int, int, int]] = set()
        for i, o in enumerate(self.unit_overrides):
            key = (o.owner, o.unit_type, o.variant)
            if not 1 <= o.owner <= len(self.players):
                raise ContractError(f'unit_overrides[{i}].owner={o.owner} out of player range')
            if key in seen:
                raise ContractError(f'duplicate unit_overrides[{i}] entry: {key}')
            seen.add(key)
            if o.health_max is None and o.shield_max is None:
                raise ContractError(f'unit_overrides[{i}] must specify at least one field')
            if o.health_max is not None and not o.health_max > 0:
                raise ContractError(f'unit_overrides[{i}].health_max must be > 0')
            if o.shield_max is not None and not o.shield_max >= 0:
                raise ContractError(f'unit_overrides[{i}].shield_max must be >= 0')
        if self.mirror is not None:
            self._validate_mirror()
        self.slot_count()

    def _validate_mirror(self) -> None:
        if self.mirror not in ('x', 'y'):
            raise ContractError(f"mirror must be None, 'x', or 'y', got {self.mirror!r}")
        if self.spawn_mode != 'explicit':
            raise ContractError("mirror is only valid for spawn_mode='explicit'")
        n, odd = divmod(len(self.spawns), 2)
        first, second = self.spawns[:n], self.spawns[n:]
        if odd or not all(
            a.owner == 1 and b.owner == 2 and (a.unit_type, a.variant) == (b.unit_type, b.variant)
            for a, b in zip(first, second, strict=True)
        ):
            raise ContractError(
                'mirror requires n Player 1 spawns followed by n Player 2 spawns with '
                'matching (unit_type, variant) pairs'
            )


# ---------------------------------------------------------------- Static data


@dataclass(frozen=True)
class WeaponSpec:
    """Weapon specification from SC2 `UnitTypeData.weapons`."""

    target: Literal['ground', 'air', 'any']
    damage: float
    attacks: int
    range: float  # edge-to-edge distance (center distance minus both radii)
    cooldown: float  # game seconds; see LOOPS_PER_GAME_SECOND
    bonus: tuple[tuple[str, float], ...] = ()  # (attribute_name, bonus_damage)


@dataclass(frozen=True)
class UnitTypeSpec:
    """Static parameters for one unit type, keyed by `(unit_type, variant)`.

    Parameters unavailable from the SC2 API (acceleration, turn rate, damage point, projectile
    speed, acquisition range, etc.) live in `measured` (populated via controlled black-box
    measurements; see `assets/measured.json`).
    """

    unit_type: int
    variant: int
    name: str
    race: str
    attributes: tuple[str, ...]
    health_max: float
    shield_max: float
    energy_max: float
    armor: float
    radius: float
    is_flying: bool
    movement_speed: float  # tiles per game second
    sight_range: float
    weapons: tuple[WeaponSpec, ...] = ()
    measured: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticData:
    """Static parameters for all unit types and their data source."""

    units: Mapping[tuple[int, int], UnitTypeSpec]
    source: str  # e.g., 'sc2 4.10'

    def get(self, unit_type: int, variant: int = Variant.STD) -> UnitTypeSpec:
        try:
            return self.units[(unit_type, variant)]
        except KeyError:
            raise ContractError(f'Unit ({unit_type}, {variant}) not in static data') from None


@dataclass(frozen=True)
class MapInfo:
    """Static map metadata extracted once from SC2 and shared across backends.

    Grids are indexed `[iy, ix]`, where `grid[iy, ix]` covers world `[ix, ix+1) x [iy, iy+1)`.
    """

    map_id: str
    sha256: str  # SHA-256 hash of the .SC2Map file
    width: int
    height: int
    playable: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    pathing: np.ndarray  # bool[H, W], True where ground units can walk
    placement: np.ndarray  # bool[H, W], True where structures can be placed
    terrain_height: np.ndarray  # float32[H, W], raw SC2 height byte values

    def validate(self) -> None:
        shape = (self.height, self.width)
        for name, dtype in (
            ('pathing', np.bool_),
            ('placement', np.bool_),
            ('terrain_height', np.float32),
        ):
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != dtype:
                raise ContractError(
                    f'MapInfo.{name} must be {np.dtype(dtype)}{shape}, got {arr.dtype}{arr.shape}'
                )


# ---------------------------------------------------------------- World state & commands

_UNIT_DTYPES: dict[str, type] = {
    'alive': np.bool_,
    'owner': np.int8,
    'unit_type': np.int32,
    'variant': np.int8,
    'x': np.float32,
    'y': np.float32,
    'facing': np.float32,  # radians
    'radius': np.float32,
    'health': np.float32,
    'health_max': np.float32,
    'shield': np.float32,
    'shield_max': np.float32,
    'energy': np.float32,
    'energy_max': np.float32,
    'weapon_cooldown': np.float32,  # loops until next attack
    'is_flying': np.bool_,
    'order_kind': np.int8,  # CommandKind; NOOP means idle
    'order_ability': np.int32,  # backend-native ability ID, 0 when idle
    'order_target_slot': np.int32,
    'order_x': np.float32,  # target point of current order, NaN if none
    'order_y': np.float32,
    'engaged_target_slot': np.int32,  # target currently engaged, NO_SLOT if none
}

_UNIT_FILL: dict[str, float] = {
    'order_target_slot': NO_SLOT,
    'engaged_target_slot': NO_SLOT,
    'order_x': np.nan,
    'order_y': np.nan,
}


@dataclass
class WorldState:
    """Complete batched world state at a given game loop."""

    game_loop: np.ndarray  # int64[B]
    alive: np.ndarray
    owner: np.ndarray
    unit_type: np.ndarray
    variant: np.ndarray
    x: np.ndarray
    y: np.ndarray
    facing: np.ndarray
    radius: np.ndarray
    health: np.ndarray
    health_max: np.ndarray
    shield: np.ndarray
    shield_max: np.ndarray
    energy: np.ndarray
    energy_max: np.ndarray
    weapon_cooldown: np.ndarray
    is_flying: np.ndarray
    order_kind: np.ndarray
    order_ability: np.ndarray
    order_target_slot: np.ndarray
    order_x: np.ndarray
    order_y: np.ndarray
    engaged_target_slot: np.ndarray
    visible: np.ndarray  # bool[B, U, P]: visible[b, u, p] is True if player p+1 sees u

    UNIT_FIELDS = tuple(_UNIT_DTYPES)

    @classmethod
    def empty(cls, batch_size: int, n_slots: int, n_players: int = 2) -> WorldState:
        shape = (batch_size, n_slots)
        units = {
            name: np.full(shape, _UNIT_FILL.get(name, 0), dtype=dtype)
            for name, dtype in _UNIT_DTYPES.items()
        }
        return cls(
            game_loop=np.zeros(batch_size, dtype=np.int64),
            visible=np.zeros((*shape, n_players), dtype=np.bool_),
            **units,
        )

    @property
    def batch_size(self) -> int:
        return int(self.game_loop.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.alive.shape[1])

    @property
    def n_players(self) -> int:
        return int(self.visible.shape[2])

    def copy(self) -> WorldState:
        return WorldState(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    def validate(self) -> None:
        if self.game_loop.ndim != 1 or self.game_loop.dtype != np.int64:
            raise ContractError(
                f'game_loop must be int64[B], got {self.game_loop.dtype}{self.game_loop.shape}'
            )
        shape = (self.batch_size, self.alive.shape[1] if self.alive.ndim == 2 else -1)
        for name, dtype in _UNIT_DTYPES.items():
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != dtype:
                raise ContractError(
                    f'{name} must be {np.dtype(dtype)}{shape}, got {arr.dtype}{arr.shape}'
                )
        if (
            self.visible.ndim != 3
            or self.visible.shape[:2] != shape
            or self.visible.dtype != np.bool_
        ):
            raise ContractError(
                f'visible must be bool[B, U, P], got {self.visible.dtype}{self.visible.shape}'
            )
        if np.any(self.alive & (self.owner == 0)):
            raise ContractError('Alive unit must have owner != 0')


@dataclass
class Command:
    """Batched unit commands for one decision step, with shape `[B, U]`."""

    kind: np.ndarray  # int8, CommandKind
    target_slot: np.ndarray  # int32, NO_SLOT if none
    target_x: np.ndarray  # float32, NaN if none
    target_y: np.ndarray
    ability: np.ndarray  # int32, backend-native ability ID, 0 when not ABILITY

    @classmethod
    def noop(cls, batch_size: int, n_slots: int) -> Command:
        shape = (batch_size, n_slots)
        return cls(
            kind=np.zeros(shape, dtype=np.int8),
            target_slot=np.full(shape, NO_SLOT, dtype=np.int32),
            target_x=np.full(shape, np.nan, dtype=np.float32),
            target_y=np.full(shape, np.nan, dtype=np.float32),
            ability=np.zeros(shape, dtype=np.int32),
        )

    def set(
        self,
        env: int,
        slot: int,
        kind: CommandKind,
        *,
        target_slot: int = NO_SLOT,
        target_xy: tuple[float, float] | None = None,
        ability: int = 0,
    ) -> Command:
        """Write one unit command in place and return self for chaining."""
        self.kind[env, slot] = kind
        self.target_slot[env, slot] = target_slot
        x, y = target_xy if target_xy is not None else (np.nan, np.nan)
        self.target_x[env, slot] = x
        self.target_y[env, slot] = y
        self.ability[env, slot] = ability
        return self

    def validate(self, batch_size: int, n_slots: int) -> None:
        shape = (batch_size, n_slots)
        for name, dtype in (
            ('kind', np.int8),
            ('target_slot', np.int32),
            ('target_x', np.float32),
            ('target_y', np.float32),
            ('ability', np.int32),
        ):
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != dtype:
                raise ContractError(
                    f'Command.{name} must be {np.dtype(dtype)}{shape}, got {arr.dtype}{arr.shape}'
                )
        if np.any((self.kind < 0) | (self.kind > max(CommandKind))):
            raise ContractError('Command.kind contains unknown values')
