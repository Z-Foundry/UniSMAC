"""Unit catalog (`UnitCatalog`): type one-hot encoding, max shield, cooldown normalization,
and Medivac detection.

Reproduces upstream SMAC rules (`get_unit_type_id`, `_init_ally_unit_types`, `unit_max_shield`,
`unit_max_cooldown`) without depending on map-private native unit type IDs:

* Upstream allied units are map-defined `*_RL` types whose native IDs are assigned consecutively
  in alphabetical order of RL unit names (e.g., on MMM: Marauder_RL=1970, Marine_RL=1971,
  Medivac_RL=1972). `ALLY_ROLES[map_type]` lists roles in that same order, so
  `native_id - first_ep_min_native_id` equals `role_index - first_ep_min_role_index`. The latter
  is used here as the unit *code*.
* Upstream derives `marine_id` and similar role IDs from `min_unit_type` to look up cooldown,
  max shield, and Medivac identity; here we invert `code - min_code` against the role table for
  identical results (including offset misalignment if a role is absent in the first episode).
* Enemy units are standard types encoded per `map_type` (upstream `get_unit_type_id(ally=False)`).

SMACv2 extensions:

* `*_gen` maps (`10gen_*`): both sides use `GEN_ROLES` indices with `min_code` fixed to 0;
  `ally_code` for agent sorting follows alphabetical unit name order (`GEN_NATIVE_ORDER`,
  matching consecutive `*_RL` native IDs starting at 1970 on `32x32_flat`).
* `ranges='per_unit'`: shoot/sight ranges use SMACv2 `use_unit_ranges` tables, with shoot range
  clamped to `max(table_value, min_attack_range)`. Looking up by role avoids the upstream
  `KeyError` on `marines` maps where `marine_id` was offset by `+4`.
* `quirks.ally_type_override`: upstream SMACv2 runs the `map_type` branch even for allied `*_RL`
  units. On `colossi_stalkers_zealots`, `bane`, and `MMM` this silently yields 2 / 1 / 2
  (reproduced when enabled); on `stalkers_and_zealots` it would raise `IndexError` and is not
  reproduced.

SMAC-Hard extensions (`type_id='standard'`):

* `new_maps` uses standard units on both sides, so `get_unit_type_id` takes the `map_type` branch
  for both sides; agent sorting and `min_unit_type` use native `unit_type`, and new map types
  (`void_ray`, `hellion`, `reaper`, `qween`) have empty role tables and `unit_type_bits == 0`.
* `quirks.role_offset`: upstream still computes role IDs as `min_unit_type + role_index`, which
  misaligns non-contiguous standard IDs (e.g. on MMM, `marauder_id=48` hits Marine and
  `medivac_id=50` matches nothing), affecting cooldown normalization and Medivac detection.
  When disabled, roles are matched directly by native unit type ID.
* `shield_max='static'`: uses each unit's own `shield_max` (upstream `unit.shield_max`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

from ...contract import ContractError, Variant

__all__ = [
    'ALLY_ROLES',
    'COOLDOWN_MAX',
    'DEFAULT_COOLDOWN_MAX',
    'GEN_NATIVE_ORDER',
    'GEN_ROLES',
    'MEDIVAC',
    'SHIELD_MAX',
    'SHOOT_RANGE',
    'SIGHT_RANGE',
    'AllyTables',
    'EnemyTables',
    'UnitCatalog',
    'ally_tables_by_env',
]

COLOSSUS, BANELING, MARINE, MARAUDER, MEDIVAC = 4, 9, 48, 51, 54
ZEALOT, STALKER, ZERGLING, HYDRALISK = 73, 74, 105, 107

# map_type -> ally roles in upstream _init_ally_unit_types order (= ascending native ID order).
ALLY_ROLES: dict[str, tuple[int, ...]] = {
    'marines': (MARINE,),
    'stalkers_and_zealots': (STALKER, ZEALOT),
    'colossi_stalkers_zealots': (COLOSSUS, STALKER, ZEALOT),
    'MMM': (MARAUDER, MARINE, MEDIVAC),
    'zealots': (ZEALOT,),
    'hydralisks': (HYDRALISK,),
    'stalkers': (STALKER,),
    'colossus': (COLOSSUS,),
    'bane': (BANELING, ZERGLING),
    # SMAC-Hard map types without upstream _init_ally_unit_types branches ('queen' is unreachable
    # due to the 'qween' spelling): no role IDs, cooldown max defaults to 15, no Medivacs.
    'void_ray': (),
    'hellion': (),
    'reaper': (),
    'qween': (),
}

# SMACv2 10gen maps: type codes (upstream get_unit_type_id *_gen branches).
GEN_ROLES: dict[str, tuple[int, ...]] = {
    'terran_gen': (MARINE, MARAUDER, MEDIVAC),
    'zerg_gen': (ZERGLING, HYDRALISK, BANELING),
    'protoss_gen': (STALKER, ZEALOT, COLOSSUS),
}
ALLY_ROLES.update(GEN_ROLES)

# Native *_RL ID order on 32x32_flat (alphabetical by unit name).
GEN_NATIVE_ORDER = (
    BANELING,
    COLOSSUS,
    HYDRALISK,
    MARAUDER,
    MARINE,
    MEDIVAC,
    STALKER,
    ZEALOT,
    ZERGLING,
)

# SMACv2 use_unit_ranges shoot / sight range tables.
SHOOT_RANGE: dict[int, float] = {
    STALKER: 6,
    ZEALOT: 0.1,
    COLOSSUS: 7,
    ZERGLING: 0.1,
    BANELING: 0.25,
    HYDRALISK: 5,
    MARINE: 5,
    MARAUDER: 6,
    MEDIVAC: 4,
}
SIGHT_RANGE: dict[int, float] = {
    STALKER: 10,
    ZEALOT: 9,
    COLOSSUS: 10,
    ZERGLING: 8,
    BANELING: 8,
    HYDRALISK: 9,
    MARINE: 9,
    MARAUDER: 10,
    MEDIVAC: 11,
}

# Fallback codes when upstream SMACv2 applies map_type branches to allied *_RL units.
_Q3_ALLY_CODE = {'colossi_stalkers_zealots': 2, 'bane': 1, 'MMM': 2}

# Upstream unit_max_cooldown (Medivac entry is energy max). Unlisted / misaligned types use 15.
COOLDOWN_MAX: dict[int, float] = {
    MARINE: 15,
    MARAUDER: 25,
    MEDIVAC: 200,
    STALKER: 35,
    ZEALOT: 22,
    COLOSSUS: 24,
    HYDRALISK: 10,
    ZERGLING: 11,
    BANELING: 1,
}
DEFAULT_COOLDOWN_MAX = 15.0

# Upstream unit_max_shield; unlisted units return None (upstream later divides by None).
SHIELD_MAX: dict[int, float] = {STALKER: 80, ZEALOT: 50, COLOSSUS: 150}


def _enemy_type_id(map_type: str, unit_type: int) -> int | None:
    """Upstream `get_unit_type_id(ally=False)`; returns None if `map_type` has no branch."""
    if map_type in GEN_ROLES:
        roles = GEN_ROLES[map_type]
        if unit_type not in roles:
            raise ContractError(f'Enemy unit type {unit_type} for {map_type} is not in {roles}')
        return roles.index(unit_type)
    if map_type == 'stalkers_and_zealots':
        return unit_type - ZEALOT
    if map_type == 'colossi_stalkers_zealots':
        return {COLOSSUS: 0, STALKER: 1}.get(unit_type, 2)
    if map_type == 'bane':
        return 0 if unit_type == BANELING else 1
    if map_type == 'MMM':
        return {MARAUDER: 0, MARINE: 1}.get(unit_type, 2)
    return None


@dataclass(frozen=True)
class AllyTables:
    """Per-unit ally constants expanded to `[B, A]` (invariant within an episode)."""

    type_id: np.ndarray  # int64; 0 when unit_type_bits == 0
    cooldown_max: np.ndarray  # float64
    shield_max: np.ndarray  # float64; NaN if no shield max
    is_medivac: np.ndarray  # bool; upstream unit_type == medivac_id (True only on MMM-like maps)
    shoot_range: np.ndarray  # float64; attack / heal range
    sight_range: np.ndarray  # float64; observation sight range


@dataclass(frozen=True)
class EnemyTables:
    type_id: np.ndarray  # int64 [B, E]
    shield_max: np.ndarray  # float64 [B, E]
    is_medivac: np.ndarray  # bool [B, E]; standard Medivac (upstream unit_type == 54)


@dataclass(frozen=True)
class UnitCatalog:
    map_type: str
    unit_type_bits: int
    shield_bits_ally: int
    shield_bits_enemy: int
    roles: tuple[int, ...]
    min_code: int | None = None  # Min ally code from first episode, populated by fit()
    ranges: str = 'fixed'  # CatalogConfig.ranges
    shoot_range: float = 6.0
    sight_range: float = 9.0
    min_attack_range: float = 2.0
    ally_type_override: bool = False  # quirks.ally_type_override
    standard: bool = False  # catalog.type_id == 'standard'
    role_offset: bool = False  # quirks.role_offset (relevant only when standard=True)
    shield_static: bool = False  # catalog.shield_max == 'static'

    @classmethod
    def create(
        cls,
        map_type: str,
        unit_type_bits: int,
        agent_race: str,
        enemy_race: str,
        *,
        ranges: str = 'fixed',
        shoot_range: float = 6.0,
        sight_range: float = 9.0,
        min_attack_range: float = 2.0,
        ally_type_override: bool = False,
        type_id: str = 'smac',
        shield_max: str = 'table',
        role_offset: bool = False,
    ) -> UnitCatalog:
        if map_type not in ALLY_ROLES:
            raise ContractError(f'Unknown map_type {map_type!r}')
        standard = type_id == 'standard'
        if standard and map_type in GEN_ROLES:
            raise ContractError("catalog.type_id='standard' does not support *_gen maps")
        if not standard and not ALLY_ROLES[map_type]:
            raise ContractError(
                f"map_type={map_type!r} is only valid with catalog.type_id='standard'"
            )
        if map_type in GEN_ROLES:
            if unit_type_bits < len(GEN_ROLES[map_type]):
                raise ContractError(
                    f'{map_type} requires unit_type_bits >= {len(GEN_ROLES[map_type])}'
                )
        elif unit_type_bits > 0 and _enemy_type_id(map_type, 0) is None:
            # Upstream raises UnboundLocalError in get_unit_type_id for this combination.
            raise ContractError(
                f'map_type={map_type!r} has no enemy type encoding; cannot use unit_type_bits > 0'
            )
        return cls(
            map_type=map_type,
            unit_type_bits=int(unit_type_bits),
            shield_bits_ally=int(agent_race == 'protoss'),
            shield_bits_enemy=int(enemy_race == 'protoss'),
            roles=ALLY_ROLES[map_type],
            min_code=0 if map_type in GEN_ROLES else None,
            ranges=ranges,
            shoot_range=float(shoot_range),
            sight_range=float(sight_range),
            min_attack_range=float(min_attack_range),
            ally_type_override=ally_type_override,
            standard=standard,
            role_offset=bool(role_offset),
            shield_static=shield_max == 'static',
        )

    @property
    def generated(self) -> bool:
        """Whether this is a SMACv2 `10gen` map (both sides encoded via `GEN_ROLES`)."""
        return self.map_type in GEN_ROLES

    @property
    def heal_possible(self) -> bool:
        """Whether heal actions exist on this map type (`MMM` and SMACv2 `terran_gen`)."""
        return self.map_type in ('MMM', 'terran_gen')

    def ally_code(self, unit_type: int) -> int:
        """Return sort code for an ally unit (role index, or `GEN_NATIVE_ORDER` for `*_gen` maps;
        native `unit_type` in `standard` mode). Raises `ContractError` if missing from roles."""
        if self.standard:
            return int(unit_type)
        try:
            if self.generated:
                self.roles.index(int(unit_type))
                return GEN_NATIVE_ORDER.index(int(unit_type))
            return self.roles.index(int(unit_type))
        except ValueError:
            raise ContractError(
                f'Ally roles {self.roles} for map_type={self.map_type!r} '
                f'do not contain unit type {unit_type}'
            ) from None

    def fit(self, ally_types: np.ndarray) -> UnitCatalog:
        """Set `min_code` from initial episode ally units (upstream `_init_ally_unit_types`)."""
        codes = [self.ally_code(int(t)) for t in np.asarray(ally_types).ravel()]
        if not codes:
            raise ContractError('No ally units present in initial episode')
        return replace(self, min_code=min(codes))

    def _require_fit(self) -> int:
        if self.min_code is None:
            raise RuntimeError('UnitCatalog is not fitted yet (call reset() first)')
        return self.min_code

    def _ally_type_id(self, unit_type: int, variant: int, off: int) -> int:
        if self.unit_type_bits == 0:
            return 0
        if not self.ally_type_override or self.generated:
            return off
        if variant != Variant.RL:
            # Standard units share enemy encoding; keep offset if map_type has no branch.
            code = _enemy_type_id(self.map_type, unit_type)
            return off if code is None else code
        return _Q3_ALLY_CODE.get(self.map_type, off)

    def _standard_role(self, unit_type: int, base: int) -> int | None:
        """Role in `standard` mode: `roles[native_id - min_unit_type]` when `role_offset` is on,
        otherwise direct native ID lookup."""
        if self.role_offset:
            off = unit_type - base
            return self.roles[off] if 0 <= off < len(self.roles) else None
        return unit_type if unit_type in self.roles else None

    def ally_tables(
        self,
        unit_type: np.ndarray,
        variant: np.ndarray | None = None,
        shield_max: np.ndarray | None = None,
    ) -> AllyTables:
        """Build `AllyTables`; `shield_max` is per-unit max shield (required if `shield_static`)."""
        base = self._require_fit()
        if self.standard:
            return self._standard_ally_tables(unit_type, base, shield_max)
        flat = np.asarray(unit_type).ravel()
        var = (
            np.full(flat.shape, int(Variant.STD))
            if variant is None
            else np.asarray(variant).ravel()
        )
        tid, cd, sh, med, shoot, sight = [], [], [], [], [], []
        for t, v in zip(flat, var, strict=True):
            if self.generated:
                self.ally_code(int(t))  # Validate role membership
                off = self.roles.index(int(t))
            else:
                off = self.ally_code(int(t)) - base
            role = self.roles[off] if 0 <= off < len(self.roles) else None
            tid.append(self._ally_type_id(int(t), int(v), off))
            cd.append(
                DEFAULT_COOLDOWN_MAX
                if role is None
                else COOLDOWN_MAX.get(role, DEFAULT_COOLDOWN_MAX)
            )
            sh.append(SHIELD_MAX.get(role, np.nan) if role is not None else np.nan)
            med.append(role == MEDIVAC)
            if self.ranges == 'per_unit':
                if role is None:
                    raise ContractError(
                        f'use_unit_ranges: unit type {int(t)} has no matching role '
                        '(upstream KeyError)'
                    )
                shoot.append(max(SHOOT_RANGE[role], self.min_attack_range))
                sight.append(SIGHT_RANGE[role])
            else:
                shoot.append(self.shoot_range)
                sight.append(self.sight_range)
        shape = np.shape(unit_type)
        out = AllyTables(
            type_id=np.array(tid, np.int64).reshape(shape),
            cooldown_max=np.array(cd, np.float64).reshape(shape),
            shield_max=np.array(sh, np.float64).reshape(shape),
            is_medivac=np.array(med, np.bool_).reshape(shape),
            shoot_range=np.array(shoot, np.float64).reshape(shape),
            sight_range=np.array(sight, np.float64).reshape(shape),
        )
        self._check(out.type_id, out.shield_max, self.shield_bits_ally, 'ally')
        return out

    def _standard_ally_tables(
        self, unit_type: np.ndarray, base: int, shield_max: np.ndarray | None
    ) -> AllyTables:
        flat = np.asarray(unit_type).ravel()
        shape = np.shape(unit_type)
        roles = [self._standard_role(int(t), base) for t in flat]
        if self.ranges == 'per_unit':
            raise ContractError("catalog.type_id='standard' does not support ranges='per_unit'")
        tid = [
            (_enemy_type_id(self.map_type, int(t)) or 0) if self.unit_type_bits > 0 else 0
            for t in flat
        ]
        out = AllyTables(
            type_id=np.array(tid, np.int64).reshape(shape),
            cooldown_max=np.array(
                [
                    DEFAULT_COOLDOWN_MAX if r is None else COOLDOWN_MAX.get(r, DEFAULT_COOLDOWN_MAX)
                    for r in roles
                ],
                np.float64,
            ).reshape(shape),
            shield_max=self._shield_max(flat, shield_max, shape),
            is_medivac=np.array([r == MEDIVAC for r in roles], np.bool_).reshape(shape),
            shoot_range=np.full(shape, self.shoot_range),
            sight_range=np.full(shape, self.sight_range),
        )
        self._check(out.type_id, out.shield_max, self.shield_bits_ally, 'ally')
        return out

    def _shield_max(
        self, flat: np.ndarray, shield_max: np.ndarray | None, shape: tuple[int, ...]
    ) -> np.ndarray:
        if self.shield_static:
            if shield_max is None:
                raise ValueError("catalog.shield_max='static' requires per-unit shield_max")
            return np.asarray(shield_max, np.float64).reshape(shape).copy()
        return np.array([SHIELD_MAX.get(int(t), np.nan) for t in flat], np.float64).reshape(shape)

    def enemy_tables(
        self, unit_type: np.ndarray, variant: np.ndarray, shield_max: np.ndarray | None = None
    ) -> EnemyTables:
        flat = np.asarray(unit_type).ravel()
        tid = [
            (_enemy_type_id(self.map_type, int(t)) or 0) if self.unit_type_bits > 0 else 0
            for t in flat
        ]
        shape = np.shape(unit_type)
        out = EnemyTables(
            type_id=np.array(tid, np.int64).reshape(shape),
            shield_max=self._shield_max(flat, shield_max, shape),
            is_medivac=(np.asarray(unit_type) == MEDIVAC) & (np.asarray(variant) == Variant.STD),
        )
        self._check(out.type_id, out.shield_max, self.shield_bits_enemy, 'enemy')
        return out

    def _check(self, type_id: np.ndarray, shield_max: np.ndarray, shield_bits: int, side: str):
        if self.unit_type_bits > 0 and np.any((type_id < 0) | (type_id >= self.unit_type_bits)):
            raise ContractError(
                f'{side} type code {np.unique(type_id)} out of range for unit_type_bits'
            )
        if shield_bits and np.any(np.isnan(shield_max)):
            raise ContractError(
                f'{side} unit missing from shield_max table (upstream divides by None)'
            )


def ally_tables_by_env(
    catalogs: Sequence[UnitCatalog],
    unit_type: np.ndarray,
    variant: np.ndarray,
    shield_max: np.ndarray,
) -> AllyTables:
    """Build `[B, A]` `AllyTables` using `catalogs[b]` per env (per-env seeded envs each have
    their own `min_code`). Reduces to a single `ally_tables` call when all catalogs are identical.

    Rust counterpart: zgame-smac `env::tables::unit_tables` (evaluates `ally_tables` once per
    unique `(catalog, unit_type, variant)` and scatters by slot), used for `accel='services'`.
    """
    first = catalogs[0]
    if all(c is first for c in catalogs):
        return first.ally_tables(unit_type, variant, shield_max)
    rows = [
        c.ally_tables(unit_type[b : b + 1], variant[b : b + 1], shield_max[b : b + 1])
        for b, c in enumerate(catalogs)
    ]
    return AllyTables(
        **{f.name: np.concatenate([getattr(r, f.name) for r in rows]) for f in fields(AllyTables)}
    )
