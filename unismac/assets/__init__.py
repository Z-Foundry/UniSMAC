"""Static assets extracted from SC2: unit static data (JSON) and map grids (NPZ).

* ``static_data.json``: ``StaticData``, with SC2 version and source map metadata.
* ``measured.json``: empirically measured parameters unavailable from the SC2 API
  (acceleration, turn rate, etc.), merged into ``UnitTypeSpec.measured`` on load; stored
  separately from ``static_data.json`` so re-exporting static data does not overwrite it.
* ``unit_overrides.json``: per-map ``Scenario.unit_overrides`` (map upgrades, etc., verified
  empirically).
* ``map_abilities.json``: per-map, per-player abilities and behaviors (Zealot Charge, Player 1
  Passive, verified empirically).
* ``maps/<map_id with / replaced by __>.npz``: ``MapInfo``, with SC2 version and map SHA-256.

Extraction (requires SC2)::

    python -m unismac.backends.sc2.extract all

Loading does not require SC2 and does not import PySC2. Grid convention (see ``extract`` module
docstring): ``grid[iy, ix]`` covers world coordinates ``[ix, ix+1) x [iy, iy+1)``, where row
index is y with no flip; ``terrain_height`` stores raw SC2 byte values (0..255), with world
height ``z = -16 + 32 * v / 255`` (see ``height_to_z``).
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import MapInfo, StaticData, UnitOverride, UnitTypeSpec, Variant, WeaponSpec

ASSETS_DIR = Path(__file__).resolve().parent
STATIC_DATA_FILE = ASSETS_DIR / 'static_data.json'
MEASURED_FILE = ASSETS_DIR / 'measured.json'
UNIT_OVERRIDES_FILE = ASSETS_DIR / 'unit_overrides.json'
MAP_ABILITIES_FILE = ASSETS_DIR / 'map_abilities.json'
MAPS_DIR = ASSETS_DIR / 'maps'
FORMAT_VERSION = 1

__all__ = [
    'ASSETS_DIR',
    'FORMAT_VERSION',
    'height_to_z',
    'load_charge_owners',
    'load_passive_owners',
    'load_map_info',
    'load_static_data',
    'load_unit_overrides',
    'map_asset_path',
    'map_asset_meta',
    'merge_measured',
    'static_data_from_json',
    'static_data_to_json',
    'with_stalker_shield_regen',
]


def height_to_z(raw: np.ndarray | float) -> np.ndarray | float:
    """Convert raw ``MapInfo.terrain_height`` to world height.

    Matches ground-unit ``pos.z`` within ~0.03.
    """
    return -16.0 + 32.0 * np.asarray(raw, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- Static data


def static_data_to_json(data: StaticData, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    units = []
    for (unit_type, variant), spec in sorted(data.units.items()):
        units.append(
            {
                'unit_type': unit_type,
                'variant': variant,
                'name': spec.name,
                'race': spec.race,
                'attributes': list(spec.attributes),
                'health_max': spec.health_max,
                'shield_max': spec.shield_max,
                'energy_max': spec.energy_max,
                'armor': spec.armor,
                'radius': spec.radius,
                'is_flying': spec.is_flying,
                'movement_speed': spec.movement_speed,
                'sight_range': spec.sight_range,
                'weapons': [
                    {
                        'target': w.target,
                        'damage': w.damage,
                        'attacks': w.attacks,
                        'range': w.range,
                        'cooldown': w.cooldown,
                        'bonus': [list(b) for b in w.bonus],
                    }
                    for w in spec.weapons
                ],
                'measured': dict(spec.measured),
            }
        )
    out: dict[str, Any] = {'format': FORMAT_VERSION, 'source': data.source}
    out.update(extra or {})
    out['units'] = units
    return out


def static_data_from_json(obj: dict[str, Any]) -> StaticData:
    if obj.get('format') != FORMAT_VERSION:
        raise ValueError(
            f'Unsupported static data format version {obj.get("format")!r} '
            f'(expected {FORMAT_VERSION})'
        )
    units: dict[tuple[int, int], UnitTypeSpec] = {}
    for u in obj['units']:
        spec = UnitTypeSpec(
            unit_type=int(u['unit_type']),
            variant=int(u['variant']),
            name=str(u['name']),
            race=str(u['race']),
            attributes=tuple(u['attributes']),
            health_max=float(u['health_max']),
            shield_max=float(u['shield_max']),
            energy_max=float(u['energy_max']),
            armor=float(u['armor']),
            radius=float(u['radius']),
            is_flying=bool(u['is_flying']),
            movement_speed=float(u['movement_speed']),
            sight_range=float(u['sight_range']),
            weapons=tuple(
                WeaponSpec(
                    target=w['target'],
                    damage=float(w['damage']),
                    attacks=int(w['attacks']),
                    range=float(w['range']),
                    cooldown=float(w['cooldown']),
                    bonus=tuple((str(a), float(b)) for a, b in w['bonus']),
                )
                for w in u['weapons']
            ),
            measured={str(k): float(v) for k, v in u.get('measured', {}).items()},
        )
        units[(spec.unit_type, spec.variant)] = spec
    return StaticData(units=units, source=str(obj['source']))


@functools.cache
def load_static_data() -> StaticData:
    """Load ``static_data.json`` merged with ``measured.json`` (cached; treat as immutable)."""
    if not STATIC_DATA_FILE.is_file():
        raise FileNotFoundError(
            f'Missing {STATIC_DATA_FILE}; run python -m unismac.backends.sc2.extract static first'
        )
    data = static_data_from_json(json.loads(STATIC_DATA_FILE.read_text(encoding='utf-8')))
    if not MEASURED_FILE.is_file():
        return data
    return merge_measured(data, json.loads(MEASURED_FILE.read_text(encoding='utf-8')))


def merge_measured(data: StaticData, obj: dict[str, Any]) -> StaticData:
    """Merge entries from ``measured.json`` into each unit's ``measured`` dict.

    Measured values override existing keys with the same name.
    """
    if obj.get('format') != FORMAT_VERSION:
        raise ValueError(f'Unsupported measured parameters format version {obj.get("format")!r}')
    units = dict(data.units)
    for entry in obj['units']:
        key = (int(entry['unit_type']), int(entry['variant']))
        if key not in units:
            raise ValueError(f'measured.json unit {key} not in static data')
        spec = units[key]
        extra = {str(k): float(v) for k, v in entry['measured'].items()}
        units[key] = dataclasses.replace(spec, measured={**spec.measured, **extra})
    return StaticData(units=units, source=data.source)


def with_stalker_shield_regen(data: StaticData) -> StaticData:
    """Add catalog shield-regen defaults for standard Stalker and Stalker_RL.

    Does not overwrite explicit values or mutate shared assets.
    """
    units = dict(data.units)
    for key in ((74, Variant.STD), (74, Variant.RL)):
        if key in units:
            spec = units[key]
            measured = {'shield_regen_delay': 160, 'shield_regen_rate': 0.125, **spec.measured}
            units[key] = dataclasses.replace(spec, measured=measured)
    return dataclasses.replace(data, units=units)


_OVERRIDE_FIELDS = ('health_max', 'shield_max')


@functools.cache
def load_unit_overrides(map_id: str) -> tuple[UnitOverride, ...]:
    """Return unit overrides for ``map_id`` from ``unit_overrides.json`` (empty if none; cached).

    Validates that each ``(unit_type, variant)`` exists in static data and only overrides
    ``health_max`` / ``shield_max``.
    """
    if not UNIT_OVERRIDES_FILE.is_file():
        return ()
    obj = json.loads(UNIT_OVERRIDES_FILE.read_text(encoding='utf-8'))
    if obj.get('format') != FORMAT_VERSION:
        raise ValueError(f'Unsupported unit overrides format version {obj.get("format")!r}')
    static = load_static_data()
    out = []
    for i, e in enumerate(obj['maps'].get(map_id, ())):
        key = (int(e['unit_type']), int(e['variant']))
        if key not in static.units:
            raise ValueError(f'unit_overrides.json {map_id}[{i}]: unit {key} not in static data')
        unknown = set(e) - {'owner', 'unit_type', 'variant', 'name', 'reason', *_OVERRIDE_FIELDS}
        if unknown:
            raise ValueError(f'unit_overrides.json {map_id}[{i}]: unknown fields {sorted(unknown)}')
        values = {f: float(e[f]) for f in _OVERRIDE_FIELDS if e.get(f) is not None}
        out.append(UnitOverride(int(e['owner']), key[0], key[1], **values))
    return tuple(out)


@functools.cache
def _map_abilities() -> dict[str, Any]:
    if not MAP_ABILITIES_FILE.is_file():
        return {}
    obj = json.loads(MAP_ABILITIES_FILE.read_text(encoding='utf-8'))
    if obj.get('format') != FORMAT_VERSION:
        raise ValueError(f'Unsupported map abilities format version {obj.get("format")!r}')
    return obj['maps']


@functools.cache
def load_charge_owners(map_id: str) -> tuple[int, ...]:
    """Return players with Zealot Charge on ``map_id`` (from ``map_abilities.json``; cached)."""
    return _owners(map_id, 'charge') or ()


@functools.cache
def load_passive_owners(map_id: str) -> tuple[int, ...] | None:
    """Return players whose units have Passive behavior on ``map_id``.

    Passive units do not auto-acquire while idle and flee rather than retaliating when hit.
    Returns ``None`` if ``map_abilities.json`` has no ``passive`` key (unmeasured), or an empty
    tuple if measured as none.
    """
    return _owners(map_id, 'passive')


_MAP_ABILITY_KEYS = frozenset({'charge', 'passive'})


def _owners(map_id: str, key: str) -> tuple[int, ...] | None:
    entry = _map_abilities().get(map_id, {})
    unknown = set(entry) - _MAP_ABILITY_KEYS
    if unknown:
        raise ValueError(f'map_abilities.json {map_id}: unknown abilities {sorted(unknown)}')
    if key not in entry:
        return None
    owners = tuple(sorted(int(o) for o in entry[key]))
    if any(o < 1 for o in owners):
        raise ValueError(f'map_abilities.json {map_id}: player IDs must be >= 1, got {owners}')
    return owners


# ---------------------------------------------------------------- Maps


def map_asset_path(map_id: str) -> Path:
    if not map_id or map_id.startswith('/') or '..' in map_id.split('/'):
        raise ValueError(f'Invalid map_id {map_id!r}')
    return MAPS_DIR / (map_id.replace('/', '__') + '.npz')


def map_asset_meta(map_id: str) -> dict[str, str]:
    """Return map asset metadata: map_id, sha256, sc2_version, and format."""
    with np.load(map_asset_path(map_id)) as z:
        return {k: str(z[k]) for k in ('map_id', 'sha256', 'sc2_version', 'format')}


@functools.cache
def load_map_info(map_id: str) -> MapInfo:
    """Load ``MapInfo`` for ``map_id`` (e.g., ``'SMAC_Maps/3m'``).

    Result is cached and arrays are read-only.
    """
    path = map_asset_path(map_id)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing map asset {path}; '
            f'run python -m unismac.backends.sc2.extract maps {map_id} first'
        )
    with np.load(path) as z:
        if int(z['format']) != FORMAT_VERSION:
            raise ValueError(f'Unsupported format version {int(z["format"])} in {path}')
        if str(z['map_id']) != map_id:
            raise ValueError(f'{path} records map_id {str(z["map_id"])!r}, expected {map_id!r}')
        x0, y0, x1, y1 = (int(v) for v in z['playable'])
        info = MapInfo(
            map_id=map_id,
            sha256=str(z['sha256']),
            width=int(z['width']),
            height=int(z['height']),
            playable=(x0, y0, x1, y1),
            pathing=z['pathing'].astype(np.bool_),
            placement=z['placement'].astype(np.bool_),
            terrain_height=z['terrain_height'].astype(np.float32),
        )
    info.validate()
    for name in ('pathing', 'placement', 'terrain_height'):
        getattr(info, name).setflags(write=False)
    return info
