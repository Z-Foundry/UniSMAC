"""Explicit initial spawn tables for portable SMAC and SMAC-Hard scenarios.

* `spawns.json`: classic SMAC initial spawns (`SMAC_Maps`).
* `spawns_hard.json`: SMAC-Hard initial spawns (`new_maps`), used by `smac-hard-portable`.

Each record stores initial `(owner, unit_type, variant, x, y)` positions observed
at `game_loop=0`, along with the SC2 version, map SHA-256, native unit type IDs,
and terrain comparison metadata. Loading requires no SC2 installation.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contract import Spawn

SPAWNS_FILE = Path(__file__).resolve().parent / 'spawns.json'
SPAWNS_FILES = {'smac': SPAWNS_FILE, 'smac-hard': SPAWNS_FILE.with_name('spawns_hard.json')}

__all__ = ['SPAWNS_FILE', 'SPAWNS_FILES', 'SpawnRecord', 'load_spawns', 'spawn_record']


@dataclass(frozen=True)
class SpawnRecord:
    """Recorded initial spawn layout for a scenario.

    * `spawns`: slot-ordered `Spawn` tuples; `variant` is the recorded value (`Variant.RL` for
      SMAC allies).
    * `native_unit_type`: maps `(unit_type, variant)` to the map's native unit type ID.
    * `portable_map_id`: matching `new_maps` map ID for `smac-portable` on SC2, or `None`.
    * `portable_terrain_same`: whether map dimensions, playable area, pathing, and height grids
      match `SMAC_Maps` identically (`None` if incomparable).
    """

    name: str
    map_id: str
    map_sha256: str
    sc2_version: str
    game_loop: int
    spawns: tuple[Spawn, ...]
    native_unit_type: dict[tuple[int, int], int]
    portable_map_id: str | None
    portable_terrain_same: bool | None
    portable_note: str


@functools.cache
def _raw(benchmark: str = 'smac') -> dict[str, Any]:
    try:
        path = SPAWNS_FILES[benchmark]
    except KeyError:
        raise KeyError(
            f'No spawn records for benchmark {benchmark!r}; available: {", ".join(SPAWNS_FILES)}'
        ) from None
    return json.loads(path.read_text(encoding='utf-8'))


def _parse(name: str, rec: dict[str, Any]) -> SpawnRecord:
    spawns = tuple(
        Spawn(
            owner=int(u['owner']),
            unit_type=int(u['unit_type']),
            x=float(u['x']),
            y=float(u['y']),
            variant=int(u['variant']),
        )
        for u in rec['units']
    )
    natives = {}
    for key, native in rec['native_unit_type'].items():
        t, v = key.split(',')
        natives[(int(t), int(v))] = int(native)
    portable = rec.get('portable') or {}
    return SpawnRecord(
        name=name,
        map_id=rec['map_id'],
        map_sha256=rec['map_sha256'],
        sc2_version=rec['sc2_version'],
        game_loop=int(rec['game_loop']),
        spawns=spawns,
        native_unit_type=natives,
        portable_map_id=portable.get('map_id'),
        portable_terrain_same=portable.get('terrain_same'),
        portable_note=portable.get('note', ''),
    )


@functools.cache
def load_spawns(benchmark: str = 'smac') -> dict[str, SpawnRecord]:
    """Return all spawn records for `benchmark` in registration order."""
    return {name: _parse(name, rec) for name, rec in _raw(benchmark)['scenarios'].items()}


def spawn_record(name: str, benchmark: str = 'smac') -> SpawnRecord:
    """Return the spawn record for `name`, or raise `KeyError` if missing."""
    records = load_spawns(benchmark)
    try:
        return records[name]
    except KeyError:
        raise KeyError(
            f'{SPAWNS_FILES[benchmark].name} has no spawn record for scenario {name!r}'
        ) from None
