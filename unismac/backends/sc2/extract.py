"""Export static assets from SC2: ``MapInfo`` (NPZ) and ``StaticData`` (JSON).

Usage::

    python -m unismac.backends.sc2.extract all
    python -m unismac.backends.sc2.extract maps SMAC_Maps/3m new_maps/3m
    python -m unismac.backends.sc2.extract static
    python -m unismac.backends.sc2.extract static --unit roach=110

**Grid orientation (verified on SC2 4.10)**: ``pathing_grid`` and ``placement_grid`` in
``game_info.start_raw`` are 1 bpp, row-packed, MSB-first (``np.unpackbits`` default), and
``terrain_height`` is 8 bpp. All three are row-major ``[H, W]`` where **row 0 is y in [0, 1)**,
requiring no flip or transpose and matching the contract convention ``grid[iy, ix]``. Verified by:

* Querying ``RequestQueryPathing`` across cells of asymmetric ``Melee/Simple64`` (88x96) and
  ``SMAC_Maps/corridor``: reachability agrees with ``pathing[iy, ix]`` at 0.97 / 0.99 vs
  0.79 / 0.43 when y-flipped (disagreements are unreachable islands and mineral lines); placement
  queries agree similarly (0.95 / 0.91 vs 0.77 / 0.42).
* Spawning a Marine on pathable cells gives mean ``pos.z`` error of 0.025 (max 1.0 on ramps)
  against ``-16 + 32 * terrain_height[iy, ix] / 255``, vs 5.2 when y-flipped.

Note that upstream SMAC reshaped 8 bpp ``terrain_height`` as ``(map_x, map_y)``, transposed, and
flipped it, which only happened to be harmless on square, vertically symmetric SMAC maps.

**Static data**: ``data_raw()`` supplies ``movement_speed``, ``armor``, ``sight_range``,
``attributes``, and weapons (``damage``, ``damage_bonus``, ``attacks``, ``range``, ``speed`` ->
``cooldown``); ``health_max``, ``shield_max``, ``energy_max``, ``radius``, and ``is_flying`` are
read from spawned units on ``32x32_flat``. Speeds and cooldowns use normal-speed game seconds:
Marine ``movement_speed=2.25`` moves 0.140625 = 2.25 / 16 per loop, and cooldown 0.861 s gives
~13.8 loops of observed ``weapon_cooldown``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...assets import (
    FORMAT_VERSION,
    MAPS_DIR,
    STATIC_DATA_FILE,
    map_asset_path,
    static_data_to_json,
)
from ...contract import (
    AgentPlayer,
    ComputerPlayer,
    ContractError,
    MapInfo,
    Scenario,
    Spawn,
    StaticData,
    UnitTypeSpec,
    Variant,
    WeaponSpec,
)
from . import _runtime
from .backend import Sc2Backend
from .session import Session

# Standard units used by SMAC / SMACv2 (SMACv2 10gen_* uses these 9 types, matching
# sc2_gen_{terran,protoss,zerg}*.yaml and starcraft2.py _init_ally_unit_types).
UNITS: dict[str, int] = {
    'marine': 48,
    'marauder': 51,
    'medivac': 54,
    'stalker': 74,
    'zealot': 73,
    'colossus': 4,
    'zergling': 105,
    'hydralisk': 107,
    'baneling': 9,
}
# Additional unit appearing in SMAC scenarios (SpineCrawler in 2s_vs_1sc is a static structure).
SMAC_EXTRA_UNITS: dict[str, int] = {
    'spinecrawler': 98,
}
# Additional large-radius ground units for collision and narrow-gap tests (unused by SMAC).
PROBE_UNITS: dict[str, int] = {
    'ultralisk': 109,
    'thor': 52,
}
ALL_UNITS: dict[str, int] = {**UNITS, **SMAC_EXTRA_UNITS, **PROBE_UNITS}
DEFAULT_MAPS = ('SMAC_Maps/32x32_flat', 'SMAC_Maps/3m', 'new_maps/3m')
STATIC_MAP = 'SMAC_Maps/32x32_flat'  # includes all 9 *_RL custom unit variants
_PLAYERS = (AgentPlayer('terran'), ComputerPlayer('terran', 'very_easy'))


def _unpack(image: Any, width: int, height: int) -> np.ndarray:
    data = np.frombuffer(image.data, dtype=np.uint8)
    if image.bits_per_pixel == 1:
        return np.unpackbits(data)[: width * height].reshape(height, width)
    if image.bits_per_pixel == 8:
        return data.reshape(height, width)
    raise ValueError(f'Unsupported bits_per_pixel={image.bits_per_pixel}')


def map_info_from_game_info(map_id: str, game_info: Any, sha256: str) -> MapInfo:
    raw = game_info.start_raw
    w, h = int(raw.map_size.x), int(raw.map_size.y)
    for name in ('pathing_grid', 'placement_grid'):
        if getattr(raw, name).bits_per_pixel != 1:
            # Upstream SMAC inverts 8 bpp pathing grids; unseen and unverified here.
            raise NotImplementedError(f'{map_id}: {name} is not 1 bpp; encoding unverified')
    p0, p1 = raw.playable_area.p0, raw.playable_area.p1
    info = MapInfo(
        map_id=map_id,
        sha256=sha256,
        width=w,
        height=h,
        playable=(int(p0.x), int(p0.y), int(p1.x), int(p1.y)),
        pathing=_unpack(raw.pathing_grid, w, h).astype(np.bool_),
        placement=_unpack(raw.placement_grid, w, h).astype(np.bool_),
        terrain_height=_unpack(raw.terrain_height, w, h).astype(np.float32),
    )
    info.validate()
    return info


def extract_map(map_id: str) -> tuple[MapInfo, str]:
    """Launch SC2 to read ``MapInfo`` for a map, returning ``(MapInfo, sc2_version)``."""
    session = Session(map_id, _PLAYERS)
    try:
        info = map_info_from_game_info(map_id, session.game_info, _runtime.map_sha256(map_id))
        return info, session.version
    finally:
        session.close()


def save_map_info(info: MapInfo, sc2_version: str, out_dir: Path = MAPS_DIR) -> Path:
    path = out_dir / map_asset_path(info.map_id).name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.int64(FORMAT_VERSION),
        map_id=np.str_(info.map_id),
        sha256=np.str_(info.sha256),
        sc2_version=np.str_(sc2_version),
        width=np.int64(info.width),
        height=np.int64(info.height),
        playable=np.array(info.playable, dtype=np.int64),
        pathing=info.pathing,
        placement=info.placement,
        terrain_height=info.terrain_height,
    )
    return path


def extract_static_data(units: dict[str, int] = ALL_UNITS) -> tuple[StaticData, dict[str, Any]]:
    """Spawn each unit (standard + *_RL variant) on ``STATIC_MAP`` and read static data."""
    from s2clientprotocol import common_pb2 as sc_common
    from s2clientprotocol import data_pb2

    backend = Sc2Backend(static_data=StaticData(units={}, source='bootstrap'))
    try:
        # Start a minimal session first to load the native type table and discover variants.
        boot = Scenario(STATIC_MAP, players=_PLAYERS, spawns=(Spawn(1, units['marine'], 4, 4),))
        backend.reset([boot])
        keys: list[tuple[int, int]] = []
        for unit_type in units.values():
            keys.append((unit_type, int(Variant.STD)))
            try:
                backend.native_type(unit_type, Variant.RL)
                keys.append((unit_type, int(Variant.RL)))
            except ContractError:
                pass
        spawns = tuple(
            Spawn(1, t, x=4.0 + 4.0 * (i % 6), y=6.0 + 6.0 * (i // 6), variant=v)
            for i, (t, v) in enumerate(keys)
        )
        state = backend.reset([Scenario(STATIC_MAP, players=_PLAYERS, spawns=spawns)])
        session = backend.session
        assert session is not None
        by_id = {u.unit_id: u for u in session.data_raw.units}
        version = session.version
        specs: dict[tuple[int, int], UnitTypeSpec] = {}
        native_ids: dict[tuple[int, int], int] = {}
        for slot, (unit_type, variant) in enumerate(keys):
            if not state.alive[0, slot]:
                raise RuntimeError(f'Failed to spawn unit {(unit_type, variant)}')
            native = backend.native_type(unit_type, variant)
            native_ids[(unit_type, variant)] = native
            d = by_id[native]
            weapons = tuple(
                WeaponSpec(
                    target=data_pb2.Weapon.TargetType.Name(w.type).lower(),  # type: ignore[arg-type]
                    damage=float(w.damage),
                    attacks=int(w.attacks),
                    range=float(w.range),
                    cooldown=float(w.speed),
                    bonus=tuple(
                        (data_pb2.Attribute.Name(b.attribute).lower(), float(b.bonus))
                        for b in w.damage_bonus
                    ),
                )
                for w in d.weapons
            )
            specs[(unit_type, variant)] = UnitTypeSpec(
                unit_type=unit_type,
                variant=variant,
                name=str(d.name),
                race=sc_common.Race.Name(d.race).lower(),
                attributes=tuple(data_pb2.Attribute.Name(a).lower() for a in d.attributes),
                health_max=float(state.health_max[0, slot]),
                shield_max=float(state.shield_max[0, slot]),
                energy_max=float(state.energy_max[0, slot]),
                armor=float(d.armor),
                radius=float(state.radius[0, slot]),
                is_flying=bool(state.is_flying[0, slot]),
                movement_speed=float(d.movement_speed),
                sight_range=float(d.sight_range),
                weapons=weapons,
            )
        data = StaticData(units=specs, source=f'sc2 {version}')
        extra = {
            'sc2_version': version,
            'source_map': STATIC_MAP,
            'source_map_sha256': _runtime.map_sha256(STATIC_MAP),
            'native_unit_type': {f'{t},{v}': n for (t, v), n in sorted(native_ids.items())},
        }
        return data, extra
    finally:
        backend.close()


def save_static_data(
    data: StaticData, extra: dict[str, Any], path: Path = STATIC_DATA_FILE
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(static_data_to_json(data, extra), ensure_ascii=False, indent=1)
    path.write_text(text + '\n', encoding='utf-8')
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m unismac.backends.sc2.extract')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_maps = sub.add_parser('maps', help='Export MapInfo (NPZ)')
    p_maps.add_argument('map_ids', nargs='*', default=list(DEFAULT_MAPS))
    p_static = sub.add_parser('static', help='Export StaticData (JSON)')
    p_static.add_argument(
        '--unit',
        action='append',
        default=[],
        metavar='NAME=ID',
        help='Extra unit to export beyond defaults (9 SMAC + large-radius units), repeatable',
    )
    sub.add_parser('all', help='Export default maps and static data')
    args = parser.parse_args(argv)

    if args.cmd in ('maps', 'all'):
        for map_id in args.map_ids if args.cmd == 'maps' else DEFAULT_MAPS:
            info, version = extract_map(map_id)
            path = save_map_info(info, version)
            print(f'{map_id}: {info.width}x{info.height} sha256={info.sha256[:12]} -> {path}')
    if args.cmd in ('static', 'all'):
        units = dict(ALL_UNITS)
        for item in getattr(args, 'unit', []):
            name, sep, value = item.partition('=')
            if not sep or not value.strip().isdigit():
                parser.error(f'--unit requires NAME=ID, got {item!r}')
            units[name.strip()] = int(value)
        data, extra = extract_static_data(units)
        path = save_static_data(data, extra)
        print(f'{len(data.units)} units ({data.source}) -> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
