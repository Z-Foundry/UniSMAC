"""Static asset loading tests (no StarCraft II installation required)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from unismac.assets import (
    height_to_z,
    load_charge_owners,
    load_map_info,
    load_passive_owners,
    load_static_data,
    map_asset_meta,
    map_asset_path,
    static_data_from_json,
    static_data_to_json,
)
from unismac.contract import ContractError, Variant

MAPS = ('SMAC_Maps/32x32_flat', 'SMAC_Maps/3m', 'new_maps/3m')
STANDARD = (48, 51, 54, 74, 73, 4, 105, 107, 9)


def test_static_data_marine() -> None:
    data = load_static_data()
    assert data.source.startswith('sc2 4.10.0')
    m = data.get(48)
    assert (m.name, m.race, m.health_max, m.armor, m.radius) == ('Marine', 'terran', 45, 0, 0.375)
    assert m.attributes == ('light', 'biological')
    assert (m.movement_speed, m.sight_range, m.is_flying) == (2.25, 9, False)
    (w,) = m.weapons
    assert (w.target, w.damage, w.attacks, w.range) == ('any', 6, 1, 5)
    assert w.cooldown == pytest.approx(0.861, abs=1e-3)


def test_static_data_covers_standard_units_and_rl_variants() -> None:
    data = load_static_data()
    for unit_type in STANDARD:
        assert data.get(unit_type).variant == Variant.STD
        rl = data.get(unit_type, Variant.RL)
        assert rl.name.endswith('_RL')
    assert data.get(54).is_flying and data.get(54).energy_max == 200
    assert data.get(74).shield_max == 80
    marauder = data.get(51).weapons[0]
    assert marauder.bonus == (('armored', 10.0),)


@pytest.mark.parametrize(
    ('unit_type', 'name', 'health', 'radius', 'sight'),
    [(18, 'CommandCenter', 1500, 2.75, 11), (24, 'Bunker', 400, 1.8125, 10)],
)
def test_static_data_buildings(
    unit_type: int, name: str, health: float, radius: float, sight: float
) -> None:
    """CommandCenter and Bunker have standard variant 0 only, no weapons, and zero speed."""
    data = load_static_data()
    s = data.get(unit_type)
    assert (s.name, s.race, s.attributes) == (
        name,
        'terran',
        ('armored', 'mechanical', 'structure'),
    )
    assert (s.health_max, s.shield_max, s.energy_max, s.armor) == (health, 0, 0, 1)
    assert (s.radius, s.sight_range, s.movement_speed, s.is_flying) == (radius, sight, 0, False)
    assert s.weapons == ()
    with pytest.raises(ContractError):
        data.get(unit_type, Variant.RL)


def test_range_radius_of_buildings() -> None:
    """Buildings store effective combat `range_radius` smaller than their visual radius."""
    data = load_static_data()
    want = {99: 0.63, 18: 2.51, 24: 1.51}
    for unit_type, r in want.items():
        s = data.get(unit_type)
        assert s.measured['range_radius'] == r
        assert r < s.radius
    have = {k for k, s in data.units.items() if 'range_radius' in s.measured}
    assert have == {(t, int(Variant.STD)) for t in want}


def test_static_data_json_roundtrip() -> None:
    data = load_static_data()
    again = static_data_from_json(static_data_to_json(data))
    assert again == data


@pytest.mark.parametrize('map_id', MAPS)
def test_map_info_layout(map_id: str) -> None:
    info = load_map_info(map_id)
    info.validate()
    assert (info.width, info.height, info.playable) == (32, 32, (2, 2, 30, 30))
    assert len(info.sha256) == 64
    assert map_asset_meta(map_id)['sc2_version'].startswith('4.10.0')
    # Border cells are unpathable; map center is pathable; indexed as [iy, ix].
    assert not info.pathing[0, 0] and info.pathing[16, 16]
    assert not info.pathing.flags.writeable
    # Flat center height level 191 corresponds to world z ~= 8.0.
    assert info.terrain_height[16, 16] == 191
    assert float(height_to_z(info.terrain_height[16, 16])) == pytest.approx(7.97, abs=0.01)


def test_map_info_3m_pathing_band() -> None:
    """`SMAC_Maps/3m` passable region spans y in [7, 24], x in [2, 29]."""
    info = load_map_info('SMAC_Maps/3m')
    ys, xs = np.nonzero(info.pathing)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (7, 24, 2, 29)


def test_map_asset_path_rejects_escape() -> None:
    with pytest.raises(ValueError):
        map_asset_path('../etc/passwd')
    assert map_asset_path('SMAC_Maps/3m').name == 'SMAC_Maps__3m.npz'


def _sc2_maps_dir() -> Path | None:
    cand = os.environ.get('SC2PATH')
    if cand and (Path(cand) / 'Maps').is_dir():
        return Path(cand) / 'Maps'
    return None


@pytest.mark.parametrize('map_id', MAPS)
def test_map_hash_matches_local_file(map_id: str) -> None:
    maps = _sc2_maps_dir()
    if maps is None or not (maps / f'{map_id}.SC2Map').is_file():
        pytest.skip('Local SC2 map files not available')
    digest = hashlib.sha256((maps / f'{map_id}.SC2Map').read_bytes()).hexdigest()
    assert load_map_info(map_id).sha256 == digest


def test_measured_parameters_are_merged() -> None:
    from unismac.assets import load_static_data

    data = load_static_data()
    medivac = data.get(54)
    assert medivac.measured['acceleration'] == 0.0087890625
    assert medivac.measured['deceleration'] == 0.009033203125
    marine = data.get(48)
    assert 'acceleration' not in marine.measured  # ground units accelerate instantaneously
    assert marine.measured['facing_rate'] > 1.0


def test_merge_measured_rejects_unknown_unit() -> None:
    from unismac.assets import load_static_data, merge_measured

    bad = {'format': 1, 'units': [{'unit_type': 999999, 'variant': 0, 'measured': {'a': 1.0}}]}
    with pytest.raises(ValueError):
        merge_measured(load_static_data(), bad)


def test_map_passive_owners() -> None:
    """Player 1 Passive metadata is present on SMAC-Hard maps except corridor."""
    for name in ('3m', '8m', '5m_vs_6m', 'MMM', '2s3z', '3s5z', '3rp_vs_5zl', 'mmmt', '7q_vs_2bc'):
        assert load_passive_owners(f'new_maps/{name}') == (1,)
    assert load_passive_owners('new_maps/corridor') == ()
    assert load_passive_owners('new_maps/3s_vs_3z') is None
    assert load_passive_owners('SMAC_Maps/3m') is None
    assert load_charge_owners('new_maps/corridor') == (1,)
    assert load_charge_owners('SMAC_Maps/3m') == ()
