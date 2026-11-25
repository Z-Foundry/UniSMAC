"""Unit parameter overrides (`Scenario.unit_overrides`): assets, env wiring, SC2 verification.

On new_maps 6m_vs_10m, mmmt, mmmt_vs_zhb, and mmmt_vs_zspi, Player 1 has ShieldWall researched,
giving Marines 55 HP (vs 45 in static data). Applies to both explicit and map_trigger spawns.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from unismac.assets import (
    UNIT_OVERRIDES_FILE,
    load_map_info,
    load_static_data,
    load_unit_overrides,
)
from unismac.contract import CapabilityError, ContractError, UnitOverride
from unismac.env import make_vec_env

UPGRADED = ('6m_vs_10m', 'mmmt', 'mmmt_vs_zhb', 'mmmt_vs_zspi')
MARINE = 48
MARINE_55 = UnitOverride(owner=1, unit_type=MARINE, health_max=55.0)


def _marines(world: Any) -> dict[int, set[tuple[float, float]]]:
    """Map owner -> set of (health, health_max) for that owner's Marines."""
    out: dict[int, set[tuple[float, float]]] = {}
    for s in range(world.alive.shape[1]):
        if world.alive[0, s] and int(world.unit_type[0, s]) == MARINE:
            hp = (float(world.health[0, s]), float(world.health_max[0, s]))
            out.setdefault(int(world.owner[0, s]), set()).add(hp)
    return out


# ---------------------------------------------------------------- Assets


def test_asset_entries():
    """Only the 4 maps' Player 1 Marines are listed; entries, map assets, and types are valid."""
    obj = json.loads(UNIT_OVERRIDES_FILE.read_text(encoding='utf-8'))
    assert sorted(obj['maps']) == sorted(f'new_maps/{m}' for m in UPGRADED)
    static = load_static_data()
    for map_id in obj['maps']:
        assert load_unit_overrides(map_id) == (MARINE_55,)
        assert static.get(MARINE).health_max == 45.0
        load_map_info(map_id)
    assert load_unit_overrides('SMAC_Maps/3m') == ()
    assert load_unit_overrides('new_maps/3m') == ()


def test_asset_rejects_bad_entries(tmp_path, monkeypatch):
    import unismac.assets as assets

    def load(entry: dict[str, Any]) -> Any:
        f = tmp_path / 'unit_overrides.json'
        f.write_text(json.dumps({'format': 1, 'maps': {'m': [entry]}}), encoding='utf-8')
        monkeypatch.setattr(assets, 'UNIT_OVERRIDES_FILE', f)
        load_unit_overrides.cache_clear()
        try:
            return load_unit_overrides('m')
        finally:
            load_unit_overrides.cache_clear()

    base = {'owner': 1, 'unit_type': MARINE, 'variant': 0, 'health_max': 55.0}
    assert load(base) == (MARINE_55,)
    with pytest.raises(ValueError, match='(?i)not in static data'):
        load({**base, 'unit_type': 99999})
    with pytest.raises(ValueError, match='(?i)unknown field'):
        load({**base, 'energy_max': 1.0})


# ---------------------------------------------------------------- Env layer


@pytest.mark.parametrize('preset', ['smac-hard', 'smac-hard-portable'])
@pytest.mark.parametrize('name', UPGRADED)
def test_env_attaches_overrides(preset, name):
    """Both explicit (portable) and map_trigger (smac-hard) Scenarios attach overrides.

    Constructing an SC2 env does not launch SC2."""
    env = make_vec_env(preset, name, backend='sc2')
    assert env.scenario is not None and env.scenario.unit_overrides == (MARINE_55,)
    off = make_vec_env(preset, name, backend='sc2', overrides={'backend.unit_overrides': 'off'})
    assert off.scenario is not None and off.scenario.unit_overrides == ()


def test_env_no_overrides_elsewhere():
    env = make_vec_env('smac-hard-portable', '3m', backend='sc2')
    assert env.scenario is not None and env.scenario.unit_overrides == ()


def test_env_requires_capability():
    """Maps with overrides raise CapabilityError if backend lacks unit_overrides;
    maps without overrides and unit_overrides='off' are unaffected."""
    from unismac.backends.pyref import PyrefBackend

    class NoOverrides(PyrefBackend):
        @property
        def capabilities(self):  # type: ignore[override]
            return dataclasses.replace(super().capabilities, unit_overrides=False)

    def backend(map_id: str) -> NoOverrides:
        return NoOverrides(load_static_data(), {map_id: load_map_info(map_id)}, 1)

    with pytest.raises(CapabilityError, match='unit_overrides'):
        make_vec_env('smac-hard-portable', '6m_vs_10m', backend=backend('new_maps/6m_vs_10m'))
    make_vec_env(
        'smac-hard-portable',
        '6m_vs_10m',
        backend=backend('new_maps/6m_vs_10m'),
        overrides={'backend.unit_overrides': 'off'},
    )
    make_vec_env('smac-hard-portable', '3m', backend=backend('new_maps/3m'))


@pytest.mark.parametrize('name', UPGRADED)
def test_pyref_applies_overrides(name):
    env = make_vec_env('smac-hard-portable', name, backend='pyref', seed=0)
    env.reset()
    assert _marines(env.world)[1] == {(55.0, 55.0)}
    if 2 in _marines(env.world):
        assert _marines(env.world)[2] == {(45.0, 45.0)}
    off = make_vec_env(
        'smac-hard-portable',
        name,
        backend='pyref',
        seed=0,
        overrides={'backend.unit_overrides': 'off'},
    )
    off.reset()
    assert _marines(off.world)[1] == {(45.0, 45.0)}


# ---------------------------------------------------------------- SC2


@pytest.mark.sc2
@pytest.mark.parametrize('preset', ['smac-hard-portable', 'smac-hard'])
def test_sc2_overrides_verified(preset):
    """On SC2, explicit/map_trigger Player 1 Marines have 55/55 HP; mismatched override
    raises ContractError on reset."""
    env = make_vec_env(preset, '6m_vs_10m', backend='sc2', seed=0)
    try:
        env.reset()
        assert _marines(env.world) == {1: {(55.0, 55.0)}, 2: {(45.0, 45.0)}}
        assert env.scenario is not None
        wrong = dataclasses.replace(
            env.scenario, unit_overrides=(dataclasses.replace(MARINE_55, health_max=60.0),)
        )
        with pytest.raises(ContractError, match=r'health_max.*\[55\.0\]'):
            env.backend.reset([wrong], seeds=[0])
    finally:
        env.close()
