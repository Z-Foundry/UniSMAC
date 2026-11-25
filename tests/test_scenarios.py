"""Unit tests for the static scenario registry (no StarCraft II required)."""

from __future__ import annotations

import dataclasses

import pytest

import unismac.scenarios as scenarios
from unismac.scenarios import ScenarioSpec


def test_counts_per_benchmark() -> None:
    assert len(scenarios.names('smac')) == 23
    assert len(scenarios.names('smacv2')) == 26
    assert len(scenarios.names('smac-hard')) == 35
    assert len(scenarios.all_specs()) == 23 + 26 + 35


def test_smac_3s5z() -> None:
    assert scenarios.get('smac', '3s5z') == ScenarioSpec(
        benchmark='smac',
        name='3s5z',
        n_agents=8,
        n_enemies=8,
        limit=150,
        a_race='P',
        b_race='P',
        unit_type_bits=2,
        map_type='stalkers_and_zealots',
        map_id='SMAC_Maps/3s5z',
        procedural=False,
    )


def test_smacv2_10gen_terran_is_procedural_on_32x32_flat() -> None:
    assert scenarios.get('smacv2', '10gen_terran') == ScenarioSpec(
        benchmark='smacv2',
        name='10gen_terran',
        n_agents=10,
        n_enemies=10,
        limit=200,
        a_race='T',
        b_race='T',
        unit_type_bits=3,
        map_type='terran_gen',
        map_id='SMAC_Maps/32x32_flat',
        procedural=True,
    )


def test_smacv2_composition() -> None:
    procedural = {s.name for s in scenarios.all_specs() if s.procedural}
    assert procedural == {'10gen_terran', '10gen_zerg', '10gen_protoss'}
    assert {scenarios.get('smacv2', n).map_id for n in procedural} == {'SMAC_Maps/32x32_flat'}
    fixed = [n for n in scenarios.names('smacv2') if n not in procedural]
    assert sorted(fixed) == sorted(scenarios.names('smac'))
    for name in fixed:
        assert scenarios.get('smacv2', name).map_id == f'SMAC_Maps/{name}'


def test_smac_hard_new_scenarios() -> None:
    assert scenarios.get('smac-hard', 'mmmt_vs_zspi') == ScenarioSpec(
        benchmark='smac-hard',
        name='mmmt_vs_zspi',
        n_agents=22,
        n_enemies=16,
        limit=400,
        a_race='T',
        b_race='P',
        unit_type_bits=4,
        map_type='MMM',
        map_id='new_maps/mmmt_vs_zspi',
        procedural=False,
    )
    unit_test = scenarios.get('smac-hard', 'unit_test')
    assert (unit_test.n_agents, unit_test.n_enemies, unit_test.limit) == (1, 30, 400)
    assert scenarios.get('smac-hard', 'pvt_large').limit == 2000
    added = set(scenarios.names('smac-hard')) - set(scenarios.names('smac'))
    assert len(added) == 12


def test_map_id_layout() -> None:
    for spec in scenarios.all_specs():
        assert not spec.map_id.endswith('.SC2Map')
        directory, _, filename = spec.map_id.partition('/')
        assert directory == ('new_maps' if spec.benchmark == 'smac-hard' else 'SMAC_Maps')
        if not spec.procedural:
            assert filename == spec.name
        assert spec.a_race in {'T', 'Z', 'P'} and spec.b_race in {'T', 'Z', 'P'}


def test_same_name_differs_by_benchmark() -> None:
    assert scenarios.get('smac', '3m').map_id == 'SMAC_Maps/3m'
    assert scenarios.get('smac-hard', '3m').map_id == 'new_maps/3m'
    keys = [(s.benchmark, s.name) for s in scenarios.all_specs()]
    assert len(keys) == len(set(keys))


def test_names_keep_upstream_order_and_lookup_errors() -> None:
    assert scenarios.names('smacv2')[:3] == ['10gen_terran', '10gen_zerg', '10gen_protoss']
    assert scenarios.names('smac')[0] == '3m'
    names = scenarios.names('smac')
    names.clear()  # returns a fresh copy
    assert scenarios.names('smac')
    with pytest.raises(KeyError):
        scenarios.get('smac', 'mmmt')
    with pytest.raises(KeyError):
        scenarios.names('bogus')
    with pytest.raises(KeyError):
        scenarios.get('bogus', '3m')


def test_spec_is_frozen() -> None:
    spec = scenarios.get('smac', '8m')
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.limit = 1  # type: ignore[misc]
