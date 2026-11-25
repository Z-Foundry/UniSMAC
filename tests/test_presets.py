"""Presets, config merging, and spawn records."""

from __future__ import annotations

import pytest

from unismac import presets, scenarios
from unismac.contract import Variant
from unismac.env import build_config
from unismac.env.config import EnvConfig, apply_overrides


def test_preset_names():
    assert presets.names() == [
        'smac',
        'smac-portable',
        'smacv2',
        'smacv2-portable',
        'smac-hard',
        'smac-hard-portable',
    ]
    with pytest.raises(KeyError):
        presets.get('nope')


def test_smac_preset_matches_upstream_defaults():
    cfg, _ = build_config('smac', '3m')
    flat = cfg.flat()
    assert flat['backend.step_mul'] == 8
    assert flat['actions.move_amount'] == 2
    assert flat['opponent.difficulty'] == 'very_hard'
    assert (flat['obs.all_health'], flat['obs.own_health'], flat['obs.last_action']) == (
        True,
        True,
        False,
    )
    assert flat['state.last_action'] and not flat['state.timestep_number']
    assert flat['reward.only_positive'] and flat['reward.only_positive_mode'] == 'abs'
    assert (flat['reward.death_value'], flat['reward.win'], flat['reward.defeat']) == (10, 200, 0)
    assert (flat['reward.negative_scale'], flat['reward.scale_rate']) == (0.5, 20)
    assert flat['quirks.timestep_pad'] and flat['quirks.terrain_height_flip']


def test_merge_priority():
    # Priority: preset < scenario defaults < user overrides
    cfg, spec = build_config('smac', 'corridor')
    assert cfg.termination.episode_limit == spec.limit == 400
    assert (cfg.players.agent_race, cfg.players.enemy_race) == ('protoss', 'zerg')
    cfg, _ = build_config('smac', 'corridor', {'episode_limit': 50, 'players.enemy_race': 'terran'})
    assert cfg.termination.episode_limit == 50 and cfg.players.enemy_race == 'terran'


def test_override_forms():
    base = EnvConfig()
    a = apply_overrides(base, {'reward_win': 100})
    b = apply_overrides(base, {'reward.win': 100})
    c = apply_overrides(base, {'reward': {'win': 100}})
    assert a == b == c and a.reward.win == 100 and base.reward.win == 200
    assert apply_overrides(base, {'difficulty': '3'}).opponent.difficulty == 'medium'
    assert apply_overrides(base, {'difficulty': 'A'}).opponent.difficulty == 'cheat_insane'


@pytest.mark.parametrize(
    'overrides',
    [
        {'no_such_key': 1},
        {'reward.nope': 1},
        {'reward_sparse': 'yes'},
        {'units.enemy_source': 'fog'},
        {'termination.episode_limit': 1.5},
        {'reward': 3},
    ],
)
def test_bad_overrides(overrides):
    with pytest.raises(ValueError):
        apply_overrides(EnvConfig(), overrides)


def test_config_frozen():
    cfg = EnvConfig()
    with pytest.raises(AttributeError):
        cfg.reward.win = 1  # type: ignore[misc]


def test_portable_preset():
    cfg, _ = build_config('smac-portable', 'MMM')
    assert cfg.backend.spawn_mode == 'explicit' and cfg.backend.spawn_variant == 'std'
    assert cfg.opponent.kind == 'opening_orders' and cfg.maps.sc2_dir == 'new_maps'
    assert cfg.actions.heal == 'mmm_rule' and cfg.units.enemy_source == 'all'
    bc = cfg.backend
    assert (bc.pyref_abilities, bc.pyref_spawn_energy, bc.pyref_heal_autocast) == (
        'heal',
        50.0,
        True,
    )


def test_spawn_records():
    records = scenarios.load_spawns()
    assert list(records) == scenarios.names('smac')
    for name, rec in records.items():
        spec = scenarios.get('smac', name)
        owners = [s.owner for s in rec.spawns]
        assert owners.count(1) == spec.n_agents and owners.count(2) == spec.n_enemies, name
        assert rec.map_id == spec.map_id and len(rec.map_sha256) == 64
        assert rec.sc2_version == '4.10.0.75689'
        assert all(s.variant == Variant.RL for s in rec.spawns if s.owner == 1), name
        assert all(s.variant == Variant.STD for s in rec.spawns if s.owner == 2), name
        for s in rec.spawns:
            assert 0 < s.x < 32 and 0 < s.y < 32
            assert (s.unit_type, s.variant) in rec.native_unit_type
    assert records['3s_vs_3z'].portable_map_id is None
    assert records['2s_vs_1sc'].portable_terrain_same is False
    assert records['3m'].portable_terrain_same is True
    with pytest.raises(KeyError):
        scenarios.spawn_record('nope')


def test_smacv2_preset_matches_upstream_defaults():
    """smacv2: upstream SMACv2 StarCraft2Env defaults; 10gen_* applies sc2_gen_*.yaml."""
    cfg, _ = build_config('smacv2', '3m')
    assert cfg.backend.spawn_mode == 'map_trigger' and cfg.opponent.kind == 'engine_ai'
    assert cfg.reward.only_positive_mode == 'clamp'
    assert cfg.termination.enemy_medivac_rule == 'no_non_medivac_left'
    assert cfg.actions.target_pad == 'mmm_rule' and cfg.actions.heal == 'mmm_rule'
    assert cfg.visibility.spotter_gate and cfg.visibility.model == 'circle'
    assert cfg.catalog.ranges == 'fixed' and not cfg.obs.own_pos
    q = cfg.quirks
    assert (q.enemy_block_reset, q.ally_type_override, q.obs_shuffle) == (True, True, True)
    assert not q.timestep_pad and q.terrain_height_flip
    assert not cfg.capability.enabled

    cfg, spec = build_config('smacv2', '10gen_zerg')
    assert spec.procedural and cfg.backend.spawn_mode == 'explicit'
    assert cfg.catalog.ranges == 'per_unit' and cfg.catalog.min_attack_range == 2
    assert cfg.obs.own_pos
    cap = cfg.capability
    assert cap.team_gen is not None and cap.start_positions is not None
    assert (cap.n_units, cap.n_enemies) == (5, 5)
    assert cap.team_gen.unit_types == ('zergling', 'baneling', 'hydralisk')
    assert cap.team_gen.weights == (0.45, 0.1, 0.45)
    assert cap.team_gen.exception_unit_types == ('baneling',)
    assert cap.start_positions.dist_type == 'surrounded_and_reflect'
    assert cap.attack is None and cap.enemy_mask is None and cap.health is None


def test_smacv2_capability_override_replaces_group():
    """capability_config replaces the whole group as upstream (no merge with yaml defaults)."""
    cap = {'n_units': 10, 'n_enemies': 11, 'team_gen': {'unit_types': ['marine'], 'weights': [1.0]}}
    cfg, _ = build_config('smacv2', '10gen_terran', {'capability_config': cap})
    assert cfg.capability.start_positions is None and cfg.capability.team_gen is not None
    assert cfg.capability.team_gen.exception_unit_types == ()
    assert (cfg.capability.n_units, cfg.capability.n_enemies) == (10, 11)
    # Dotted sub-key overrides still merge on top of yaml defaults
    cfg, _ = build_config('smacv2', '10gen_terran', {'capability.n_enemies': 7})
    assert cfg.capability.n_enemies == 7 and cfg.capability.start_positions is not None
    assert cfg.capability.team_gen is not None
    assert cfg.capability.team_gen.unit_types == ('marine', 'marauder', 'medivac')


def test_smacv2_portable_preset():
    cfg, _ = build_config('smacv2-portable', 'MMM')
    assert cfg.backend.spawn_mode == 'explicit' and cfg.backend.spawn_variant == 'std'
    assert cfg.opponent.kind == 'opening_orders' and cfg.maps.sc2_dir == 'new_maps'
    assert cfg.units.enemy_source == 'all' and cfg.actions.heal == 'mmm_rule'
    bc = cfg.backend
    assert (bc.pyref_abilities, bc.pyref_spawn_energy, bc.pyref_heal_autocast) == (
        'heal',
        50.0,
        True,
    )
    assert cfg.visibility.spotter_gate and cfg.reward.only_positive_mode == 'clamp'
    q = cfg.quirks
    assert not (
        q.enemy_block_reset or q.ally_type_override or q.obs_shuffle or q.terrain_height_flip
    )
    cfg, _ = build_config('smacv2-portable', '10gen_protoss')
    assert cfg.capability.team_gen is not None
    assert cfg.backend.spawn_clamp and cfg.maps.sc2_dir == 'new_maps'
    assert cfg.capability.team_gen.unit_types == ('stalker', 'zealot', 'colossus')


def test_smac_hard_preset_matches_upstream_defaults():
    """smac-hard: upstream SMAC-Hard defaults (scripted opponent), with use_ability off."""
    cfg, spec = build_config('smac-hard', 'MMM')
    assert spec.benchmark == 'smac-hard' and spec.map_id == 'new_maps/MMM'
    assert cfg.backend.spawn_mode == 'map_trigger' and cfg.backend.sc2_slot_order == 'tag'
    assert cfg.opponent.kind == 'scripted' and cfg.opponent.script_abilities == 'issue'
    assert not cfg.actions.ability and cfg.actions.target_pad == 'max_units_9'
    assert not cfg.actions.range_check and cfg.actions.heal == 'mmm_rule'
    assert (cfg.units.enemy_source, cfg.units.ally_order, cfg.units.enemy_order) == (
        'all',
        'type_xy',
        'type_xy',
    )
    assert (cfg.catalog.type_id, cfg.catalog.shield_max) == ('standard', 'static')
    assert cfg.reward.only_positive_mode == 'clamp'
    assert cfg.termination.enemy_medivac_rule == 'last_unit_is_medivac'
    q = cfg.quirks
    assert q.timestep_pad and q.terrain_height_flip and q.role_offset
    assert build_config('smac-hard', '3m', {'use_ability': True})[0].actions.ability is True


def test_smac_hard_portable_preset():
    cfg, _ = build_config('smac-hard-portable', '7q_vs_2bc')
    bc = cfg.backend
    assert bc.spawn_mode == 'explicit' and bc.spawn_variant == 'std' and bc.spawn_clamp
    assert (bc.pyref_commands, bc.pyref_command_delay, bc.pyref_move_lag) == ('delayed', 1, 1)
    assert (bc.pyref_abilities, bc.pyref_spawn_energy, bc.pyref_heal_autocast) == (
        'heal',
        50.0,
        True,
    )
    assert cfg.opponent.kind == 'scripted' and cfg.opponent.script_abilities == 'drop'
    assert not cfg.actions.ability and cfg.actions.target_pad == 'max_units_9'
    q = cfg.quirks
    assert not (q.timestep_pad or q.terrain_height_flip or q.role_offset)


def test_smac_hard_spawn_records():
    from unismac.env.scripted import POOLS

    records = scenarios.load_spawns('smac-hard')
    assert list(records) == [n for n in scenarios.names('smac-hard') if n in records]
    assert len(records) == 27 and set(records) <= set(POOLS)
    assert not {'3s_vs_3z', '3s_vs_4z', 'unit_test', 'pvt_large'} & set(records)
    for name, rec in records.items():
        spec = scenarios.get('smac-hard', name)
        owners = [s.owner for s in rec.spawns]
        assert owners.count(1) == spec.n_agents and owners.count(2) == spec.n_enemies, name
        assert rec.map_id == spec.map_id and rec.portable_map_id == spec.map_id
        assert all(s.variant == Variant.STD for s in rec.spawns), name
        assert all(rec.native_unit_type[k] == k[0] for k in rec.native_unit_type), name
    with pytest.raises(KeyError, match='spawns_hard'):
        scenarios.spawn_record('2m_vs_1z', 'smac-hard')


def test_scenario_overrides_below_user():
    cfg, _ = build_config(
        'smacv2', '10gen_terran', {'obs_own_pos': False, 'use_unit_ranges': False}
    )
    assert not cfg.obs.own_pos and cfg.catalog.ranges == 'fixed'
