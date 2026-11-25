"""Env-layer tests for `smacv2` / `smacv2-portable`: dims, co-existence, FovTurn,
stochastic_attack, issue-time range checks, and procedural episodes on pyref.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac import scenarios
from unismac.adapters.pymarl import PyMARLEnv
from unismac.assets import load_static_data
from unismac.backends.sc2.backend import CAPABILITIES as SC2_CAPABILITIES
from unismac.contract import CapabilityError, CommandKind, WorldState
from unismac.env import make_vec_env

# Upstream SMACv2 defaults (10gen_* with sc2_gen_*.yaml):
# (get_obs_size, get_state_size, n_actions, get_cap_size).
SMACV2_DIMS = {
    '10gen_terran': (82, 120, 11, 3),
    '10gen_zerg': (82, 120, 11, 3),
    '10gen_protoss': (92, 130, 11, 3),
    '3m': (30, 48, 9, 0),
    '8m': (80, 168, 14, 0),
    '25m': (250, 950, 31, 0),
    '5m_vs_6m': (55, 98, 12, 0),
    '8m_vs_9m': (85, 179, 15, 0),
    '10m_vs_11m': (105, 243, 17, 0),
    '27m_vs_30m': (285, 1170, 36, 0),
    'MMM': (160, 290, 16, 3),
    'MMM2': (176, 322, 18, 3),
    '2s3z': (80, 120, 11, 2),
    '3s5z': (128, 216, 14, 2),
    '3s5z_vs_3s6z': (136, 230, 15, 2),
    '3s_vs_3z': (36, 54, 9, 0),
    '3s_vs_4z': (42, 61, 10, 0),
    '3s_vs_5z': (48, 68, 11, 0),
    '1c3s5z': (162, 270, 15, 3),
    '2m_vs_1z': (16, 26, 7, 0),
    'corridor': (156, 282, 30, 0),
    '6h_vs_8z': (78, 140, 14, 0),
    '2s_vs_1sc': (17, 27, 7, 0),
    'so_many_baneling': (202, 397, 38, 0),
    'bane_vs_bane': (336, 984, 30, 2),
    '2c_vs_64zg': (332, 342, 70, 0),
}


class _StubSc2:
    """Construction-only stub declaring SC2 backend capabilities."""

    capabilities = SC2_CAPABILITIES
    batch_size = 1
    static_data = load_static_data()

    def reset(self, scenarios, *, env_ids=None, seeds=None) -> WorldState:
        raise AssertionError('unexpected reset')

    def step(self, command, n_loops) -> WorldState:
        raise AssertionError('unexpected step')

    def close(self) -> None:
        pass


def _dims(env):
    cap_size = int(env.config.capability.attack is not None) + env.catalog.unit_type_bits
    return (
        env.obs_layout.reported_size,
        env.state_layout.reported_size,
        env.n_actions,
        cap_size,
    )


def test_scenario_list_complete():
    assert list(SMACV2_DIMS) == scenarios.names('smacv2')


@pytest.mark.parametrize('name', list(SMACV2_DIMS))
def test_smacv2_dims(name):
    env = make_vec_env('smacv2', name, backend=_StubSc2())
    assert _dims(env) == SMACV2_DIMS[name]
    info = PyMARLEnv(env).get_env_info()
    assert info['cap_shape'] == SMACV2_DIMS[name][3]
    assert info['n_agents'] == env.n_agents


@pytest.mark.parametrize('name', list(SMACV2_DIMS))
def test_smacv2_portable_dims(name):
    env = make_vec_env('smacv2-portable', name, backend='pyref')
    assert _dims(env) == SMACV2_DIMS[name]


def test_smac_and_smacv2_in_one_process():
    """Both presets construct and run in one process (upstream smac/smacv2 cannot coexist)."""
    v1 = make_vec_env('smac', 'MMM', backend=_StubSc2())
    v2 = make_vec_env('smacv2', 'MMM', backend=_StubSc2())
    v2g = make_vec_env('smacv2', '10gen_terran', backend=_StubSc2())
    assert (
        v1.obs_layout.reported_size,
        v2.obs_layout.reported_size,
        v2g.obs_layout.reported_size,
    ) == (160, 160, 82)
    assert 'cap_shape' not in PyMARLEnv(v1).get_env_info()
    assert PyMARLEnv(v2).get_env_info()['cap_shape'] == 3
    assert v1.config.reward.only_positive_mode == 'abs'
    assert v2.config.reward.only_positive_mode == 'clamp'
    rng = np.random.default_rng(0)
    for preset, name in (('smac-portable', '3m'), ('smacv2-portable', '10gen_zerg')):
        env = make_vec_env(preset, name, backend='pyref', batch_size=2, seed=0)
        env.reset()
        for _ in range(5):
            avail = env.avail_actions()
            acts = np.array([[rng.choice(np.flatnonzero(r)) for r in rows] for rows in avail])
            env.step(acts)
        env.close()


def test_smacv2_portable_10gen_not_on_sc2():
    with pytest.raises(CapabilityError, match='new_maps'):
        make_vec_env('smacv2-portable', '10gen_terran', backend='sc2')


def test_procedural_needs_explicit_spawns():
    with pytest.raises(ValueError, match='explicit'):
        make_vec_env(
            'smacv2',
            '10gen_terran',
            backend=_StubSc2(),
            overrides={'backend.spawn_mode': 'map_trigger'},
        )
    with pytest.raises(ValueError, match='team_gen'):
        make_vec_env(
            'smacv2', '10gen_terran', backend=_StubSc2(), overrides={'capability_config': {}}
        )
    with pytest.raises(ValueError):
        make_vec_env(
            'smacv2', '3m', backend=_StubSc2(), overrides={'capability_config': {'n_units': 4}}
        )


def _env(overrides, seed=0, batch=1, name='10gen_protoss'):
    return make_vec_env(
        'smacv2-portable', name, backend='pyref', batch_size=batch, seed=seed, overrides=overrides
    )


class _Recorder:
    """Wrapper around pyref backend recording commands passed to step()."""

    def __init__(self, inner):
        self.inner = inner
        self.commands = []
        self.capabilities = inner.capabilities
        self.batch_size = inner.batch_size
        self.static_data = inner.static_data

    def reset(self, scenarios, *, env_ids=None, seeds=None):
        return self.inner.reset(scenarios, env_ids=env_ids, seeds=seeds)

    def step(self, command, n_loops):
        self.commands.append(command)
        return self.inner.step(command, n_loops)

    def close(self):
        self.inner.close()


def _recorded(name, overrides, seed=0):
    from unismac.assets import load_map_info
    from unismac.backends.pyref import PyrefBackend
    from unismac.env import build_config
    from unismac.env.core import resolve_map_ids

    config, spec = build_config('smacv2-portable', name, overrides)
    map_id = resolve_map_ids(config, spec, 'pyref')
    backend = _Recorder(PyrefBackend(load_static_data(), {map_id: load_map_info(map_id)}, 1))
    return make_vec_env(
        'smacv2-portable', name, backend=backend, seed=seed, overrides=overrides
    ), backend


def test_fov_turn_sets_direction_without_command():
    env, be = _recorded('10gen_protoss', {'conic_fov': True, 'num_fov_actions': 4})
    assert env.n_actions == 6 + 4 + env.n_enemies and env._fov is not None
    env.reset()
    np.testing.assert_array_equal(env._fov.dirs[0], np.tile([1.0, 0.0], (env.n_agents, 1)))
    acts = np.full((1, env.n_agents), 1)  # STOP
    acts[0, 0] = 6 + 1  # direction 1: pi/2
    env.step(acts)
    np.testing.assert_allclose(env._fov.dirs[0, 0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_array_equal(env._fov.dirs[0, 1:], np.tile([1.0, 0.0], (env.n_agents - 1, 1)))
    kinds = be.commands[-1].kind[0]
    sc0 = env.runtime.last_scenarios()[0]
    assert sc0 is not None
    ally_slots = [i for i, sp in enumerate(sc0.spawns) if sp.owner == 1]
    assert sorted(kinds[ally_slots].tolist()).count(int(CommandKind.NOOP)) == 1
    names = env.obs_layout.own
    assert 'fov_x' in names and 'fov_y' in names


def test_stochastic_attack_drops_commands():
    """Zero attack probability drops attack commands (upstream get_agent_action returns None)."""
    cap = {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {'unit_types': ['stalker'], 'weights': [1.0]},
        'start_positions': {'dist_type': 'surrounded'},
        'attack': {'lower_bound': [0.0], 'upper_bound': [1e-12]},
    }
    env, be = _recorded('10gen_protoss', {'capability_config': cap})
    env.reset()
    n_no = env._actions.n_no_attack
    issued = 0
    attacking = np.zeros(env.n_agents, bool)
    for _ in range(40):
        avail = env.avail_actions()[0]
        acts = np.array([[np.flatnonzero(r)[-1] for r in avail]])
        attacking = acts[0] >= n_no
        step = env.step(acts)
        kinds = be.commands[-1].kind[0]
        issued += int((kinds == CommandKind.ATTACK_UNIT).sum())
        if attacking.any():
            break
        if step.terminated[0] or step.truncated[0]:
            env.reset()
    assert attacking.any() and issued == 0


def test_range_check_at_issue_time():
    """action_mask=False allows out-of-range targets in mask, but issue-time check drops them."""
    cap = {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {'unit_types': ['stalker'], 'weights': [1.0]},
        'start_positions': {'dist_type': 'reflect_position'},
    }
    env, be = _recorded('10gen_protoss', {'capability_config': cap, 'action_mask': False})
    env.reset()
    n_no = env._actions.n_no_attack
    avail = env.avail_actions()[0]
    assert (avail[:, n_no:] == 1).all()  # all live enemies selectable
    acts = np.array([[n_no] * env.n_agents])
    env.step(acts)
    # Initial distance > 6 (reflect: ally x <= 15, enemy x >= 17), all blocked by range check
    assert (be.commands[-1].kind[0] != CommandKind.ATTACK_UNIT).all()


@pytest.mark.parametrize('name', ['10gen_terran', '10gen_zerg', '10gen_protoss'])
def test_generated_pyref_episodes(name):
    env = _env({}, seed=3, batch=4, name=name)
    env.reset()
    rng = np.random.default_rng(0)
    ends = 0
    for _ in range(400):
        avail = env.avail_actions()
        acts = np.array([[rng.choice(np.flatnonzero(r)) for r in rows] for rows in avail])
        step = env.step(acts)
        done = step.terminated | step.truncated
        assert np.isfinite(step.reward).all() and np.isfinite(env.obs()).all()
        if done.any():
            ends += int(done.sum())
            env.reset(np.flatnonzero(done).tolist())
    assert ends >= 4
    caps = env.capabilities().reshape(4, env.n_agents, -1)
    assert (caps.sum(-1) == 1).all()  # one-hot unit type
    for sc in env.runtime.last_scenarios():
        assert sc is not None
        x0, y0, x1, y1 = (2, 2, 30, 30)
        assert all(x0 <= s.x <= x1 and y0 <= s.y <= y1 for s in sc.spawns)  # spawn_clamp


def test_health_capability_rejected():
    cap = {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {'unit_types': ['marine'], 'weights': [1.0]},
        'health': {},
    }
    with pytest.raises(CapabilityError):
        make_vec_env(
            'smacv2', '10gen_terran', backend=_StubSc2(), overrides={'capability_config': cap}
        )


def test_smacv2_portable_mmm_heal_on_pyref():
    """smacv2-portable enables Medivac heal like smac-portable (pyref abilities='heal')."""
    env = make_vec_env('smacv2-portable', 'MMM', backend='pyref', batch_size=2, seed=0)
    assert env._actions.heal_enabled and env.runtime.backend.capabilities.abilities
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(60):
        avail = env.avail_actions()
        acts = np.array([[rng.choice(np.flatnonzero(r)) for r in rows] for rows in avail])
        step = env.step(acts)
        done = step.terminated | step.truncated
        if done.any():
            env.reset(np.flatnonzero(done).tolist())
    env.close()
