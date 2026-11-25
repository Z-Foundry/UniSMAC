"""Env layer: dim matrix, MultiAgentEnv/VecEnv APIs, and smac-portable episodes on pyref."""

from __future__ import annotations

import numpy as np
import pytest

from unismac import scenarios
from unismac.assets import load_static_data
from unismac.backends.sc2.backend import CAPABILITIES as SC2_CAPABILITIES
from unismac.contract import CapabilityError, CommandKind, WorldState
from unismac.env import UnismacEnv, VecEnv, make_env, make_vec_env

# Upstream SMAC default (get_obs_size, get_state_size, n_actions); per-scenario parity
# is checked in subprocess by upstream_parity/test_smac_parity.py::test_dims_match_upstream.
SMAC_DIMS = {
    '3m': (30, 48, 9),
    '8m': (80, 168, 14),
    '25m': (250, 950, 31),
    '5m_vs_6m': (55, 98, 12),
    '8m_vs_9m': (85, 179, 15),
    '10m_vs_11m': (105, 243, 17),
    '27m_vs_30m': (285, 1170, 36),
    'MMM': (160, 290, 16),
    'MMM2': (176, 322, 18),
    '2s3z': (80, 120, 11),
    '3s5z': (128, 216, 14),
    '3s5z_vs_3s6z': (136, 230, 15),
    '3s_vs_3z': (36, 54, 9),
    '3s_vs_4z': (42, 61, 10),
    '3s_vs_5z': (48, 68, 11),
    '1c3s5z': (162, 270, 15),
    '2m_vs_1z': (16, 26, 7),
    'corridor': (156, 282, 30),
    '6h_vs_8z': (78, 140, 14),
    '2s_vs_1sc': (17, 27, 7),
    'so_many_baneling': (202, 397, 38),
    'bane_vs_bane': (336, 984, 30),
    '2c_vs_64zg': (332, 342, 70),
}
# Missing from new_maps; smac-portable on SC2 is unavailable for these scenarios.
SC2_PORTABLE_MISSING = {'3s_vs_3z', '3s_vs_4z'}


class _StubSc2:
    """Construction-only stub (dims are known before reset) declaring SC2 capabilities."""

    capabilities = SC2_CAPABILITIES
    batch_size = 1
    static_data = load_static_data()

    def reset(self, scenarios, *, env_ids=None, seeds=None) -> WorldState:
        raise AssertionError('unexpected reset')

    def step(self, command, n_loops) -> WorldState:
        raise AssertionError('unexpected step')

    def close(self) -> None:
        pass


def _random_actions(rng, avail: np.ndarray) -> np.ndarray:
    return np.array([[rng.choice(np.flatnonzero(row)) for row in env_rows] for env_rows in avail])


def test_scenario_list_complete():
    assert list(SMAC_DIMS) == scenarios.names('smac')


@pytest.mark.parametrize('name', list(SMAC_DIMS))
def test_smac_dims(name):
    env = make_vec_env('smac', name, backend=_StubSc2())
    assert (env.get_obs_size(), env.get_state_size(), env.get_total_actions()) == SMAC_DIMS[name]
    assert env.obs_dim == env.get_obs_size() and env.state_dim == env.get_state_size()
    info = env.get_env_info()
    assert info['n_agents'] == scenarios.get('smac', name).n_agents
    assert info['episode_limit'] == scenarios.get('smac', name).limit


@pytest.mark.parametrize('name', list(SMAC_DIMS))
def test_portable_dims_equal_smac(name):
    env = make_vec_env('smac-portable', name, backend='pyref')
    assert (env.get_obs_size(), env.get_state_size(), env.get_total_actions()) == SMAC_DIMS[name]


def test_timestep_quirk_q1():
    """Q1 quirk: smac with obs_timestep_number makes actual obs 1 larger than get_obs_size()."""
    env = make_vec_env('smac', '3m', backend=_StubSc2(), overrides={'obs_timestep_number': True})
    assert (env.get_obs_size(), env.obs_dim) == (31, 32)
    fixed = make_vec_env(
        'smac',
        '3m',
        backend=_StubSc2(),
        overrides={'obs_timestep_number': True, 'quirks.timestep_pad': False},
    )
    assert (fixed.get_obs_size(), fixed.obs_dim) == (31, 31)


def test_smac_preset_needs_sc2_capabilities():
    with pytest.raises(CapabilityError):
        make_env('smac', '3m', backend='pyref')
    with pytest.raises(CapabilityError):
        make_env('smac', '3m', backend='sc2', batch_size=2)


@pytest.mark.parametrize('name', sorted(SC2_PORTABLE_MISSING))
def test_portable_sc2_missing_new_maps(name):
    with pytest.raises(CapabilityError, match='new_maps'):
        make_env('smac-portable', name, backend='sc2')


def test_portable_pyref_episode_batch1():
    env = make_env('smac-portable', '3m', backend='pyref', seed=0)
    assert isinstance(env, UnismacEnv)
    rng = np.random.default_rng(0)
    for _ in range(2):
        obs, state = env.reset()
        assert len(obs) == 3 and all(o.shape == (30,) and o.dtype == np.float32 for o in obs)
        assert state.shape == (48,) and state.dtype == np.float32
        terminated, steps = False, 0
        while not terminated:
            avail = env.get_avail_actions()
            assert isinstance(avail, list) and len(avail[0]) == env.get_total_actions()
            assert env.get_avail_agent_actions(0) == avail[0]
            acts = [int(rng.choice(np.flatnonzero(a))) for a in avail]
            reward, terminated, info = env.step(acts)
            assert isinstance(reward, float) and isinstance(terminated, bool)
            assert set(info) >= {'battle_won', 'dead_allies', 'dead_enemies'}
            steps += 1
        assert steps <= env.episode_limit
        np.testing.assert_array_equal(env.get_obs_agent(1), env.get_obs()[1])
    stats = env.get_stats()
    assert stats['battles_game'] == 2
    env.close()


def test_portable_pyref_episode_batch8():
    env = make_env('smac-portable', '3m', backend='pyref', batch_size=8, seed=0)
    assert isinstance(env, VecEnv)
    obs, state = env.reset()
    assert obs.shape == (8, 3, 30) and state.shape == (8, 48)
    assert env.get_avail_actions().shape == (8, 3, 9)
    rng = np.random.default_rng(0)
    ended_at = np.full(8, -1)
    t = 0
    while not env.done.all():
        reward, terminated, infos = env.step(_random_actions(rng, env.get_avail_actions()))
        t += 1
        newly = terminated & (ended_at < 0)
        ended_at[newly] = t
        # Terminated envs stay frozen: reward=0, terminated=True, info={}.
        frozen = (ended_at >= 0) & (ended_at < t)
        assert np.all(reward[frozen] == 0) and np.all(terminated[frozen])
        assert all(infos[b] == {} for b in np.flatnonzero(frozen))
    assert t <= env.episode_limit and env.get_stats()['battles_game'] == 8
    # Partial reset: reset only envs 0 and 3.
    before = env.get_obs().copy()
    obs, _ = env.reset(env_ids=[0, 3])
    assert not env.done[0] and not env.done[3] and env.done[1]
    np.testing.assert_array_equal(obs[1], before[1])
    reward, terminated, _ = env.step(_random_actions(rng, env.get_avail_actions()))
    assert terminated[1] and reward[1] == 0
    env.close()


def test_pyref_deterministic():
    def run():
        env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=3)
        env.reset()
        rng = np.random.default_rng(5)
        out = []
        while not env.done.all():
            reward, _, _ = env.step(_random_actions(rng, env.get_avail_actions()))
            out.append((reward.copy(), env.get_state().copy()))
        return out

    a, b = run(), run()
    assert len(a) == len(b)
    for (ra, sa), (rb, sb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(sa, sb)


def test_unavailable_action_rejected():
    env = make_env('smac-portable', '3m', backend='pyref')
    env.reset()
    with pytest.raises(ValueError, match='(?i)action'):
        env.step([0, 0, 0])  # live units cannot take no-op
    with pytest.raises(ValueError):
        env.step([99, 1, 1])


class _HealRecorder:
    """Wrapper around a backend counting Medivac heal commands received by step()."""

    def __init__(self, inner):
        self.inner = inner
        self.capabilities = inner.capabilities
        self.batch_size = inner.batch_size
        self.static_data = inner.static_data
        self.heals = 0

    def reset(self, scenarios, *, env_ids=None, seeds=None):
        return self.inner.reset(scenarios, env_ids=env_ids, seeds=seeds)

    def step(self, command, n_loops):
        self.heals += int((command.kind == CommandKind.ABILITY).sum())
        return self.inner.step(command, n_loops)

    def close(self):
        self.inner.close()


def test_portable_mmm_heal_on_pyref():
    """smac-portable MMM: Medivac heal runs on pyref (starts at 50 energy, drops on heal)."""
    from unismac.assets import load_map_info
    from unismac.backends.pyref import PyrefBackend, PyrefConfig
    from unismac.env import build_config
    from unismac.env.core import resolve_map_ids

    config, spec = build_config('smac-portable', 'MMM')
    map_id = resolve_map_ids(config, spec, 'pyref')
    pyref = PyrefConfig(abilities='heal', spawn_energy=50.0, heal_autocast=True)
    inner = PyrefBackend(load_static_data(), {map_id: load_map_info(map_id)}, 2, pyref)
    backend = _HealRecorder(inner)
    env = make_vec_env('smac-portable', 'MMM', backend=backend, batch_size=2, seed=0)
    by_name = make_vec_env('smac-portable', 'MMM', backend='pyref', batch_size=2)
    assert by_name.backend.capabilities.abilities  # string backend also enables abilities
    assert env.actions.heal_enabled
    rng = np.random.default_rng(0)
    env.reset()
    world = env._world
    assert world is not None
    medivac = world.alive & (world.unit_type == 54)
    assert medivac.sum() == 4 and np.all(world.energy[medivac] == 50.0)
    low = 50.0
    done_any = np.zeros(2, bool)
    for _ in range(env.episode_limit):
        _, done, _ = env.step(_random_actions(rng, env.get_avail_actions()))
        world = env._world
        assert world is not None
        mine = world.alive & (world.unit_type == 54) & (world.owner == 1)
        if mine.any():
            low = min(low, float(world.energy[mine].min()))
        done_any |= done
        if done.all():
            break
    assert backend.heals > 0 and low < 50.0
    assert done_any.all()
