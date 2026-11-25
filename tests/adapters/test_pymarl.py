"""Zero-dependency contract tests for the PyMARL adapter (`unismac.adapters.pymarl`)."""

from __future__ import annotations

import numpy as np
import pytest

from unismac.adapters import PyMARLEnv, UnismacEnv, make_pymarl_env
from unismac.adapters.pymarl import PyMARLEnv as DirectPyMARLEnv
from unismac.adapters.pymarl import make_pymarl_env as direct_make_pymarl_env
from unismac.env import make_vec_env


def test_pymarl_exports_and_wrapper_validation() -> None:
    """`PyMARLEnv` and `UnismacEnv` alias match; wrapper enforces `batch_size == 1`."""
    assert UnismacEnv is PyMARLEnv
    assert DirectPyMARLEnv is PyMARLEnv
    assert direct_make_pymarl_env is make_pymarl_env

    with make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=0) as vec:
        with pytest.raises(ValueError, match='batch_size=1'):
            PyMARLEnv(vec)


def test_pymarl_single_sided_episode_and_surface() -> None:
    """Full `MultiAgentEnv` API: reset, step, masks, env_info, stats, frozen-slot semantics."""
    with make_pymarl_env('smac-portable', '3m', backend='pyref', seed=0) as env:
        assert isinstance(env, PyMARLEnv)
        assert 'PyMARLEnv' in repr(env) or 'UnismacEnv' in repr(env)
        assert env.accel in ('python', 'services', 'rust')
        assert env.dual is False
        assert env.mirror is None
        assert env.obs_layout.size == 30
        assert env.state_layout.size == 48
        assert env.n_agents_for(1) == 3
        assert env.n_actions_for(1) == 9
        assert env.seed() == 0
        assert (env.get_obs_size(), env.get_state_size(), env.get_total_actions()) == (30, 48, 9)

        info_meta = env.get_env_info()
        assert info_meta['obs_shape'] == 30
        assert info_meta['state_shape'] == 48
        assert info_meta['n_actions'] == 9
        assert info_meta['n_agents'] == 3
        assert info_meta['episode_limit'] == 60
        assert 'cap_shape' not in info_meta

        rng = np.random.default_rng(0)
        for _ in range(2):
            obs, state = env.reset()
            assert len(obs) == 3 and all(o.shape == (30,) and o.dtype == np.float32 for o in obs)
            assert state.shape == (48,) and state.dtype == np.float32
            assert env.max_reward is not None and env.max_reward > 0
            alive = env.alive()
            assert alive.dtype == np.bool_ and alive.shape == (3,) and alive.all()

            terminated, steps = False, 0
            while not terminated:
                avail = env.get_avail_actions()
                assert isinstance(avail, list) and len(avail) == 3
                assert len(avail[0]) == env.get_total_actions()
                assert env.get_avail_agent_actions(0) == avail[0]
                acts = [int(rng.choice(np.flatnonzero(a))) for a in avail]
                reward, terminated, info = env.step(acts)
                assert isinstance(reward, float) and isinstance(terminated, bool)
                assert set(info) >= {'battle_won', 'dead_allies', 'dead_enemies'}
                steps += 1
            assert 0 < steps <= env.episode_limit
            np.testing.assert_array_equal(env.get_obs_agent(1), env.get_obs()[1])

            # Upstream MultiAgentEnv contract: stepping a finished episode returns (0.0, True, {}).
            r_post, t_post, i_post = env.step([0, 0, 0])
            assert r_post == 0.0 and t_post is True and i_post == {}

        stats = env.get_stats()
        assert stats['battles_game'] == 2
        assert set(stats) >= {'battles_won', 'battles_game', 'battles_draw', 'win_rate', 'timeouts'}

        with pytest.raises(NotImplementedError):
            env.render()
        with pytest.raises(NotImplementedError):
            env.save_replay()


def test_pymarl_invalid_actions_rejected() -> None:
    """Unavailable or out-of-range actions raise `ValueError`."""
    with make_pymarl_env('smac-portable', '3m', backend='pyref', seed=0) as env:
        env.reset()
        with pytest.raises(ValueError, match='(?i)action'):
            env.step([0, 0, 0])
        with pytest.raises(ValueError):
            env.step([99, 1, 1])


def test_pymarl_two_sided_mode() -> None:
    """`dual=True` exposes side=2 observations/masks and two-sided step rewards."""
    with make_pymarl_env(
        'smac-portable', '3m', profile='minimal', backend='pyref', dual=True, seed=1
    ) as env:
        assert env.dual is True
        assert env.mirror == 'x'
        env.reset()
        assert len(env.get_obs(2)) == 3
        assert env.get_state(2).shape == (48,)
        assert len(env.get_avail_actions(2)) == 3
        assert env.alive(2).shape == (3,)
        _, _, info = env.step([1, 1, 1], [1, 1, 1])
        assert 'reward_enemy' in info and 'battle_lost' in info


def test_pymarl_smacv2_capabilities_and_timestep_quirk() -> None:
    """SMACv2 `cap_shape`/`get_capabilities` and Q1 `obs_timestep_number` reported size match."""
    with make_pymarl_env('smacv2-portable', '10gen_terran', backend='pyref', seed=0) as env:
        info = env.get_env_info()
        assert info['cap_shape'] == 3 and env.get_cap_size() == 3
        env.reset()
        caps = env.get_capabilities().reshape(env.n_agents, 3)
        assert (caps.sum(-1) == 1).all()

    with make_pymarl_env(
        'smac-portable',
        '3m',
        backend='pyref',
        overrides={'obs_timestep_number': True, 'quirks.timestep_pad': True},
    ) as env:
        assert env.get_obs_size() == 31 and env.obs_layout.size == 32


def test_pymarl_matches_kernel_bitwise() -> None:
    """`PyMARLEnv` produces bitwise-identical trajectories to direct `VecEnv`."""
    with (
        make_pymarl_env('smac-portable', '3m', backend='pyref', seed=4) as pm,
        make_vec_env('smac-portable', '3m', backend='pyref', batch_size=1, seed=4) as vec,
    ):
        pm_obs, pm_state = pm.reset()
        vec.reset()
        np.testing.assert_array_equal(np.stack(pm_obs), vec.obs()[0])
        np.testing.assert_array_equal(pm_state, vec.state()[0])

        rng = np.random.default_rng(7)
        for _ in range(30):
            avail = vec.avail_actions()
            assert pm.get_avail_actions() == avail[0].tolist()
            acts = [int(rng.choice(np.flatnonzero(row))) for row in avail[0]]
            r_pm, d_pm, i_pm = pm.step(acts)
            step = vec.step(np.array([acts], dtype=np.int64))
            assert r_pm == float(step.reward[0])
            assert d_pm == bool(step.terminated[0] or step.truncated[0])
            assert i_pm == step.infos[0]
            np.testing.assert_array_equal(np.stack(pm.get_obs()), vec.obs()[0])
            np.testing.assert_array_equal(pm.get_state(), vec.state()[0])
            if d_pm:
                break
