"""Official Gymnasium 1.0+ contract tests for `unismac.adapters.gymnasium`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('gymnasium', reason="requires optional 'gymnasium' extra")
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

from unismac.adapters.gymnasium import (
    GymnasiumEnv,
    GymnasiumVecEnv,
    make_gym_env,
    make_gym_vec_env,
)
from unismac.env import make_vec_env

pytestmark = pytest.mark.adapters


def _valid_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = (1.0 + rng.random(mask.shape)) * mask
    return scores.argmax(axis=-1).astype(np.int64)


@pytest.mark.parametrize(
    ('preset', 'scenario', 'obs_mode', 'include_state'),
    [
        ('smac-portable', '3m', 'dict', True),
        ('smac-portable', '3m', 'dict', False),
        ('smac-portable', '3m', 'obs_only', True),
        ('smac-portable', 'MMM', 'dict', True),
        ('smacv2-portable', '10gen_terran', 'dict', True),
        ('smac-hard-portable', '2s3z', 'dict', True),
    ],
)
def test_gymnasium_official_check_env(
    preset: str, scenario: str, obs_mode: str, include_state: bool
) -> None:
    """`GymnasiumEnv` passes Gymnasium's official `check_env` across all 3 portable benchmarks."""
    with make_gym_env(
        preset,
        scenario,
        backend='pyref',
        obs_mode=obs_mode,  # type: ignore[arg-type]
        include_state=include_state,
        invalid_action_mode='fallback',
        seed=0,
    ) as env:
        assert isinstance(env, GymnasiumEnv)
        check_env(env, skip_render_check=True)


def test_gymnasium_terminated_vs_truncated_and_invalid_actions() -> None:
    """Timeout yields `truncated=True, terminated=False`; invalid action modes behave as spec'd."""
    with make_gym_env(
        'smac-portable',
        '3m',
        backend='pyref',
        seed=0,
        overrides={'termination.episode_limit': 3},
    ) as env:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict) and {'obs', 'state', 'action_mask'} <= set(obs)
        assert obs['action_mask'].dtype == np.int8
        assert isinstance(info, dict)

        # Default invalid_action_mode='raise' rejects no-op on live agents.
        with pytest.raises(ValueError, match='(?i)action'):
            env.step(np.zeros(env.vec.n_agents, dtype=np.int64))

        stop = np.ones(env.vec.n_agents, dtype=np.int64)
        for step_idx in range(3):
            _, _, terminated, truncated, _ = env.step(stop)
            if step_idx < 2:
                assert not terminated and not truncated
            else:
                assert terminated is False and truncated is True

    # invalid_action_mode='fallback' remaps illegal actions safely and records count in info.
    with make_gym_env(
        'smac-portable',
        '3m',
        backend='pyref',
        invalid_action_mode='fallback',
        seed=0,
    ) as env:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.vec.n_agents, dtype=np.int64))
        assert info.get('invalid_actions_remapped', 0) == env.vec.n_agents


def test_gymnasium_rejects_dual_mode() -> None:
    """Gymnasium single-team adapters reject two-sided (`dual=True`) environments."""
    with make_vec_env(
        'smac-portable', '3m', backend='pyref', profile='sc2-symmetric', dual=True, batch_size=1
    ) as vec1:
        with pytest.raises(ValueError, match='(?i)dual|two-sided|pettingzoo'):
            GymnasiumEnv(vec1)
    with make_vec_env(
        'smac-portable', '3m', backend='pyref', profile='sc2-symmetric', dual=True, batch_size=2
    ) as vec2:
        with pytest.raises(ValueError, match='(?i)dual|two-sided|pettingzoo'):
            GymnasiumVecEnv(vec2)
    with pytest.raises(ValueError, match='(?i)dual|two-sided|pettingzoo'):
        make_gym_env(
            'smac-portable',
            '3m',
            backend='pyref',
            profile='sc2-symmetric',
            overrides={'opponent.kind': 'external'},
        )


@pytest.mark.parametrize(
    'mode',
    [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP, AutoresetMode.DISABLED],
)
def test_gymnasium_vec_env_autoreset_modes(mode: AutoresetMode) -> None:
    """`GymnasiumVecEnv` honors NEXT_STEP, SAME_STEP, and DISABLED autoreset semantics."""
    with make_gym_vec_env(
        'smac-portable',
        '3m',
        backend='pyref',
        batch_size=3,
        autoreset_mode=mode,
        seed=0,
        overrides={'termination.episode_limit': 4},
    ) as env:
        assert isinstance(env, GymnasiumVecEnv)
        assert env.num_envs == 3
        obs, infos = env.reset(seed=0)
        assert obs['obs'].shape == (3, 3, 30)
        assert obs['state'].shape == (3, 48)
        assert obs['action_mask'].shape == (3, 3, 9) and obs['action_mask'].dtype == np.int8
        assert isinstance(infos, dict)

        stop = np.ones((3, 3), dtype=np.int64)
        for _ in range(3):
            _, _, term, trunc, _ = env.step(stop)
            assert not term.any() and not trunc.any()

        # Step 4 hits episode_limit on all 3 slots.
        obs4, _, term4, trunc4, info4 = env.step(stop)
        assert not term4.any() and trunc4.all()
        if mode is AutoresetMode.SAME_STEP:
            assert 'final_obs' in info4 and 'final_info' in info4
            assert info4['_final_obs'].all() and info4['_final_info'].all()
            # Slot was reset immediately; step 5 runs fresh episode step 1.
            _, _, term5, trunc5, _ = env.step(stop)
            assert not term5.any() and not trunc5.any()
        elif mode is AutoresetMode.NEXT_STEP:
            # Step 5 performs the deferred autoreset with zero reward and no done flags.
            obs5, rew5, term5, trunc5, _ = env.step(stop)
            assert np.all(rew5 == 0.0) and not term5.any() and not trunc5.any()
            assert obs5['obs'].shape == obs4['obs'].shape
        else:
            # DISABLED keeps slots frozen until explicit reset_mask.
            _, rew5, term5, trunc5, _ = env.step(stop)
            assert np.all(rew5 == 0.0) and not term5.any() and not trunc5.any()
            env.reset(options={'reset_mask': np.array([True, False, False])})
            assert env.vec.runtime.done().tolist() == [False, True, True]


def test_gymnasium_wrapper_and_seeded_reproducibility() -> None:
    """Explicit `GymnasiumEnv(vec)` wrapper and `reset(seed=...)` are bitwise reproducible."""
    with make_vec_env('smac-portable', '3m', backend='pyref', batch_size=1, seed=5) as vec:
        env = GymnasiumEnv(vec)
        o1, _ = env.reset(seed=17)
        o2, _ = env.reset(seed=17)
        np.testing.assert_array_equal(o1['obs'], o2['obs'])
        np.testing.assert_array_equal(o1['state'], o2['state'])
        np.testing.assert_array_equal(o1['action_mask'], o2['action_mask'])
