"""Official PettingZoo ParallelEnv & AEC contract tests for `unismac.adapters.pettingzoo`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('pettingzoo', reason="requires optional 'pettingzoo' extra")
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils.conversions import parallel_to_aec

from unismac.adapters.pettingzoo import PettingZooParallelEnv, make_pettingzoo_env
from unismac.env import make_vec_env

pytestmark = pytest.mark.adapters


@pytest.mark.parametrize(
    ('preset', 'scenario', 'dual', 'dead_agent_mode', 'obs_with_mask'),
    [
        ('smac-portable', '3m', False, 'keep', True),
        ('smac-portable', '3m', False, 'remove', True),
        ('smac-portable', '3m', False, 'keep', False),
        ('smacv2-portable', '10gen_terran', False, 'keep', True),
        ('smac-portable', '3m', True, 'keep', True),
        ('smac-portable', '3m', True, 'remove', True),
        ('smac-portable', '5m_vs_6m', True, 'keep', True),
    ],
)
def test_pettingzoo_official_parallel_api_test(
    preset: str,
    scenario: str,
    dual: bool,
    dead_agent_mode: str,
    obs_with_mask: bool,
) -> None:
    """`PettingZooParallelEnv` passes official `parallel_api_test` across single/dual/asymmetric."""
    profile = 'sc2-symmetric' if dual else 'minimal'
    with make_pettingzoo_env(
        preset,
        scenario,
        backend='pyref',
        profile=profile,
        dual=dual,
        dead_agent_mode=dead_agent_mode,  # type: ignore[arg-type]
        obs_with_mask=obs_with_mask,
        invalid_action_mode='fallback',
        seed=0,
    ) as env:
        assert isinstance(env, PettingZooParallelEnv)
        parallel_api_test(env, num_cycles=200)


def test_pettingzoo_parallel_to_aec_official_api_test() -> None:
    """`parallel_to_aec(env)` passes official PettingZoo `api_test`."""
    with make_pettingzoo_env(
        'smac-portable',
        '3m',
        backend='pyref',
        profile='minimal',
        invalid_action_mode='fallback',
        seed=0,
    ) as par:
        aec = parallel_to_aec(par)
        api_test(aec, num_cycles=100)


def test_pettingzoo_dual_asymmetric_spaces_state_and_masks() -> None:
    """Two-sided asymmetric `5m_vs_6m` exposes distinct spaces, `int8` masks, and per-side state."""
    with make_pettingzoo_env(
        'smac-portable',
        '5m_vs_6m',
        backend='pyref',
        profile='sc2-symmetric',
        dual=True,
        seed=3,
    ) as env:
        assert env.possible_agents == [
            'ally_0',
            'ally_1',
            'ally_2',
            'ally_3',
            'ally_4',
            'enemy_0',
            'enemy_1',
            'enemy_2',
            'enemy_3',
            'enemy_4',
            'enemy_5',
        ]
        # Cached space objects & distinct action sizes for 5m vs 6m.
        assert env.observation_space('ally_0') is env.observation_space('ally_0')
        assert env.action_space('ally_0') is env.action_space('ally_0')
        assert env.action_space('ally_0').n == 12
        assert env.action_space('enemy_0').n == 11

        obs, infos = env.reset(seed=3)
        assert set(obs) == set(env.possible_agents)
        assert set(infos) == set(env.possible_agents)
        assert obs['ally_0']['action_mask'].dtype == np.int8
        assert obs['enemy_0']['action_mask'].dtype == np.int8

        st1 = env.state()
        np.testing.assert_array_equal(st1, env.state_for(1))
        st2 = env.state_for(2)
        assert st1.shape == env.state_space.shape
        assert st2.shape == env.state_space_for(2).shape

        # Default invalid_action_mode='raise' rejects no-op on live agents.
        with pytest.raises(ValueError, match='(?i)action'):
            env.step({a: 0 for a in env.agents})

        stop = {a: 1 for a in env.agents}
        next_obs, rewards, terms, truncs, step_infos = env.step(stop)
        assert set(next_obs) == set(env.possible_agents)
        assert set(rewards) == set(env.possible_agents)
        assert set(terms) == set(env.possible_agents)
        assert set(truncs) == set(env.possible_agents)
        assert step_infos['ally_0']['side'] == 1 and 'battle_won' in step_infos['ally_0']
        assert step_infos['enemy_0']['side'] == 2 and 'battle_won' in step_infos['enemy_0']


def test_pettingzoo_wrapper_and_seeded_reproducibility() -> None:
    """Explicit `PettingZooParallelEnv(vec)` wrapper and `reset(seed=...)` are reproducible."""
    with make_vec_env('smac-portable', '3m', backend='pyref', batch_size=1, seed=0) as vec:
        env = PettingZooParallelEnv(vec)
        o1, _ = env.reset(seed=11)
        o2, _ = env.reset(seed=11)
        for a in env.possible_agents:
            np.testing.assert_array_equal(o1[a]['observation'], o2[a]['observation'])
            np.testing.assert_array_equal(o1[a]['action_mask'], o2[a]['action_mask'])
