"""Env-layer tests for smac-hard / smac-hard-portable presets (pyref, capabilities, SC2 duel)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from unismac import scenarios
from unismac.adapters.pymarl import PyMARLEnv, make_pymarl_env
from unismac.contract import CapabilityError
from unismac.env import make_vec_env
from unismac.env.opponents.scripted import POOLS, ScriptedOpponent


def _scripted(env) -> ScriptedOpponent:
    opp = env._opponent
    assert isinstance(opp, ScriptedOpponent)
    return opp


def _random_actions(env, rng) -> np.ndarray:
    avail = env.avail_actions()
    return np.array(
        [[int(rng.choice(np.flatnonzero(row))) for row in avail[b]] for b in range(len(avail))]
    )


@pytest.mark.parametrize('name', ['3m', 'MMM', '2vr_vs_3sc', '7q_vs_2bc', 'mmmt_vs_zhb'])
def test_portable_pyref_runs(name):
    """Run 3+ episodes on pyref with batch=4; scripted opponent samples from scenario pool."""
    env = make_vec_env('smac-hard-portable', name, backend='pyref', batch_size=4, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    done_eps = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(300):
            assert set(_scripted(env).names) == {0, 1, 2, 3}
            assert set(_scripted(env).names.values()) <= set(POOLS[name])
            step = env.step(_random_actions(env, rng))
            done = step.terminated | step.truncated
            if done.any():
                done_eps += int(done.sum())
                env.reset(np.flatnonzero(done).tolist())
    assert done_eps >= 3


def _run(env, rng, n: int) -> None:
    env.reset()
    for _ in range(n):
        step = env.step(_random_actions(env, rng))
        if (step.terminated | step.truncated).any():
            env.reset()


@pytest.mark.parametrize('mode', ['drop', 'issue'])
def test_portable_script_abilities(mode):
    """mmmt_zspi_bio issues non-heal abilities: 'drop' ignores them on pyref; 'issue' raises
    CapabilityError (pyref supports heal only)."""
    over = {'opponent.scripts': ('mmmt_zspi_bio',), 'opponent.script_abilities': mode}
    env = make_vec_env(
        'smac-hard-portable', 'mmmt_vs_zspi', backend='pyref', seed=0, overrides=over
    )
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if mode == 'drop':
            _run(env, rng, 120)
        else:
            with pytest.raises(CapabilityError, match='ability'):
                _run(env, rng, 120)


def test_use_ability_needs_ability_query():
    with pytest.raises(CapabilityError, match='use_ability=False'):
        make_vec_env('smac-hard-portable', '3m', backend='pyref', overrides={'use_ability': True})


@pytest.mark.parametrize('name', ['3s_vs_3z', '2m_vs_1z', 'bane_vs_bane', 'unit_test'])
def test_unavailable_scenarios(name):
    with pytest.raises((CapabilityError, ValueError)):
        make_vec_env('smac-hard-portable', name, backend='pyref')


def test_script_pool_override_and_unknown():
    env = make_vec_env(
        'smac-hard-portable', '3m', backend='pyref', overrides={'opponent.scripts': ('atkn',)}
    )
    env.reset()
    assert _scripted(env).names == {0: 'atkn'}
    with pytest.raises(ValueError, match='(?i)unknown script'):
        make_vec_env(
            'smac-hard-portable', '3m', backend='pyref', overrides={'opponent.scripts': ('nope',)}
        )


def test_registered_smac_hard_scenarios():
    names = scenarios.names('smac-hard')
    assert len(names) == 35 and {'3m', 'mmmt_vs_zspi', 'pvt_large'} <= set(names)


# ---------------------------------------------------------------- SC2
def _episode(env, rng) -> tuple[int, dict]:
    obs, state = env.reset()
    assert len(obs) == env.n_agents and len(obs[0]) == env.get_obs_size()
    assert len(state) == env.get_state_size()
    terminated, steps, info = False, 0, {}
    while not terminated:
        acts = [int(rng.choice(np.flatnonzero(a))) for a in env.get_avail_actions()]
        _, terminated, info = env.step(acts)
        steps += 1
    return steps, info


@pytest.mark.sc2
def test_smac_hard_3m_sc2_duel():
    """smac-hard 3m: two-process duel (two AgentPlayers), scripted opponent, 2 episodes."""
    env = make_pymarl_env('smac-hard', '3m', backend='sc2', seed=0)
    try:
        # Upstream sizing: padding = max(3, 3, 9) = 9, n_actions = 6 + 9 + 9
        assert (env.get_obs_size(), env.get_state_size(), env.get_total_actions()) == (
            30,
            48 + 3 * 15,
            24,
        )
        assert isinstance(env, PyMARLEnv)
        vec = env.vec
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            steps, info = _episode(env, np.random.default_rng(0))
            session = vec.runtime.backend.session  # type: ignore[attr-defined]
            assert session is not None and len(session.controllers) == 2
            assert 0 < steps <= env.episode_limit and 'battle_won' in info
            first = _scripted(vec).names[0]
            steps, info = _episode(env, np.random.default_rng(1))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
        assert env.get_stats()['battles_game'] == 2
        assert first in POOLS['3m'] and _scripted(vec).names[0] in POOLS['3m']
    finally:
        env.close()


@pytest.mark.sc2
def test_smac_hard_portable_3m_sc2():
    """smac-hard-portable on SC2: explicit spawns (spawns_hard.json) and two AgentPlayers."""
    env = make_pymarl_env('smac-hard-portable', '3m', backend='sc2', seed=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            steps, info = _episode(env, np.random.default_rng(0))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
    finally:
        env.close()
