"""Smoke tests for the env layer on real SC2 (`pytest -m sc2`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.contract import CapabilityError
from unismac.env import make_env

pytestmark = pytest.mark.sc2


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


def test_smac_3m_random_episode():
    env = make_env('smac', '3m', backend='sc2', seed=0)
    try:
        assert env.get_env_info()['obs_shape'] == 30
        steps, info = _episode(env, np.random.default_rng(0))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
        steps, _ = _episode(env, np.random.default_rng(1))  # restart protocol
        assert env.get_stats()['battles_game'] == 2
    finally:
        env.close()


def test_smac_portable_3m_random_episode():
    env = make_env('smac-portable', '3m', backend='sc2', seed=0)
    try:
        assert (env.get_obs_size(), env.get_state_size(), env.get_total_actions()) == (30, 48, 9)
        steps, info = _episode(env, np.random.default_rng(0))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
    finally:
        env.close()


def test_smac_portable_mmm_heal():
    """smac-portable MMM: standard Medivac starts at 50 energy and sends heal commands.

    Engine responses match smac (Medivac_RL): Success, TooMuchLife (full HP), or
    AlreadyTargeted (already healing); upstream heal masking likewise ignores target HP."""
    from s2clientprotocol import error_pb2

    from unismac.contract import CommandKind
    from unismac.env import make_vec_env

    allowed = {
        error_pb2.Success,
        error_pb2.TooMuchLife,
        error_pb2.AlreadyTargeted,
    }
    env = make_vec_env('smac-portable', 'MMM', backend='sc2', seed=0)
    try:
        env.reset()
        world = env._world
        assert world is not None
        medivac = world.alive[0] & (world.unit_type[0] == 54)
        assert medivac.sum() == 2
        assert np.all(np.abs(world.energy[0, medivac] - 50.0) < 0.5)  # 1 loop ran on reset
        rng = np.random.default_rng(0)
        backend: Any = env.backend  # Sc2Backend-specific last_action_results
        results: list[int] = []
        terminated, info = False, [{}]
        while not terminated:
            acts = _heal_first(env, rng)
            _, done, info = env.step(acts)
            terminated = bool(done[0])
            results += [r for _, k, r in backend.last_action_results if k == CommandKind.ABILITY]
        assert 'battle_won' in info[0]
        assert results and set(results) <= allowed, results
        assert error_pb2.Success in results
    finally:
        env.close()


def _heal_first(env, rng) -> np.ndarray:
    """Random actions; redirect Medivac heals to injured targets for reliable Success."""
    avail = env.get_avail_actions()[0]
    allies = env._frame.allies
    heals = env.actions.heals(env._ally_tables)[0]
    n_no = env.actions.n_no_attack
    acts = []
    for a, row in enumerate(avail):
        act = int(rng.choice(np.flatnonzero(row)))
        if act >= n_no and heals[a]:
            targets = np.flatnonzero(row[n_no : n_no + env.n_agents])
            hurt = [t for t in targets if 0 < allies.health[0, t] < allies.health_max[0, t]]
            if hurt:
                act = n_no + int(rng.choice(hurt))
        acts.append(act)
    return np.array([acts])


def test_smacv2_10gen_terran_random_episode():
    """smacv2 10gen_terran 5v5: explicit spawns (*_RL allies + std enemies) vs engine_ai."""
    env = make_env('smacv2', '10gen_terran', backend='sc2', seed=0)
    try:
        info = env.get_env_info()
        assert (info['obs_shape'], info['state_shape'], info['n_actions']) == (82, 120, 11)
        assert info['cap_shape'] == 3
        steps, info = _episode(env, np.random.default_rng(0))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
        caps = env.get_capabilities().reshape(env.n_agents, -1)
        assert (caps.sum(-1) == 1).all()
        steps, _ = _episode(env, np.random.default_rng(1))  # 2nd ep: kill and respawn
        assert env.get_stats()['battles_game'] == 2
    finally:
        env.close()


def test_smacv2_3m_random_episode():
    env = make_env('smacv2', '3m', backend='sc2', seed=0)
    try:
        steps, info = _episode(env, np.random.default_rng(0))
        assert 0 < steps <= env.episode_limit and 'battle_won' in info
    finally:
        env.close()


def test_smacv2_portable_10gen_random_episode():
    """smacv2-portable 10gen: opening_orders needs 2 user slots, while SMAC_Maps/32x32_flat
    has only 1 (JoinGame NoSpaceForUser) and new_maps lacks 32x32_flat: CapabilityError."""
    with pytest.raises(CapabilityError, match='new_maps'):
        make_env('smacv2-portable', '10gen_protoss', backend='sc2', seed=0)
