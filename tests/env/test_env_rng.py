"""`unismac.env.semantics.rng`: stream key/counter layout and bitwise reset reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from unismac.env import make_vec_env
from unismac.env.semantics import rng as rng_mod
from unismac.env.semantics.generators import ScenarioGenerator

GEN_SCENARIO = '10gen_terran'
STOCH = {
    'capability_config': {
        'n_units': 5,
        'n_enemies': 6,
        'team_gen': {
            'unit_types': ['marine', 'marauder', 'medivac'],
            'weights': [0.45, 0.45, 0.1],
            'exception_unit_types': ['medivac'],
        },
        'start_positions': {'dist_type': 'surrounded_and_reflect'},
        'attack': {'lower_bound': [0.3], 'upper_bound': [0.9], 'observe': True},
        'enemy_mask': {'mask_probability': 0.3},
    },
    'prob_obs_enemy': 0.5,
    'conic_fov': True,
    'quirks.obs_shuffle': True,
}


def test_stream_is_pure_function_of_key():
    a = rng_mod.stream(3, 'generators', 2, 5, 7).random(16)
    b = rng_mod.stream(3, 'generators', 2, 5, 7).random(16)
    np.testing.assert_array_equal(a, b)
    others = [
        rng_mod.stream(4, 'generators', 2, 5, 7),
        rng_mod.stream(3, 'attack', 2, 5, 7),
        rng_mod.stream(3, 'generators', 3, 5, 7),
        rng_mod.stream(3, 'generators', 2, 6, 7),
        rng_mod.stream(3, 'generators', 2, 5, 8),
    ]
    for g in others:
        assert not np.array_equal(g.random(16), a)
    # Seeds use low 64 bits: negative values match their two's complement
    np.testing.assert_array_equal(
        rng_mod.stream(-1, 'attack', 0, 0).random(4),
        rng_mod.stream(2**64 - 1, 'attack', 0, 0).random(4),
    )


def test_stream_bad_args():
    with pytest.raises(ValueError):
        rng_mod.stream(0, 'nope', 0, 0)
    with pytest.raises(ValueError):
        rng_mod.stream(0, 'attack', -1, 0)


def test_shuffle_is_permutation():
    items = list(range(10))
    rng_mod.shuffle(rng_mod.stream(0, 'obs_shuffle', 0, 0), items)
    assert sorted(items) == list(range(10)) and items != list(range(10))


def _rollout(batch: int, seed: int, n_steps: int, partial: bool):
    env = make_vec_env(
        'smacv2-portable',
        GEN_SCENARIO,
        backend='pyref',
        batch_size=batch,
        seed=seed,
        overrides=STOCH,
    )
    act_rng = np.random.default_rng(0)
    out: list[tuple[np.ndarray, ...]] = []
    env.reset()
    out.append((env.get_obs().copy(), env.get_state().copy()))
    spawns = [dict(env.last_scenarios)]
    for t in range(n_steps):
        avail = env.get_avail_actions()
        acts = np.array([[act_rng.choice(np.flatnonzero(r)) for r in rows] for rows in avail])
        reward, done, _ = env.step(acts)
        out.append((reward.copy(), done.copy(), env.get_obs().copy(), env.get_state().copy()))
        ids = np.flatnonzero(done).tolist()
        if partial and t % 7 == 6:
            ids = sorted(set(ids) | {t % batch})
        if ids:
            env.reset(ids)
            spawns.append(dict(env.last_scenarios))
            out.append((env.get_obs().copy(), env.get_state().copy()))
    env.close()
    return env, out, spawns


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for u, v in zip(x, y, strict=True):
            np.testing.assert_array_equal(u, v)


def test_rollout_reproducible_with_partial_resets():
    """Same seed and action sequence yield bitwise-identical obs/state/reward/done and resets
    (stochastic_attack, spotter prob_obs_enemy, obs shuffle, and scenario gen all use streams)."""
    _, a, sa = _rollout(4, 11, 60, partial=True)
    _, b, sb = _rollout(4, 11, 60, partial=True)
    _same(a, b)
    assert sa == sb
    _, c, _ = _rollout(4, 12, 60, partial=True)
    assert not all(np.array_equal(x[0], y[0]) for x, y in zip(a, c, strict=False))


def test_scenarios_keyed_by_env_and_episode():
    """Reset k of env b depends only on (seed, b, k), independent of batch size or other resets."""
    seed = 21
    env = make_vec_env(
        'smacv2-portable', GEN_SCENARIO, backend='pyref', batch_size=3, seed=seed, overrides=STOCH
    )
    gen = ScenarioGenerator(env.config.capability, env.n_agents, env.n_enemies)

    def expected(b, k):
        return env._generated_scenario(gen.draw(rng_mod.stream(seed, 'generators', b, k)))

    env.reset()
    first = dict(env.last_scenarios)
    assert env.attack_prob is not None and env.enemy_mask is not None
    attack0 = env.attack_prob.copy()
    assert all(first[b] == expected(b, 0) for b in range(3))
    assert len({first[b] for b in range(3)}) == 3
    env.reset([1])
    assert env.last_scenarios == {1: expected(1, 1)}
    np.testing.assert_array_equal(env.attack_prob[[0, 2]], attack0[[0, 2]])
    assert not np.array_equal(env.attack_prob[1], attack0[1])
    env.reset([1, 2])
    assert env.last_scenarios == {1: expected(1, 2), 2: expected(2, 1)}
    assert env.reset_count.tolist() == [1, 3, 2]

    single = make_vec_env(
        'smacv2-portable', GEN_SCENARIO, backend='pyref', seed=seed, overrides=STOCH
    )
    single.reset()
    assert single.attack_prob is not None and single.enemy_mask is not None
    assert single.last_scenarios[0] == first[0]
    np.testing.assert_array_equal(single.attack_prob[0], attack0[0])
    np.testing.assert_array_equal(single.enemy_mask[0], env.enemy_mask[0])
    env.close()
    single.close()


def test_unseeded_env_draws_entropy():
    a = make_vec_env('smacv2-portable', GEN_SCENARIO, backend='pyref')
    b = make_vec_env('smacv2-portable', GEN_SCENARIO, backend='pyref')
    a.reset()
    b.reset()
    assert a._rng_seed != b._rng_seed
