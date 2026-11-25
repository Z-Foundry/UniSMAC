"""Per-env seeding: `make_vec_env(seeds=...)` and `reset(..., seeds=...)`.

Checks that row b of a batch=4 `VecEnv` matches a standalone batch=1 env with `seeds[b]` bitwise
under identical actions (including procedural generation, stochastic_attack, spotter, obs
shuffle, backend seeds, script draws, and re-seeding on reset) across pyref, zgame, and all
three portable presets. Unseeded invariance is tested in `test_env_invariance.py`.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from unismac.contract import ContractError, WorldState
from unismac.env import VecEnv, make_vec_env
from unismac.env import rng as rng_mod
from unismac.profiles import profile_overrides

# All SMACv2 stochastic features enabled (same as test_env_rng)
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
SEEDS = [11, 7, 123456789, 2**40 + 5]
# (preset, scenario, overrides or profile, n_steps). acc-ram-cov-axo enables backend RNG
# (cooldown jitter, spawn facing) to verify per-env backend seeding.
CASES: list[tuple[str, str, Any, int]] = [
    ('smac-portable', '3m', None, 120),
    ('smac-portable', '3m', 'acc-ram-cov-axo', 70),
    ('smacv2-portable', '10gen_protoss', None, 120),
    ('smacv2-portable', '10gen_terran', STOCH, 120),
    ('smac-hard-portable', '2s3z', None, 120),
    ('smac-hard-portable', 'MMM', None, 100),
]


def _overrides(over: Any) -> dict[str, Any] | None:
    return profile_overrides(over)[1] if isinstance(over, str) else over


def _same_row(big: VecEnv, b: int, one: VecEnv) -> None:
    """Assert row b of big matches batch=1 env one bitwise (including all WorldState fields)."""
    pairs = [
        ('obs', big.get_obs()[b], one.get_obs()[0]),
        ('state', big.get_state()[b], one.get_state()[0]),
        ('avail', big.get_avail_actions()[b], one.get_avail_actions()[0]),
        ('max_reward', big.max_rewards[b], one.max_rewards[0]),
        ('reset_count', big.reset_count[b], one.reset_count[0]),
    ]
    pairs += [
        (f.name, getattr(big.world, f.name)[b], getattr(one.world, f.name)[0])
        for f in fields(WorldState)
    ]
    for name, x, y in pairs:
        np.testing.assert_array_equal(x, y, err_msg=f'env {b} {name}')
    assert big.opponent_names()[b] == one.opponent_names()[0]
    if b in big.last_scenarios and 0 in one.last_scenarios:
        assert big.last_scenarios[b] == one.last_scenarios[0]


def _actions(rngs: list[np.random.Generator], avail: np.ndarray) -> np.ndarray:
    return np.array(
        [[int(rngs[b].choice(np.flatnonzero(r))) for r in avail[b]] for b in range(len(rngs))]
    )


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
@pytest.mark.parametrize(
    ('preset', 'scenario', 'over', 'n_steps'),
    CASES,
    ids=[f'{p}/{s}' + ('/axo' if o == 'acc-ram-cov-axo' else '') for p, s, o, _ in CASES],
)
def test_batch_rows_match_single_envs(
    preset: str, scenario: str, over: Any, n_steps: int, backend: str
) -> None:
    """Each batch row with per-env seeds matches a standalone `batch_size=1` env bitwise."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    k = len(SEEDS)
    over = _overrides(over)
    kw: dict[str, Any] = {'backend': backend, 'overrides': over}
    big = make_vec_env(preset, scenario, batch_size=k, seeds=SEEDS, **kw)
    by_seed: list[dict[str, Any]] = [
        {'seed': s} if b % 2 == 0 else {'seeds': [s]} for b, s in enumerate(SEEDS)
    ]
    ones = [make_vec_env(preset, scenario, **one, **kw) for one in by_seed]
    scripted = big.config.opponent.kind == 'scripted'
    rngs = [np.random.default_rng(100 + b) for b in range(k)]
    big.reset()
    for one in ones:
        one.reset()
    for b in range(k):
        _same_row(big, b, ones[b])
    resets = 0
    for t in range(n_steps):
        acts = _actions(rngs, big.get_avail_actions())
        reward, term, infos = big.step(acts)
        for b, one in enumerate(ones):
            r1, d1, i1 = one.step(acts[b : b + 1])
            assert (r1[0], d1[0], i1[0]) == (reward[b], term[b], infos[b]), f'step {t} env {b}'
            _same_row(big, b, one)
        ids = set(np.flatnonzero(term).tolist())
        new_seeds: dict[int, int] = {}
        names: dict[int, str] = {}
        if t == 30:
            ids |= {1, 3}
            new_seeds = {1: 9001, 3: 9003}
        elif t % 17 == 16:
            ids.add(t % k)
        if t == 45 and scripted:
            ids.add(0)
            names = {0: 'atkn'}
        if ids:
            order = sorted(ids)
            big.reset(
                order,
                seeds=[new_seeds.get(b) for b in order],
                opponents=[names.get(b) for b in order] if names else None,
            )
            for b in order:
                ones[b].reset(
                    seeds=[new_seeds.get(b)],
                    opponents=[names[b]] if b in names else None,
                )
                _same_row(big, b, ones[b])
            if names:
                assert big.opponent_names()[0] == 'atkn'
            resets += len(order)
    assert resets >= 8
    assert big.env_seeds == (SEEDS[0], 9001, SEEDS[2], 9003)
    big.close()
    for one in ones:
        one.close()


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_reseeded_row_matches_fresh_env(backend: str) -> None:
    """Re-seeding a single slot via `reset([1], seeds=[77])` matches a fresh `seed=77` env."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    kw: dict[str, Any] = {'backend': backend, 'overrides': STOCH}
    env = make_vec_env('smacv2-portable', '10gen_terran', batch_size=3, seed=5, **kw)
    ref = make_vec_env('smacv2-portable', '10gen_terran', batch_size=3, seed=5, **kw)
    rngs = [np.random.default_rng(b) for b in range(3)]
    env.reset()
    ref.reset()
    for _ in range(4):
        acts = _actions(rngs, env.get_avail_actions())
        env.step(acts)
        ref.step(acts)
    env.reset([1], seeds=[77])
    ref.reset([1])
    fresh = make_vec_env('smacv2-portable', '10gen_terran', seed=77, **kw)
    fresh.reset()
    assert env.env_seeds == (None, 77, None) and env.reset_count.tolist() == [1, 1, 1]
    ref_rng = np.random.default_rng(9)
    for t in range(12):
        _same_row(env, 1, fresh)
        for b in (0, 2):
            np.testing.assert_array_equal(env.get_obs()[b], ref.get_obs()[b])
            np.testing.assert_array_equal(env.world.x[b], ref.world.x[b])
        acts = _actions(rngs, env.get_avail_actions())
        _, term, _ = env.step(acts)
        fresh.step(acts[1:2])
        acts[1] = [int(ref_rng.choice(np.flatnonzero(r))) for r in ref.get_avail_actions()[1]]
        ref.step(acts)
        assert not term.any(), t


def test_backend_seeds_per_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend seeds: default `seed + b` (reading `env.seed` on reset); per-env seeded slots use
    their own seed; slots re-seeded on reset keep the new seed thereafter."""

    def recorder(env: VecEnv) -> list[list[int | None]]:
        calls: list[list[int | None]] = []
        orig = env.backend.reset

        def reset(scenarios: Any, *, env_ids: Any = None, seeds: Any = None) -> WorldState:
            calls.append(list(seeds))
            return orig(scenarios, env_ids=env_ids, seeds=seeds)

        monkeypatch.setattr(env.backend, 'reset', reset)
        return calls

    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=3, seed=10)
    calls = recorder(env)
    env.reset()
    env.reset([2])
    env.seed = 100
    env.reset([1])
    env.reset([0, 1], seeds=[5, None])
    env.reset([0])
    assert calls == [[10, 11, 12], [12], [101], [5, 101], [5]]

    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=3, seeds=[4, 5, 6])
    calls = recorder(env)
    env.reset()
    env.reset([1])
    assert env.seed is None and env.env_seeds == (4, 5, 6)
    assert calls == [[4, 5, 6], [5]]

    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2)
    calls = recorder(env)
    env.reset()
    assert calls == [[None, None]] and env.env_seeds == (None, None)


def test_rng_stream_keys() -> None:
    """Stream keys: per-env seeded slots use (seed, stream, 0, ep, frame); others use env_id."""
    env = make_vec_env('smacv2-portable', '10gen_terran', backend='pyref', batch_size=3, seed=21)
    env.reset()
    env.reset([0, 2])
    env.reset([2], seeds=[8])
    env.reset([2])

    def same(a: np.random.Generator, b: np.random.Generator) -> bool:
        return bool(np.array_equal(a.random(8), b.random(8)))

    assert same(env.rng_stream('attack', 0, 3), rng_mod.stream(21, 'attack', 0, 1, 3))
    assert same(env.rng_stream('generators', 1), rng_mod.stream(21, 'generators', 1, 0))
    assert same(env.rng_stream('opponent', 2, 5), rng_mod.stream(8, 'opponent', 0, 1, 5))


# Procedural gen on non-*_gen maps: catalog min_code is fit from episode 0 allies (seeds 2 and 8
# spawn all Zealots on ep 0, giving min_code=1 and upstream-shifted Zealot cooldown_max=35).
TEAM_GEN_2S3Z = {
    'capability_config': {
        'n_units': 3,
        'n_enemies': 3,
        'team_gen': {'unit_types': ['stalker', 'zealot'], 'weights': [0.3, 0.7]},
        'start_positions': {'dist_type': 'surrounded_and_reflect'},
    }
}


def test_unit_catalog_fitted_per_env() -> None:
    """Per-env seeded slots fit min_code from their own ep 0 (batch-shared would differ on 1, 3)."""
    seeds = [0, 2, 3, 8]
    kw: dict[str, Any] = {'backend': 'pyref', 'overrides': TEAM_GEN_2S3Z}
    big = make_vec_env('smacv2-portable', '2s3z', batch_size=4, seeds=seeds, **kw)
    ones = [make_vec_env('smacv2-portable', '2s3z', seed=s, **kw) for s in seeds]
    big.reset()
    for one in ones:
        one.reset()
    assert [one.catalog.min_code for one in ones] == [0, 1, 0, 1]
    assert big.catalog.min_code == 0  # batch-shared catalog rule (without seeds) unchanged
    cd = big._require()[2].cooldown_max
    for b, one in enumerate(ones):
        np.testing.assert_array_equal(cd[b], one._require()[2].cooldown_max[0])
    rngs = [np.random.default_rng(b) for b in range(4)]
    for t in range(10):
        for b, one in enumerate(ones):
            _same_row(big, b, one)
        acts = _actions(rngs, big.get_avail_actions())
        _, term, _ = big.step(acts)
        for b, one in enumerate(ones):
            one.step(acts[b : b + 1])
        assert not term.any(), t

    # Default (batch seed) still fits from env 0 of first reset and shares across batch
    env = make_vec_env('smacv2-portable', '2s3z', batch_size=4, seed=3, **kw)
    env.reset()
    assert env.catalog.min_code == 0 and not env._row_catalogs


def test_seed_argument_errors() -> None:
    with pytest.raises(ValueError, match='(?i)seed.*seeds'):
        make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=1, seeds=[1, 2])
    with pytest.raises(ValueError, match='batch_size'):
        make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seeds=[1])
    with pytest.raises(TypeError):
        make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seeds=[1, 2.5])  # type: ignore[list-item]
    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=1)
    env.reset()
    before = env.reset_count.copy()
    with pytest.raises(ValueError, match='seeds'):
        env.reset([0], seeds=[1, 2])
    with pytest.raises(ContractError):
        env.reset([0, 0], seeds=[1, 2])
    with pytest.raises(ContractError):
        env.reset([2])
    np.testing.assert_array_equal(env.reset_count, before)
    assert env.env_seeds == (None, None)
