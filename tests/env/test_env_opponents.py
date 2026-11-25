"""Per-env scripted opponent selection on reset: `VecEnv.reset(opponents=...)` and `script_pool`.

Checks that pinning script X under the same seed is bitwise-identical to an episode that drew X
randomly, and leaves other envs' and later episodes' draws unchanged. Names missing from
`SCRIPTS`, outside the scenario's script pool, or non-None for non-scripted opponents raise
`ValueError` without mutating environment state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unismac.env import build_config, make_vec_env
from unismac.env.opponents.scripted import POOLS, SCRIPTS, script_pool
from unismac.env.semantics import rng as rng_mod

PRESET, SCENARIO, SEED = 'smac-hard-portable', '2s3z', 3
Plan = dict[tuple[int, int], str]


def _run(backend: str, plan: Plan, n_steps: int = 150) -> list[list[Any]]:
    """B = 3, seed=SEED, per-env action RNGs; plan[(env_id, ep)] pins a script. Returns per-env
    traces of (script_name, obs, state) on reset and (reward, term, obs, state, info) per step."""
    env = make_vec_env(PRESET, SCENARIO, backend=backend, batch_size=3, seed=SEED)  # type: ignore[arg-type]
    rngs = [np.random.default_rng(b) for b in range(3)]
    rows: list[list[Any]] = [[], [], []]

    def reset(ids: list[int]) -> None:
        counts = env.runtime.reset_count()
        given = [plan.get((b, int(counts[b]))) for b in ids]
        env.reset(ids, opponents=given if any(given) else None)
        obs, state, names = env.obs(), env.state(), env.runtime.opponent_names()
        for b in ids:
            rows[b].append((names[b], obs[b].copy(), state[b].copy()))

    reset([0, 1, 2])
    for _ in range(n_steps):
        avail = env.avail_actions()
        acts = np.array(
            [[int(rngs[b].choice(np.flatnonzero(r))) for r in avail[b]] for b in range(3)]
        )
        step = env.step(acts)
        term = step.terminated | step.truncated
        obs, state = env.obs(), env.state()
        for b in range(3):
            rows[b].append((step.reward[b], term[b], obs[b].copy(), state[b].copy(), step.infos[b]))
        ids = np.flatnonzero(term).tolist()
        if ids:
            reset(ids)
    env.close()
    return rows


def _same(a: list[Any], b: list[Any]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert len(x) == len(y)
        for u, v in zip(x, y, strict=True):
            if isinstance(u, np.ndarray):
                np.testing.assert_array_equal(u, v)
            else:
                assert u == v


def _names(rows: list[Any]) -> list[str]:
    return [r[0] for r in rows if len(r) == 3]


def _drawn(b: int, k: int) -> str:
    """Script drawn for env b, episode k when unspecified (`ScriptedOpponent` draw rule)."""
    pool = POOLS[SCENARIO]
    return pool[int(rng_mod.stream(SEED, 'opponent', b, k, 0).integers(len(pool)))]


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_named_episode_matches_drawn_episode(backend: str) -> None:
    """Pinning the script that would have been drawn yields a bitwise-identical trajectory."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    base = _run(backend, {})
    plan = {(b, k): _drawn(b, k) for b in range(3) for k in range(12)}
    for b in range(3):
        names = _names(base[b])
        assert len(names) >= 2
        assert names == [plan[(b, k)] for k in range(len(names))]
    named = _run(backend, plan)
    for b in range(3):
        _same(base[b], named[b])


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_naming_one_episode_leaves_other_draws(backend: str) -> None:
    """Pinning slot 0 leaves slots 1 and 2 and subsequent draws in slot 0 unchanged."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    base = _run(backend, {})
    other = sorted(set(POOLS[SCENARIO]) - {_drawn(0, 0)})[0]
    changed = _run(backend, {(0, 0): other})
    for b in (1, 2):
        _same(base[b], changed[b])
    a, c = _names(base[0]), _names(changed[0])
    assert c[0] == other != a[0]
    n = min(len(a), len(c))
    assert n >= 2 and c[1:n] == a[1:n]


def test_later_draws_unchanged_after_naming() -> None:
    """Reset-only check: pinning overrides only that draw; later and other envs match default."""

    def draws(first: str | None) -> list[list[str | None]]:
        env = make_vec_env(PRESET, SCENARIO, backend='pyref', batch_size=3, seed=SEED)
        env.reset(opponents=[first, None, None])
        seq = [env.runtime.opponent_names()]
        for _ in range(6):
            env.reset([0, 2])
            seq.append(env.runtime.opponent_names())
        env.close()
        return seq

    base = draws(None)
    named = draws('atkw' if base[0][0] != 'atkw' else 'atkn')
    assert named[0][0] != base[0][0]
    assert named[0][1:] == base[0][1:]
    assert named[1:] == base[1:]


def test_opponents_validation() -> None:
    env = make_vec_env(PRESET, SCENARIO, backend='pyref', batch_size=2, seed=1)
    env.reset()
    before = (env.runtime.reset_count(), env.runtime.opponent_names(), env.obs().copy())
    assert 'MMM' in SCRIPTS and 'MMM' not in POOLS[SCENARIO]
    for bad, msg in (
        (['nope', None], '(?i)unknown script'),
        ([None, 'MMM'], '(?i)script pool'),
        (['base'], 'opponents'),
    ):
        with pytest.raises(ValueError, match=msg):
            env.reset(opponents=bad)
    with pytest.raises(ValueError, match='(?i)script pool'):
        env.reset([1], opponents=['MMM'])
    np.testing.assert_array_equal(env.runtime.reset_count(), before[0])
    assert env.runtime.opponent_names() == before[1]
    np.testing.assert_array_equal(env.obs(), before[2])
    env.reset([1], opponents=['2s3z_1'])
    assert env.runtime.opponent_names()[1] == '2s3z_1'

    # User-overridden script pool: names outside the custom pool also raise
    pooled = make_vec_env(
        PRESET, SCENARIO, backend='pyref', seed=1, overrides={'opponent.scripts': ('atkn',)}
    )
    with pytest.raises(ValueError, match='(?i)script pool'):
        pooled.reset(opponents=['base'])
    pooled.reset(opponents=['atkn'])

    # Non-scripted opponent: non-None raises; all-None is equivalent to omitting
    plain = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=1)
    with pytest.raises(ValueError, match='not a scripted opponent'):
        plain.reset(opponents=['base', None])
    plain.reset(opponents=[None, None])
    obs = plain.obs()
    ref_env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=1)
    ref_env.reset()
    np.testing.assert_array_equal(obs, ref_env.obs())
    assert plain.runtime.opponent_names() == [None, None]
    ref_env.close()


def test_script_pool_public() -> None:
    """`script_pool(config.opponent, scenario)`: upstream pool (weighted) or user override."""
    config, spec = build_config(PRESET, SCENARIO)
    pool = script_pool(config.opponent, spec.name)
    assert isinstance(pool, tuple) and pool == POOLS[SCENARIO]
    assert all(isinstance(n, str) and n in SCRIPTS for n in pool)
    config2, _ = build_config(PRESET, '2c_vs_64zg')
    assert script_pool(config2.opponent, '2c_vs_64zg').count('2c_vs_64zg') == 2
    custom, _ = build_config(PRESET, SCENARIO, {'opponent.scripts': ('atkn', 'base')})
    assert script_pool(custom.opponent, SCENARIO) == ('atkn', 'base')
    with pytest.raises(ValueError):
        script_pool(config.opponent, 'so_many_baneling')
    env = make_vec_env(PRESET, SCENARIO, backend='pyref', seed=0)
    assert env._opponent.pool == pool  # type: ignore[attr-defined]
