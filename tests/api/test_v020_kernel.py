"""Purified v0.2.0 environment kernel invariant tests.

Verifies:
1. Top-level `unismac` and `unismac.env` expose only the purified kernel (`StepBatch`, `VecEnv`,
   `RuntimeView`, `make_vec_env`) with `UnismacEnv` and `make_env` cleanly removed.
2. AST static check proving `VecEnv` has zero public-to-public and zero private-to-public calls.
3. `StepBatch` immutability, non-iterability, `terminated` vs `truncated` separation, and frozen
   slot `(False, False)` semantics.
4. SMACv2 `_spotter_stamp` same-frame idempotency across partial resets and first-call partial
   `reset([0])`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import unismac
import unismac.env as env_pkg
from unismac import StepBatch, VecEnv, make_vec_env
from unismac.env import RuntimeView


def test_v020_top_level_surface_purified() -> None:
    """Top-level `unismac` and `unismac.env` drop legacy `UnismacEnv` and `make_env`."""
    assert unismac.__version__ == '0.2.0'
    assert unismac.StepBatch is StepBatch
    assert not hasattr(unismac, 'UnismacEnv')
    assert not hasattr(unismac, 'make_env')
    assert not hasattr(env_pkg, 'UnismacEnv')
    assert not hasattr(env_pkg, 'make_env')
    assert not (Path(unismac.__file__).resolve().parent / 'env' / 'single.py').exists()

    expected_top = {
        'CONTRACT_VERSION',
        'STANDARD_PROFILE',
        'StepBatch',
        'VecEnv',
        'list_profiles',
        'make_vec_env',
    }
    assert set(unismac.__all__) == expected_top
    assert RuntimeView is env_pkg.RuntimeView

    removed_pymarl_methods = (
        'get_obs',
        'get_state',
        'get_avail_actions',
        'get_capabilities',
        'get_obs_size',
        'get_state_size',
        'get_total_actions',
        'get_env_info',
        'get_cap_size',
        'get_stats',
        'max_reward',
    )
    for name in removed_pymarl_methods:
        assert not hasattr(VecEnv, name), f'legacy method still on VecEnv: {name}'


def test_vec_env_ast_zero_public_intercalls() -> None:
    """AST invariant: no method on `VecEnv` calls another public method on `self`."""
    core_path = Path(unismac.__file__).resolve().parent / 'env' / 'core.py'
    tree = ast.parse(core_path.read_text(encoding='utf-8'), filename=str(core_path))
    vec_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VecEnv')
    public_methods = {
        item.name
        for item in vec_cls.body
        if isinstance(item, ast.FunctionDef)
        and not item.name.startswith('_')
        and not any(isinstance(d, ast.Name) and d.id == 'property' for d in item.decorator_list)
    }
    assert {'reset', 'step', 'obs', 'state', 'avail_actions', 'alive', 'capabilities', 'close'} <= (
        public_methods
    )

    violations: list[str] = []
    for item in vec_cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if item.name == '__exit__':
            continue
        for node in ast.walk(item):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'self'
                and node.func.attr in public_methods
            ):
                violations.append(f'VecEnv.{item.name}:L{node.lineno} -> self.{node.func.attr}()')
    assert not violations, '\n'.join(violations)


def test_step_batch_contract_and_terminated_vs_truncated() -> None:
    """`StepBatch` is frozen, non-iterable, splits terminated/truncated, and clears frozen slots."""
    with make_vec_env(
        'smac-portable',
        '3m',
        backend='pyref',
        batch_size=2,
        seed=0,
        overrides={'termination.episode_limit': 3},
    ) as env:
        assert env.reset() is None
        assert isinstance(env.runtime, RuntimeView)
        assert not env.runtime.done().any()

        stop = np.ones((2, 3), dtype=np.int64)
        s1 = env.step(stop)
        assert isinstance(s1, StepBatch)
        assert s1.reward_enemy is None
        with pytest.raises(TypeError):
            iter(s1)  # type: ignore[arg-type]

        env.step(stop)
        s3 = env.step(stop)
        # Step 3 reaches episode_limit: truncated=True, terminated=False.
        assert not s3.terminated.any() and s3.truncated.all()
        assert env.runtime.done().all()

        # Stepping already-frozen slots emits reward=0, terminated=False, truncated=False, info={}.
        s4 = env.step(stop)
        assert np.all(s4.reward == 0.0)
        assert not s4.terminated.any() and not s4.truncated.any()
        assert s4.infos == [{}, {}]
        assert env.runtime.done().all()


def test_smacv2_first_call_partial_reset_and_spotter_idempotency() -> None:
    """First-call `reset([0])` does not crash, and partial reset keeps other slots' spotter gate."""
    stoch = {
        'capability_config': {
            'n_units': 5,
            'n_enemies': 5,
            'team_gen': {
                'unit_types': ['marine', 'marauder', 'medivac'],
                'weights': [0.45, 0.45, 0.1],
                'exception_unit_types': ['medivac'],
            },
            'start_positions': {'dist_type': 'surrounded_and_reflect'},
        },
        'prob_obs_enemy': 0.5,
    }
    # 1. First-call partial reset on SMACv2 must not crash with episode=-1.
    with make_vec_env(
        'smacv2-portable', '10gen_terran', backend='pyref', batch_size=2, seed=5, overrides=stoch
    ) as env:
        env.reset([0])
        assert env.runtime.reset_count().tolist() == [1, 0]

    # 2. Same-frame spotter idempotency: resetting slot 1 keeps slot 0 obs & gate bitwise equal.
    with make_vec_env(
        'smacv2-portable', '10gen_terran', backend='pyref', batch_size=2, seed=7, overrides=stoch
    ) as env:
        env.reset()
        rng = np.random.default_rng(1)
        for _ in range(5):
            avail = env.avail_actions()
            acts = np.array([[int(rng.choice(np.flatnonzero(r))) for r in rows] for rows in avail])
            env.step(acts)
        obs0_before = env.obs()[0].copy()
        state0_before = env.state()[0].copy()
        env.reset([1])
        np.testing.assert_array_equal(env.obs()[0], obs0_before)
        np.testing.assert_array_equal(env.state()[0], state0_before)
