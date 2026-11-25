"""Public API surface tests for UniSMAC (packages, profiles, CLI, VecEnv, and zgame metadata)."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from typing import Any, Literal

import numpy as np
import pytest

import unismac
import unismac.profiles as profiles
from unismac.contract import CapabilityError
from unismac.env import RuntimeView, StepBatch, VecEnv, make_vec_env
from unismac.env import accel as accel_mod
from unismac.profiles import merge_overrides, profile_overrides

PROFILE_API = (
    'LEGACY_PROFILES',
    'PROFILES',
    'PUBLIC_PROFILES',
    'Profile',
    'STANDARD_EXPLORE_PROFILE',
    'STANDARD_PROFILE',
    'STANDARD_PROFILES',
    'STANDARD_SYMMETRIC_PROFILE',
    'backend_label',
    'list_profiles',
    'merge_overrides',
    'parse_pyref_options',
    'profile_overrides',
    'pyref_extra_overrides',
)


def _require_zgame() -> Any:
    from unismac.backends.zgame.backend import _module

    try:
        return _module()
    except ImportError as exc:
        pytest.skip(f'zgame extension unavailable: {exc}')


def test_package_surface() -> None:
    assert unismac.make_vec_env is make_vec_env and unismac.VecEnv is VecEnv
    assert unismac.StepBatch is StepBatch
    assert re.fullmatch(r'\d+\.\d+\.\d+', unismac.__version__)
    assert re.fullmatch(r'\d+\.\d+', unismac.CONTRACT_VERSION)
    assert issubclass(CapabilityError, Exception)
    params = inspect.signature(make_vec_env).parameters
    for name in (
        'backend',
        'profile',
        'dual',
        'batch_size',
        'seed',
        'seeds',
        'overrides',
        'zgame_threads',
        'accel',
    ):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert params['accel'].default == 'auto'
    reset = inspect.signature(VecEnv.reset).parameters
    assert reset['env_ids'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert reset['env_ids'].default is None
    for name in ('opponents', 'seeds'):
        assert reset[name].kind is inspect.Parameter.KEYWORD_ONLY and reset[name].default is None
    for name in ('step', 'obs', 'avail_actions', 'state', 'alive', 'capabilities', 'close'):
        assert callable(getattr(VecEnv, name)), name
    assert isinstance(VecEnv.runtime, property)
    for name in ('world', 'backend', 'healers', 'env_seeds'):
        assert isinstance(getattr(RuntimeView, name), property), name
    for name in (
        'frame',
        'game_loop',
        'done',
        'episode_steps',
        'episode_count',
        'reset_count',
        'max_rewards',
        'counters',
        'opponent_names',
        'opponent_errors',
        'last_scenarios',
        'rng_stream',
        'close_backend',
    ):
        assert callable(getattr(RuntimeView, name)), name
    with pytest.raises(CapabilityError):
        make_vec_env('smac-portable', '3m', backend='sc2', batch_size=2)


def test_profile_helpers() -> None:
    backend, over = profile_overrides('sc2-native')
    assert backend == 'pyref' and isinstance(over, dict) and over
    assert over['backend.pyref_options']['native_share_rule'] == 'priority_tree'
    assert over['backend.pyref_options']['native_assist_on_hit'] is True
    _, over2 = profile_overrides('aligned-v4', {'move_lag': 2})
    assert over2['backend.pyref_options']['move_lag'] == 2
    prof = {'backend.pyref_options': {'a': 1, 'b': 1}, 'x': 1}
    merged = merge_overrides(prof, {'backend.pyref_options': {'b': 2}, 'x': 2, 'y': 3})
    assert merged == {'backend.pyref_options': {'a': 1, 'b': 2}, 'x': 2, 'y': 3}
    assert merge_overrides(prof, None) == prof
    assert prof == {'backend.pyref_options': {'a': 1, 'b': 1}, 'x': 1}


def test_profiles_module_surface() -> None:
    """Verify exported names, semantic public profiles, and legacy aliases in `unismac.profiles`."""
    assert sorted(profiles.__all__) == sorted(PROFILE_API)
    assert all(hasattr(profiles, name) for name in PROFILE_API)
    assert profiles.STANDARD_PROFILE == 'sc2-native'
    assert profiles.STANDARD_EXPLORE_PROFILE == 'sc2-explore'
    assert profiles.STANDARD_SYMMETRIC_PROFILE == 'sc2-symmetric'
    assert profiles.STANDARD_PROFILES == (
        profiles.STANDARD_PROFILE,
        profiles.STANDARD_EXPLORE_PROFILE,
        profiles.STANDARD_SYMMETRIC_PROFILE,
    )
    assert profiles.list_profiles() == [
        'sc2-native',
        'sc2-explore',
        'sc2-symmetric',
        'sc2-live',
        'minimal',
    ]
    assert len(profiles.PUBLIC_PROFILES) == 5
    assert len(profiles.LEGACY_PROFILES) == 17
    assert len(profiles.list_profiles(include_legacy=True)) == 22
    assert profile_overrides('sc2-native') == profile_overrides('rghf-nat-cbt')
    assert profile_overrides('sc2-explore') == profile_overrides('rghf-nat')
    assert profile_overrides('sc2-symmetric') == profile_overrides('rghf-nat-cbt-sym')
    assert profile_overrides('sc2-live') == profile_overrides('sc2')
    assert profile_overrides('minimal') == profile_overrides('preset')
    for name in (
        'sc2-live',
        'minimal',
        'sc2',
        'preset',
        'aligned-v4',
        'acc-ram-cov-axo',
        'acc-ram-cov-axo-rghf',
        *profiles.STANDARD_PROFILES,
    ):
        assert isinstance(profiles.PROFILES[name], profiles.Profile), name
    assert profiles.backend_label('aligned-v4', 'zgame') == 'zgame-aligned-v4'
    assert profiles.backend_label(profiles.STANDARD_PROFILE, 'zgame') == 'zgame-sc2-native'
    assert profiles.backend_label(profiles.STANDARD_EXPLORE_PROFILE, 'zgame') == 'zgame-sc2-explore'
    assert profiles.backend_label('sc2-live') == 'sc2'
    assert profiles.backend_label('sc2') == 'sc2'
    assert profiles.parse_pyref_options(None) is None


def test_cli_surface(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify public CLI subcommands (`info`, `list-scenarios`, `list-profiles`, `bench`)."""
    from unismac import cli

    assert cli.main([]) == 0
    assert 'unismac' in capsys.readouterr().out.lower()

    assert cli.main(['info']) == 0
    info_out = capsys.readouterr().out
    assert f'unismac: {unismac.__version__}' in info_out
    assert f'contract_version: {unismac.CONTRACT_VERSION}' in info_out

    assert cli.main(['list-profiles']) == 0
    prof_out = capsys.readouterr().out
    assert 'sc2-native' in prof_out and 'rghf-nat-cbt' not in prof_out

    assert cli.main(['list-profiles', '--all']) == 0
    all_prof_out = capsys.readouterr().out
    assert 'sc2-native' in all_prof_out and 'rghf-nat-cbt' in all_prof_out

    assert cli.main(['list-scenarios', '--preset', 'smac-portable']) == 0
    scen_out = capsys.readouterr().out
    assert '3m' in scen_out.splitlines() and '8m' in scen_out.splitlines()

    assert (
        cli.main(
            [
                'bench',
                '--preset',
                'smac-portable',
                '--scenario',
                '3m',
                '--backend',
                'pyref',
                '--profile',
                'minimal',
                '--batch-size',
                '2',
                '--steps',
                '2',
            ]
        )
        == 0
    )
    bench_out = capsys.readouterr().out
    assert 'env-steps/s' in bench_out

    with pytest.raises(SystemExit) as exc:
        cli.main(['unknown-subcommand'])
    assert exc.value.code == 2

    r = subprocess.run(
        [sys.executable, '-m', 'unismac', '--version'], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0 and r.stdout.strip() == unismac.__version__, r.stderr


def test_dual_surface() -> None:
    """Verify two-sided self-play API signatures and symmetric profile exports."""
    step = inspect.signature(VecEnv.step).parameters
    assert list(step) == ['self', 'actions', 'enemy_actions']
    assert step['enemy_actions'].default is None
    for name in ('obs', 'state', 'avail_actions', 'alive'):
        side = inspect.signature(getattr(VecEnv, name)).parameters['side']
        assert side.default == 1, name
    for name in (
        'obs_layout_for',
        'state_layout_for',
        'n_agents_for',
        'n_actions_for',
    ):
        assert list(inspect.signature(getattr(VecEnv, name)).parameters) == ['self', 'side'], name
    frame_params = inspect.signature(RuntimeView.frame).parameters
    assert list(frame_params) == ['self', 'side'] and frame_params['side'].default == 1
    assert profiles.STANDARD_SYMMETRIC_PROFILE == 'sc2-symmetric'
    assert profiles.STANDARD_SYMMETRIC_PROFILE in profiles.PROFILES
    for name in (
        'aligned-v4-sym',
        'acc-ram-cov-axo-sym',
        'rghf-nat-cbt-sym',
        profiles.STANDARD_SYMMETRIC_PROFILE,
    ):
        assert isinstance(profiles.PROFILES[name], profiles.Profile), name
    from unismac.contract import Capabilities, Scenario, mirror_facing

    assert Scenario('m').mirror is None and Capabilities.__dataclass_fields__['mirror_spawn']
    assert callable(mirror_facing) and unismac.CONTRACT_VERSION == '0.2'


def _acts(env: VecEnv, rng: np.random.Generator) -> np.ndarray:
    avail = env.avail_actions()
    return np.array([[int(rng.choice(np.flatnonzero(r))) for r in rows] for rows in avail])


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_vec_env_surface(backend: str) -> None:
    if backend == 'zgame':
        _require_zgame()
    b = 2
    env = make_vec_env('smac-portable', '3m', backend=backend, batch_size=b, seed=0)  # type: ignore[arg-type]
    a, k = env.n_agents, env.n_actions
    assert (a, env.n_enemies, k, env.episode_limit) == (3, 3, 9, 60)
    assert all(isinstance(v, int) for v in (a, k, env.episode_limit))
    assert env.config.backend.step_mul == 8
    lay = env.obs_layout
    for name in ('move', 'enemy', 'ally', 'own', 'timestep'):
        block = getattr(lay, name)
        assert isinstance(block, tuple) and all(isinstance(n, str) for n in block), name
    assert (lay.n_enemies, lay.n_allies) == (env.n_enemies, a - 1)
    size = len(lay.move) + 3 * len(lay.enemy) + 2 * len(lay.ally) + len(lay.own) + len(lay.timestep)
    assert lay.size == size
    assert env.reset() is None
    obs, state = env.obs(), env.state()
    assert obs.dtype == np.float32 and obs.shape == (b, a, lay.size)
    assert state.dtype == np.float32 and state.shape == (b, env.state_layout.size)
    avail = env.avail_actions()
    assert avail.dtype == np.int64 and avail.shape == (b, a, k)

    health = env.runtime.frame().allies.health
    assert health.dtype == np.float64 and health.shape == (b, a)
    alive = env.alive()
    assert alive.dtype == np.bool_ and alive.all()
    np.testing.assert_array_equal(alive, health > 0)
    loops = env.runtime.game_loop()
    assert loops.dtype == np.int64 and loops.shape == (b,)
    assert env.runtime.world is not None
    np.testing.assert_array_equal(loops, env.runtime.world.game_loop)
    loops[:] = -1  # fresh array copy
    assert (env.runtime.world.game_loop >= 0).all()

    rng = np.random.default_rng(0)
    start = env.runtime.game_loop()
    step = env.step(_acts(env, rng))
    assert isinstance(step, StepBatch)
    assert step.reward.dtype == np.float64 and step.reward.shape == (b,)
    assert step.terminated.dtype == np.bool_ and step.terminated.shape == (b,)
    assert step.truncated.dtype == np.bool_ and step.truncated.shape == (b,)
    assert step.reward_enemy is None
    assert isinstance(step.infos, list) and len(step.infos) == b
    assert {'battle_won', 'dead_allies', 'dead_enemies'} <= set(step.infos[0])
    np.testing.assert_array_equal(env.runtime.game_loop(), start + 8)
    for _ in range(80):
        step = env.step(_acts(env, rng))
        done_now = step.terminated | step.truncated
        np.testing.assert_array_equal(env.alive(), env.runtime.frame().allies.health > 0)
        for i in np.flatnonzero(done_now):
            assert step.infos[i] == {} or step.infos[i]['dead_allies'] == int(
                (~env.alive()[i]).sum()
            )
        if done_now.any():
            env.reset(np.flatnonzero(done_now).tolist())
    assert callable(env.runtime.backend.close)

    # close_backend shuts down the backend while preserving VecEnv state until the next reset
    last = env.runtime.game_loop()
    env.runtime.close_backend()
    step = env.step(env.avail_actions().argmax(-1))
    assert not step.terminated.any() and not step.truncated.any() and env.runtime.done().all()
    assert not step.reward.any() and step.infos == [{}] * b
    np.testing.assert_array_equal(env.runtime.game_loop(), last)
    env.reset()
    obs = env.obs()
    assert obs.shape == (b, a, lay.size) and env.alive().all()
    env.step(_acts(env, rng))
    env.close()


@pytest.mark.parametrize('accel', ['auto', 'services'])
def test_opponent_names_and_errors(accel: Literal['auto', 'services']) -> None:
    if accel == 'services' and not accel_mod.native_available():
        pytest.skip('zgame native env services unavailable')
    env = make_vec_env(
        'smac-hard-portable', '2s3z', backend='pyref', batch_size=2, seed=1, accel=accel
    )
    env.reset()
    opp: Any = env._opponent
    assert isinstance(opp.names, dict) and isinstance(opp.errors, dict)
    assert env.runtime.opponent_names() == [opp.names[0], opp.names[1]]
    assert all(n in opp.pool for n in env.runtime.opponent_names())
    errs = env.runtime.opponent_errors()
    assert errs.dtype == np.int64 and errs.tolist() == [0, 0]
    if env.accel == 'rust':
        assert opp.scripts == {}
        return

    def boom(*args: Any) -> None:
        raise ZeroDivisionError

    opp.scripts[1].script = boom
    with pytest.warns(RuntimeWarning):
        env.step(_acts(env, np.random.default_rng(0)))
    assert opp.errors == {1: 1} and env.runtime.opponent_errors().tolist() == [0, 1]

    plain = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=1)
    assert plain.runtime.opponent_names() == [None, None]
    plain.reset()
    assert plain.runtime.opponent_names() == [None, None]
    assert plain.runtime.opponent_errors().tolist() == [0, 0]


def test_sc2_backend_session_surface() -> None:
    """`Sc2Backend.session` and `close()` work without launching StarCraft II."""
    from unismac.backends.sc2 import Sc2Backend

    assert isinstance(Sc2Backend.session, property)
    be = Sc2Backend()
    assert be.session is None
    be.close()
    assert be.session is None


def test_zgame_build_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """`backends.zgame.build_info()` reads metadata directly from `zgame._zgame.build_info()`."""
    from unismac.backends.zgame import build_info

    z = _require_zgame()
    raw = z.build_info()
    assert set(raw) == {'version', 'commit', 'dirty', 'built', 'profile'}
    assert raw['version'] == z.__version__ and raw['profile'] in ('release', 'debug')
    assert raw['commit'] is None or re.fullmatch(r'[0-9a-f]{7,40}', raw['commit'])
    assert raw['dirty'] in (None, True, False)
    assert raw['built'] is None or re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', raw['built'])
    build_info.cache_clear()
    try:
        info = build_info()
        assert {'version', 'module', 'sha256', 'commit', 'dirty', 'built', 'source'} == set(info)
        assert info['source'] == 'module'
        assert (info['commit'], info['dirty'], info['built']) == (
            raw['commit'],
            raw['dirty'],
            raw['built'],
        )
        monkeypatch.delattr(z, 'build_info')
        build_info.cache_clear()
        old = build_info()
        assert old['source'] is None
        assert old['commit'] is None and old['dirty'] is None and old['built'] is None
    finally:
        build_info.cache_clear()
