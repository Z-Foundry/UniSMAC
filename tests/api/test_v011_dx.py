"""Tests for v0.1.1 developer-experience, API ergonomics, CLI, metadata, and manifest features."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import unismac
from unismac import cli, presets, profiles, scenarios
from unismac.assets import MapInfo, StaticData
from unismac.backends.pyref import PyrefConfig
from unismac.backends.pyref.options_meta import (
    OPTION_GROUPS,
    OPTION_TIERS,
    OPTIONS_META,
    describe_pyref_options,
)
from unismac.contract import CapabilityError
from unismac.env import UnismacEnv, build_config, make_env, make_vec_env
from unismac.env import core as core_mod
from unismac.env import single as single_mod
from unismac.env.accel import classic as classic_mod
from unismac.env.accel import manifest as manifest_mod


def test_top_level_v011_exports() -> None:
    """Top-level package exposes v0.1.1 version and convenience profile helpers."""
    assert unismac.__version__ == '0.1.1'
    assert unismac.CONTRACT_VERSION == '0.2'
    assert unismac.STANDARD_PROFILE == profiles.STANDARD_PROFILE == 'sc2-native'
    assert unismac.list_profiles is profiles.list_profiles
    assert unismac.list_profiles() == [
        'sc2-native',
        'sc2-explore',
        'sc2-symmetric',
        'sc2-live',
        'minimal',
    ]


def test_make_env_profile_dual_and_auto_backend() -> None:
    """``profile=``, ``dual=``, and ``backend='auto'`` work while preserving 0.1.0 defaults."""
    for fn in (make_vec_env, make_env):
        params = inspect.signature(fn).parameters
        assert params['profile'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['dual'].kind is inspect.Parameter.KEYWORD_ONLY

    # Default with no backend/profile/dual remains 'sc2' (0.1.0 compatibility).
    with make_vec_env('smac-portable', '3m', batch_size=1) as sc2_env:
        assert type(sc2_env.backend).__name__ == 'Sc2Backend'
    with pytest.raises(CapabilityError):
        make_vec_env('smac-portable', '3m', batch_size=2)

    # profile= resolves backend automatically without awkward tuple unpacking.
    with make_vec_env(
        'smac-portable',
        '3m',
        profile='sc2-native',
        batch_size=1,
        seed=0,
        overrides={'backend.pyref_options': {'projectile_3d': False}},
    ) as env:
        assert type(env.backend).__name__ in ('ZgameBackend', 'PyrefBackend')
        opts = dict(env.config.backend.pyref_options)
        assert opts['projectile_3d'] is False
        assert opts['native_assist_on_hit'] is True

    # Explicit backend='auto' picks zgame when installed, else pyref.
    with make_vec_env('smac-portable', '3m', backend='auto', batch_size=1, seed=0) as env:
        assert type(env.backend).__name__ in ('ZgameBackend', 'PyrefBackend')

    # dual=True configures external opponent cleanly.
    with make_vec_env(
        'smac-portable',
        '3m',
        profile='sc2-symmetric',
        backend='pyref',
        dual=True,
        batch_size=2,
        seed=0,
    ) as env:
        assert env.dual is True
        assert env.config.opponent.kind == 'external'

    # Conflicting arguments raise clear ValueError.
    with pytest.raises(ValueError, match='dual|external'):
        make_vec_env(
            'smac-portable',
            '3m',
            backend='pyref',
            dual=True,
            batch_size=1,
            overrides={'opponent.kind': 'scripted'},
        )
    with pytest.raises(ValueError, match='sc2'):
        make_vec_env('smac-portable', '3m', profile='sc2-live', backend='zgame', batch_size=1)


def test_context_manager_repr_and_unismac_env_parity() -> None:
    """``VecEnv`` and ``UnismacEnv`` support context managers, ``repr``, and aligned properties."""
    assert single_mod.UnismacEnv is UnismacEnv
    assert core_mod.UnismacEnv is UnismacEnv

    with make_vec_env(
        'smac-portable', '3m', profile='minimal', backend='pyref', batch_size=2, seed=0
    ) as vec:
        text = repr(vec)
        assert 'VecEnv' in text and '3m' in text and 'pyref' in text
        vec.reset()

    with make_env(
        'smac-portable', '3m', profile='minimal', backend='pyref', dual=True, seed=0
    ) as env:
        text = repr(env)
        assert 'UnismacEnv' in text and '3m' in text
        assert env.accel in ('python', 'services', 'rust')
        assert env.dual is True
        assert env.mirror == 'x'
        assert env.obs_layout.size == 30
        assert env.state_layout.size == 48
        assert env.n_agents_for(1) == 3
        assert env.n_actions_for(1) == 9
        env.reset()
        alive = env.alive()
        assert alive.dtype == np.bool_ and alive.shape == (3,) and alive.all()


def test_profiles_net_config_and_unknown_suggestion() -> None:
    """Public profiles stay key-equivalent while dropping redundant default resets."""
    assert profiles.profile_overrides('sc2-native') == profiles.profile_overrides('rghf-nat-cbt')
    assert profiles.profile_overrides('sc2-explore') == profiles.profile_overrides('rghf-nat')
    assert profiles.profile_overrides('sc2-symmetric') == profiles.profile_overrides(
        'rghf-nat-cbt-sym'
    )

    defaults = PyrefConfig()
    _, over = profiles.profile_overrides('sc2-native')
    for key, val in over['backend.pyref_options'].items():
        assert val != getattr(defaults, key), f'redundant default key in sc2-native: {key}'

    with pytest.raises(ValueError) as exc:
        profiles.profile_overrides('sc2-nativ')
    msg = str(exc.value)
    assert 'sc2-native' in msg
    assert 'include_legacy=True' in msg
    assert 'rghf-nat-cbt' not in msg


def test_difflib_suggestions_and_helpful_errors() -> None:
    """Unknown scenarios, presets, config keys, and PyrefConfig options emit actionable hints."""
    with pytest.raises(KeyError, match='8m_vs_9m'):
        scenarios.get('smac', '8m_vs_9')

    with pytest.raises(KeyError, match='smac-portable'):
        presets.get('smac-portabl')

    assert 'zgame' in presets.get('smac-portable').description.lower()

    with pytest.raises(ValueError, match='episode_limit'):
        build_config('smac-portable', '3m', {'termination.episode_limt': 50})

    with pytest.raises(ValueError, match='backend.pyref_options'):
        build_config('smac-portable', '3m', {'backend.pyref_options.projectile_3d': True})

    with pytest.raises(ValueError, match='projectile_3d'):
        make_vec_env(
            'smac-portable',
            '3m',
            backend='pyref',
            batch_size=1,
            overrides={'backend.pyref_options': {'projectil_3d': True}},
        )

    cfg, spec = build_config('smac-portable', '3m', {'termination.episode_limit': 12})
    blockers = classic_mod.rust_blockers(cfg, spec)
    assert blockers and any('continuing_episode' in b for b in blockers)


def test_cli_v011_subcommands_and_capability_error(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI catches ``CapabilityError`` in ``bench`` and exposes new discovery subcommands."""
    rc = cli.main(
        [
            'bench',
            '--preset',
            'smac-hard-portable',
            '--scenario',
            '3s_vs_3z',
            '--backend',
            'pyref',
            '--batch-size',
            '1',
            '--steps',
            '1',
        ]
    )
    assert rc != 0
    captured = capsys.readouterr()
    assert captured.err or captured.out

    assert cli.main(['list-scenarios', '--preset', 'smac-hard-portable', '--runnable']) == 0
    runnable_lines = set(capsys.readouterr().out.splitlines())
    assert '3m' in runnable_lines and '3s_vs_3z' not in runnable_lines

    assert cli.main(['list-scenarios', '--preset', 'smac-portable', '--details']) == 0
    details_out = capsys.readouterr().out
    assert '3m' in details_out and '8m' in details_out

    assert cli.main(['show-profile', 'sc2-native']) == 0
    show_out = capsys.readouterr().out
    assert 'sc2-native' in show_out and 'native_assist_on_hit' in show_out

    assert cli.main(['show-profile', 'sc2-native', '--raw']) == 0
    raw_out = capsys.readouterr().out
    assert 'native_assist_on_hit' in raw_out

    assert cli.main(['show-options', '--group', 'attack', '--zgame-only']) == 0
    opts_out = capsys.readouterr().out
    assert 'attack' in opts_out

    assert cli.main(['info', '--json']) == 0
    info_payload = json.loads(capsys.readouterr().out)
    assert info_payload['unismac'] == unismac.__version__
    assert 'sc2_path' in info_payload


def test_pyref_options_meta_and_native_mirror() -> None:
    """``options_meta`` covers all 164 ``PyrefConfig`` fields and ``native/`` mirrors 7 modules."""
    cfg = PyrefConfig()
    field_names = {f.name for f in dataclasses.fields(PyrefConfig)}
    assert len(field_names) == 164
    assert set(OPTIONS_META) == field_names
    for name, meta in OPTIONS_META.items():
        assert meta.name == name
        assert meta.group in OPTION_GROUPS
        assert meta.tier in OPTION_TIERS
        assert meta.default == getattr(cfg, name)
        assert isinstance(meta.zgame_supported, bool)
        assert meta.summary

    attack_opts = describe_pyref_options(group='attack')
    assert attack_opts and all(m.group == 'attack' for m in attack_opts)
    zgame_opts = describe_pyref_options(zgame_only=True)
    assert zgame_opts and all(m.zgame_supported for m in zgame_opts)
    with pytest.raises(ValueError, match='attack'):
        describe_pyref_options(group='attak')

    native_pkg = importlib.import_module('unismac.backends.pyref.subsystems.native')
    for sub in ('q12', 'wall', 'avoid', 'stop', 'air', 'steer', 'contact'):
        mod = importlib.import_module(f'unismac.backends.pyref.subsystems.native.{sub}')
        assert mod is not None
    for sym in ('steer', 'contact', 'WallMap', 'build_wall_map'):
        assert hasattr(native_pkg, sym), sym


def test_manifest_v011_cli_and_test_paths(capsys: pytest.CaptureFixture[str]) -> None:
    """Manifest CLI exposes ``--list-locked``, readable multi-file diffs, and layered test paths."""
    assert manifest_mod.main(['--list-locked']) == 0
    locked_out = capsys.readouterr().out
    assert 'env/config.py' in locked_out and 'VecEnv.step' in locked_out
    assert 'env/semantics/reward.py' in locked_out

    want = manifest_mod.current()
    got = {
        **want,
        'files': {k: ('0' * 64 if i < 7 else v) for i, (k, v) in enumerate(want['files'].items())},
    }
    diffs = manifest_mod.diff(want, got)
    assert diffs
    manifest_mod._print_cli_diff(want, got)
    cli_diff_out = capsys.readouterr().out
    assert '[sources]' in cli_diff_out and '(7):' in cli_diff_out
    assert 'python -m unismac.env.accel.manifest --write' in cli_diff_out

    env_dir = Path(unismac.__file__).resolve().parent / 'env'
    top_py = sorted(p.name for p in env_dir.glob('*.py'))
    top_dirs = sorted(p.name for p in env_dir.iterdir() if p.is_dir() and p.name != '__pycache__')
    assert top_py == ['__init__.py', 'config.py', 'core.py', 'dual.py', 'single.py']
    assert top_dirs == ['accel', 'opponents', 'semantics']
    assert (env_dir / 'accel' / 'rust_manifest.json').is_file()

    tests_root = Path(__file__).resolve().parents[1]
    assert (tests_root / 'env' / 'test_env_accel.py').is_file()
    assert (tests_root / 'env' / 'test_env_scripted_rust.py').is_file()


def test_conftest_session_fixtures(
    static_data: StaticData, map_info: Callable[[str], MapInfo]
) -> None:
    """Session-cached asset fixtures return stable shared instances."""
    assert static_data is not None
    m1 = map_info('SMAC_Maps/32x32_flat')
    m2 = map_info('SMAC_Maps/32x32_flat')
    assert m1 is m2
