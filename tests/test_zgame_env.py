"""zgame backend integrated with the env layer: make_vec_env end-to-end, pyref parity, errors."""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pytest

pytest.importorskip('zgame', reason="requires the optional 'zgame' extension")

from unismac.contract import CapabilityError
from unismac.env import VecEnv, make_vec_env


def _random_actions(rng: np.random.Generator, avail: np.ndarray) -> np.ndarray:
    out = np.zeros(avail.shape[:2], np.int64)
    for b in range(avail.shape[0]):
        for a in range(avail.shape[1]):
            out[b, a] = int(rng.choice(np.flatnonzero(avail[b, a])))
    return out


def test_portable_zgame_episode_batch8_and_partial_reset() -> None:
    env = make_vec_env('smac-portable', '3m', backend='zgame', batch_size=8, seed=0)
    assert isinstance(env, VecEnv)
    obs, state = env.reset()
    assert obs.shape == (8, 3, 30) and state.shape == (8, 48)
    rng = np.random.default_rng(0)
    t = 0
    while not env.done.all():
        _, terminated, infos = env.step(_random_actions(rng, env.get_avail_actions()))
        t += 1
        for b in np.flatnonzero(terminated):
            assert infos[b] == {} or 'battle_won' in infos[b]
    assert t <= env.episode_limit and env.get_stats()['battles_game'] == 8
    # Partially reset a subset of finished envs and step until all finish again
    before = env.get_obs().copy()
    obs, _ = env.reset(env_ids=[1, 6])
    assert not env.done[1] and not env.done[6] and env.done[0]
    np.testing.assert_array_equal(obs[0], before[0])
    while not env.done.all():
        env.step(_random_actions(rng, env.get_avail_actions()))
    assert env.get_stats()['battles_game'] == 10
    env.reset()
    assert not env.done.any()
    env.step(_random_actions(rng, env.get_avail_actions()))
    env.close()


@pytest.mark.parametrize('scenario', ['3m', '8m'])
def test_portable_env_matches_pyref(scenario: str) -> None:
    """smac-portable preset with identical seed and actions: avail, obs, state, reward,
    terminated, and wins match pyref bitwise (env features ignore facing)."""

    def run(backend: str) -> list[tuple[np.ndarray, ...]]:
        env = make_vec_env('smac-portable', scenario, backend=backend, batch_size=3, seed=4)  # type: ignore[arg-type]
        obs, state = env.reset()
        out: list[tuple[np.ndarray, ...]] = [(obs, state)]
        rng = np.random.default_rng(1)
        while not env.done.all():
            avail = env.get_avail_actions()
            reward, terminated, _ = env.step(_random_actions(rng, avail))
            out.append((avail, reward, terminated, env.get_obs(), env.get_state()))
        out.append((np.array([env.get_stats()['battles_won']]),))
        env.close()
        return out

    a, b = run('pyref'), run('zgame')
    assert len(a) == len(b)
    for xa, xb in zip(a, b, strict=True):
        for u, v in zip(xa, xb, strict=True):
            np.testing.assert_array_equal(u, v)


_M1_FIELDS: frozenset[str] = frozenset(
    {
        'commands',
        'command_delay',
        'move_lag',
        'spawn_facing',
        'kinematics',
        'rl_turn',
        'arrival_radius',
        'collision',
        'collision_iterations',
        'contact_slack',
        'idle_push_share',
        'mover_push_share',
        'mover_push_power',
        'mover_push_cap',
        'mover_speed_cap',
        'push_clearance',
        'arrival_clears_waypoint',
        'pin_engaged',
        'pin_after_fire',
        'pin_lag',
        'pin_pair_share',
        'idle_return',
        'return_threshold',
        'return_far',
        'return_delay',
        'idle_return_rule',
        'return_ratio',
        'crowd_arrival',
        'crowd_goal_tol',
        'crowd_progress',
        'crowd_loops',
        'crowd_chain',
        'crowd_cover',
        'native_relation',
        'native_enemy_relation',
        'native_idle_share',
        'native_speed_rule',
        'native_second_pass',
        'reissue_unit',
    }
)

_M2_FIELDS: frozenset[str] = _M1_FIELDS | frozenset(
    {
        'targeting',
        'auto_acquire',
        'auto_acquire_owners',
        'acquire_start_loop',
        'acquire_margin',
        'acquire_min_range',
        'scan_period',
        'scan_phase',
        'retarget',
        'retarget_busy',
        'retarget_busy_lock',
        'retarget_defer',
        'retarget_defer_memory',
        'retarget_ready',
        'retarget_ready_margin',
        'retarget_acquire',
        'retarget_lock',
        'retarget_armed_only',
        'kill_rescan_delay',
        'kill_rescan_lock',
        'kill_rescan_order',
        'scan_after_move',
        'retarget_after_move',
        'hit_assist',
        'hit_assist_range',
        'hit_assist_pick',
        'hit_assist_tie',
        'hit_assist_cap',
        'hit_keeps_kill_wait',
        'acquire_center',
        'kill_hold',
        'report_engaged_as_attack_unit',
        'retreat_owners',
        'retreat_distance',
        'kill_no_retreat',
        'kill_no_retreat_end',
        'passive_from_map',
        'attack',
        'damage',
        'chase_lock',
        'chase_lock_target',
        'follow_lock',
        'follow_slop',
        'slop_on_arrival',
        'attack_commit',
        'attack_on_arrival',
        'charge',
        'amove_needs_weapon',
        'attack_unit_needs_weapon',
        'dead_target_move',
        'slow_arrival',
        'hit_slow_owner1_only',
        'projectile_3d',
        'min_damage',
        'regen_race_fallback',
        'range_radius',
        'visibility',
        'vision_levels',
        'reveal_loops',
        'lost_vision',
        'lost_vision_giveup',
        'abilities',
        'stop_keeps_heal',
        'heal_counts_armed',
        'spawn_energy',
        'heal_autocast',
        'heal_on_arrival',
        'heal_facing',
    }
)

_SUB_PROFILES: dict[str, str] = {'v4': 'aligned-v4', 'axo': 'acc-ram-cov-axo'}


def _profile_subset_options(name: str) -> dict[str, Any]:
    """Return non-default PyrefConfig options for a named subset ('v4', 'axo', '*-m1', '*-m2')."""
    from unismac.profiles import profile_overrides

    if name.endswith('-m1'):
        prof, keep = name[:-3], _M1_FIELDS
    elif name.endswith('-m2'):
        prof, keep = name[:-3], _M2_FIELDS
    else:
        prof, keep = name, None
    _, over = profile_overrides(_SUB_PROFILES[prof])
    out: dict[str, Any] = {}
    for key, value in over.items():
        if key == 'backend.pyref_options':
            out.update(value)
        elif key.startswith('backend.pyref_'):
            out[key.removeprefix('backend.pyref_')] = value
    if keep is not None:
        out = {k: v for k, v in out.items() if k in keep}
    return out


def _m1_overrides() -> dict[str, object]:
    opts = _profile_subset_options('axo-m1')
    over: dict[str, object] = {
        f'backend.pyref_{k}': opts.pop(k) for k in ('commands', 'command_delay', 'move_lag')
    }
    over['backend.pyref_options'] = opts
    return over


@pytest.mark.parametrize('scenario', ['3m', '2s3z'])
def test_portable_env_m1_matches_pyref(scenario: str) -> None:
    """smac-portable with axo movement/collision subset (delayed commands, random spawn facing,
    accel, hard collision): avail, reward, terminated, wins match bitwise; obs/state <= 1e-4
    (ulp position diffs from float32 arctan2; see test_zgame.M1_POS_ATOL)."""

    def run(backend: str) -> list[tuple[np.ndarray, ...]]:
        env = make_vec_env(
            'smac-portable',
            scenario,
            backend=backend,  # type: ignore[arg-type]
            batch_size=3,
            seed=4,
            overrides=_m1_overrides(),
        )
        obs, state = env.reset()
        out: list[tuple[np.ndarray, ...]] = [(obs, state)]
        rng = np.random.default_rng(2)
        while not env.done.all():
            avail = env.get_avail_actions()
            reward, terminated, _ = env.step(_random_actions(rng, avail))
            out.append((avail, reward, terminated, env.get_obs(), env.get_state()))
        out.append((np.array([env.get_stats()['battles_won']]),))
        env.close()
        return out

    a, b = run('pyref'), run('zgame')
    assert len(a) == len(b)
    for xa, xb in zip(a, b, strict=True):
        for u, v in zip(xa, xb, strict=True):
            if u.dtype.kind == 'f' and u.ndim >= 2:  # obs [B, A, F] and state [B, S]
                np.testing.assert_allclose(u, v, rtol=0, atol=1e-4)
            else:
                np.testing.assert_array_equal(u, v)


def _m2_overrides(profile: str) -> dict[str, object]:
    """Build env overrides for the combat/abilities subset of profile (excluding steering,
    pathing, and avoidance)."""
    opts = _profile_subset_options(f'{profile}-m2')
    over: dict[str, object] = {
        f'backend.pyref_{k}': opts.pop(k) for k in ('commands', 'command_delay', 'move_lag')
    }
    over['backend.pyref_options'] = opts
    return over


@pytest.mark.parametrize('profile', ['v4', 'axo'])
def test_portable_env_m2_mmm_matches_pyref(profile: str) -> None:
    """smac-portable MMM (Medivac heal/autocast, energy) with combat/abilities subset of
    profile: avail, reward, terminated, wins match bitwise; obs/state <= 1e-4."""

    def run(backend: str) -> list[tuple[np.ndarray, ...]]:
        env = make_vec_env(
            'smac-portable',
            'MMM',
            backend=backend,  # type: ignore[arg-type]
            batch_size=2,
            seed=5,
            overrides=_m2_overrides(profile),
        )
        obs, state = env.reset()
        out: list[tuple[np.ndarray, ...]] = [(obs, state)]
        rng = np.random.default_rng(3)
        while not env.done.all():
            avail = env.get_avail_actions()
            reward, terminated, _ = env.step(_random_actions(rng, avail))
            out.append((avail, reward, terminated, env.get_obs(), env.get_state()))
        out.append((np.array([env.get_stats()['battles_won']]),))
        env.close()
        return out

    a, b = run('pyref'), run('zgame')
    assert len(a) == len(b)
    for xa, xb in zip(a, b, strict=True):
        for u, v in zip(xa, xb, strict=True):
            if u.dtype.kind == 'f' and u.ndim >= 2:
                np.testing.assert_allclose(u, v, rtol=0, atol=1e-4)
            else:
                np.testing.assert_array_equal(u, v)


def _v4_run(
    backend: str, scenario: str, seed: int, profile: str = 'aligned-v4'
) -> list[tuple[np.ndarray, ...]]:
    """Run smac-portable with a full profile (default aligned-v4) under random actions until
    all envs terminate."""
    from unismac.profiles import profile_overrides

    _, over = profile_overrides(profile)
    env = make_vec_env(
        'smac-portable',
        scenario,
        backend=backend,  # type: ignore[arg-type]
        batch_size=2,
        seed=seed,
        overrides=over,
    )
    obs, state = env.reset()
    out: list[tuple[np.ndarray, ...]] = [(obs, state)]
    rng = np.random.default_rng(seed)
    while not env.done.all():
        avail = env.get_avail_actions()
        reward, terminated, _ = env.step(_random_actions(rng, avail))
        out.append((avail, reward, terminated, env.get_obs(), env.get_state()))
    out.append((np.array([env.get_stats()['battles_won']]),))
    env.close()
    return out


def _v4_compare(scenario: str, seed: int, *, exact: bool, profile: str = 'aligned-v4') -> int:
    """Exact mode requires bitwise parity throughout. In-process mode checks that the first
    divergence is <= 1e-6 (ulp) in obs/state, and discrete outputs stay bitwise-equal while
    obs/state diff <= 1e-4; beyond 1e-4 avoidance/collision thresholds can amplify ulp diffs
    and comparison stops. Returns number of steps compared."""
    a = _v4_run('pyref', scenario, seed, profile)
    b = _v4_run('zgame', scenario, seed, profile)
    if exact:
        assert len(a) == len(b)
        for xa, xb in zip(a, b, strict=True):
            for u, v in zip(xa, xb, strict=True):
                np.testing.assert_array_equal(u, v)
        return len(a)
    first = True
    for t, (xa, xb) in enumerate(zip(a[:-1], b[:-1], strict=False)):
        worst = 0.0
        for u, v in zip(xa, xb, strict=True):
            if u.dtype.kind == 'f' and u.ndim >= 2:  # obs [B, A, F] and state [B, S]
                worst = max(worst, float(np.max(np.abs(u.astype(np.float64) - v), initial=0.0)))
            else:
                np.testing.assert_array_equal(u, v, err_msg=f'step {t}')
        if worst > 0 and first:
            assert worst <= 1e-6, (t, worst)
            first = False
        if worst > 1e-4:
            return t
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[-1][0], b[-1][0])
    return len(a)


V4_SCENARIOS = ('3m', '8m', '2s3z', 'MMM', 'corridor')


@pytest.mark.parametrize('scenario', V4_SCENARIOS)
def test_portable_env_v4_matches_pyref(scenario: str) -> None:
    """smac-portable with full aligned-v4 (navmesh, sc2_avoid, pin_engaged, all 26 options),
    in-process: see `_v4_compare` (first diff ulp-level; discrete outputs bitwise <= 1e-4)."""
    assert _v4_compare(scenario, 6, exact=False) >= 5


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_portable_env_v4_bitwise_without_avx512() -> None:
    """Same as above, bitwise-identical obs and state in a subprocess without AVX-512."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        f'for s in {V4_SCENARIOS!r}:\n'
        '    m["_v4_compare"](s, 6, exact=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# acc-ram-cov-axo-rghf: 2s3z exercises Stalker_RL/Zealot_RL fallback shield regen; MMM exercises
# Medivac health-fraction autocast heal
RGHF_SCENARIOS = ('2s3z', 'MMM')


@pytest.mark.parametrize('scenario', RGHF_SCENARIOS)
def test_portable_env_rghf_matches_pyref(scenario: str) -> None:
    """smac-portable with full acc-ram-cov-axo-rghf in-process (same policy as `_v4_compare`)."""
    assert _v4_compare(scenario, 6, exact=False, profile='acc-ram-cov-axo-rghf') >= 5


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_portable_env_rghf_bitwise_without_avx512() -> None:
    """Same as above, bitwise-identical throughout in a subprocess without AVX-512."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        f'for s in {RGHF_SCENARIOS!r}:\n'
        '    m["_v4_compare"](s, 6, exact=True, profile="acc-ram-cov-axo-rghf")\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


def test_zgame_env_config_errors() -> None:
    with pytest.raises(CapabilityError):
        make_vec_env(
            'smac-portable',
            '3m',
            backend='zgame',
            overrides={'backend.pyref_options': {'steering': 'separation'}},
        )
    with pytest.raises(ValueError):
        make_vec_env(
            'smac-portable', '3m', backend='zgame', overrides={'backend.pyref_options': {'nope': 1}}
        )
    with pytest.raises(ValueError, match='sc2 / pyref / zgame'):
        make_vec_env('smac-portable', '3m', backend='nope')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='zgame_threads'):
        make_vec_env('smac-portable', '3m', backend='pyref', zgame_threads=2)


def test_zgame_threads_passed_to_backend() -> None:
    from unismac.backends.zgame import ZgameBackend

    for threads, want in ((3, 3), (None, 1)):
        env = make_vec_env(
            'smac-portable', '3m', backend='zgame', batch_size=4, zgame_threads=threads
        )
        be = env.backend
        assert isinstance(be, ZgameBackend) and be.threads == want


# Two-sided mode (opponent.kind=external + symmetric profile): bitwise equal without AVX-512
DUAL_CASES = (
    ('smac-portable', '8m', 'aligned-v4-sym'),
    ('smac-portable', 'corridor', 'aligned-v4-sym'),
    ('smac-portable', 'MMM', 'acc-ram-cov-axo-sym'),
    ('smac-portable', '3s_vs_5z', 'aligned-v4-sym'),
    ('smac-hard-portable', '2s3z', 'acc-ram-cov-axo-sym'),
    ('smac-hard-portable', 'MMM', 'aligned-v4-sym'),
)


def _dual_run(backend: str, preset: str, scenario: str, profile: str, seed: int) -> list[tuple]:
    """Run both sides with random legal actions until done, recording obs/state/avail/rewards."""
    from unismac.profiles import profile_overrides

    over = {**profile_overrides(profile)[1], 'opponent.kind': 'external'}
    env = make_vec_env(preset, scenario, backend=backend, batch_size=2, seed=seed, overrides=over)  # type: ignore[arg-type]
    obs, state = env.reset()
    out: list[tuple] = [(obs, state, env.get_obs(2), env.get_state(2))]
    rng = np.random.default_rng(seed)
    while not env.done.all():
        av1, av2 = env.get_avail_actions(), env.get_avail_actions(2)
        reward, terminated, infos = env.step(_random_actions(rng, av1), _random_actions(rng, av2))
        side2 = np.array([i.get('reward_enemy', 0.0) for i in infos])
        lost = np.array([i.get('battle_lost', False) for i in infos])
        out.append((av1, av2, reward, side2, lost, terminated))
        out.append((env.get_obs(), env.get_state(), env.get_obs(2), env.get_state(2)))
    out.append((np.array([env.get_stats()['battles_won']]),))
    env.close()
    return out


def _dual_compare(preset: str, scenario: str, profile: str, seed: int) -> int:
    a = _dual_run('pyref', preset, scenario, profile, seed)
    b = _dual_run('zgame', preset, scenario, profile, seed)
    assert len(a) == len(b), (scenario, len(a), len(b))
    for t, (xa, xb) in enumerate(zip(a, b, strict=True)):
        for u, v in zip(xa, xb, strict=True):
            np.testing.assert_array_equal(u, v, err_msg=f'{preset}/{scenario} item {t}')
    return len(a)


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_dual_bitwise_without_avx512() -> None:
    """Two-sided mode: pyref and zgame match bitwise on both sides' avail, obs, state, rewards,
    termination, and wins (subprocess with NumPy AVX-512 disabled)."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        f'for case in {DUAL_CASES!r}:\n'
        '    m["_dual_compare"](*case, 3)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- Native movement & combat parity

_M12_NAT_FULL_PO: dict[str, object] = {
    'pin_sidestep': False,
    'pin_sidestep_drift': False,
    'pin_tie_grid': False,
    'avoid_parallel_faster_only': False,
    'avoid_cross_cos': 0.9,
    'avoid_share': 'full',
    'avoid_share_power': 1.0,
    'avoid_side_check': False,
    'contact_orbit': False,
    'mover_push_share': 'half',
    'mover_speed_cap': False,
    'pin_slide': False,
    'crowd_arrival': False,
    'crowd_cover': False,
    'push_clearance': False,
    'pin_after_fire': False,
    'pin_lag': False,
    'steering': 'native',
    'collision': 'native',
    'native_share_rule': 'priority_tree',
    'native_batch_flags': True,
    'native_nbr_order': 'quantized',
    'native_steer_avoid': True,
    'native_move_reissue': True,
    'native_stop': True,
    'native_air_sep': True,
    'native_wall_clearance': True,
    'contact_slack': 0.0,
    'idle_push_share': 0.5,
}

_M12_RGHF_NAT_CBT_PO: dict[str, object] = {
    **_M12_NAT_FULL_PO,
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'native_backswing': True,
    'native_retarget': True,
    'hit_assist': True,
    'native_assist_on_hit': True,
    'native_scan_cadence': True,
    'acquire_start_loop': 0,
    'native_loop_order': True,
    'kill_rescan_delay': 0,
    'kill_rescan_order': True,
}

M12_CBT_SCENARIOS = (
    '3m',
    '8m',
    '2s3z',
    '3s5z',
    'MMM',
    '5m_vs_6m',
    'corridor',
    '2c_vs_64zg',
    'bane_vs_bane',
    '1c3s5z',
)

M12_COMBO_CASES: tuple[tuple[str, str, str, dict[str, object]], ...] = (
    ('smac-portable', '2s3z', 'aligned-v4', _M12_NAT_FULL_PO),
    ('smac-portable', '2s3z', 'acc-ram-cov-axo-rghf', _M12_NAT_FULL_PO),
    (
        'smac-portable',
        '2s3z',
        'acc-ram-cov-axo-rghf',
        {**_M12_RGHF_NAT_CBT_PO, 'scan_phase': 'tag_rotate', 'scan_period': 8},
    ),
    ('smacv2-portable', '10gen_terran', 'acc-ram-cov-axo-rghf', _M12_RGHF_NAT_CBT_PO),
    ('smac-hard-portable', 'MMM', 'acc-ram-cov-axo-rghf', _M12_RGHF_NAT_CBT_PO),
)

M12_SYM_DUAL_CASES: tuple[tuple[str, str, str, dict[str, object]], ...] = (
    (
        'smac-portable',
        '2s3z',
        'acc-ram-cov-axo-rghf',
        {
            **_M12_RGHF_NAT_CBT_PO,
            'auto_acquire_owners': (),
            'retreat_owners': (1, 2),
            'hit_slow_owner1_only': False,
        },
    ),
    (
        'smac-portable',
        'MMM',
        'acc-ram-cov-axo-rghf',
        {
            **_M12_RGHF_NAT_CBT_PO,
            'auto_acquire_owners': (),
            'retreat_owners': (1, 2),
            'hit_slow_owner1_only': False,
        },
    ),
)


def _m12_env_compare(
    preset: str,
    scenario: str,
    profile: str,
    pyref_opts: dict[str, object],
    *,
    seed: int = 6,
    batch_size: int = 1,
    dual: bool = False,
) -> int:
    from unismac.profiles import profile_overrides

    _, over = profile_overrides(profile, pyref_opts)
    if dual:
        over = {**over, 'opponent.kind': 'external'}

    def run(backend: str) -> list[tuple[np.ndarray, ...]]:
        env = make_vec_env(
            preset,
            scenario,
            backend=backend,  # type: ignore[arg-type]
            batch_size=batch_size,
            seed=seed,
            overrides=over,
        )
        obs, state = env.reset()
        out: list[tuple[np.ndarray, ...]] = [
            (obs, state, env.get_obs(2), env.get_state(2)) if dual else (obs, state)
        ]
        rng = np.random.default_rng(seed)
        step = 0
        while not env.done.all():
            avail = env.get_avail_actions()
            act = _random_actions(rng, avail)
            if dual:
                av2 = env.get_avail_actions(2)
                reward, terminated, infos = env.step(act, _random_actions(rng, av2))
                side2 = np.array([i.get('reward_enemy', 0.0) for i in infos])
                lost = np.array([i.get('battle_lost', False) for i in infos])
                out.append((avail, av2, reward, side2, lost, terminated))
                out.append((env.get_obs(), env.get_state(), env.get_obs(2), env.get_state(2)))
            else:
                reward, terminated, _ = env.step(act)
                out.append((avail, reward, terminated, env.get_obs(), env.get_state()))
            if batch_size >= 2 and step == 10:
                p_obs, p_state = env.reset(env_ids=[0])
                out.append((p_obs, p_state))
            step += 1
        out.append((np.array([env.get_stats()['battles_won']]),))
        env.close()
        return out

    a, b = run('pyref'), run('zgame')
    assert len(a) == len(b), (preset, scenario, len(a), len(b))
    for t, (xa, xb) in enumerate(zip(a, b, strict=True)):
        for u, v in zip(xa, xb, strict=True):
            np.testing.assert_array_equal(u, v, err_msg=f'{preset}/{scenario} item {t}')
    return len(a)


@pytest.mark.parametrize('scenario', M12_CBT_SCENARIOS)
def test_portable_env_rghf_nat_cbt_bitwise(scenario: str) -> None:
    """10 SMAC scenarios with native movement/combat: bitwise parity in-process (AVX-512 ON)."""
    bs = 2 if scenario in ('3m', '8m', '2s3z') else 1
    assert (
        _m12_env_compare(
            'smac-portable', scenario, 'acc-ram-cov-axo-rghf', _M12_RGHF_NAT_CBT_PO, batch_size=bs
        )
        >= 5
    )


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'opts'),
    M12_COMBO_CASES,
    ids=[f'{p}/{s}/{prof}' for p, s, prof, _ in M12_COMBO_CASES],
)
def test_m12_combo_variants_bitwise(
    preset: str, scenario: str, profile: str, opts: dict[str, object]
) -> None:
    """Bitwise parity for v4-nat, rghf-nat, rot, smacv2-portable, and smac-hard-portable combos."""
    assert _m12_env_compare(preset, scenario, profile, opts, batch_size=2) >= 5


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'opts'),
    M12_SYM_DUAL_CASES,
    ids=[f'{p}/{s}' for p, s, _, _ in M12_SYM_DUAL_CASES],
)
def test_m12_sym_dual_bitwise(
    preset: str, scenario: str, profile: str, opts: dict[str, object]
) -> None:
    """rghf-nat-cbt-sym two-sided mirror mode: bitwise parity between pyref and zgame."""
    assert _m12_env_compare(preset, scenario, profile, opts, seed=3, batch_size=2, dual=True) >= 5


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_m12_env_bitwise_without_avx512() -> None:
    """Bitwise parity for rghf-nat-cbt and combo variants in a subprocess without AVX-512."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        'po = m["_M12_RGHF_NAT_CBT_PO"]\n'
        'for s in ("3m", "8m", "2s3z", "MMM", "1c3s5z"):\n'
        '    m["_m12_env_compare"]("smac-portable", s, "acc-ram-cov-axo-rghf", po, batch_size=2)\n'
        'for p, s, prof, opts in m["M12_COMBO_CASES"]:\n'
        '    m["_m12_env_compare"](p, s, prof, opts, batch_size=1)\n'
        'for p, s, prof, opts in m["M12_SYM_DUAL_CASES"]:\n'
        '    m["_m12_env_compare"](p, s, prof, opts, seed=3, batch_size=1, dual=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# Named profiles directly (no extra pyref_options) match bitwise over full episodes
NATIVE_PROFILE_SINGLE_CASES = (
    ('2s3z', 'rghf-nat'),
    ('2s3z', 'rghf-nat-cbt'),
    ('MMM', 'rghf-nat-cbt'),
)
NATIVE_PROFILE_DUAL_CASES = (
    ('smac-portable', '2s3z', 'rghf-nat-cbt-sym'),
    ('smac-portable', 'MMM', 'rghf-nat-cbt-sym'),
    ('smac-hard-portable', '2s3z', 'rghf-nat-cbt-sym'),
)


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_native_named_profiles_bitwise_without_avx512() -> None:
    """Named profiles rghf-nat / rghf-nat-cbt / rghf-nat-cbt-sym (no extra pyref_options)
    match bitwise between zgame and pyref without AVX-512 (single- and two-sided)."""
    import os
    import subprocess
    import sys

    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        f'for scen, prof in {NATIVE_PROFILE_SINGLE_CASES!r}:\n'
        '    m["_v4_compare"](scen, 6, exact=True, profile=prof)\n'
        f'for preset, scen, prof in {NATIVE_PROFILE_DUAL_CASES!r}:\n'
        '    m["_dual_compare"](preset, scen, prof, 3)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ------------------------------------------------------------ Per-map passive behavior end-to-end


def test_b02_corridor_rghf_nat_cbt_bitwise_rust_and_still_wins() -> None:
    """smac-hard-portable/corridor under rghf-nat-cbt and rghf-nat:
    1) pyref and zgame both select accel='rust' and match bitwise on obs/state/avail/reward/
       done/WorldState;
    2) still policy wins >= 1 episode (vs 0 wins and 0 reward with passive_from_map=False)."""
    import dataclasses

    def still_policy(env: VecEnv, _rng: np.random.Generator) -> np.ndarray:
        avail = env.get_avail_actions()
        return np.where(avail[..., 1] == 1, 1, 0).astype(np.int64)

    from unismac import profiles

    # 1. Random-action bitwise parity + accel='rust' for rghf-nat-cbt and rghf-nat
    for prof_name in (profiles.STANDARD_PROFILE, profiles.STANDARD_EXPLORE_PROFILE):
        _, over = profiles.profile_overrides(prof_name)
        ep = make_vec_env(
            'smac-hard-portable',
            'corridor',
            backend='pyref',
            batch_size=2,
            seed=6,
            overrides=over,
            accel='auto',
        )
        ez = make_vec_env(
            'smac-hard-portable',
            'corridor',
            backend='zgame',
            batch_size=2,
            seed=6,
            overrides=over,
            accel='auto',
        )
        try:
            assert ep.accel == 'rust' and ez.accel == 'rust'
            obs_p, st_p = ep.reset()
            obs_z, st_z = ez.reset()
            np.testing.assert_array_equal(obs_p, obs_z)
            np.testing.assert_array_equal(st_p, st_z)
            rng = np.random.default_rng(6)
            for _ in range(30):
                av_p, av_z = ep.get_avail_actions(), ez.get_avail_actions()
                np.testing.assert_array_equal(av_p, av_z)
                acts = _random_actions(rng, av_p)
                rp, tp, ip = ep.step(acts)
                rz, tz, iz = ez.step(acts)
                np.testing.assert_array_equal(rp, rz)
                np.testing.assert_array_equal(tp, tz)
                assert ip == iz
                for f in dataclasses.fields(ep.world):
                    ap, az = getattr(ep.world, f.name), getattr(ez.world, f.name)
                    if ap.dtype.kind == 'f':
                        assert np.array_equal(ap, az, equal_nan=True), (prof_name, f.name)
                    else:
                        assert np.array_equal(ap, az), (prof_name, f.name)
                if tp.all():
                    break
        finally:
            ep.close()
            ez.close()

    # 2. Still policy over 12 eps: rghf-nat-cbt wins >= 1; pyref matches zgame on first 4 eps
    _, on_over = profiles.profile_overrides(profiles.STANDARD_PROFILE)
    _, off_over = profiles.profile_overrides(profiles.STANDARD_PROFILE, {'passive_from_map': False})

    def run_still(
        backend: str, over: dict[str, object], n_eps: int
    ) -> list[tuple[bool, int, float]]:
        env = make_vec_env(
            'smac-hard-portable',
            'corridor',
            backend=backend,  # type: ignore[arg-type]
            batch_size=1,
            seed=0,
            overrides=over,
            accel='auto',
        )
        try:
            assert env.accel == 'rust'
            out: list[tuple[bool, int, float]] = []
            for e in range(n_eps):
                env.reset_count[0] = e
                env.seed = e
                env.reset()
                rng = np.random.default_rng([0, 2, e])
                tot_r, steps, won = 0.0, 0, False
                while not env.done[0]:
                    r, term, infos = env.step(still_policy(env, rng))
                    tot_r += float(r[0])
                    steps += 1
                    if term[0]:
                        won = bool(infos[0].get('battle_won', False))
                out.append((won, steps, tot_r))
            return out
        finally:
            env.close()

    z_on = run_still('zgame', on_over, 12)
    z_off = run_still('zgame', off_over, 12)
    p_on = run_still('pyref', on_over, 4)
    assert z_on[:4] == p_on
    assert sum(w for w, _, _ in z_on) >= 1 and float(np.mean([r for _, _, r in z_on])) > 10.0
    assert sum(w for w, _, _ in z_off) == 0 and all(r == 0.0 for _, _, r in z_off)


_B02_NON_CORRIDOR: tuple[tuple[str, str], ...] = (
    ('smac-portable', '3m'),
    ('smac-portable', '8m'),
    ('smac-portable', '2s3z'),
    ('smac-hard-portable', '8m'),
    ('smac-hard-portable', '2s3z'),
    ('smac-hard-portable', '3s5z'),
    ('smac-hard-portable', 'MMM'),
)


@pytest.mark.parametrize(
    ('preset', 'scenario'),
    _B02_NON_CORRIDOR,
    ids=[f'{p}/{s}' for p, s in _B02_NON_CORRIDOR],
)
def test_b02_non_corridor_passive_from_map_zero_diff(preset: str, scenario: str) -> None:
    """Non-corridor scenarios (SMAC_Maps/* and new_maps/{8m,2s3z,3s5z,MMM}) show zero diff
    between passive_from_map=True and False on both pyref and zgame, using accel='rust'."""
    import dataclasses

    from unismac import profiles

    _, on_over = profiles.profile_overrides(profiles.STANDARD_PROFILE)
    _, off_over = profiles.profile_overrides(profiles.STANDARD_PROFILE, {'passive_from_map': False})
    for backend in ('pyref', 'zgame'):
        e_on = make_vec_env(
            preset,
            scenario,
            backend=backend,  # type: ignore[arg-type]
            batch_size=2,
            seed=4,
            overrides=on_over,
            accel='auto',
        )
        e_off = make_vec_env(
            preset,
            scenario,
            backend=backend,  # type: ignore[arg-type]
            batch_size=2,
            seed=4,
            overrides=off_over,
            accel='auto',
        )
        try:
            assert e_on.accel == 'rust' and e_off.accel == 'rust'
            obs1, st1 = e_on.reset()
            obs0, st0 = e_off.reset()
            np.testing.assert_array_equal(obs1, obs0)
            np.testing.assert_array_equal(st1, st0)
            rng = np.random.default_rng(7)
            for _ in range(25):
                av1, av0 = e_on.get_avail_actions(), e_off.get_avail_actions()
                np.testing.assert_array_equal(av1, av0)
                acts = _random_actions(rng, av1)
                r1, t1, i1 = e_on.step(acts)
                r0, t0, i0 = e_off.step(acts)
                np.testing.assert_array_equal(r1, r0)
                np.testing.assert_array_equal(t1, t0)
                assert i1 == i0
                for f in dataclasses.fields(e_on.world):
                    a1, a0 = getattr(e_on.world, f.name), getattr(e_off.world, f.name)
                    if a1.dtype.kind == 'f':
                        assert np.array_equal(a1, a0, equal_nan=True), (backend, f.name)
                    else:
                        assert np.array_equal(a1, a0), (backend, f.name)
                if t1.all():
                    break
        finally:
            e_on.close()
            e_off.close()


# ---------------------------------------- Multi-level vision/3D projectiles & Charge env parity

_M13_B01: dict[str, object] = {'visibility': 'sc2', 'lost_vision': 'sc2', 'projectile_3d': True}
_M13_B03: dict[str, object] = {'charge': True}
_M13_BOTH: dict[str, object] = {**_M13_B01, **_M13_B03}


def _m13_over(profile: str, extra: dict[str, object]) -> dict[str, object]:
    from unismac import profiles

    _, base = profiles.profile_overrides(profile)
    out = dict(base)
    raw = out.get('backend.pyref_options', {})
    assert isinstance(raw, dict)
    out['backend.pyref_options'] = {**raw, **extra}
    return out


_M13_ENV_CASES: tuple[tuple[str, str, str, str, dict[str, object]], ...] = (
    ('smac-hard-portable', 'mmmt', 'rghf-nat-cbt', 'b01', _M13_B01),
    ('smac-hard-portable', '3st_vs_5zl', 'rghf-nat-cbt', 'b01+b03', _M13_BOTH),
    ('smac-hard-portable', '3rp_vs_5zl', 'rghf-nat-cbt', 'b01+b03', _M13_BOTH),
    ('smac-hard-portable', 'corridor', 'rghf-nat-cbt', 'b03', _M13_B03),
    ('smac-portable', 'MMM2', 'rghf-nat-cbt', 'b01', _M13_B01),
    ('smac-hard-portable', 'mmmt', 'aligned-v5', 'b01', _M13_B01),
)


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'tag', 'extra'),
    _M13_ENV_CASES,
    ids=[f'{p}/{s}/{pr}/{t}' for p, s, pr, t, _ in _M13_ENV_CASES],
)
def test_m13_env_full_episode_matches_pyref(
    preset: str, scenario: str, profile: str, tag: str, extra: dict[str, object]
) -> None:
    """Multi-level vision/3D projectiles and Charge: in-process VecEnv(accel='auto') selects
    accel='rust' and matches pyref bitwise across obs, state, avail, reward, done, infos, and
    all 24 WorldState fields (including visible)."""
    import dataclasses

    over = _m13_over(profile, extra)
    ep = make_vec_env(
        preset, scenario, backend='pyref', batch_size=2, seed=4, overrides=over, accel='auto'
    )
    ez = make_vec_env(
        preset, scenario, backend='zgame', batch_size=2, seed=4, overrides=over, accel='auto'
    )
    try:
        assert ep.accel == 'rust' and ez.accel == 'rust', (preset, scenario, tag)
        op, sp = ep.reset()
        oz, sz = ez.reset()
        np.testing.assert_array_equal(op, oz)
        np.testing.assert_array_equal(sp, sz)
        rng = np.random.default_rng(13)
        while not ep.done.all():
            avp, avz = ep.get_avail_actions(), ez.get_avail_actions()
            np.testing.assert_array_equal(avp, avz)
            acts = _random_actions(rng, avp)
            rp, tp, ip = ep.step(acts)
            rz, tz, iz = ez.step(acts)
            np.testing.assert_array_equal(rp, rz)
            np.testing.assert_array_equal(tp, tz)
            assert ip == iz
            np.testing.assert_array_equal(ep.get_obs(), ez.get_obs())
            np.testing.assert_array_equal(ep.get_state(), ez.get_state())
            for f in dataclasses.fields(ep.world):
                ap, az = getattr(ep.world, f.name), getattr(ez.world, f.name)
                if ap.dtype.kind == 'f':
                    assert np.array_equal(ap, az, equal_nan=True), (scenario, tag, f.name)
                else:
                    assert np.array_equal(ap, az), (scenario, tag, f.name)
        assert ez.done.all()
        assert ep.get_stats()['battles_won'] == ez.get_stats()['battles_won']
    finally:
        ep.close()
        ez.close()


@pytest.mark.parametrize(
    ('preset', 'scenario', 'extra'),
    [
        ('smac-hard-portable', 'mmmt', _M13_B01),
        ('smac-hard-portable', '3st_vs_5zl', _M13_BOTH),
    ],
    ids=['mmmt+b01', '3st_vs_5zl+b01+b03'],
)
def test_m13_accel_three_tiers_bitwise(
    preset: str, scenario: str, extra: dict[str, object]
) -> None:
    """With visibility='sc2' and charge=True, python/services/rust tiers match bitwise,
    verifying _zgame.env (SEMANTICS == 3) and Engine.write_rows work seamlessly."""
    over = _m13_over('rghf-nat-cbt', extra)
    envs = {
        tier: make_vec_env(
            preset,
            scenario,
            backend='zgame',
            batch_size=2,
            seed=7,
            overrides=over,
            accel=tier,  # type: ignore[arg-type]
        )
        for tier in ('python', 'services', 'rust')
    }
    try:
        resets = {t: e.reset() for t, e in envs.items()}
        for t in ('services', 'rust'):
            np.testing.assert_array_equal(resets['python'][0], resets[t][0])
            np.testing.assert_array_equal(resets['python'][1], resets[t][1])
            np.testing.assert_array_equal(envs['python'].world.visible, envs[t].world.visible)
        rng = np.random.default_rng(19)
        for _ in range(40):
            if envs['python'].done.all():
                break
            av0 = envs['python'].get_avail_actions()
            for t in ('services', 'rust'):
                np.testing.assert_array_equal(av0, envs[t].get_avail_actions())
            acts = _random_actions(rng, av0)
            out = {t: e.step(acts) for t, e in envs.items()}
            for t in ('services', 'rust'):
                np.testing.assert_array_equal(out['python'][0], out[t][0])
                np.testing.assert_array_equal(out['python'][1], out[t][1])
                assert out['python'][2] == out[t][2]
                np.testing.assert_array_equal(envs['python'].get_obs(), envs[t].get_obs())
                np.testing.assert_array_equal(envs['python'].get_state(), envs[t].get_state())
                np.testing.assert_array_equal(envs['python'].world.visible, envs[t].world.visible)
    finally:
        for e in envs.values():
            e.close()
