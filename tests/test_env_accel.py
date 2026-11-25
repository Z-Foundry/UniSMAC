"""Acceleration-tier parity tests (`python` / `services` / `rust`) for the environment layer.

Verifies per-function (E0) and `VecEnv`-level (E1) bitwise equivalence across all
acceleration tiers, classic recipe eligibility, and the semantic-layer source manifest.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import platform
import subprocess
import sys
import warnings
import zlib
from collections import Counter
from typing import Any

import numpy as np
import pytest

from unismac.contract import Command
from unismac.env import VecEnv, accel, build_config, make_env, make_vec_env
from unismac.env import classic as classic_mod
from unismac.env import generators as generators_mod
from unismac.env import manifest as manifest_mod
from unismac.env import rng as rng_mod
from unismac.env import services as services_mod
from unismac.env.abilities import ABILITY_DICT
from unismac.env.actions import ActionSpace, FovState
from unismac.env.catalog import AllyTables
from unismac.env.config import (
    ActionConfig,
    AttackConfig,
    CapabilityConfig,
    EnemyMaskConfig,
    StartPositionsConfig,
    TeamGenConfig,
)
from unismac.env.opponents import OpeningOrders
from unismac.env.scripted import POOLS, SCRIPTS
from unismac.env.units import Frame, Side
from unismac.profiles import profile_overrides

pytest.importorskip('zgame', reason="requires the optional 'zgame' extension")
native = pytest.importorskip(
    'unismac.env.native',
    reason="requires the optional 'zgame' extension with a compatible 'env' submodule",
    exc_type=ImportError,
)
native_rust = pytest.importorskip('unismac.env.native_rust', exc_type=ImportError)
ZgameBackend = native_rust.ZgameBackend

PRESETS = ('smac-portable', 'smacv2-portable', 'smac-hard-portable')
SCENARIOS = ('3m', '8m', '2s3z', '3s5z', 'MMM', '5m_vs_6m', 'corridor')
PROFILES = ('aligned-v4', 'acc-ram-cov-axo')
# All SMACv2 stochastic features enabled (same as test_env_seeds)
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
# All stochastic features handled by rust tier (conic FOV uses np.arctan, stays on services)
STOCH_CIRCLE = {k: v for k, v in STOCH.items() if k != 'conic_fov'}
# SMACv2 action_mask=False: targets ignore range in mask, checked on issue (to_command reach)
NO_MASK = {'action_mask': False, 'obs_pathing_grid': True, 'obs_terrain_height': True}
GEN_SCENARIOS = ('10gen_terran', '10gen_protoss', '10gen_zerg')
# Two-sided mode: external opponent with symmetric profiles
EXTERNAL = {'opponent.kind': 'external'}
DUAL_PROFILES = ('aligned-v4-sym', 'acc-ram-cov-axo-sym')
DUAL_V4 = {**profile_overrides('aligned-v4-sym')[1], **EXTERNAL}
DUAL_AXO = {**profile_overrides('acc-ram-cov-axo-sym')[1], **EXTERNAL}
# SMACv2 stochastic options valid in two-sided mode (attack/enemy_mask are Player 1 only)
DUAL_STOCH = {
    'capability_config': {
        k: v for k, v in STOCH['capability_config'].items() if k not in ('attack', 'enemy_mask')
    },
    'prob_obs_enemy': 0.5,
    'quirks.obs_shuffle': True,
    **EXTERNAL,
}
SEMANTIC = (
    'draw',
    'avail',
    'observe_spotter',
    'encode_obs',
    'encode_state',
    'game_end',
    'game_end_dual',
    'reward',
    'spotter_after_step',
    'opponent_reset',
    'opponent_issue',
)


# ---------------------------------------------------------------- Bitwise comparison


def _same(a: Any, b: Any, where: str) -> None:
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), where
        assert (a.dtype, a.shape) == (b.dtype, b.shape), (
            f'{where}: {a.dtype}{a.shape} / {b.dtype}{b.shape}'
        )
        assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes(), where
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), where
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), f'{where}.{f.name}')
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f'{where}[{i}]')
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for k in a:
            _same(a[k], b[k], f'{where}[{k!r}]')
    else:
        assert type(a) is type(b) and a == b, f'{where}: {a!r} / {b!r}'


def _run(fn: Any, *args: Any, **kw: Any) -> tuple[Any, BaseException | None]:
    try:
        return fn(*args, **kw), None
    except Exception as exc:  # exceptions must also match
        return None, exc


def _same_outcome(ref: tuple[Any, Any], got: tuple[Any, Any], where: str) -> Any:
    (r, re_), (g, ge) = ref, got
    if re_ is not None or ge is not None:
        assert type(re_) is type(ge) and str(re_) == str(ge), f'{where}: {re_!r} / {ge!r}'
        raise ge  # type: ignore[misc]
    _same(r, g, where)
    return g


class Checking:
    """Kernels for one tier (`services` or `rust`) checked against reference on every call (E0).
    `calls` counts invocations per kernel name."""

    def __init__(self, tier: str = 'services') -> None:
        self.tier = tier
        self.rust = accel.kernels(tier)
        self.calls: Counter[str] = Counter()

    def kernels(self) -> accel.Kernels:
        ref = accel.PYTHON
        rust = self.rust

        def pure(name: str, f_ref: Any, f_rust: Any) -> Any:
            def call(*args: Any, **kw: Any) -> Any:
                self.calls[name] += 1
                return _same_outcome(_run(f_ref, *args, **kw), _run(f_rust, *args, **kw), name)

            return call

        def to_command(space, *args, fov=None, out=None, **kw):
            """When `out` is given (Player 2 merge), run reference on a copy of `out`."""
            self.calls['to_command'] += 1
            self.calls['to_command_p2'] += out is not None
            fov_ref, out_ref = copy.deepcopy(fov), copy.deepcopy(out)
            r = _run(ActionSpace.to_command, space, *args, fov=fov_ref, out=out_ref, **kw)
            g = _run(rust.to_command, space, *args, fov=fov, out=out, **kw)
            _same_outcome(r, g, 'to_command')
            if fov is not None and fov_ref is not None:
                _same(fov_ref.dirs, fov.dirs, 'to_command fov.dirs')
            return g[0]

        def assign(view, world, env_ids, ally_code):
            self.calls['assign'] += 1
            self.calls['assign_p2'] += view.owners == (2, 1)
            view_ref = copy.deepcopy(view)
            r = _run(ref.assign, view_ref, world, env_ids, ally_code)
            g = _run(rust.assign, view, world, env_ids, ally_code)
            _same_outcome(r, g, 'assign')
            for name in ('ally_slots', 'enemy_slots', '_fresh'):
                _same(getattr(view_ref, name), getattr(view, name), f'assign {name}')

        def frame(view, world):
            self.calls['frame'] += 1
            self.calls['frame_p2'] += view.owners == (2, 1)
            view_ref = copy.deepcopy(view)
            r = _run(ref.frame, view_ref, world)
            g = _same_outcome(r, _run(rust.frame, view, world), 'frame')
            _same(view_ref._fresh, view._fresh, 'frame _fresh')
            _same(dict(sorted(view_ref._last.items())), dict(sorted(view._last.items())), '_last')
            return g

        def copy_rows(dst, src, rows):
            self.calls['copy_rows'] += 1
            dst_ref = copy.deepcopy(dst)
            r = _run(ref.copy_rows, dst_ref, src, rows)
            _same_outcome(r, _run(rust.copy_rows, dst, src, rows), 'copy_rows')
            _same(dst_ref, dst, 'copy_rows dst')

        def advance(backend, command, n_loops, world, rows):
            """Compare zgame fast path (writing directly into world) against copying rows from
            `_output()`; other backends compare via copy_rows."""
            self.calls['advance'] += 1
            world_ref = copy.deepcopy(world)
            if rust.advance is native_rust.advance and type(backend) is ZgameBackend:
                rust.advance(backend, command, n_loops, world, rows)
                self.calls['advance_rows'] += 1
                ref.copy_rows(world_ref, backend._output(), rows)
            else:
                out = backend.step(command, n_loops)
                ref.copy_rows(world_ref, out, rows)
                rust.copy_rows(world, out, rows)
            _same(world_ref, world, 'advance world')

        def fill_max_rewards(out, *args):
            self.calls['fill_max_rewards'] += 1
            out_ref = out.copy()
            r = _run(ref.fill_max_rewards, out_ref, *args)
            _same_outcome(r, _run(rust.fill_max_rewards, out, *args), 'fill_max_rewards')
            _same(out_ref, out, 'max_rewards')

        stats = ('battles_game', 'battles_won', 'timeouts', 'episode_count', 'done')

        def settle(env, active, end, cur, reward):
            self.calls['settle'] += 1
            before = {k: getattr(env, k).copy() for k in stats}
            r = _run(ref.settle, env, active, end, cur, reward)
            after = {k: getattr(env, k).copy() for k in stats}
            for k in stats:
                getattr(env, k)[...] = before[k]
            g = _same_outcome(r, _run(rust.settle, env, active, end, cur, reward), 'settle')
            for k in stats:
                _same(after[k], getattr(env, k), f'settle {k}')
            return g

        def settle_enemy(active, end, reward, max_rewards, config, infos):
            self.calls['settle_enemy'] += 1
            infos_ref = copy.deepcopy(infos)
            r = _run(ref.settle_enemy, active, end, reward, max_rewards, config, infos_ref)
            g = _run(rust.settle_enemy, active, end, reward, max_rewards, config, infos)
            g = _same_outcome(r, g, 'settle_enemy')
            _same(infos_ref, infos, 'settle_enemy infos')
            return g

        rs, ps = rust.services, ref.services
        svc = services_mod.Services(
            'checking',
            pure('pairwise', ps.pairwise, rs.pairwise),
            pure('grid_lookup', ps.grid_lookup, rs.grid_lookup),
            pure('ordered_sum', ps.ordered_sum, rs.ordered_sum),
        )
        semantic = {k: getattr(rust, k) for k in SEMANTIC}
        if self.tier == 'rust':
            semantic = self._semantic(pure)
        return accel.Kernels(
            tier=self.tier,
            services=svc,
            to_command=to_command,
            assign=assign,
            frame=frame,
            copy_rows=copy_rows,
            advance=advance,
            ally_tables=pure('ally_tables', ref.ally_tables, rust.ally_tables),
            enemy_tables=pure('enemy_tables', ref.enemy_tables, rust.enemy_tables),
            fill_max_rewards=fill_max_rewards,
            settle=settle,
            settle_enemy=settle_enemy,
            **semantic,
        )

    def _semantic(self, pure: Any) -> dict[str, Any]:
        """`rust`-tier semantics: run reference on copies of stateful components and Command."""
        ref, rust = accel.PYTHON, self.rust

        def on_copy(name: str, attrs: tuple[str, ...]) -> Any:
            def call(obj, *args, **kw):
                self.calls[name] += 1
                obj_ref = copy.deepcopy(obj)
                r = _run(getattr(ref, name), obj_ref, *args, **kw)
                g = _same_outcome(r, _run(getattr(rust, name), obj, *args, **kw), name)
                for a in attrs:
                    _same(getattr(obj_ref, a), getattr(obj, a), f'{name} {a}')
                return g

            return call

        def observe_spotter(ctx, spotter, streams, shuffle):
            self.calls['observe_spotter'] += 1
            sp_ref = copy.deepcopy(spotter)
            r = _run(ref.observe_spotter, ctx, sp_ref, streams, shuffle)
            g = _run(rust.observe_spotter, ctx, spotter, streams, shuffle)
            g = _same_outcome(r, g, 'observe_spotter')
            for a in ('seen', 'tag', 'gate'):
                _same(getattr(sp_ref, a), getattr(spotter, a), f'observe_spotter {a}')
            return g

        opening = ('target', 'pending')

        def scripted_by_rust() -> bool:
            """Return True when the scripted opponent is backed by Rust (`RustKernels._script`)."""
            rk = getattr(rust.opponent_issue, '__self__', None)
            memo = getattr(rk, '_script', None)
            return memo is not None and memo[1] is not None

        # Scripted opponent: Rust state is not copyable, so only count Rust-backed calls here;
        # per-call parity is in test_env_scripted_rust.py, full-episode parity is in E1.
        def opponent_reset(opponent, world, env_ids, names=None):
            if type(opponent) is not OpeningOrders:
                g = rust.opponent_reset(opponent, world, env_ids, names)
                self.calls['opponent_reset'] += scripted_by_rust()
                return g
            return on_copy('opponent_reset', opening)(opponent, world, env_ids, names)

        def opponent_issue(opponent, world, command, active):
            if type(opponent) is not OpeningOrders:
                g = rust.opponent_issue(opponent, world, command, active)
                self.calls['opponent_issue'] += scripted_by_rust()
                return g
            self.calls['opponent_issue'] += 1
            o_ref, c_ref = copy.deepcopy(opponent), copy.deepcopy(command)
            r = _run(ref.opponent_issue, o_ref, world, c_ref, active)
            g = _run(rust.opponent_issue, opponent, world, command, active)
            _same_outcome(r, g, 'opponent_issue')
            _same(c_ref, command, 'opponent_issue command')
            for a in opening:
                _same(getattr(o_ref, a), getattr(opponent, a), f'opponent_issue {a}')

        return dict(
            draw=pure('draw', ref.draw, rust.draw),
            avail=pure('avail', ref.avail, rust.avail),
            observe_spotter=observe_spotter,
            encode_obs=pure('encode_obs', ref.encode_obs, rust.encode_obs),
            encode_state=pure('encode_state', ref.encode_state, rust.encode_state),
            game_end=pure('game_end', ref.game_end, rust.game_end),
            game_end_dual=pure('game_end_dual', ref.game_end_dual, rust.game_end_dual),
            reward=on_copy('reward', ('dead_ally', 'dead_enemy')),
            spotter_after_step=on_copy('spotter_after_step', ('seen', 'tag', 'gate')),
            opponent_reset=opponent_reset,
            opponent_issue=opponent_issue,
        )


def _install(env: VecEnv, k: accel.Kernels) -> None:
    """Swap env kernels and update services held by semantic components."""
    env._k = k
    env.actions.ops = k.services
    env.rewarder.ops = k.services
    if env._p2 is not None:
        env._p2.actions.ops = k.services
        env._p2.rewarder.ops = k.services
    object.__setattr__(env.terrain, 'ops', k.services)


# ---------------------------------------------------------------- Parallel tier runs


def _envs(
    preset: str,
    scenario: str,
    over: Any,
    backend: Any,
    batch: int,
    tiers: tuple[str, ...] = ('python', 'services'),
    **kw: Any,
) -> list[VecEnv]:
    """Build envs for multiple tiers on the same config. `rust` starts from `services` and swaps
    in Rust kernels directly (bypassing classic/manifest checks tested in `test_resolve_*`)."""
    over = profile_overrides(over)[1] if isinstance(over, str) else over
    envs = []
    for t in tiers:
        a: Any = 'services' if t == 'rust' else t
        env = make_vec_env(
            preset, scenario, backend=backend, batch_size=batch, overrides=over, accel=a, **kw
        )
        assert env.accel == a
        if t == 'rust':
            _install(env, accel.kernels('rust'))
            env.accel = 'rust'
        envs.append(env)
    return envs


def _pair(preset: str, scenario: str, over: Any, backend: Any, batch: int, **kw: Any):
    return _envs(preset, scenario, over, backend, batch, **kw)


_COUNTERS = (
    'max_rewards',
    'reset_count',
    'episode_steps',
    'episode_count',
    'done',
    'battles_won',
    'battles_game',
    'timeouts',
)


def _compare(py: VecEnv, sv: VecEnv, where: str) -> None:
    """Compare obs, state, avail, and all system/semantic state: WorldState, views, tables,
    per-env counters, generator draws, spotter, and opponent state."""
    _same(py.get_obs(), sv.get_obs(), f'{where} obs')
    _same(py.get_state(), sv.get_state(), f'{where} state')
    _same(py.get_avail_actions(), sv.get_avail_actions(), f'{where} avail')
    _same(py._require(), sv._require(), f'{where} WorldState, frame, tables')
    _same(py.view.ally_slots, sv.view.ally_slots, f'{where} ally_slots')
    _same(py.view.enemy_slots, sv.view.enemy_slots, f'{where} enemy_slots')
    for name in _COUNTERS:
        _same(getattr(py, name), getattr(sv, name), f'{where} {name}')
    _same(py.attack_prob, sv.attack_prob, f'{where} attack_prob')
    _same(py.enemy_mask, sv.enemy_mask, f'{where} enemy_mask')
    _same(py.rewarder.dead_ally, sv.rewarder.dead_ally, f'{where} dead_ally')
    _same(py.rewarder.dead_enemy, sv.rewarder.dead_enemy, f'{where} dead_enemy')
    if py.spotter is not None and sv.spotter is not None:
        for a in ('seen', 'tag', 'gate'):
            _same(getattr(py.spotter, a), getattr(sv.spotter, a), f'{where} spotter.{a}')
    if type(py.opponent) is OpeningOrders and type(sv.opponent) is OpeningOrders:
        _same(py.opponent.target, sv.opponent.target, f'{where} opponent.target')
        _same(py.opponent.pending, sv.opponent.pending, f'{where} opponent.pending')
    assert py.opponent_names() == sv.opponent_names(), f'{where} opponent_names'
    _same(py.opponent_errors(), sv.opponent_errors(), f'{where} opponent_errors')
    warned = [getattr(e.opponent, '_warned', None) for e in (py, sv)]
    assert warned[0] == warned[1], f'{where} script warning state'
    p2, s2 = py._p2, sv._p2
    assert (p2 is None) == (s2 is None), where
    if p2 is not None and s2 is not None:  # two-sided mode: same checks for Player 2
        _same(py.get_obs(2), sv.get_obs(2), f'{where} p2 obs')
        _same(py.get_state(2), sv.get_state(2), f'{where} p2 state')
        _same(py.get_avail_actions(2), sv.get_avail_actions(2), f'{where} p2 avail')
        _same(p2.require(), s2.require(), f'{where} p2 frame and tables')
        for name in ('ally_slots', 'enemy_slots'):
            _same(getattr(p2.view, name), getattr(s2.view, name), f'{where} p2 {name}')
        for name in ('max_rewards', 'last_action'):
            _same(getattr(p2, name), getattr(s2, name), f'{where} p2 {name}')
        for name in ('dead_ally', 'dead_enemy'):
            _same(getattr(p2.rewarder, name), getattr(s2.rewarder, name), f'{where} p2 {name}')
        if p2.spotter is not None and s2.spotter is not None:
            for a in ('seen', 'tag', 'gate'):
                _same(getattr(p2.spotter, a), getattr(s2.spotter, a), f'{where} p2 spotter.{a}')


def drive(
    envs: list[VecEnv],
    n_steps: int,
    *,
    seed: int = 0,
    reset_every: int = 13,
    reseed: dict[int, dict[int, int]] | None = None,
    scripts: dict[int, dict[int, str]] | None = None,
) -> dict[str, int]:
    """Step all envs with identical actions for n_steps, comparing every frame against the first;
    reset finished envs immediately plus one partial reset every reset_every steps. reseed and
    scripts apply per-step overrides on reset. Returns episode and step counts."""
    rng = np.random.default_rng(seed)
    py, others = envs[0], envs[1:]
    for e in envs:
        e.reset()
    for sv in others:
        _compare(py, sv, f'reset ({sv.accel})')
    b_size = py.batch_size
    episodes = 0
    for t in range(n_steps):
        avail = py.get_avail_actions()
        acts = (rng.random(avail.shape) * avail).argmax(-1)
        step: tuple[Any, ...] = (acts,)
        if py.dual:  # two-sided mode: Player 2 also picks random legal actions
            avail2 = py.get_avail_actions(2)
            step = (acts, (rng.random(avail2.shape) * avail2).argmax(-1))
        r1, d1, i1 = py.step(*step)
        for sv in others:
            r2, d2, i2 = sv.step(*step)
            where = f'step {t} ({sv.accel})'
            _same(r1, r2, f'{where} reward')
            _same(d1, d2, f'{where} terminated')
            assert json.dumps(i1) == json.dumps(i2), where
            _same(i1, i2, f'{where} infos')
            _compare(py, sv, where)
        ids = set(np.flatnonzero(d1).tolist())
        episodes += len(ids)
        if t % reset_every == reset_every - 1:
            ids.add(t % b_size)
        new_seeds = (reseed or {}).get(t, {})
        names = (scripts or {}).get(t, {})
        ids |= set(new_seeds) | set(names)
        if ids:
            order = sorted(ids)
            kw: dict[str, Any] = {}
            if new_seeds:
                kw['seeds'] = [new_seeds.get(b) for b in order]
            if names:
                kw['opponents'] = [names.get(b) for b in order]
            o1, s1 = py.reset(order, **kw)
            for sv in others:
                where = f'step {t} ({sv.accel})'
                o2, s2 = sv.reset(order, **kw)
                _same((o1, s1), (o2, s2), f'{where} reset')
                _compare(py, sv, f'{where} after reset')
                assert py.opponent_names() == sv.opponent_names(), where
    for sv in others:
        _same(py.max_rewards, sv.max_rewards, 'max_rewards')
        assert py.get_stats() == sv.get_stats()
    return {'episodes': episodes, 'steps': n_steps * b_size}


# ---------------------------------------------------------------- Tiers


@pytest.fixture
def manifest_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat manifest as current so tier tests ignore local edits (see test_manifest_current)."""
    monkeypatch.setattr(manifest_mod, 'mismatches', lambda: ())
    monkeypatch.setattr(accel, '_warned', set())


def test_resolve_tiers(manifest_ok: None) -> None:
    assert accel.resolve('python') == 'python'
    assert accel.resolve('services') == 'services'
    assert accel.resolve('auto') == 'services'  # rust requires a config
    with pytest.raises(ValueError, match='make_vec_env'):
        accel.resolve('rust')
    with pytest.raises(ValueError, match='accel'):
        accel.resolve('fast')
    with pytest.raises(ValueError, match='accel'):
        make_vec_env('smac-portable', '3m', backend='pyref', accel='fast')  # type: ignore[arg-type]
    # Classic recipes: the 3 presets + scenario defaults; backend.* / seeds / batch are allowed
    aligned = profile_overrides('aligned-v4')[1]
    for preset, scenario, over in [
        ('smac-portable', '3m', None),
        ('smac-portable', 'corridor', aligned),
        ('smacv2-portable', '10gen_terran', None),
        ('smacv2-portable', '8m', aligned),
        ('smac-hard-portable', '3m', None),
        ('smac-hard-portable', 'mmmt', aligned),
    ]:
        env = make_vec_env(preset, scenario, backend='pyref', batch_size=2, seed=0, overrides=over)
        assert env.accel == 'rust', (preset, scenario)
        assert classic_mod.rust_blockers(env.config, env.spec) == []
    env = make_vec_env(
        'smac-portable', '3m', backend='zgame', seeds=[4, 9], batch_size=2, accel='rust'
    )
    assert env.accel == 'rust'
    one = make_env('smac-portable', '3m', backend='pyref', seed=0, accel='rust')
    assert one.vec.accel == 'rust'  # type: ignore[union-attr]
    # Non-classic recipes: auto falls back to services; explicit rust raises naming the blocker
    for preset, scenario, over, why in [
        ('smac-hard-portable', '3m', {'opponent.scripts': ('8m',)}, 'opponent.scripts'),
        ('smac-hard-portable', '3m', {'opponent.script_abilities': 'issue'}, 'script_abilities'),
        ('smac-portable', '3m', {'quirks.obs_shuffle': True}, 'quirks.obs_shuffle'),
        ('smacv2-portable', '10gen_terran', {'conic_fov': True}, 'visibility.model=cone'),
        ('smacv2-portable', '10gen_terran', {'prob_obs_enemy': 0.5}, 'smacv2-portable'),
    ]:
        env = make_vec_env(preset, scenario, backend='pyref', seed=0, overrides=over)
        assert env.accel == 'services', (preset, scenario, over)
        with pytest.raises(ValueError, match=why):
            make_vec_env(preset, scenario, backend='pyref', seed=0, overrides=over, accel='rust')
    env = make_vec_env('smac-portable', '3m', backend='pyref', seed=0, accel='python')
    assert env.accel == 'python'
    env = make_vec_env('smac-portable', '3m', backend='pyref', seed=0, accel='services')
    assert env.accel == 'services'
    # Two-sided mode: external opponent + symmetric profile is classic (opponent group skipped);
    # modifying other groups still disqualifies it
    for preset, scenario, over in [
        ('smac-portable', '3m', DUAL_V4),
        ('smacv2-portable', '10gen_terran', DUAL_AXO),
        ('smac-hard-portable', '2s3z', DUAL_V4),
        ('smac-hard-portable', 'MMM', {**DUAL_AXO, 'opponent.scripts': ('MMM',)}),
    ]:
        env = make_vec_env(preset, scenario, backend='pyref', batch_size=2, seed=0, overrides=over)
        assert env.dual and env.accel == 'rust', (preset, scenario)
    # Allowed key: continuing_episode only affects timeout info; classic in both single & dual
    for preset, scenario, over in [
        ('smac-portable', '3m', {**aligned, 'termination.continuing_episode': True}),
        ('smac-hard-portable', '2s3z', {'continuing_episode': True}),
        ('smac-hard-portable', 'MMM', {**DUAL_V4, 'termination.continuing_episode': True}),
    ]:
        env = make_vec_env(preset, scenario, backend='pyref', seed=0, overrides=over)
        assert env.config.termination.continuing_episode, (preset, scenario)
        assert env.accel == 'rust', (preset, scenario)
    dual_obs = {**DUAL_V4, 'quirks.obs_shuffle': True}
    env = make_vec_env('smac-portable', '3m', backend='pyref', overrides=dual_obs)
    assert env.accel == 'services'
    with pytest.raises(ValueError, match='quirks.obs_shuffle'):
        make_vec_env('smac-portable', '3m', backend='pyref', overrides=dual_obs, accel='rust')
    # Configs invalid on pyref/zgame: test blocker logic directly (ability segment, SC2 preset)
    for preset, scenario, over, why in [
        ('smac-hard-portable', '3m', {'use_ability': True}, 'actions.ability'),
        ('smac-hard', '3m', None, 'quirks.timestep_pad'),
    ]:
        config, spec = build_config(preset, scenario, over)
        assert any(why in m for m in classic_mod.rust_blockers(config, spec)), (preset, over)


class _CustomScript(SCRIPTS['8m']):  # type: ignore[misc, valid-type]
    """Custom script class (same behavior as built-in 8m)."""


def test_resolve_script_registry(manifest_ok: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifying the smac-hard-portable script registry or pool disqualifies classic rust."""
    with monkeypatch.context() as m:
        m.setitem(SCRIPTS, '8m', _CustomScript)
        env = make_vec_env('smac-hard-portable', '3m', backend='pyref', seed=0)
        assert env.accel == 'services'
        with pytest.raises(ValueError, match='(?i)built-in'):
            make_vec_env('smac-hard-portable', '3m', backend='pyref', seed=0, accel='rust')
        # Scenarios not using 8m are unaffected
        assert make_vec_env('smac-hard-portable', '2s3z', backend='pyref', seed=0).accel == 'rust'
    with monkeypatch.context() as m:
        m.setitem(POOLS, '3m', ('base', 'atkw'))
        with pytest.raises(ValueError, match='(?i)script pool'):
            make_vec_env('smac-hard-portable', '3m', backend='pyref', seed=0, accel='rust')
    assert make_vec_env('smac-hard-portable', '3m', backend='pyref', seed=0).accel == 'rust'


def test_resolve_stale_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale manifest: auto falls back to services with a one-time warning; explicit rust raises."""
    why = 'semantic source differs from manifest: env/reward.py (...)'
    monkeypatch.setattr(manifest_mod, 'mismatches', lambda: (why,))
    monkeypatch.setattr(accel, '_warned', set())
    with pytest.warns(RuntimeWarning, match='env/reward.py'):
        env = make_vec_env('smac-portable', '3m', backend='pyref', seed=0)
    assert env.accel == 'services'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert make_vec_env('smac-portable', '8m', backend='pyref', seed=0).accel == 'services'
    with pytest.raises(ValueError, match='env/reward.py'):
        make_vec_env('smac-portable', '3m', backend='pyref', seed=0, accel='rust')


def test_auto_without_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, 'native_available', lambda: False)
    assert accel.resolve('auto') == 'python'
    with pytest.raises(ImportError, match='zgame'):
        accel.resolve('services')
    with pytest.raises(ImportError, match='zgame'):
        accel.resolve('rust')
    env = make_vec_env('smac-portable', '3m', backend='pyref', seed=0)
    assert env.accel == 'python'
    with pytest.raises(ImportError, match='zgame'):
        make_vec_env('smac-portable', '3m', backend='pyref', seed=0, accel='rust')


def test_manifest_current() -> None:
    """Checked-in manifest matches current semantic-layer sources, NumPy range, and zgame."""
    problems = manifest_mod.mismatches()
    assert not problems, '\n'.join(problems)


def test_manifest_diff_names_what_changed() -> None:
    got = manifest_mod.current()
    assert manifest_mod.diff(got, got) == []
    assert 'unismac.env.manifest --write' in manifest_mod.diff(None, got)[0]
    want = copy.deepcopy(got)
    want['files']['env/reward.py'] = '0' * 64
    want['functions']['env/core.py:VecEnv.step'] = '0' * 64
    want['numpy_min'] = '99.0'
    msgs = manifest_mod.diff(want, got)
    assert any('env/reward.py' in m and 'unismac.env.manifest --write' in m for m in msgs), msgs
    assert any('VecEnv.step' in m for m in msgs), msgs
    assert any('numpy' in m and '99.0' in m for m in msgs), msgs
    assert sorted(got['files']) == sorted(manifest_mod.SOURCES)
    # LF normalization produces identical hashes for CRLF vs LF inputs
    assert manifest_mod._sha(b'a\r\nb\r\n') == manifest_mod._sha(b'a\nb\n')


# ---------------------------------------------------------------- E0: per-function


E0_CASES: list[tuple[str, str, Any, str]] = [
    ('smac-portable', '8m', 'aligned-v4', 'zgame'),
    ('smac-portable', 'MMM', 'acc-ram-cov-axo', 'pyref'),
    ('smac-portable', '2s3z', NO_MASK, 'zgame'),
    ('smacv2-portable', '10gen_terran', STOCH, 'zgame'),
    ('smacv2-portable', '10gen_protoss', None, 'pyref'),
    ('smac-hard-portable', 'MMM', 'aligned-v4', 'zgame'),
    ('smac-hard-portable', '3s5z', 'acc-ram-cov-axo', 'pyref'),
]


E0_RUST_CASES: list[tuple[str, str, Any, str]] = [
    ('smac-portable', '8m', 'aligned-v4', 'zgame'),
    ('smac-portable', 'MMM', 'acc-ram-cov-axo', 'pyref'),
    ('smac-portable', 'MMM', 'aligned-v4', 'zgame'),
    ('smac-portable', '2s3z', NO_MASK, 'zgame'),
    ('smacv2-portable', '10gen_terran', STOCH_CIRCLE, 'zgame'),
    ('smacv2-portable', '10gen_protoss', None, 'pyref'),
    ('smacv2-portable', '10gen_zerg', 'aligned-v4', 'zgame'),
    ('smac-hard-portable', 'MMM', 'aligned-v4', 'zgame'),
    ('smac-hard-portable', '3s5z', 'acc-ram-cov-axo', 'pyref'),
    ('smac-hard-portable', '7q_vs_2bc', None, 'zgame'),
]


def e0_case(
    preset: str,
    scenario: str,
    over: Any,
    backend: str,
    tier: str = 'services',
    n_steps: int = 60,
    strict: bool = True,
) -> tuple[Counter, Counter]:
    """Run one tier wrapped in Checking alongside python (also E1); return per-function call
    counts and fallback counts (must be 0 when strict)."""
    py, sv = _envs(preset, scenario, over, backend, 4, ('python', tier), seeds=[3, 5, 8, 13])
    chk = Checking(tier)
    _install(sv, chk.kernels())
    scripted = sv.config.opponent.kind == 'scripted'
    scripts = {20: {1: sv.opponent.pool[0]}} if scripted else None  # type: ignore[attr-defined]
    before = Counter(native.fallbacks)
    drive([py, sv], n_steps, reseed={30: {2: 99}}, scripts=scripts)
    fell = native.fallbacks - before
    if strict:
        assert not fell, fell
    return chk.calls, fell


@pytest.mark.parametrize(('preset', 'scenario', 'over', 'backend'), E0_CASES)
def test_e0_functions_on_trajectories(preset: str, scenario: str, over: Any, backend: str) -> None:
    calls, _ = e0_case(preset, scenario, over, backend)
    for name in ('to_command', 'assign', 'frame', 'copy_rows', 'advance', 'settle', 'pairwise'):
        assert calls[name] > 0, (name, calls)
    for name in ('ally_tables', 'enemy_tables', 'fill_max_rewards', 'grid_lookup', 'ordered_sum'):
        assert calls[name] > 0, (name, calls)


@pytest.mark.parametrize(('preset', 'scenario', 'over', 'backend'), E0_RUST_CASES)
def test_e0_rust_functions_on_trajectories(
    preset: str, scenario: str, over: Any, backend: str
) -> None:
    calls, _ = e0_case(preset, scenario, over, backend, tier='rust')
    names = ['to_command', 'assign', 'frame', 'advance', 'settle', 'avail', 'encode_obs']
    names += ['encode_state', 'game_end', 'reward', 'opponent_reset', 'opponent_issue']
    if scenario.startswith('10gen'):
        names.append('draw')
    if over is STOCH_CIRCLE:
        names += ['observe_spotter', 'spotter_after_step']
    if backend == 'zgame':
        names.append('advance_rows')
    for name in names:
        assert calls[name] > 0, (name, calls)


E0_DUAL_CASES: list[tuple[str, str, Any, str]] = [
    ('smac-portable', '8m', DUAL_V4, 'zgame'),
    ('smac-portable', 'MMM', DUAL_AXO, 'pyref'),
    ('smac-portable', '3s_vs_5z', DUAL_V4, 'zgame'),
    ('smacv2-portable', '10gen_terran', DUAL_STOCH, 'zgame'),
    ('smac-hard-portable', '2s3z', DUAL_AXO, 'pyref'),
    ('smac-hard-portable', 'MMM', DUAL_V4, 'zgame'),
]
E0_DUAL_IDS = ['/'.join((p, s, b)) for p, s, _, b in E0_DUAL_CASES]


@pytest.mark.parametrize('tier', ['services', 'rust'])
@pytest.mark.parametrize(('preset', 'scenario', 'over', 'backend'), E0_DUAL_CASES, ids=E0_DUAL_IDS)
def test_e0_dual(preset: str, scenario: str, over: Any, backend: str, tier: str) -> None:
    """Two-sided mode: check Player 2 views, merged commands, and settlement (plus rust-tier
    termination, avail, obs, state, reward) per-function on trajectories with zero fallbacks."""
    calls, _ = e0_case(preset, scenario, over, backend, tier=tier)
    names = ['to_command_p2', 'assign_p2', 'frame_p2', 'settle', 'settle_enemy']
    if tier == 'rust':
        names += ['game_end_dual', 'avail', 'encode_obs', 'encode_state', 'reward']
    if over is DUAL_STOCH and tier == 'rust':
        names += ['observe_spotter', 'spotter_after_step', 'draw']
    for name in names:
        assert calls[name] > 0, (name, calls)
    assert calls['game_end'] == 0, calls  # two-sided mode uses game_end_dual only


@pytest.mark.parametrize(
    ('preset', 'scenario', 'over', 'allowed'),
    [
        (
            'smacv2-portable',
            '10gen_terran',
            STOCH,
            {'avail', 'observe_spotter', 'encode_obs', 'encode_state'},
        ),
        ('smac-hard-portable', 'MMM', {'opponent.scripts': ('MMM', 'custom_mmm')}, set()),
    ],
    ids=['cone', 'custom-script'],
)
def test_e0_rust_unsupported_falls_back(
    preset: str, scenario: str, over: Any, allowed: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported Rust configs (conic FOV; custom script class in pool): affected functions
    fall back to reference while the rest stay on Rust, producing bitwise-identical results."""
    monkeypatch.setitem(SCRIPTS, 'custom_mmm', type('CustomMmm', (SCRIPTS['MMM'],), {}))
    calls, fell = e0_case(preset, scenario, over, 'zgame', tier='rust', n_steps=30, strict=False)
    assert set(fell) <= allowed, fell
    if over is STOCH:
        assert fell['encode_obs'] > 0 and fell['observe_spotter'] > 0, fell
    else:
        assert calls['opponent_issue'] == 0  # custom scripted opponent runs Python method directly
    assert calls['game_end'] > 0 and calls['reward'] > 0 and calls['encode_state'] > 0


def _synthetic(rng: np.random.Generator, b_size: int, n_agents: int, n_enemies: int, n_fov: int):
    cfg = ActionConfig(
        target_pad='max_units_9', ability=True, mask_by_range=False, range_check=True
    )
    space = ActionSpace(
        n_agents, n_enemies, cfg, heal_enabled=True, heal_map=True, n_fov=n_fov, ops=native.SERVICES
    )
    n_slots = n_agents + n_enemies + 2
    perm = np.array([rng.permutation(n_slots) for _ in range(b_size)])

    def side(slots: np.ndarray) -> Side:
        shape = slots.shape
        f = lambda lo, hi: rng.uniform(lo, hi, shape)  # noqa: E731
        health = np.where(rng.random(shape) < 0.3, 0.0, f(1, 50))
        return Side(
            slots, health > 0, np.full(shape, 48, np.int32), np.ones(shape, np.int8),
            f(0, 7), f(0, 7), health, f(50, 60), f(0, 5), f(5, 6), f(0, 200), f(0, 15),
        )  # fmt: skip

    frame = Frame(
        side(perm[:, :n_agents].copy()), side(perm[:, n_agents : n_agents + n_enemies].copy())
    )
    shape = (b_size, n_agents)
    tables = AllyTables(
        np.zeros(shape, np.int64), np.full(shape, 15.0), np.full(shape, np.nan),
        rng.random(shape) < 0.3, rng.uniform(2, 9, shape), np.full(shape, 9.0),
    )  # fmt: skip
    fov = FovState(b_size, n_agents, n_fov) if n_fov else None
    if fov is not None:
        fov.dirs[:] = rng.normal(size=fov.dirs.shape)
    keys = rng.integers(0, 2**63, (2, b_size), dtype=np.uint64)
    streams = rng_mod.Streams(
        keys[0], keys[1], rng.integers(0, 60, b_size).astype(np.uint64),
        rng.integers(0, 9, b_size).astype(np.uint64),
    )  # fmt: skip
    choices = np.array([0, *ABILITY_DICT])
    choice = choices[rng.integers(0, len(choices), shape)]
    return space, frame, tables, fov, streams, choice, n_slots


@pytest.mark.parametrize('n_fov', [0, 4])
def test_e0_to_command_synthetic(n_fov: int) -> None:
    """All to_command branches (including ability segment and heal): on random frames/actions,
    Rust matches reference Command, fov dirs, and exceptions bitwise."""
    rng = np.random.default_rng(n_fov)
    outcomes: Counter[str] = Counter()
    for trial in range(300):
        b_size, n_agents, n_enemies = (
            int(rng.integers(1, 5)),
            int(rng.integers(1, 7)),
            int(rng.integers(1, 8)),
        )
        space, frame, tables, fov, streams, choice, n_slots = _synthetic(
            rng, b_size, n_agents, n_enemies, n_fov
        )
        k = space.n_actions
        acts = rng.integers(0, k, (b_size, n_agents))
        if trial % 7 == 0:
            acts[0, 0] = k + 1 if trial % 2 else -1  # out of range
        avail = (rng.random((b_size, n_agents, k)) < (0.6 if trial % 5 == 0 else 1.01)).astype(
            np.int64
        )
        active = rng.random(b_size) < 0.8
        prob = rng.random((b_size, n_agents))
        kw = dict(attack_prob=prob, attack_rng=streams) if trial % 2 else {}
        fov_ref = copy.deepcopy(fov)
        r = _run(
            ActionSpace.to_command, space, acts, active, avail, frame, tables, n_slots,
            fov=fov_ref, ability_choice=choice, **kw,
        )  # fmt: skip
        g = _run(
            native.to_command, space, acts, active, avail, frame, tables, n_slots,
            fov=fov, ability_choice=choice, **kw,
        )  # fmt: skip
        try:
            _same_outcome(r, g, f'trial {trial}')
            outcomes['ok'] += 1
        except AssertionError:
            raise
        except Exception as exc:
            outcomes[type(exc).__name__] += 1
        if fov is not None and fov_ref is not None:
            _same(fov_ref.dirs, fov.dirs, f'trial {trial} fov')
    assert outcomes['ok'] > 100 and len(outcomes) > 1, outcomes


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_e0_services_random() -> None:
    """Numeric services match NumPy bitwise on random inputs (0, subnormals, huge, inf, NaN, -0)."""
    rng = np.random.default_rng(11)
    special = np.array([0.0, -0.0, 5e-324, -1e-310, 1e300, -1e300, np.inf, -np.inf, np.nan, 1.0])

    def coords(shape: tuple[int, ...]) -> np.ndarray:
        x = rng.uniform(-40, 40, shape) * 10.0 ** rng.integers(-3, 4, shape)
        mask = rng.random(shape) < 0.05
        x[mask] = rng.choice(special, int(mask.sum()))
        return x

    for b, a, n in [(1, 1, 1), (3, 5, 7), (64, 8, 8), (128, 10, 11), (2, 1, 9)]:
        ax, ay, bx, by = (coords(s) for s in [(b, a), (b, a), (b, n), (b, n)])
        _same(native.pairwise(ax, ay, bx, by), services_mod.pairwise(ax, ay, bx, by), 'pairwise')
        got = native.pairwise(ax, ay, ax, ay, exclude_self=True)
        _same(got, services_mod.pairwise(ax, ay, ax, ay, exclude_self=True), 'exclude_self')
        values, mask = coords((b, n)), rng.random((b, n)) < 0.7
        _same(native.ordered_sum(values, mask), services_mod.ordered_sum(values, mask), 'sum')
    for gx, gy in [(1, 1), (32, 32), (17, 40)]:
        grid_b, grid_f = rng.random((gx, gy)) < 0.5, rng.random((gx, gy))
        xi, yi = rng.integers(-3, gx + 3, (64, 9)), rng.integers(-3, gy + 3, (64, 9))
        for grid, outside in ((grid_b, False), (grid_b, 1.0), (grid_f, 1.0), (grid_f, -0.5)):
            want = services_mod.grid_lookup(grid, xi, yi, outside)
            _same(native.grid_lookup(grid, xi, yi, outside), want, 'grid_lookup')
    assert not native.fallbacks['pairwise'] and not native.fallbacks['grid_lookup']


def _philox(col: list[int]) -> np.random.Generator:
    key = np.array(col[:2], np.uint64)
    return np.random.Generator(
        np.random.Philox(key=key, counter=np.array([0, 0, col[2], col[3]], np.uint64))
    )


def _np_call(g: np.random.Generator, call: tuple[Any, ...]) -> Any:
    name, *args = call
    if name == 'permutation':
        return g.permutation(args[0]).tolist()
    if name == 'integers':
        return int(g.integers(args[0], args[1]))
    if name == 'choice':
        return g.choice(args[0], size=(args[1],), replace=False).tolist()
    if name == 'choice_p':
        return g.choice(len(args[0]), size=(args[1],), p=args[0]).tolist()
    if name == 'multinomial':
        return g.multinomial(args[0], args[1]).tolist()
    if name == 'uniform':
        return float(g.uniform(args[0], args[1]))
    return float(g.random())


def _random_call(r: np.random.Generator) -> tuple[Any, ...]:
    k = int(r.integers(0, 7))
    if k == 0:
        return ('permutation', int(r.integers(0, 12)))
    if k == 1:
        lo = int(r.integers(-(2**40), 5))
        span = [1, 2, 7, 2**32 - 1, 2**32, 2**32 + 1, 2**33, 2**62]
        return ('integers', lo, lo + int(r.choice(span)))
    if k == 2:
        pop = int(r.integers(1, 10))
        return ('choice', pop, int(r.integers(0, pop + 1)))
    if k == 3:
        m = int(r.integers(1, 6))
        p = r.random(m) * (r.random(m) > 0.2) + 1e-3
        return ('choice_p', (p / p.sum()).tolist(), int(r.integers(0, 9)))
    if k == 4:
        m = int(r.integers(1, 6))
        return ('multinomial', int(r.integers(0, 40)), r.dirichlet(np.ones(m)).tolist())
    if k == 5:
        lo = float(r.uniform(-10, 10))
        return ('uniform', lo, lo + float(r.uniform(0, 30)))
    return ('random',)


def _same_value(want: Any, got: Any, where: Any) -> None:
    if isinstance(want, float):
        assert np.float64(want).tobytes() == np.float64(got).tobytes(), where
    else:
        assert want == got, where


def test_e0_rust_distributions() -> None:
    """zgame-core `npgen` (NumPy 2.5.4 `Generator` distributions): matches bitwise across random
    streams/args/call sequences; invalid args raise Fallback so reference raises the same error."""
    env = native._env
    for seed in range(1500):
        r = np.random.default_rng(seed)
        col = [int(x) for x in r.integers(0, 2**63, 2)]
        col += [int(r.integers(0, 50)), int(r.integers(0, 9))]
        calls = [_random_call(r) for _ in range(30)]
        g = _philox(col)
        want = [_np_call(g, c) for c in calls]
        got = env.np_calls(col, calls)
        for c, w, x in zip(calls, want, got, strict=True):
            _same_value(w, x, (seed, c))
    bad = [
        ('integers', 5, 5),
        ('choice_p', [0.5, 0.6], 3),
        ('choice_p', [1.5, -0.5], 3),
        ('choice_p', [float('nan'), 1.0], 1),
        ('multinomial', 3, [0.7, 0.4, 0.1]),
        ('multinomial', 3, [-0.1, 1.1]),
        ('uniform', 1.0, 0.0),
        ('uniform', -1e308, 1e308),
        ('choice', 3, 4),
    ]
    for call in bad:
        with pytest.raises(Exception):  # noqa: B017  NumPy raises varied exception types
            _np_call(_philox([1, 2, 3, 4]), call)
        with pytest.raises(native.Fallback):
            env.np_calls([1, 2, 3, 4], [call])
    # NumPy only validates the first d - 1 weights: last entry outside [0, 1] is accepted
    col = [1, 2, 3, 4]
    assert env.np_calls(col, [('multinomial', 3, [0.7, 0.7])]) == [
        _np_call(_philox(col), ('multinomial', 3, [0.7, 0.7]))
    ]


def _random_generator(r: np.random.Generator) -> generators_mod.ScenarioGenerator:
    names = list(generators_mod.UNIT_NAMES)
    n_types = int(r.integers(1, 5))
    types = tuple(str(t) for t in r.choice(names, n_types, replace=False))
    exc = tuple(t for t in types if r.random() < 0.3)
    if set(types) <= set(exc):
        exc = exc[1:]
    w = r.random(n_types) * (r.random(n_types) > 0.3)
    w[[t not in exc for t in types].index(True)] += (
        0.3  # ensure non-exception weight > 0 so reference draw terminates
    )
    if r.random() < 0.9:
        w = w / w.sum()
    kind = str(r.choice(['surrounded_and_reflect', 'surrounded', 'reflect_position']))
    lo = float(r.uniform(-0.2, 1))
    hi = lo + float(r.uniform(-0.1, 0.8))
    cap = CapabilityConfig(
        n_units=int(r.integers(1, 11)),
        n_enemies=int(r.integers(1, 13)),
        team_gen=TeamGenConfig(
            unit_types=types, weights=tuple(w.tolist()), exception_unit_types=exc
        )
        if r.random() < 0.9
        else None,
        start_positions=StartPositionsConfig(
            dist_type=kind,  # type: ignore[arg-type]
            p=float(r.choice([0.0, 0.5, 1.0, r.random()])),
            map_x=float(r.choice([32, 33, 40])),
            map_y=float(r.choice([32, 28, 64])),
        )
        if r.random() < 0.9
        else None,
        attack=AttackConfig(lower_bound=(lo,), upper_bound=(hi,)) if r.random() < 0.7 else None,
        enemy_mask=EnemyMaskConfig(mask_probability=float(r.choice([0.0, 1.0, r.random(), 1.2])))
        if r.random() < 0.7
        else None,
    )
    assert cap.n_units is not None and cap.n_enemies is not None
    return generators_mod.ScenarioGenerator(cap, cap.n_units, cap.n_enemies)


def test_e0_rust_draw_random() -> None:
    """Randomized ScenarioGenerator configs (types, zero/unnormalized weights, exceptions, all 3
    spawn dists + p, map sizes, reversed attack bounds, mask prob 0/1/>1): bitwise parity."""
    r = np.random.default_rng(5)
    outcomes: Counter[str] = Counter()
    for trial in range(400):
        gen = _random_generator(r)
        b_size = int(r.integers(1, 4))
        keys = r.integers(0, 2**63, (2, b_size), dtype=np.uint64)
        streams = rng_mod.Streams(
            keys[0], keys[1], np.zeros(b_size, np.uint64),
            r.integers(0, 9, b_size).astype(np.uint64),
        )  # fmt: skip
        ids = sorted(set(r.integers(0, b_size, b_size).tolist()))
        ref = _run(generators_mod.draw_batch, gen, streams, ids)
        got = _run(native_rust.RustKernels().draw, gen, streams, ids)
        try:
            _same_outcome(ref, got, f'trial {trial}')
            outcomes['ok'] += 1
        except AssertionError:
            raise
        except Exception as exc:
            outcomes[type(exc).__name__] += 1
    assert outcomes['ok'] > 250 and len(outcomes) > 1, outcomes


def test_e0_streams_match_rng_stream() -> None:
    """`rng_streams` row b matches `rng_stream`, and Rust's double draws match NumPy (verified
    indirectly via stochastic_attack; directly checks stream keys here)."""
    env = make_vec_env('smacv2-portable', '10gen_terran', backend='pyref', batch_size=3, seed=21)
    env.reset()
    env.reset([2], seeds=[8])
    env.reset([0])
    frames = np.array([4, 0, 7])
    st = env.rng_streams('attack', frames)
    for b in range(3):
        a, c = st(b).random(5), env.rng_stream('attack', b, int(frames[b])).random(5)
        assert np.array_equal(a, c)
    assert st.table().dtype == np.uint64 and st.table().shape == (4, 3)


def test_e0_fallbacks_give_reference_errors() -> None:
    """Inputs unhandled by Rust fall back to reference with identical errors and unchanged state."""
    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=0)
    env.reset()
    acts = np.zeros((2, env.n_agents), np.int64)  # live units cannot take no-op
    with pytest.raises(ValueError, match='(?i)action'):
        env.step(acts)
    world = env.world
    view = copy.deepcopy(env.view)
    bad = dataclasses.replace(world, x=world.x.copy())
    bad.x[0, 0] = np.nan  # NaN sort key falls back to reference
    k = accel.kernels('services')
    before = Counter(native.fallbacks)
    k.assign(view, bad, [0], env.catalog.ally_code)
    assert native.fallbacks['assign'] == before['assign'] + 1
    ref = copy.deepcopy(env.view)
    accel.PYTHON.assign(ref, bad, [0], env.catalog.ally_code)
    _same(ref.ally_slots, view.ally_slots, 'assign with NaN')
    gone = dataclasses.replace(world, alive=world.alive.copy())
    gone.alive[1, 0] = False
    with pytest.raises(Exception, match='(?i)scenario requires') as e_rust:
        k.assign(copy.deepcopy(env.view), gone, [1], env.catalog.ally_code)
    with pytest.raises(Exception, match='(?i)scenario requires') as e_ref:
        accel.PYTHON.assign(copy.deepcopy(env.view), gone, [1], env.catalog.ally_code)
    assert type(e_rust.value) is type(e_ref.value) and str(e_rust.value) == str(e_ref.value)
    # Unsupported dtype (float32) falls back to reference and matches NumPy
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    _same(native.pairwise(x, x, x, x), services_mod.pairwise(x, x, x, x), 'float32 pairwise')
    cmd = Command.noop(1, 1)
    assert cmd.kind.dtype == np.int8


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 CPU feature toggle is only applicable on x86_64',
)
@pytest.mark.parametrize('tier', ['services', 'rust'])
def test_e0_without_avx512(tier: str) -> None:
    """Re-run E0 parity checks in a subprocess with NumPy AVX-512 dispatch disabled."""
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    if tier == 'services':
        body = (
            'for case in m["E0_CASES"] + m["E0_DUAL_CASES"]:\n'
            '    m["e0_case"](*case)\n'
            'm["test_e0_to_command_synthetic"](4)\n'
            'm["test_e0_services_random"]()\n'
        )
    else:
        body = (
            'for case in m["E0_RUST_CASES"] + m["E0_DUAL_CASES"]:\n'
            '    m["e0_case"](*case, tier="rust")\n'
            'm["test_e0_rust_distributions"]()\n'
            'm["test_e0_rust_draw_random"]()\n'
        )
    code = (
        'import runpy\n'
        'import importlib\n'
        'umath = importlib.import_module("numpy._core._multiarray_umath")\n'
        'assert not umath.__cpu_features__.get("AVX512_SKX")\n'
        f'm = runpy.run_path({__file__!r})\n'
    ) + body
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- E1: VecEnv level


E1_CELLS = [(p, s, prof) for p in PRESETS for s in SCENARIOS for prof in PROFILES]
E1_CELLS += [('smacv2-portable', s, prof) for s in GEN_SCENARIOS for prof in PROFILES]
# In the E1 grid, only E1_PYREF cells run on pyref in default `make test`; the remaining pyref
# cells are marked slow (`make test-slow`). Each cell runs 3 tiers (pyref 2-66s vs zgame
# 0.1-0.4s), and env-layer code is backend-agnostic, so zgame runs all cells and pyref runs one
# per preset.
E1_PYREF = (
    ('smac-portable', 'MMM', 'aligned-v4'),
    ('smacv2-portable', '10gen_zerg', 'acc-ram-cov-axo'),
    ('smac-hard-portable', '3s5z', 'acc-ram-cov-axo'),
)


def _e1_tiers(preset: str) -> tuple[str, ...]:
    rust = ('rust',) if preset in classic_mod.CLASSIC_PRESETS else ()
    return ('python', 'services', *rust)


def _grid(cells: list[tuple[str, str, str]], keep: tuple, ids: list[str]) -> list[Any]:
    """Parameterize (preset, scenario, profile, backend) with slow mark on non-kept pyref cells."""
    assert set(keep) <= set(cells), set(keep) - set(cells)
    return [
        pytest.param(
            *c,
            b,
            id=f'{i}-{b}',
            marks=[pytest.mark.slow] if b == 'pyref' and c not in keep else [],
        )
        for c, i in zip(cells, ids, strict=True)
        for b in ('pyref', 'zgame')
    ]


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'backend'),
    _grid(E1_CELLS, E1_PYREF, ['/'.join(c) + '/' + '+'.join(_e1_tiers(c[0])) for c in E1_CELLS]),
)
def test_e1_tiers_match(preset: str, scenario: str, profile: str, backend: str) -> None:
    """Run all 3 classic presets across all applicable acceleration tiers."""
    envs = _envs(preset, scenario, profile, backend, 3, _e1_tiers(preset), seed=7)
    if 'rust' in _e1_tiers(preset):
        assert classic_mod.rust_blockers(envs[0].config, envs[0].spec) == []
    before = Counter(native.fallbacks)
    drive(envs, 40, seed=zlib.crc32(f'{preset}/{scenario}/{profile}'.encode()))
    assert native.fallbacks == before


# Race-fallback regen and health-fraction heal profile cells on zgame
RGHF_CELLS = [('smac-portable', '2s3z'), ('smac-portable', 'MMM'), ('smac-hard-portable', 'MMM')]

_NAT_FULL_PO = {
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

_NATIVE_CBT_PO = {
    **_NAT_FULL_PO,
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

NATIVE_CBT_CELLS = [
    ('smac-portable', '8m', _NATIVE_CBT_PO, False),
    ('smac-portable', '2s3z', _NATIVE_CBT_PO, False),
    ('smac-portable', 'MMM', _NATIVE_CBT_PO, False),
    ('smacv2-portable', '10gen_terran', _NATIVE_CBT_PO, False),
    ('smac-hard-portable', 'MMM', _NATIVE_CBT_PO, False),
    ('smac-portable', '2s3z', _NAT_FULL_PO, False),
    (
        'smac-portable',
        '2s3z',
        {**_NATIVE_CBT_PO, 'scan_phase': 'tag_rotate', 'scan_period': 8},
        False,
    ),
    (
        'smac-portable',
        '2s3z',
        {
            **_NATIVE_CBT_PO,
            'auto_acquire_owners': (),
            'retreat_owners': (1, 2),
            'hit_slow_owner1_only': False,
        },
        True,
    ),
]


@pytest.mark.parametrize(
    ('preset', 'scenario'), RGHF_CELLS, ids=[f'{p}/{s}' for p, s in RGHF_CELLS]
)
def test_e1_tiers_match_rghf(preset: str, scenario: str) -> None:
    """`acc-ram-cov-axo-rghf` is a classic recipe matching bitwise across all 3 tiers."""
    envs = _envs(preset, scenario, 'acc-ram-cov-axo-rghf', 'zgame', 3, _e1_tiers(preset), seed=7)
    assert classic_mod.rust_blockers(envs[0].config, envs[0].spec) == []
    before = Counter(native.fallbacks)
    drive(envs, 40, seed=zlib.crc32(f'{preset}/{scenario}/rghf'.encode()))
    assert native.fallbacks == before


def _native_cell_id(p: str, s: str, po: dict[str, Any], d: bool) -> str:
    tag = (
        'sym'
        if d
        else ('rot' if 'scan_phase' in po else ('nat' if 'native_backswing' not in po else 'cbt'))
    )
    return f'{p}/{s}/{tag}'


@pytest.mark.parametrize(
    ('preset', 'scenario', 'po', 'dual'),
    NATIVE_CBT_CELLS,
    ids=[_native_cell_id(p, s, po, d) for p, s, po, d in NATIVE_CBT_CELLS],
)
def test_e1_tiers_match_native_combat(
    preset: str, scenario: str, po: dict[str, Any], dual: bool
) -> None:
    """Native movement/combat options select `accel='rust'` and match bitwise across all 3 tiers."""
    _, over = profile_overrides('acc-ram-cov-axo-rghf', po)
    if dual:
        over = {**over, 'opponent.kind': 'external'}
    auto_env = make_vec_env(preset, scenario, backend='zgame', batch_size=2, overrides=over, seed=7)
    assert auto_env.accel == 'rust'
    assert classic_mod.rust_blockers(auto_env.config, auto_env.spec) == []
    auto_env.close()
    envs = _envs(preset, scenario, over, 'zgame', 3, _e1_tiers(preset), seed=7)
    before = Counter(native.fallbacks)
    drive(envs, 40, seed=zlib.crc32(f'{preset}/{scenario}/native/{dual}'.encode()))
    assert native.fallbacks == before


# Public semantic profiles (and legacy aliases) directly without extra pyref_options
NATIVE_PROFILE_CELLS = [
    ('smac-portable', '2s3z', 'sc2-explore', False),
    ('smac-portable', '2s3z', 'sc2-native', False),
    ('smac-portable', 'MMM', 'sc2-native', False),
    ('smacv2-portable', '10gen_terran', 'sc2-native', False),
    ('smac-hard-portable', 'MMM', 'sc2-native', False),
    ('smac-portable', '2s3z', 'sc2-symmetric', True),
    ('smac-hard-portable', 'MMM', 'sc2-symmetric', True),
    ('smac-portable', '2s3z', 'rghf-nat-cbt', False),
]


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'dual'),
    NATIVE_PROFILE_CELLS,
    ids=[f'{p}/{s}/{prof}' for p, s, prof, _ in NATIVE_PROFILE_CELLS],
)
def test_e1_tiers_match_native_profiles(
    preset: str, scenario: str, profile: str, dual: bool
) -> None:
    """Named profiles (`sc2-explore`, `sc2-native`, `sc2-symmetric`) hit `accel='rust'`."""
    _, over = profile_overrides(profile)
    if dual:
        over = {**over, **EXTERNAL}
    auto_env = make_vec_env(preset, scenario, backend='zgame', batch_size=2, overrides=over, seed=7)
    assert auto_env.accel == 'rust'
    assert classic_mod.rust_blockers(auto_env.config, auto_env.spec) == []
    auto_env.close()
    envs = _envs(preset, scenario, over, 'zgame', 3, _e1_tiers(preset), seed=7)
    before = Counter(native.fallbacks)
    drive(envs, 40, seed=zlib.crc32(f'{preset}/{scenario}/{profile}/{dual}'.encode()))
    assert native.fallbacks == before


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_e1_seeds_scripts_and_generation(backend: str) -> None:
    """Per-env seeds (init & reseed), per-env scripts, and SMACv2 full stochastic generation."""
    py, sv = _pair('smacv2-portable', '10gen_terran', STOCH, backend, 4, seeds=[11, 7, 5, 2])
    drive([py, sv], 60, reseed={25: {1: 9001, 3: 9003}})
    py, sv = _pair('smac-hard-portable', '2s3z', 'aligned-v4', backend, 3, seed=3)
    pool = sv.opponent.pool  # type: ignore[attr-defined]
    drive([py, sv], 60, scripts={10: {0: pool[-1]}, 30: {1: pool[0], 2: pool[-1]}})


# smac-hard-portable scenarios with maps and script pools (3s_vs_3z, 3s_vs_4z lack maps)
HARD_SCENARIOS = tuple(s for s in POOLS if s not in ('3s_vs_3z', '3s_vs_4z'))
HARD_CELLS = [(s, 'zgame') for s in HARD_SCENARIOS]
HARD_CELLS += [(s, 'pyref') for s in ('27m_vs_30m', '2c_vs_64zg')]


def hard_scripts(pool: tuple[str, ...], b_size: int, start: int = 3) -> dict[int, dict[int, str]]:
    """Assign each script in pool once (every 3 steps starting at start, cycling across envs)."""
    names = sorted(set(pool))
    return {start + 3 * i: {i % b_size: n} for i, n in enumerate(names)}


@pytest.mark.parametrize(('scenario', 'backend'), HARD_CELLS, ids=['/'.join(c) for c in HARD_CELLS])
def test_e1_rust_smac_hard(scenario: str, backend: str) -> None:
    """smac-hard-portable across all 3 tiers: scripted opponent sampling, per-env pinning
    (every script in pool once), re-seeding, and error counts."""
    seed = zlib.crc32(scenario.encode())
    tiers = ('python', 'services', 'rust')
    envs = _envs('smac-hard-portable', scenario, None, backend, 3, tiers, seed=seed % 1000)
    assert classic_mod.rust_blockers(envs[0].config, envs[0].spec) == []
    before = Counter(native.fallbacks)
    pool = envs[0].opponent.pool  # type: ignore[attr-defined]
    n = 3 * len(set(pool)) + 12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # script warnings (_compare checks _warned)
        drive(
            envs,
            n,
            seed=seed,
            reseed={4: {1: 77}, n - 5: {0: 5, 2: 6}},
            scripts=hard_scripts(pool, 3),
        )
    assert native.fallbacks == before
    rk = envs[2]._k.opponent_issue.__self__  # type: ignore[attr-defined]
    assert rk._script is not None and rk._script[1] is not None  # backed by Rust opponent


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_e1_rust_script_warnings(backend: str) -> None:
    """Script exception warnings match across all 3 tiers in message, category, and location."""
    got = []
    for tier in ('python', 'services', 'rust'):
        (env,) = _envs('smac-hard-portable', '7q_vs_2bc', None, backend, 2, (tier,), seed=1)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            env.reset()
            for _ in range(3):
                env.step(np.zeros((2, env.n_agents), np.int64) + 1)
            env.reset([0], opponents=['7q_vs_2bc_c'])
            env.step(np.zeros((2, env.n_agents), np.int64) + 1)
        got.append([(str(w.message), w.category, w.filename, w.lineno) for w in rec])
        got[-1].append(env.opponent_errors().tolist())  # type: ignore[arg-type]
    assert got[0][0][2].endswith('accel.py'), got[0]
    assert got[0] == got[1] == got[2]


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_e1_rust_seeds_and_generation(backend: str) -> None:
    """All 3 tiers: per-env seeds (init & reseed), SMACv2 non-cone stochastic features, and
    classic 10gen scenarios."""
    tiers = ('python', 'services', 'rust')
    seeds = [11, 7, 5, 2]
    envs = _envs('smacv2-portable', '10gen_terran', STOCH_CIRCLE, backend, 4, tiers, seeds=seeds)
    before = Counter(native.fallbacks)
    drive(envs, 60, reseed={25: {1: 9001, 3: 9003}})
    envs = _envs('smacv2-portable', '10gen_zerg', 'aligned-v4', backend, 3, tiers, seeds=[4, 0, 4])
    drive(envs, 60, reseed={10: {0: 77}, 40: {1: 4, 2: 5}})
    envs = _envs('smac-portable', 'MMM', 'acc-ram-cov-axo', backend, 2, tiers, seed=12)
    drive(envs, 50, reset_every=7, reseed={20: {1: 3}})
    envs = _envs('smac-hard-portable', '2s3z', 'aligned-v4', backend, 3, tiers, seed=3)
    pool = envs[0].opponent.pool  # type: ignore[attr-defined]
    drive(envs, 60, scripts={10: {0: pool[-1]}, 30: {1: pool[0], 2: pool[-1]}})
    seeds = [1, 2, 3, 4]
    envs = _envs('smac-hard-portable', 'MMM', 'acc-ram-cov-axo', backend, 4, tiers, seeds=seeds)
    drive(envs, 60, reset_every=5, reseed={8: {0: 11}, 20: {1: 12, 3: 13}})
    assert native.fallbacks == before


# Two-sided mode: core scenarios across all 3 presets (plus SMACv2 10gen) x symmetric profiles,
# plus two large smac-hard-portable maps
DUAL_CELLS = [(p, s, prof) for p in PRESETS for s in SCENARIOS for prof in DUAL_PROFILES]
DUAL_CELLS += [('smacv2-portable', s, prof) for s in GEN_SCENARIOS for prof in DUAL_PROFILES]
DUAL_CELLS += [
    ('smac-hard-portable', s, prof) for s in ('27m_vs_30m', '2c_vs_64zg') for prof in DUAL_PROFILES
]
# Two-sided pyref cells kept in default `make test` (same rationale as E1_PYREF)
DUAL_PYREF = (
    ('smac-hard-portable', 'MMM', 'aligned-v4-sym'),
    ('smacv2-portable', '10gen_terran', 'acc-ram-cov-axo-sym'),
)


@pytest.mark.parametrize(
    ('preset', 'scenario', 'profile', 'backend'),
    _grid(DUAL_CELLS, DUAL_PYREF, ['/'.join(c) for c in DUAL_CELLS]),
)
def test_e1_rust_dual_tiers_match(preset: str, scenario: str, profile: str, backend: str) -> None:
    """Two-sided 3-tier parity: obs, state, avail, views, tables, rewards, infos (incl.
    reward_enemy, battle_lost) match bitwise with partial resets every 13 steps, zero fallbacks."""
    over = {**profile_overrides(profile)[1], **EXTERNAL}
    tiers = ('python', 'services', 'rust')
    envs = _envs(preset, scenario, over, backend, 3, tiers, seed=7)
    before = Counter(native.fallbacks)
    drive(envs, 40, seed=zlib.crc32(f'dual/{preset}/{scenario}/{profile}'.encode()))
    assert native.fallbacks == before


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
@pytest.mark.parametrize(
    'over', [{'termination.episode_limit': 8}, DUAL_V4], ids=['single', 'dual']
)
def test_e1_rust_continuing_episode(backend: str, over: dict[str, Any]) -> None:
    """`termination.continuing_episode` (allowed key for classic): all 3 tiers run to timeout
    and emit `episode_limit=True` on the timeout step while staying bitwise-identical otherwise.
    Two-sided mode holds both sides still to reach episode_limit; single-sided shrinks limit
    to 8 so episodes time out before the scripted opponent finishes combat."""
    tiers = ('python', 'services', 'rust')
    on = {**over, 'termination.continuing_episode': True}
    envs = _envs('smac-hard-portable', '3m', on, backend, 2, tiers, seed=5)
    off = _envs('smac-hard-portable', '3m', over, backend, 2, ('rust',), seed=5)[0]
    assert all(e.accel == t for e, t in zip(envs, tiers, strict=True)) and off.accel == 'rust'
    for e in (*envs, off):
        e.reset()
    limit = envs[0].episode_limit

    def stop(side: int) -> np.ndarray:
        """Issue STOP when available for alive units, else NOOP."""
        avail = envs[0].get_avail_actions(side)
        return np.where(avail[..., 1] == 1, 1, 0).astype(np.int64)

    timeouts = 0
    for t in range(limit):
        step: tuple[Any, ...] = (stop(1), stop(2)) if envs[0].dual else (stop(1),)
        outs = [e.step(*step) for e in envs]
        r0, d0, i0 = off.step(*step)
        for (r, d, i), e in zip(outs, envs, strict=True):
            _same(r, r0, f'step {t} ({e.accel}) reward')
            _same(d, d0, f'step {t} ({e.accel}) terminated')
            assert json.dumps(i) == json.dumps(outs[0][2]), (t, e.accel)
            _compare(envs[0], e, f'step {t} ({e.accel})')
        # Only difference when enabled is episode_limit on the timeout step
        assert all('episode_limit' not in x for x in i0), (t, i0)
        hit = [x.pop('episode_limit', None) for x in outs[0][2]]
        assert all(h in (None, True) for h in hit), (t, hit)
        assert json.dumps(i0) == json.dumps(outs[0][2]), t
        timeouts += sum(h is True for h in hit)
        if t < limit - 1:
            assert not any(hit), t
    assert timeouts == int(envs[0].timeouts.sum()) == int(off.timeouts.sum()) > 0
    if envs[0].dual:
        assert timeouts == 2


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_e1_rust_dual_seeds_and_generation(backend: str) -> None:
    """Two-sided 3-tier parity with per-env seeds (init & reseed), SMACv2 two-sided stochastic
    features, and frequent partial resets."""
    tiers = ('python', 'services', 'rust')
    before = Counter(native.fallbacks)
    seeds = [11, 7, 5, 2]
    envs = _envs('smacv2-portable', '10gen_terran', DUAL_STOCH, backend, 4, tiers, seeds=seeds)
    drive(envs, 60, reseed={25: {1: 9001, 3: 9003}})
    envs = _envs('smacv2-portable', '10gen_zerg', DUAL_AXO, backend, 3, tiers, seeds=[4, 0, 4])
    drive(envs, 60, reseed={10: {0: 77}, 40: {1: 4, 2: 5}})
    seeds = [1, 2, 3, 4]
    envs = _envs('smac-hard-portable', 'MMM', DUAL_AXO, backend, 4, tiers, seeds=seeds)
    drive(envs, 60, reset_every=5, reseed={8: {0: 11}, 20: {1: 12, 3: 13}})
    assert native.fallbacks == before
