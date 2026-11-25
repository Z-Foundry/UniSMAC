"""Bitwise parity between Rust SMAC-Hard scripted opponents (`accel='rust'`) and `ScriptedOpponent`.

E0 (per-script): runs a reference shadow alongside the Rust opponent (same env and inputs),
comparing every reset and issue: Command arrays, per-env script names, `errors`, `_warned`,
warnings (message and category), and `opponent` RNG stream draws (env, step, length, index).
Two input sources are used:

* **Trajectories**: real env rollouts (pyref, zgame) rotating Player 1 policies (random, attack
  first/last enemy, hold, retreat), with episode-end and random partial resets, per-env script
  selection (`reset(opponents=)` cycling through the pool), and re-seeding;
* **Synthetic**: perturbed initial states (random walks, damage/deaths, visibility, coinciding
  positions, step jumps around multiples of 50, random unit drops on reset, random ability
  queries, both `script_abilities` modes) to cover rare branches such as script exceptions.

Also includes randomized parity tests for Python semantics (`x ** 2`, `sum`, `sorted`,
`min`/`max`, `np.argmax`/`argmin`, `center`, `distance_to`). Running this file directly
(`python tests/env/test_env_scripted_rust.py --coverage out.json`) runs large-scale E0 and collects
line coverage for `env/opponents/scripted/` via `sys.settrace`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import platform
import random
import sys
import types
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests._support.simd import run_without_avx512
from unismac.contract import Command, WorldState
from unismac.env import accel, make_vec_env
from unismac.env.config import OpponentConfig
from unismac.env.opponents import scripted as scripted_pkg
from unismac.env.opponents.scripted.base import POOLS, SCRIPTS, Script, register
from unismac.env.opponents.scripted.opponent import ScriptedOpponent, StreamChoice, script_pool
from unismac.env.semantics import rng as rng_mod

pytest.importorskip('zgame', reason="requires the optional 'zgame' extension")
native = pytest.importorskip(
    'unismac.env.accel.services',
    reason="requires the optional 'zgame' extension with a compatible 'env' submodule",
    exc_type=ImportError,
)
native_rust = pytest.importorskip('unismac.env.accel.semantics', exc_type=ImportError)
_env = native._env

RUST_SCRIPTS = frozenset(_env.script_names())
SCRIPTED_DIR = Path(scripted_pkg.__file__).parent
# smac-hard-portable scenarios with maps on both pyref and zgame (3s_vs_3z and 3s_vs_4z lack
# maps, so their scripts are tested on 3s_vs_5z via opponent.scripts)
SCENARIOS = tuple(s for s in POOLS if s not in ('3s_vs_3z', '3s_vs_4z'))
# noop is not in any pool; run it on 3m
EXTRA_POOLS = {
    '3s_vs_5z': ('3s_vs_3z_a', '3s_vs_3z_g', '3s_vs_4z_a', '3s_vs_4z_g'),
    '3m': ('noop', 'base'),
}
POLICIES = ('random', 'attack', 'focus_last', 'hold', 'retreat')
# Ability IDs sampled for synthetic queries (7q_vs_2bc reads 401; 2vr_vs_3sc reads 1727, 1731)
SYN_ABILITIES = (401, 1727, 1731, 2116, 3675, 1, 23, 386)
# Morph flips for synthetic states (Spore Crawler root/uproot, Siege Tank siege/unsiege;
# under 'drop' ability commands are not issued, so trajectories never reach the other mode)
SYN_FLIP = {99: 140, 140: 99, 33: 32, 32: 33}


def rust_pool(scenario: str) -> tuple[str, ...]:
    """Subset of scenario's script pool available in Rust (full pool once all are ported)."""
    return tuple(n for n in POOLS[scenario] if n in RUST_SCRIPTS)


# ---------------------------------------------------------------- Bitwise comparison


def _same_arrays(a: Any, b: Any, where: str) -> None:
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f'{where}.{f.name}'
        assert x.tobytes() == y.tobytes(), (
            f'{where}.{f.name}: diff at {np.argwhere(x.view(np.uint8) != y.view(np.uint8))[:4]}'
        )


def _call(fn: Any, *args: Any) -> tuple[Any, list[tuple[str, type]]]:
    """Invoke fn and record warnings (message, category) and any raised exception."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            out: Any = fn(*args)
        except Exception as exc:  # exceptions must also match
            out = exc
    return out, [(str(w.message), w.category) for w in rec]


class Recorder:
    """Reference `opponent` stream: logs (env, step, n, k) for each `integers(n)` from frame 1."""

    def __init__(
        self, gen: np.random.Generator, log: list[tuple[int, int, int, int]], b: int, f: int
    ):
        self.gen, self.log, self.b, self.f = gen, log, b, f

    def integers(self, n: int) -> Any:
        k = self.gen.integers(n)
        if self.f > 0:
            self.log.append((self.b, self.f - 1, int(n), int(k)))
        return k


class Shadow:
    """Pair a Rust-backed `ScriptedOpponent` (`opp`) with a reference shadow (`ref`) on identical
    inputs. `stats` counts resets, issues, script exceptions (by type), and RNG choices."""

    def __init__(self, env: Any, opp: ScriptedOpponent, kernels: Any) -> None:
        self.env, self.opp, self.k = env, opp, kernels
        self.ref = ScriptedOpponent(opp.config)
        self.ref.bind(env)
        self.choices: list[tuple[int, int, int, int]] = []
        self.ref._stream = self._stream  # type: ignore[method-assign]
        self.stats: Counter[str] = Counter()

    def _stream(self, b: int, frame: int) -> Any:
        return Recorder(self.env._rng_stream('opponent', b, frame), self.choices, b, frame)

    def _rust(self) -> Any:
        rk = self.k.opponent_reset.__self__
        assert rk._script is not None and rk._script.opponent is self.opp, 'opponent not using Rust'
        assert rk._script.native is not None, 'opponent not using Rust'
        return rk._script.native

    def _same_state(self, where: str) -> None:
        assert self.ref.names == self.opp.names, where
        assert self.ref.errors == self.opp.errors, where
        assert self.ref._warned == self.opp._warned, where
        got = self._rust().take_choices()
        assert [tuple(c) for c in got] == self.choices, f'{where} choices: {got} / {self.choices}'
        self.stats['choice'] += len(got)
        self.choices = []

    def reset(self, opponent: Any, world: WorldState, env_ids: Any, names: Any = None) -> None:
        assert opponent is self.opp
        r = _call(accel.PYTHON.opponent_reset, self.ref, world, env_ids, names)
        g = _call(self.k.opponent_reset, opponent, world, env_ids, names)
        self._same_outcome(r, g, 'reset')
        self._same_state('reset')
        self.stats['reset'] += len(env_ids)

    def issue(self, opponent: Any, world: WorldState, command: Command, active: np.ndarray) -> None:
        assert opponent is self.opp
        c_ref = copy.deepcopy(command)
        before = dict(self.ref.errors)
        r = _call(accel.PYTHON.opponent_issue, self.ref, world, c_ref, active)
        g = _call(self.k.opponent_issue, opponent, world, command, active)
        where = f'issue (steps={self.env._episode_steps.tolist()})'
        self._same_outcome(r, g, where)
        _same_arrays(c_ref, command, f'{where} command')
        self._same_state(where)
        self.stats['issue'] += int(np.count_nonzero(active))
        self.stats['cmd'] += int(np.count_nonzero(c_ref.kind[world.owner == 2]))
        for b, n in self.ref.errors.items():
            if n != before.get(b, 0):
                self.stats['errors'] += 1
        for msg, _ in r[1]:
            self.stats['warn:' + msg.split('raised ')[1].split('(')[0]] += 1

    @staticmethod
    def _same_outcome(r: Any, g: Any, where: str) -> None:
        (ro, rw), (go, gw) = r, g
        if isinstance(ro, Exception) or isinstance(go, Exception):
            assert type(ro) is type(go) and str(ro) == str(go), f'{where}: {ro!r} / {go!r}'
        assert rw == gw, f'{where} warnings: {rw} / {gw}'


def install_shadow(env: Any) -> Shadow:
    """Replace opponent kernels in a rust-tier env with shadow-checking wrappers."""
    k = env._k
    sh = Shadow(env, env._opponent, k)
    env._k = dataclasses.replace(k, opponent_reset=sh.reset, opponent_issue=sh.issue)
    return sh


# ---------------------------------------------------------------- Trajectories


def rust_env(
    scenario: str, backend: Any, batch: int, pool: tuple[str, ...] | None = None, **kw: Any
):
    """smac-hard-portable env with rust-tier opponent (rest services); optional pool override."""
    over = {} if pool is None else {'opponent.scripts': pool}
    env = make_vec_env(
        'smac-hard-portable',
        scenario,
        backend=backend,
        batch_size=batch,
        overrides=over,
        accel='services',
        **kw,
    )
    k = accel.kernels('rust')
    env._k = dataclasses.replace(
        env._k, opponent_reset=k.opponent_reset, opponent_issue=k.opponent_issue
    )
    return env, install_shadow(env)


def policy_actions(avail: np.ndarray, pol: list[str], dirs: np.ndarray, rng: Any) -> np.ndarray:
    rnd = (rng.random(avail.shape) * avail).argmax(-1)
    out = rnd.copy()
    for b, p in enumerate(pol):
        for a in range(avail.shape[1]):
            av = avail[b, a]
            atk = np.flatnonzero(av[6:])
            if p == 'attack' and len(atk):
                out[b, a] = 6 + atk[0]
            elif p == 'focus_last' and len(atk):
                out[b, a] = 6 + atk[-1]
            elif p == 'hold' and av[1]:
                out[b, a] = 1
            elif p == 'retreat' and av[dirs[b]]:
                out[b, a] = dirs[b]
    return out


def run_trajectory(
    scenario: str,
    backend: str,
    *,
    batch: int = 6,
    n_steps: int = 120,
    seed: int = 0,
    pool: tuple[str, ...] | None = None,
    reset_every: int = 17,
) -> Counter[str]:
    """Trajectory E0 for one scenario: rotating policies, partial resets, script/seed overrides."""
    pool = pool if pool is not None else rust_pool(scenario)
    env, sh = rust_env(scenario, backend, batch, pool, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    names = sorted(set(pool))
    pol = [str(rng.choice(POLICIES)) for _ in range(batch)]
    dirs = rng.integers(2, 6, batch)
    k = 0
    for t in range(n_steps):
        acts = policy_actions(env.avail_actions(), pol, dirs, rng)
        step = env.step(acts)
        done = step.terminated | step.truncated
        ids = set(np.flatnonzero(done).tolist())
        if t % reset_every == reset_every - 1:
            ids.add(int(rng.integers(batch)))
        if ids:
            order = sorted(ids)
            kw: dict[str, Any] = {}
            if rng.random() < 0.6:
                kw['opponents'] = [
                    None if rng.random() < 0.25 else names[(k + i) % len(names)]
                    for i in range(len(order))
                ]
                k += len(order)
            if rng.random() < 0.15:
                kw['seeds'] = [int(rng.integers(1 << 20)) for _ in order]
            if rng.random() < 0.05:  # unknown or out-of-pool name must raise ValueError cleanly
                bad = 'no_such_script' if rng.random() < 0.5 else _not_in_pool(pool)
                with pytest.raises(ValueError, match=bad):
                    env.reset(order, opponents=[bad] * len(order))
            env.reset(order, **kw)
            for b in order:
                pol[b] = str(rng.choice(POLICIES))
                dirs[b] = rng.integers(2, 6)
    assert sh.opp.names == sh.ref.names
    env.close()
    sh.stats['steps'] = n_steps
    return sh.stats


def _not_in_pool(pool: tuple[str, ...]) -> str:
    return next(n for n in sorted(SCRIPTS) if n not in pool)


# ---------------------------------------------------------------- Synthetic


class FakeEnv:
    """Minimal synthetic env exposing only attributes read by scripted opponents."""

    def __init__(self, name: str, seed: int, enemy_slots: np.ndarray, ally_slots: np.ndarray):
        b = enemy_slots.shape[0]
        self.spec = types.SimpleNamespace(name=name)
        self._view = types.SimpleNamespace(enemy_slots=enemy_slots, ally_slots=ally_slots)
        self._seed = seed
        self._episode_steps = np.zeros(b, np.int64)
        self._reset_count = np.zeros(b, np.int64)
        self.abilities: list[list[list[int]]] | None = None

    def _rng_stream(self, name: str, b: int, frame: int = 0) -> np.random.Generator:
        return rng_mod.stream(self._seed, name, b, int(self._reset_count[b]) - 1, frame)

    def _rng_streams(self, name: str, frames: np.ndarray) -> rng_mod.Streams:
        keys = [rng_mod.stream_key(self._seed, name, b) for b in range(len(self._reset_count))]
        k0, k1 = (np.array(k, np.uint64) for k in zip(*keys, strict=True))
        ep = np.maximum(self._reset_count - 1, 0).astype(np.uint64)
        return rng_mod.Streams(k0, k1, np.asarray(frames).astype(np.uint64), ep)

    def _query_abilities(self) -> list[list[list[int]]] | None:
        return self.abilities


def _base_world(scenario: str, seed: int) -> tuple[WorldState, np.ndarray, np.ndarray]:
    env = make_vec_env('smac-hard-portable', scenario, backend='zgame', batch_size=1, seed=seed)
    env.reset()
    world = copy.deepcopy(env.runtime.world)
    es, al = env._view.enemy_slots, env._view.ally_slots
    assert world is not None and es is not None and al is not None
    out = world, es.copy(), al.copy()
    env.close()
    return out


def _tile(world: WorldState, b: int) -> WorldState:
    return dataclasses.replace(
        world,
        **{f.name: np.repeat(getattr(world, f.name), b, axis=0) for f in dataclasses.fields(world)},
    )


def _fresh_row(world: WorldState, base: WorldState, b: int, rng: Any) -> None:
    """Reset row b to base state, randomly shift/spread squads, and drop random units."""
    for f in dataclasses.fields(world):
        getattr(world, f.name)[b] = getattr(base, f.name)[0]
    if rng.random() < 0.4:
        for owner in (1, 2):
            side = world.alive[b] & (world.owner[b] == owner)
            if not side.any():
                continue
            k, dx, dy = rng.uniform(1.0, 3.5), rng.uniform(-8, 8), rng.uniform(-8, 8)
            for arr, d in ((world.x, dx), (world.y, dy)):
                c = arr[b, side].mean()
                arr[b, side] = np.clip(c + (arr[b, side] - c) * k + d, 0, 40).astype(np.float32)
    if rng.random() < 0.5:
        alive = world.alive[b]
        kill = alive & (rng.random(alive.shape) < rng.random() * 0.8)
        alive[kill] = False
        world.health[b, kill] = 0.0


def _perturb(world: WorldState, rng: Any) -> None:
    """Apply one step of random perturbations: positions, damage/deaths, visibility, overlaps."""
    al = world.alive
    step = rng.normal(0.0, 0.8, world.x.shape).astype(np.float32)
    world.x[al] = np.clip(world.x[al] + step[al], 0, 40).astype(np.float32)
    step = rng.normal(0.0, 0.8, world.y.shape).astype(np.float32)
    world.y[al] = np.clip(world.y[al] + step[al], 0, 40).astype(np.float32)
    hit = al & (rng.random(al.shape) < 0.25)
    dmg = (rng.random(al.shape) * world.health_max * 0.3).astype(np.float32)
    sh = np.minimum(world.shield, dmg)
    world.shield[hit] -= sh[hit]
    world.health[hit] = np.maximum(world.health[hit] - (dmg - sh)[hit], 0).astype(np.float32)
    dead = al & ((world.health <= 0) | (rng.random(al.shape) < 0.02))
    al[dead] = False
    world.health[dead] = 0.0
    world.energy[al] = np.minimum(world.energy[al] + 1.0, 200).astype(np.float32)
    world.weapon_cooldown[...] = (rng.random(al.shape) < 0.5) * rng.random(al.shape).astype(
        np.float32
    )
    world.visible[...] = rng.random(world.visible.shape) < 0.75
    flip = al & np.isin(world.unit_type, list(SYN_FLIP)) & (rng.random(al.shape) < 0.1)
    for b, u in zip(*np.nonzero(flip), strict=True):
        world.unit_type[b, u] = SYN_FLIP[int(world.unit_type[b, u])]
    for b in range(world.alive.shape[0]):
        if rng.random() < 0.05:  # force two live units to coincide
            idx = np.flatnonzero(al[b])
            if len(idx) >= 2:
                i, j = rng.choice(idx, 2, replace=False)
                world.x[b, j], world.y[b, j] = world.x[b, i], world.y[b, i]


def run_synthetic(
    scenario: str,
    pool: tuple[str, ...],
    *,
    batch: int = 6,
    n_steps: int = 120,
    seed: int = 0,
    drop: bool = True,
) -> Counter[str]:
    """Synthetic E0: randomly perturb a scenario's initial state while cycling scripts in pool."""
    rng = np.random.default_rng(seed)
    base, es, al = _base_world(scenario, seed)
    world = _tile(base, batch)
    env = FakeEnv(scenario, seed, np.repeat(es, batch, 0), np.repeat(al, batch, 0))
    cfg = OpponentConfig(
        kind='scripted', scripts=pool, script_abilities='drop' if drop else 'issue'
    )
    opp = ScriptedOpponent(cfg)
    opp.bind(env)  # type: ignore[arg-type]
    k = accel.kernels('rust')
    sh = Shadow(env, opp, k)
    names = sorted(set(pool))

    def reset(ids: list[int], first: bool = False) -> None:
        for b in ids:
            if not first:
                _fresh_row(world, base, b, rng)
            env._reset_count[b] += 1
            env._episode_steps[b] = 0
        chosen = None
        if rng.random() < 0.7:
            chosen = [names[int(rng.integers(len(names)))] for _ in ids]
        sh.reset(opp, world, ids, chosen)

    reset(list(range(batch)), first=True)
    for _ in range(n_steps):
        _perturb(world, rng)
        if rng.random() < 0.3:
            env.abilities = [
                [
                    sorted(set(rng.choice(SYN_ABILITIES, rng.integers(0, 4)).tolist()))
                    for _ in range(world.n_slots)
                ]
                for _ in range(batch)
            ]
        else:
            env.abilities = None
        active = rng.random(batch) < 0.9
        cmd = Command.noop(batch, world.n_slots)
        sh.issue(opp, world, cmd, active)
        env._episode_steps[active] += 1
        jump = rng.random(batch) < 0.03
        env._episode_steps[jump] = rng.choice(
            [49, 50, 99, 100, 149, 150, 199, 200], int(jump.sum())
        )
        gone = [b for b in range(batch) if not world.alive[b].any() or rng.random() < 0.02]
        ended = [
            b
            for b in range(batch)
            if not (world.alive[b] & (world.owner[b] == 1)).any()
            or not (world.alive[b] & (world.owner[b] == 2)).any()
        ]
        ids = sorted(set(gone) | set(ended))
        if ids:
            reset(ids)
    sh.stats['steps'] = n_steps
    return sh.stats


# ---------------------------------------------------------------- Coverage


def executable_lines(path: Path) -> set[int]:
    """Executable lines inside function/lambda bodies (excluding module/class bodies and Python
    3.14 compiler-generated `__annotate__` frames)."""
    code = compile(path.read_text(), str(path), 'exec')
    out: set[int] = set()

    def walk(co: types.CodeType, in_func: bool) -> None:
        func = bool(co.co_flags & 0x1) and co.co_name != '__annotate__'  # CO_OPTIMIZED: function
        if func:
            for _, _, line in co.co_lines():
                if line is not None and line != co.co_firstlineno:
                    out.add(line)
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk(c, func or in_func)

    walk(code, False)
    return out


class Coverage:
    """Line coverage for `env/opponents/scripted/` via `sys.settrace` (filtered to that dir)."""

    def __init__(self) -> None:
        self.hit: dict[str, set[int]] = {}
        self.root = str(SCRIPTED_DIR)

    def _local(self, frame: Any, event: str, arg: Any) -> Any:
        if event == 'line':
            self.hit.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return self._local

    def _global(self, frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename.startswith(self.root):
            return self._local
        return None

    def __enter__(self) -> Coverage:
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.settrace(None)

    def report(self) -> dict[str, dict[str, Any]]:
        out = {}
        for path in sorted(SCRIPTED_DIR.rglob('*.py')):
            lines = executable_lines(path)
            if not lines:
                continue
            hit = self.hit.get(str(path), set()) & lines
            out[str(path.relative_to(SCRIPTED_DIR))] = {
                'lines': len(lines),
                'hit': len(hit),
                'missed': sorted(lines - hit),
            }
        return out


# ---------------------------------------------------------------- Test cases


def test_e0_py_semantics_random() -> None:
    """Bitwise parity with CPython/NumPy for `x ** 2`, `sum`, `sorted` (<64 or no NaN),
    `min`/`max`, `np.argmax`/`argmin`, `center`, and `distance_to` on random float lists
    (float32 values, small ints, +-0, NaN)."""
    rng = random.Random(0)

    def lst(n: int) -> list[float]:
        nan = rng.random() < 0.3
        out = []
        for _ in range(n):
            r = rng.random()
            if nan and r < 0.15:
                out.append(math.nan)
            elif r < 0.3:
                out.append(float(rng.randint(-3, 3)))
            elif r < 0.35:
                out.append(-0.0)
            else:
                out.append(float(np.float32(rng.uniform(-40, 40))))
        return out

    def ref(fn: Any) -> Any:
        try:
            return fn()
        except Exception as e:
            return (type(e).__name__, repr(e))

    def same(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return (a == b and math.copysign(1, a) == math.copysign(1, b)) or (
                math.isnan(a) and math.isnan(b)
            )
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
        return type(a) is type(b) and a == b

    for _ in range(4000):
        n = rng.randint(0, 80)
        xs = lst(n)
        got = _env.py_ops('sq', xs)
        assert all(same(ref(lambda x=x: x**2), g) for x, g in zip(xs, got, strict=True)), xs
        if sys.version_info >= (3, 12):
            assert same(sum(xs), _env.py_ops('sum', xs)), xs
        if n < 64 or not any(math.isnan(x) for x in xs):
            assert sorted(range(n), key=xs.__getitem__) == _env.py_ops('sort', xs), xs
        for op, fn in (('min', min), ('max', max)):
            want = ref(lambda fn=fn, n=n, xs=xs: fn(range(n), key=xs.__getitem__))
            assert same(want, _env.py_ops(op, xs)), (op, xs)
        for op in ('argmax', 'argmin'):
            want = ref(lambda op=op, xs=xs: int(getattr(np, op)(xs)))
            assert same(want, _env.py_ops(op, xs)), (op, xs)
        pts = xs[: (n // 2) * 2]
        if sys.version_info >= (3, 12):
            want = ref(lambda p=pts: (sum(p[0::2]) / len(p[0::2]), sum(p[1::2]) / len(p[1::2])))
            assert same(want, _env.py_ops('center', pts)), pts
        if n >= 4:
            a = xs[:4]
            want = ref(lambda a=a: math.sqrt((a[0] - a[2]) ** 2 + (a[1] - a[3]) ** 2))
            assert same(want, _env.py_ops('distance', a)), a


def test_e0_choice_stream_random() -> None:
    """Direct parity test for `self.random.choice` (`StreamChoice`) across random stream keys,
    episodes, frames, and sequence lengths (incl. 0, 1, and large values), since the fallback
    branches calling it in MMM/mmmt_fap are unreachable in live episodes."""
    rng = np.random.default_rng(0)
    for i in range(3000):
        b = int(rng.integers(6))
        key = rng_mod.stream_key(int(rng.integers(1 << 31)), 'opponent', b)
        episode, frame = int(rng.integers(0, 50)), int(rng.integers(0, 400))
        k0, k1 = (np.array([k], np.uint64) for k in key)
        streams = rng_mod.Streams(
            k0, k1, np.array([frame], np.uint64), np.array([episode], np.uint64)
        )
        col = [int(v) for v in streams.table()[:, 0]]
        lens = [int(n) for n in rng.choice([0, 1, 2, 3, 7, 30, 1000, 1 << 33], rng.integers(1, 12))]
        choice = StreamChoice()
        choice.rng = streams(0)
        want: list[Any] = []
        for n in lens:
            try:
                want.append(choice.choice(range(n)))
            except IndexError as e:
                want.append(('IndexError', repr(e)))
        assert [tuple(x) if isinstance(x, list) else x for x in _env.script_choices(col, lens)] == [
            tuple(x) if isinstance(x, tuple) else x for x in want
        ], (i, lens)


def test_registry_names_match() -> None:
    """Rust script registry matches reference SCRIPTS (all 45 ported); error paths match."""
    assert RUST_SCRIPTS == set(SCRIPTS)
    error_paths()


TRAJ_CASES = [
    ('3m', 'pyref'),
    ('2s3z', 'pyref'),
    ('MMM', 'pyref'),
    *[(s, 'zgame') for s in SCENARIOS],
]


@pytest.mark.parametrize(('scenario', 'backend'), TRAJ_CASES)
def test_e0_trajectory(scenario: str, backend: str) -> None:
    pool = rust_pool(scenario)
    n = 40 if backend == 'pyref' else 100
    stats = run_trajectory(scenario, backend, n_steps=n, seed=len(scenario))
    assert stats['issue'] > 0 and stats['reset'] > 0, stats
    if scenario in EXTRA_POOLS and backend == 'zgame':
        extra = tuple(n for n in EXTRA_POOLS[scenario] if n in RUST_SCRIPTS)
        if extra:
            run_trajectory(scenario, backend, n_steps=n, seed=1, pool=pool + extra)


SYN_CASES = [*SCENARIOS, *[f'{s}+extra' for s in EXTRA_POOLS]]


@pytest.mark.parametrize('case', SYN_CASES)
def test_e0_synthetic(case: str) -> None:
    scenario = case.removesuffix('+extra')
    pool = rust_pool(scenario)
    if case.endswith('+extra'):
        pool = tuple(n for n in EXTRA_POOLS[scenario] if n in RUST_SCRIPTS) or pool
    for i, drop in enumerate((True, False)):
        stats = run_synthetic(scenario, pool, n_steps=80, seed=100 + i, drop=drop)
        assert stats['issue'] > 0, stats


@pytest.mark.parity
@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 CPU feature toggle is only applicable on x86_64',
)
def test_e0_without_avx512() -> None:
    """Re-run scripted-opponent E0 parity checks with NumPy AVX-512 dispatch disabled."""
    body = (
        'm["test_e0_py_semantics_random"]()\n'
        'for s in ("3m", "MMM", "2c_vs_64zg", "mmmt_vs_zhb", "3s_vs_5z", "7q_vs_2bc"):\n'
        '    m["run_trajectory"](s, "zgame", n_steps=60, seed=5)\n'
        '    m["run_synthetic"](s, m["rust_pool"](s), n_steps=60, seed=6)\n'
    )
    run_without_avx512(__file__, body, verify_disabled=True)


# ---------------------------------------------------------------- Large-scale E0 & coverage


def error_paths() -> None:
    """Framework error paths (shared Python code: `script_pool`, unbound calls, missing RNG)."""
    cfg = OpponentConfig(kind='scripted')
    with pytest.raises(ValueError, match='(?i)no SMAC-Hard script'):
        script_pool(cfg, '2m_vs_1z')
    with pytest.raises(ValueError, match='(?i)unknown script'):
        script_pool(OpponentConfig(kind='scripted', scripts=('no_such',)), '3m')
    with pytest.raises(RuntimeError, match='(?i)not bound|bind'):
        ScriptedOpponent(cfg)._require_env()
    with pytest.raises(RuntimeError, match='(?i)stream'):
        StreamChoice().choice([1])
    with pytest.raises(RuntimeError, match='(?i)random'):
        SCRIPTS['MMM']('MMM').random.choice([1])
    with pytest.raises(NotImplementedError):
        Script('3m').script({}, {}, [], {'red': [], 'blue': []}, 0)
    with pytest.raises(ValueError, match='(?i)duplicate'):
        register('8m')(Script)


def main(argv: list[str]) -> None:
    """`--coverage out.json [--quick]`: run trajectories (both backends, multiple seeds) and
    synthetic cases across all scenarios, writing line coverage to JSON."""
    out = Path(argv[argv.index('--coverage') + 1])
    quick = '--quick' in argv
    seeds = (0,) if quick else (0, 1, 2)
    total: Counter[str] = Counter()
    per: dict[str, dict[str, int]] = {}
    with Coverage() as cov:
        error_paths()
        test_e0_choice_stream_random()
        for scenario in SCENARIOS:
            pool = rust_pool(scenario)
            for seed in seeds:
                for backend in ('zgame',) if quick else ('zgame', 'pyref'):
                    if backend == 'pyref' and (
                        seed > 0 or scenario not in ('3m', '2s3z', 'MMM', '27m_vs_30m', 'corridor')
                    ):
                        continue
                    n = 60 if backend == 'pyref' else 200
                    st = run_trajectory(scenario, backend, n_steps=n, seed=seed, batch=8)
                    total.update(st)
                    per[f'traj/{scenario}/{backend}/{seed}'] = dict(st)
                for drop in (True, False):
                    st = run_synthetic(
                        scenario, pool, n_steps=150, seed=seed * 2 + drop, batch=8, drop=drop
                    )
                    total.update(st)
                    per[f'syn/{scenario}/{seed}/{drop}'] = dict(st)
            if scenario in EXTRA_POOLS:
                extra = tuple(n for n in EXTRA_POOLS[scenario] if n in RUST_SCRIPTS)
                if extra:
                    for seed in seeds:
                        st = run_trajectory(
                            scenario, 'zgame', n_steps=200, seed=seed, batch=8, pool=extra
                        )
                        total.update(st)
                        st = run_synthetic(scenario, extra, n_steps=150, seed=seed, batch=8)
                        total.update(st)
    rep = cov.report()
    lines = sum(v['lines'] for v in rep.values())
    hit = sum(v['hit'] for v in rep.values())
    out.write_text(
        json.dumps(
            {'total': dict(total), 'coverage': rep, 'lines': lines, 'hit': hit, 'runs': per},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f'coverage {hit}/{lines} = {hit / lines:.1%}; {dict(total)}')


if __name__ == '__main__':
    main(sys.argv[1:])
