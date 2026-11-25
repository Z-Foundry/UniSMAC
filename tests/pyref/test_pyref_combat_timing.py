"""Combat timing tests for `pyref`: `attack='sc2'`, `damage='sc2'`, `move_lag`,
`spawn_facing`, and `auto_acquire_owners`.

Covers combat timing parameters from `assets/measured.json`:
- Firing: turn toward target (`facing_rate`), then wait `attack_windup` loops once within
  `attack_arc`.
- Cooldown: add `cooldown * 16 + U(-1, 2)` per shot, preserving negative remainder.
- Projectiles: track target with distance-based hit delay; Zealot twin strikes 4 loops apart.
- Concussive Shells: 50% movement slow lasting 24 loops.
- Regeneration: shields regen 0.125/loop after 160 loops out of combat; Zerg HP 70/4096/loop.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig, rng
from unismac.contract import Command, CommandKind, Scenario, Spawn, StaticData

MARINE, MARAUDER, ZEALOT, STALKER, ZERGLING, HYDRALISK = 48, 51, 73, 74, 105, 107
MAP_ID = 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {MAP_ID: load_map_info(MAP_ID)}
P5 = PyrefConfig(attack='sc2', damage='sc2')


def make(cfg: PyrefConfig = P5, static: StaticData = STATIC) -> PyrefBackend:
    return PyrefBackend(static, MAPS, 1, cfg)


def pair(a: int, b: int, edge: float, *, x: float = 10.0, y: float = 16.0) -> Scenario:
    """Spawn player 1 `a` at (x, y) and player 2 `b` at edge distance `edge` along +x."""
    gap = STATIC.get(a).radius + STATIC.get(b).radius + edge
    return Scenario(map_id=MAP_ID, spawns=(Spawn(1, a, x, y), Spawn(2, b, x + gap, y)))


def attack(n: int = 2, slot: int = 0, target: int = 1) -> Command:
    return Command.noop(1, n).set(0, slot, CommandKind.ATTACK_UNIT, target_slot=target)


def noop(n: int = 2) -> Command:
    return Command.noop(1, n)


def run(backend: PyrefBackend, first: Command, loops: int) -> list:
    """Issue `first` on loop 0 then noops; return post-step states (index k = frame k + 1)."""
    out = [backend.step(first, 1)]
    out += [backend.step(noop(first.kind.shape[1]), 1) for _ in range(loops - 1)]
    return out


def fire_frames(states: list, slot: int = 0) -> list[int]:
    """Frames (index + 1) where `weapon_cooldown` increases."""
    cd = [0.0] + [float(s.weapon_cooldown[0, slot]) for s in states]
    return [k for k in range(1, len(cd)) if cd[k] > cd[k - 1]]


def ehp(states: list, slot: int = 1) -> list[float]:
    return [float(s.health[0, slot] + s.shield[0, slot]) for s in states]


# ---------------------------------------------------------------- RNG


def test_rng_deterministic_and_keyed() -> None:
    a = rng.uniform(7, 'cooldown', 10, 3)
    assert a == rng.uniform(7, 'cooldown', 10, 3)
    others = {
        rng.uniform(8, 'cooldown', 10, 3),
        rng.uniform(7, 'spawn_facing', 10, 3),
        rng.uniform(7, 'cooldown', 11, 3),
        rng.uniform(7, 'cooldown', 10, 4),
    }
    assert a not in others and len(others) == 4
    xs = [rng.uniform(1, 'cooldown', k, 0) for k in range(2000)]
    assert 0.0 <= min(xs) and max(xs) < 1.0
    assert abs(float(np.mean(xs)) - 0.5) < 0.03
    with pytest.raises(ValueError):
        rng.uniform(1, 'nope', 0, 0)


# ---------------------------------------------------------------- Windup and facing


@pytest.mark.parametrize(
    ('unit', 'windup'), [(MARAUDER, 0), (MARINE, 1), (STALKER, 3), (HYDRALISK, 4)]
)
def test_windup_after_facing(unit: int, windup: int) -> None:
    """Facing 0 toward target along +x: aligned on loop 0, fires after `windup` loops."""
    assert STATIC.get(unit).measured['attack_windup'] == windup
    backend = make()
    backend.reset([pair(unit, MARINE, 1.0)])
    states = run(backend, attack(), 10)
    assert fire_frames(states)[0] == 1 + windup
    for s in states[:windup]:
        assert s.weapon_cooldown[0, 0] == 0, 'observed cooldown stays 0 during windup'


def test_turn_before_attack() -> None:
    """Target behind (delta pi): turns 1.0923/loop, attacks on loop 3 when delta <= 0.17."""
    backend = make()
    gap = 2 * STATIC.get(MARINE).radius + 1.0
    backend.reset(
        [
            Scenario(
                MAP_ID, spawns=(Spawn(1, MARINE, 20.0, 16.0), Spawn(2, MARINE, 20.0 - gap, 16.0))
            )
        ]
    )
    states = run(backend, attack(), 8)
    facing = [float(s.facing[0, 0]) for s in states]
    assert facing[0] == pytest.approx(-1.0923, abs=1e-4) or facing[0] == pytest.approx(
        1.0923, abs=1e-4
    )
    assert fire_frames(states)[0] == 3 + 1  # attack begins on loop 2 with 1-loop windup


def test_no_measured_keys_fires_like_instant() -> None:
    """Without measured keys (no windup, facing gate, or jitter), matches instant/standard."""
    bare = StaticData(
        units={k: dataclasses.replace(s, measured={}) for k, s in STATIC.units.items()},
        source='bare',
    )
    a, b = make(PyrefConfig(), bare), make(P5, bare)
    for be in (a, b):
        be.reset([pair(MARINE, ZEALOT, 1.0)])
    sa, sb = run(a, attack(), 40), run(b, attack(), 40)
    assert fire_frames(sa)[0] == fire_frames(sb)[0] == 1
    assert ehp(sa)[:14] == ehp(sb)[:14]


# ---------------------------------------------------------------- Cooldown jitter


def _intervals(seed: int) -> list[int]:
    backend = make()
    backend.reset([pair(MARINE, ZEALOT, 1.0)], seeds=[seed])
    f = fire_frames(run(backend, attack(), 300))
    return [b - a for a, b in zip(f, f[1:], strict=False)]


def test_cooldown_jitter_distribution() -> None:
    """Marine cooldown 13.77: intervals in 12..16, mean ~14.3, deterministic per seed."""
    iv = _intervals(1) + _intervals(2) + _intervals(3)
    assert set(iv) <= {12, 13, 14, 15, 16}
    assert len(set(iv)) >= 3
    assert 14.0 <= float(np.mean(iv)) <= 14.7
    assert _intervals(1) == _intervals(1)
    assert _intervals(1) != _intervals(2)


# ---------------------------------------------------------------- Projectiles, bursts, slow


@pytest.mark.parametrize(('edge', 'lag'), [(0.1, 2), (2.0, 3), (4.0, 5), (5.0, 6), (6.0, 7)])
def test_marauder_projectile_lag(edge: float, lag: int) -> None:
    """Marauder vs stationary Marine: fire-to-impact frame lag scales with edge distance."""
    backend = make()
    backend.reset([pair(MARAUDER, MARINE, edge)])
    states = run(backend, attack(), 15)
    fire = fire_frames(states)[0]
    hp = [45.0, *ehp(states)]
    hit = next(k for k in range(1, len(hp)) if hp[k] < hp[k - 1])
    assert hit - fire == lag


def test_zealot_second_attack_four_loops_later() -> None:
    backend = make()
    backend.reset([pair(ZEALOT, MARINE, 0.05)])
    hp = [45.0, *ehp(run(backend, attack(), 8))]
    drops = [(k, hp[k - 1] - hp[k]) for k in range(1, len(hp)) if hp[k] < hp[k - 1]]
    assert drops == [(1, 8.0), (5, 8.0)]


def test_concussive_slow() -> None:
    """After Marauder hit, Zealot moves at half speed starting 2 loops later for 24 loops."""
    backend = make()
    s0 = backend.reset([pair(MARAUDER, ZEALOT, 2.0)])
    cmd = attack().set(0, 1, CommandKind.MOVE, target_xy=(29.0, 16.0))
    states = [s0, *run(backend, cmd, 40)]  # index = frame
    hp = ehp(states)
    hit = next(k for k in range(1, len(hp)) if hp[k] < hp[k - 1])
    x = [float(s.x[0, 1]) for s in states]
    step = {k: x[k] - x[k - 1] for k in range(1, len(x))}  # observed delta on frame k
    full = 0.140625
    assert step[hit] == pytest.approx(full, abs=1e-5)
    slowed = [k for k, v in step.items() if v == pytest.approx(full / 2, abs=1e-5)]
    first = [k for k in slowed if k <= hit + 26]  # before second hit
    assert first == list(range(hit + 2, hit + 26))
    assert step[hit + 26] == pytest.approx(full, abs=1e-5)


# ---------------------------------------------------------------- Regeneration


def test_shield_regen_after_160_loops() -> None:
    """After one Marine shot and STOP, Zealot shield stays flat 160 loops then +0.125/loop."""
    backend = make()
    backend.reset([pair(MARINE, ZEALOT, 1.0)])
    backend.step(attack(), 1)  # first shot starts windup on loop 0 and fires on loop 1
    s = backend.step(noop(), 1)
    stop = Command.noop(1, 2).set(0, 0, CommandKind.STOP)
    s = backend.step(stop, 1)
    hit_loop = 1
    shield = float(s.shield[0, 1])
    assert shield == 44.0
    s = backend.step(noop(), 160 - (int(s.game_loop[0]) - hit_loop))  # advance to loop 160
    assert float(s.shield[0, 1]) == shield, 'no regen within 160 loops'
    s = backend.step(noop(), 1)
    assert float(s.shield[0, 1]) == shield + 0.125
    s = backend.step(noop(), 8)
    assert float(s.shield[0, 1]) == shield + 9 * 0.125


def test_zerg_health_regen_and_fallback() -> None:
    """Zergling regens 70/4096 per loop; Hydralisk regens only with `regen_race_fallback`."""
    rate = 70 / 4096
    for unit, cfg, expect in (
        (ZERGLING, P5, rate),
        (HYDRALISK, P5, 0.0),
        (HYDRALISK, dataclasses.replace(P5, regen_race_fallback=True), rate),
    ):
        backend = make(cfg)
        backend.reset([pair(MARINE, unit, 1.0)])
        s0 = backend.step(attack(), 3)
        h0 = float(s0.health[0, 1])
        s1 = backend.step(Command.noop(1, 2).set(0, 0, CommandKind.STOP), 20)
        assert float(s1.health[0, 1]) == pytest.approx(h0 + 20 * expect, abs=1e-4)


def test_stalker_shield_fallback() -> None:
    for cfg, regen in ((P5, False), (dataclasses.replace(P5, regen_race_fallback=True), True)):
        backend = make(cfg)
        backend.reset([pair(MARINE, STALKER, 1.0)])
        s0 = backend.step(attack(), 3)
        sh0 = float(s0.shield[0, 1])
        s1 = backend.step(Command.noop(1, 2).set(0, 0, CommandKind.STOP), 200)
        assert (float(s1.shield[0, 1]) > sh0) == regen


# ---------------------------------------------------------------- Movement lag


def _move_states(cfg: PyrefConfig, loops: int = 20) -> list:
    backend = make(cfg)
    backend.reset([Scenario(MAP_ID, spawns=(Spawn(1, MARINE, 15.0, 16.0),))])
    cmd = Command.noop(1, 1).set(0, 0, CommandKind.MOVE, target_xy=(5.0, 16.0))
    return run(backend, cmd, loops)


def test_move_lag_splits_orders_and_motion() -> None:
    """With `command_delay=1` and `move_lag=1`, orders and facing update on frame 2 while
    displacement starts on frame 3, matching `command_delay=2` positions bit-for-bit."""
    lag = _move_states(PyrefConfig(commands='delayed', command_delay=1, move_lag=1))
    c2 = _move_states(PyrefConfig(commands='delayed', command_delay=2))
    c1 = _move_states(PyrefConfig(commands='delayed', command_delay=1))
    kinds = [int(s.order_kind[0, 0]) for s in lag]
    assert kinds[0] == CommandKind.NOOP and kinds[1] == CommandKind.MOVE
    assert lag[1].x[0, 0] == 15.0 and lag[2].x[0, 0] < 15.0
    assert lag[1].facing[0, 0] != 0.0, 'facing already turns on activation loop'
    assert [float(s.x[0, 0]) for s in lag] == [float(s.x[0, 0]) for s in c2]
    assert c1[1].x[0, 0] < 15.0


def test_move_lag_redirect_keeps_old_velocity_one_loop() -> None:
    backend = make(PyrefConfig(commands='delayed', command_delay=1, move_lag=1))
    backend.reset([Scenario(MAP_ID, spawns=(Spawn(1, MARINE, 15.0, 16.0),))])
    go = Command.noop(1, 1).set(0, 0, CommandKind.MOVE, target_xy=(25.0, 16.0))
    up = Command.noop(1, 1).set(0, 0, CommandKind.MOVE, target_xy=(15.0, 26.0))
    backend.step(go, 5)
    before = backend.step(up, 1)
    s1 = backend.step(Command.noop(1, 1), 1)  # new order activates: still moves along +x
    s2 = backend.step(Command.noop(1, 1), 1)  # subsequent loop moves along new heading
    assert s1.x[0, 0] - before.x[0, 0] == pytest.approx(0.140625) and s1.y[0, 0] == 16.0
    assert s2.y[0, 0] > s1.y[0, 0]


# ---------------------------------------------------------------- Auto-acquire and facing


def test_auto_acquire_owners_and_start_loop() -> None:
    cfg = dataclasses.replace(P5, auto_acquire=True, auto_acquire_owners=(2,), acquire_start_loop=2)
    backend = make(cfg)
    backend.reset([pair(MARINE, MARINE, 2.0)])
    states = run(backend, noop(), 30)
    eng = [(int(s.engaged_target_slot[0, 0]), int(s.engaged_target_slot[0, 1])) for s in states]
    assert eng[0] == eng[1] == (-1, -1)
    assert eng[2] == (-1, 0), 'player 2 idle unit engages starting on loop 2 (frame 3)'
    assert fire_frames(states, 0) == [], 'player 1 idle unit does not auto-acquire'
    assert fire_frames(states, 1)


def test_spawn_facing_random() -> None:
    cfg = PyrefConfig(spawn_facing='random')
    sc = pair(MARINE, MARINE, 2.0)
    faces = []
    for seed in (1, 1, 2):
        backend = make(cfg)
        s = backend.reset([sc], seeds=[seed])
        faces.append(tuple(float(f) for f in s.facing[0]))
        later = backend.step(noop(), 5)
        assert tuple(float(f) for f in later.facing[0]) == faces[-1], 'idle keeps spawn facing'
    assert faces[0] == faces[1] != faces[2]
    assert all(-math.pi <= f < math.pi for f in faces[2])
    assert make(PyrefConfig()).reset([sc]).facing[0].tolist() == [0.0, 0.0]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match='move_lag'):
        PyrefConfig(move_lag=1)
    with pytest.raises(ValueError, match='move_lag'):
        PyrefConfig(commands='delayed', command_delay=1, move_lag=2)
    with pytest.raises(ValueError, match='spawn_facing'):
        PyrefConfig(spawn_facing='north')
    with pytest.raises(ValueError, match='acquire_start_loop'):
        PyrefConfig(acquire_start_loop=-1)
