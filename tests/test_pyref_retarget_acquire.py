"""Tests for pyref `retarget_acquire`: center-distance retargeting within acquire range.

When an engaged target leaves weapon range on a scan frame, any enemy within acquire range that is
closer by center distance is also a retarget candidate regardless of weapon cooldown or margin.

F (player 2 Marine) is idle or on ATTACK_MOVE with engaged target A in +x and enemy B in -y;
periodic scan frames are loops 2, 10, 18, ... (acquire_start_loop=2, period=8). Disabled by
default with bitwise-identical behavior.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim, f32, is_enemy, range_edge, weapon_for
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn

from .test_pyref_batch import _sim as _crowd

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, HELLION, ULTRALISK = 48, 53, 109
F, A, B = 0, 1, 2
RNG = STATIC.get(MARINE).weapons[0].range
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2, 'retarget': True}  # fmt: skip


def _r(unit: int) -> float:
    return STATIC.get(unit).radius


def _sim(ta: int, tb: int, ea: float, eb: float, kind: CommandKind, **config: object) -> Sim:
    """Place A (type ta) at range + ea and B (type tb) at range + eb (edge-to-edge); F is
    engaged with A on cooldown, last fired loop 0 (scan frame 10 is outside deferral)."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    rf = _r(MARINE)
    sc = Scenario(
        FLAT,
        spawns=(
            Spawn(2, MARINE, 16.0, 16.0),
            Spawn(1, ta, 16.0 + rf + _r(ta) + RNG + ea, 16.0),
            Spawn(1, tb, 16.0, 16.0 - rf - _r(tb) - RNG - eb),
        ),
    )
    be.reset([sc], seeds=[0])
    sim = be._sim  # noqa: SLF001 - test mutates state directly
    assert sim is not None
    w = sim.world
    w.order_kind[0, F] = kind
    if kind == CommandKind.ATTACK_MOVE:
        w.order_x[0, F], w.order_y[0, F] = np.float32(28), np.float32(16)
    if kind == CommandKind.ATTACK_UNIT:
        w.order_target_slot[0, F] = A
    w.engaged_target_slot[0, F] = A
    w.weapon_cooldown[0, F] = np.float32(5)
    sim.combat.last_fire_loop[0, F] = 0
    return sim


def _at(sim: Sim, loop: int) -> int:
    sim.world.game_loop[:] = loop
    targeting.sc2(sim)
    return int(sim.world.engaged_target_slot[0, F])


def test_defaults_and_validation() -> None:
    assert not PyrefConfig().retarget_acquire
    with pytest.raises(ValueError, match='retarget_acquire.*retarget'):
        PyrefConfig(retarget_acquire=True)


@pytest.mark.parametrize('kind', [CommandKind.NOOP, CommandKind.ATTACK_MOVE])
@pytest.mark.parametrize('cooldown', [0.0, 5.0])
@pytest.mark.parametrize('ready', [False, True])
def test_switch_to_center_closer_in_acquire_range(
    kind: CommandKind, cooldown: float, ready: bool
) -> None:
    """A is 0.6 past range (Ultralisk, center range + 1.6) and B is 0.3 past range inside
    acquire range (Hellion, center range + 0.925): enabled switches to B on scan frame 10
    at any cooldown; disabled switches only if ready and cooldown == 0 (0.3 > 0.2)."""
    for acquire in (False, True):
        sim = _sim(ULTRALISK, HELLION, 0.6, 0.3, kind, retarget_acquire=acquire,
                   retarget_ready=ready)  # fmt: skip
        sim.world.weapon_cooldown[0, F] = np.float32(cooldown)
        assert _at(sim, 9) == A  # not a scan frame
        want = acquire or (ready and cooldown == 0)
        assert _at(sim, 10) == (B if want else A)


def test_center_not_edge() -> None:
    """B closer by 0.2 edge-to-edge but 0.175 farther center-to-center (A Hellion +0.4,
    B Ultralisk +0.2): stays on A; vice versa (A Ultralisk +0.3, B Hellion +0.4): swaps."""
    sim = _sim(HELLION, ULTRALISK, 0.4, 0.2, CommandKind.NOOP, retarget_acquire=True)
    assert _at(sim, 10) == A
    sim = _sim(ULTRALISK, HELLION, 0.3, 0.4, CommandKind.NOOP, retarget_acquire=True)
    assert _at(sim, 10) == B


@pytest.mark.parametrize(
    ('ea', 'eb', 'kind'),
    [
        (0.8, 0.6, CommandKind.NOOP),  # B outside acquire range (range + 0.5)
        (-0.1, 0.3, CommandKind.NOOP),  # A still in weapon range
        (0.4, 0.3, CommandKind.ATTACK_UNIT),  # ATTACK_UNIT does not retarget
    ],
)
def test_no_switch(ea: float, eb: float, kind: CommandKind) -> None:
    sim = _sim(ULTRALISK, HELLION, ea, eb, kind, retarget_acquire=True)
    assert _at(sim, 10) == A


def test_deferred_like_other_candidates() -> None:
    """Scan frame 10 inside fire deferral (loop 5): defers switch to fire + 8."""
    sim = _sim(ULTRALISK, HELLION, 0.4, 0.3, CommandKind.NOOP, retarget_acquire=True)
    sim.combat.last_fire_loop[0, F] = 5
    assert [_at(sim, t) for t in range(10, 14)] == [A, A, A, B]


def _center(sim: Sim, b: int, u: int, t: int) -> np.float32:
    w = sim.world
    dx, dy = w.x[b, t] - w.x[b, u], w.y[b, t] - w.y[b, u]
    return f32(np.sqrt(dx * dx + dy * dy))


def _ref_switch(sim: Sim, b: int, u: int) -> int:
    """Per-unit reference for an engaged unit whose target is out of range on a scan frame.
    Candidates in acquire range: in-range and closer edge-to-edge; if retarget_ready and
    cooldown == 0, closer edge-to-edge by > margin; if retarget_acquire, closer by center.
    Ranked by armed first, then smallest edge distance, then lowest slot."""
    cfg, w, p = sim.config, sim.world, sim.params
    cur = int(w.engaged_target_slot[b, u])
    ce, cc = range_edge(sim, b, u, cur), _center(sim, b, u, cur)
    ready = cfg.retarget_ready and not w.weapon_cooldown[b, u] > 0
    best, best_key = cur, None
    for t in range(sim.n_slots):
        wi = weapon_for(sim, b, u, t)
        if t == u or not is_enemy(w, b, u, t) or wi < 0:
            continue
        e, reach = range_edge(sim, b, u, t), p.weapon_range[b, u, wi]
        if not e <= max(reach + f32(cfg.acquire_margin), f32(cfg.acquire_min_range)):
            continue
        ok = bool(e <= reach and e < ce)
        ok |= bool(ready and e < ce - f32(cfg.retarget_ready_margin))
        ok |= bool(cfg.retarget_acquire and _center(sim, b, u, t) < cc)
        key = (not p.has_weapon[b, t].any(), e, t)
        if ok and (best_key is None or key < best_key):
            best, best_key = t, key
    return best


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('ready', [False, True])
def test_batch_matches_per_unit(seed: int, ready: bool) -> None:
    """Randomly scattered 3s5z (4 envs, distinct Stalker/Zealot radii) on a scan frame
    outside deferral: batch `targeting.sc2` matches per-unit reference bitwise for engaged
    units with and without `retarget_acquire`; unjudged units match across configs."""
    sim = _crowd('3s5z', 4, seed, **BASE, retarget_lock='none', retarget_ready=ready)
    sim.config = dataclasses.replace(sim.config, auto_acquire_owners=None)
    rng = np.random.default_rng(300 + seed)
    w, n = sim.world, sim.n_slots
    shape = w.x.shape
    cx, cy = float(np.nanmean(w.x)), float(np.nanmean(w.y))
    w.x[:] = (cx + rng.uniform(-6, 6, shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-6, 6, shape)).astype(np.float32)
    w.order_kind[:] = rng.choice([int(CommandKind.NOOP), int(CommandKind.ATTACK_MOVE)], shape)
    foe = (w.owner[:, None, :] != w.owner[:, :, None]) & (w.owner[:, None, :] != 0)
    enemy = np.argmax(np.where(foe, rng.random(foe.shape), -1), axis=2)
    w.engaged_target_slot[:] = np.where(rng.random(shape) < 0.9, enemy, NO_SLOT)
    cd = np.where(rng.random(shape) < 0.4, 0, rng.uniform(0.5, 10, shape))
    w.weapon_cooldown[:] = cd.astype(np.float32)
    sim.combat.last_fire_loop[:] = -1
    w.game_loop[:] = 10
    out, judged = {}, np.zeros(shape, bool)
    for acquire in (False, True):
        s = copy.deepcopy(sim)
        s.config = dataclasses.replace(s.config, retarget_acquire=acquire)
        ref = copy.deepcopy(s)
        targeting.sc2(s)
        out[acquire] = s.world.engaged_target_slot.copy()
        for b in range(ref.batch_size):
            for u in range(n):
                cur = int(ref.world.engaged_target_slot[b, u])
                if not ref.world.alive[b, u] or cur == NO_SLOT:
                    continue
                if not targeting._in_sight(ref, b, u, cur):  # noqa: SLF001
                    continue
                judged[b, u] = True
                want = cur if targeting.in_weapon_range(ref, b, u, cur) else _ref_switch(ref, b, u)
                assert out[acquire][b, u] == want, (acquire, b, u)
    np.testing.assert_array_equal(out[False][~judged], out[True][~judged])
    assert (out[False] != out[True]).sum() >= 5  # option has observable effect


def _episode(ebs: list[float], seeds: list[int], acquire: bool) -> list:
    """F idles, A (Ultralisk) starts at range - 0.2, B (Hellion) at range + eb per env;
    F shoots A, then on loop 6 A MOVEs 1.2 farther along +x while F chases (40 loops)."""
    cfg = PyrefConfig(**{**BASE, 'retarget_lock': 'none', 'retarget_acquire': acquire})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(ebs), cfg)
    rf, ra, rb = _r(MARINE), _r(ULTRALISK), _r(HELLION)
    scs = [
        Scenario(
            FLAT,
            spawns=(
                Spawn(2, MARINE, 16.0, 16.0),
                Spawn(1, ULTRALISK, 16.0 + rf + ra + RNG - 0.2, 16.0),
                Spawn(1, HELLION, 16.0, 16.0 - rf - rb - RNG - eb),
            ),
        )
        for eb in ebs
    ]
    st = be.reset(scs, seeds=seeds)
    ax = 16.0 + rf + ra + RNG + 1.0
    states = [st]
    for loop in range(40):
        cmd = Command.noop(len(ebs), 3)
        for b in range(len(ebs)):
            if loop == 0:
                cmd.set(b, A, CommandKind.HOLD)
                cmd.set(b, B, CommandKind.HOLD)
            if loop == 6:
                cmd.set(b, A, CommandKind.MOVE, target_xy=(ax, 16.0))
        states.append(be.step(cmd, 1))
    be.close()
    return states


def test_batch_matches_single_episode() -> None:
    """Batched run (varying eb/seeds) matches single-env runs bitwise; switches only on."""
    ebs, seeds = [0.1, 0.3, 0.45], [0, 1, 2]
    for acquire in (False, True):
        every = _episode(ebs, seeds, acquire)
        switched = [any(int(s.engaged_target_slot[b, F]) == B for s in every) for b in range(3)]
        assert switched == [acquire] * 3
        for b, (eb, seed) in enumerate(zip(ebs, seeds, strict=True)):
            one = _episode([eb], [seed], acquire)
            for s_all, s_one in zip(every, one, strict=True):
                for f in ('x', 'y', 'weapon_cooldown', 'engaged_target_slot', 'health'):
                    a, o = getattr(s_all, f)[b], getattr(s_one, f)[0]
                    assert np.array_equal(a, o, equal_nan=True)
