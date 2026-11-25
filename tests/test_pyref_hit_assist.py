"""Tests for pyref `hit_assist` (allied retaliation when a nearby ally is hit).

When a unit is hit, idle and ATTACK_MOVE allies of the same owner within edge-to-edge
distance <= `hit_assist_range` (default 4) that have no engaged target and are not in
post-kill wait engage the attacker on the hit loop; HOLD and already-engaged allies do not.

V (player 2 Marine) is at (16, 16) and S (player 1 Stalker) is at edge distance 6 along +x
(outside V's acquire range, inside Stalker's weapon range); V's allied Marines sit around V.
Disabled by default, leaving default behavior bit-identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, STALKER = 48, 74
RM, RS = STATIC.get(MARINE).radius, STATIC.get(STALKER).radius
V, S = 0, 1
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2}  # fmt: skip


def _around(edge: float, angle: float) -> tuple[float, float]:
    """Center of an allied Marine at edge distance `edge` and bearing `angle` deg from V."""
    d, a = 2 * RM + edge, math.radians(angle)
    return 16.0 + d * math.cos(a), 16.0 + d * math.sin(a)


def _scenario(helpers: list[tuple[float, float]]) -> Scenario:
    """V, S, helpers at (edge, angle) from slot 2, and a distant player 1 Marine T."""
    spawns = [Spawn(2, MARINE, 16.0, 16.0), Spawn(1, STALKER, 16.0 + RM + RS + 6.0, 16.0)]
    spawns += [Spawn(2, MARINE, *_around(e, a)) for e, a in helpers]
    spawns.append(Spawn(1, MARINE, 3.0, 3.0))
    return Scenario(FLAT, spawns=tuple(spawns))


def _sim(helpers: list[tuple[float, float]], **config: object) -> tuple[PyrefBackend, Sim]:
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([_scenario(helpers)], seeds=[0])
    sim = be._sim  # noqa: SLF001 - direct state setup for test
    assert sim is not None
    return be, sim


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.hit_assist is False
    assert cfg.hit_assist_range == 4.0
    with pytest.raises(ValueError, match="hit_assist.*targeting='sc2'"):
        PyrefConfig(hit_assist=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'hit_assist' not in opts


@pytest.mark.parametrize('assist', [False, True])
def test_on_hit_who_responds(assist: bool) -> None:
    """After `on_hit`: idle (slot 2) and ATTACK_MOVE (slot 7) allies at edge 3.9 engage S;
    allies at edge 4.1 (slot 3), on HOLD (slot 4), engaged on T (slot 5), or in post-kill wait
    (slot 6) do not. V still retaliates. When disabled, only V engages S."""
    helpers = [(3.9, 180.0), (4.1, 270.0), (2.0, 90.0), (2.0, 225.0), (2.0, 135.0), (3.9, 200.0)]
    be, sim = _sim(helpers, hit_assist=assist)
    try:
        w, c = sim.world, sim.combat
        t = len(helpers) + 2
        w.game_loop[:] = 10
        w.order_kind[0, 4] = CommandKind.HOLD
        w.engaged_target_slot[0, 5] = t
        c.scan_loop[0, 6] = 14
        w.order_kind[0, 7] = CommandKind.ATTACK_MOVE
        w.order_x[0, 7], w.order_y[0, 7] = w.x[0, 7], w.y[0, 7]
        targeting.on_hit(sim, 0, S, V)
        got = [int(e) for e in w.engaged_target_slot[0, :t]]
        if assist:
            assert got == [S, NO_SLOT, S, NO_SLOT, NO_SLOT, t, NO_SLOT, S]
            assert int(c.scan_loop[0, 6]) == 14
            assert not np.isnan(c.leash_x[0, 2])  # idle responder records leash
        else:
            assert got == [S, NO_SLOT, NO_SLOT, NO_SLOT, NO_SLOT, t, NO_SLOT, NO_SLOT]
    finally:
        be.close()


def test_range_and_victim_state() -> None:
    """With hit_assist_range=4.2 an ally at edge 4.1 responds; busy V still triggers assist."""
    be, sim = _sim([(4.1, 270.0)], hit_assist=True, hit_assist_range=4.2)
    try:
        w = sim.world
        w.engaged_target_slot[0, V] = 3
        targeting.on_hit(sim, 0, S, V)
        assert int(w.engaged_target_slot[0, 2]) == S
        assert int(w.engaged_target_slot[0, V]) == 3
    finally:
        be.close()


def _run(edges: list[float], assist: bool, seeds: list[int] | None = None) -> list[WorldState]:
    """One idle ally per env at edge edges[b] (180 deg); S attacks V on loop 0; 60 loops."""
    cfg = PyrefConfig(**{**BASE, 'hit_assist': assist})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(edges), cfg)
    try:
        scs = [_scenario([(e, 180.0)]) for e in edges]
        rows = [be.reset(scs, seeds=seeds or list(range(len(edges))))]
        for loop in range(60):
            cmd = Command.noop(len(edges), rows[0].x.shape[1])
            if loop == 0:
                for b in range(len(edges)):
                    cmd.set(b, S, CommandKind.ATTACK_UNIT, target_slot=V)
                    cmd.set(b, 3, CommandKind.HOLD)
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def _hit_and_engage(rows: list[WorldState], b: int) -> tuple[int, int | None]:
    hp = [float(s.health[b, V] + s.shield[b, V]) for s in rows]
    hit = next(k for k in range(1, len(rows)) if hp[k] < hp[k - 1])
    eng = next((k for k, s in enumerate(rows) if int(s.engaged_target_slot[b, 2]) == S), None)
    return hit, eng


@pytest.mark.parametrize('edge', [1.0, 3.5])
def test_episode_engage_at_hit_row(edge: float) -> None:
    """Ally engages S on the hit loop (outside acquire range); never engages if disabled."""
    hit, eng = _hit_and_engage(_run([edge], True), 0)
    assert eng == hit
    assert _hit_and_engage(_run([edge], False), 0)[1] is None


def test_batch_matches_single() -> None:
    """Batched run (varying helper edges and seeds) matches single-env bit-for-bit; 4.5 ignores."""
    edges = [1.0, 3.5, 4.5]
    every = _run(edges, True)
    for b, e in enumerate(edges):
        one = _run([e], True, seeds=[b])
        for s_all, s_one in zip(every, one, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True), f
        hit, eng = _hit_and_engage(every, b)
        assert (eng == hit) if e < 4 else eng is None
