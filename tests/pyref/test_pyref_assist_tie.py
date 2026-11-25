"""Tests for pyref `hit_assist_tie` and `hit_assist_cap`.

`hit_assist_tie='spawn_order'`: when multiple attackers hit on the same loop at equal center
distance from a responder, pick the earliest in spawn order (modeling SC2 tag order); default
'first' keeps the first resolved. `hit_assist_cap`: per victim per loop, only the first N hits
in update order trigger ally assist, flushed after all hits resolve (`targeting.assist_flush`);
default 0 is unlimited.

Setup: V (Player 2 Marine, slot 0) at (16, 16); responder H (slot 3, Player 2 Marine) at
(14.5, 16); Player 1 Stalkers S1 (slot 1) at (21, 19) and S2 (slot 2) at (21, 13) are
equidistant from H; S3 (slot 4, Player 1 Stalker) at (18.5, 16) is closest to H.
"""

from __future__ import annotations

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
V, S1, S2, H, S3 = 0, 1, 2, 3, 4
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'damage': 'sc2', 'auto_acquire': True,
        'auto_acquire_owners': (2,), 'acquire_start_loop': 2, 'hit_assist': True,
        'hit_assist_pick': 'nearest'}  # fmt: skip
SPAWNS = (Spawn(2, MARINE, 16.0, 16.0), Spawn(1, STALKER, 21.0, 19.0),
          Spawn(1, STALKER, 21.0, 13.0), Spawn(2, MARINE, 14.5, 16.0),
          Spawn(1, STALKER, 18.5, 16.0))  # fmt: skip


def _sim(seed: int = 0, **config: object) -> tuple[PyrefBackend, Sim]:
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([Scenario(FLAT, spawns=SPAWNS)], seeds=[seed])
    sim = be._sim  # noqa: SLF001 - test mutates internal state directly
    assert sim is not None
    sim.world.game_loop[:] = 11  # non-scan loop (acquire_start_loop=2, period 8)
    return be, sim


def _d(sim: Sim, a: int) -> float:
    w = sim.world
    return float(np.hypot(w.x[0, a] - w.x[0, H], w.y[0, a] - w.y[0, H]))


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.hit_assist_tie == 'first'
    assert cfg.hit_assist_cap == 0
    with pytest.raises(ValueError, match='hit_assist_tie'):
        PyrefConfig(**{**BASE, 'hit_assist_tie': 'tag'})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hit_assist_tie.*hit_assist_pick='nearest'"):
        PyrefConfig(targeting='sc2', hit_assist=True, hit_assist_tie='spawn_order')
    with pytest.raises(ValueError, match='hit_assist_cap'):
        PyrefConfig(**{**BASE, 'hit_assist_cap': -1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hit_assist_cap.*hit_assist.*damage='sc2'"):
        PyrefConfig(**{**BASE, 'damage': 'standard', 'hit_assist_cap': 2})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hit_assist_cap.*hit_assist.*damage='sc2'"):
        PyrefConfig(targeting='sc2', damage='sc2', hit_assist_cap=2)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'hit_assist_tie' not in opts
        assert 'hit_assist_cap' not in opts


def test_spawn_before_is_total_order() -> None:
    """spawn_before is a strict total order matching the scan_phase='per_unit' permutation."""
    be, sim = _sim(seed=7, scan_phase='per_unit')
    try:
        n = sim.n_slots
        order = sorted(
            range(n), key=lambda u: sum(targeting.spawn_before(sim, 0, k, u) for k in range(n))
        )
        for i, u in enumerate(order):
            for k in order[i + 1 :]:
                assert targeting.spawn_before(sim, 0, u, k)
                assert not targeting.spawn_before(sim, 0, k, u)
        assert [int(sim.combat.scan_shift[0, u]) for u in order] == [r % 8 for r in range(n)]
    finally:
        be.close()


@pytest.mark.parametrize('order', [(S1, S2), (S2, S1)])
def test_tie_same_loop(order: tuple[int, int]) -> None:
    """Equidistant S1 and S2 hit on the same loop: 'first' picks first resolved,
    'spawn_order' picks earlier spawn regardless of resolution order."""
    seen = set()
    for seed in range(8):
        for tie in ('first', 'spawn_order'):
            be, sim = _sim(seed=seed, hit_assist_tie=tie)
            try:
                assert _d(sim, S1) == _d(sim, S2)
                for a in order:
                    targeting.on_hit(sim, 0, a, V)
                got = int(sim.world.engaged_target_slot[0, H])
                if tie == 'first':
                    assert got == order[0]
                else:
                    want = S1 if targeting.spawn_before(sim, 0, S1, S2) else S2
                    assert got == want
                    seen.add(got)
            finally:
                be.close()
    assert seen == {S1, S2}  # both spawn orders occur across 8 seeds


def test_tie_later_loop_keeps() -> None:
    """Equal-distance hits on a later loop do not switch (same-loop only); closer still does."""
    for seed in range(8):
        be, sim = _sim(seed=seed, hit_assist_tie='spawn_order')
        try:
            first, second = (S2, S1) if targeting.spawn_before(sim, 0, S1, S2) else (S1, S2)
            targeting.on_hit(sim, 0, first, V)  # later-spawned attacker hits first
            sim.world.game_loop[:] = 12
            targeting.on_hit(sim, 0, second, V)
            assert int(sim.world.engaged_target_slot[0, H]) == first
            targeting.on_hit(sim, 0, S3, V)
            assert int(sim.world.engaged_target_slot[0, H]) == S3
        finally:
            be.close()


@pytest.mark.parametrize('cap', [0, 2, 3])
def test_cap_drops_last_in_update_order(cap: int) -> None:
    """S1, S2, S3 hit V on the same loop: with cap=2, the last hit in update order is dropped
    and H picks the nearest of the first two; with cap=0 or 3, H picks nearest S3. When
    cap > 0, on_hit only records hits and assist_flush engages."""
    seen = set()
    for seed in range(12):
        be, sim = _sim(seed=seed, hit_assist_tie='spawn_order', hit_assist_cap=cap)
        try:
            for a in (S1, S2, S3):
                targeting.on_hit(sim, 0, a, V)
            if cap > 0:
                assert int(sim.world.engaged_target_slot[0, H]) == NO_SLOT
                assert sim.buf.assist_hits[0, V, [S1, S2, S3]].all()
                targeting.assist_flush(sim)
                assert not sim.buf.assist_hits.any()
            got = int(sim.world.engaged_target_slot[0, H])
            if cap != 2:
                assert got == S3
                continue
            by_upd = sorted((S1, S2, S3), key=lambda a: sum(
                targeting.updates_before(sim, 0, k, a) for k in (S1, S2, S3)))  # fmt: skip
            kept = by_upd[:2]
            dmin = min(_d(sim, a) for a in kept)
            tied = [a for a in kept if _d(sim, a) == dmin]
            want = (
                tied[0]
                if len(tied) == 1
                else (S1 if targeting.spawn_before(sim, 0, S1, S2) else S2)
            )
            assert got == want
            assert int(sim.combat.assist_target[0, H]) == want
            seen.add(by_upd[-1])
        finally:
            be.close()
    if cap == 2:
        assert S3 in seen  # S3 ranked last (dropped) at least once


def _episode(**config: object) -> list[WorldState]:
    """S1 ATTACK_UNITs V on loop 0 while S2 and S3 HOLD far away, for 60 loops."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    spawns = (SPAWNS[0], Spawn(1, STALKER, 23.0, 16.0), Spawn(1, STALKER, 28.0, 28.0),
              SPAWNS[3], Spawn(1, STALKER, 28.0, 4.0))  # fmt: skip
    try:
        rows = [be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])]
        for loop in range(60):
            cmd = Command.noop(1, len(spawns))
            if loop == 0:
                cmd.set(0, S1, CommandKind.ATTACK_UNIT, target_slot=V)
                cmd.set(0, S2, CommandKind.HOLD)
                cmd.set(0, S3, CommandKind.HOLD)
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def test_single_attacker_same_as_default() -> None:
    """With one attacker, enabling hit_assist_tie and hit_assist_cap matches default
    bit-for-bit, and H engages S1 on hit."""
    a = _episode()
    for opts in ({'hit_assist_tie': 'spawn_order'}, {'hit_assist_cap': 2},
                 {'hit_assist_tie': 'spawn_order', 'hit_assist_cap': 1}):  # fmt: skip
        b = _episode(**opts)
        for s_a, s_b in zip(a, b, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_a, f), getattr(s_b, f), equal_nan=True), (opts, f)
    assert any(int(s.engaged_target_slot[0, H]) == S1 for s in a)
