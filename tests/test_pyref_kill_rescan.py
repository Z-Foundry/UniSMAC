"""Tests for pyref `kill_rescan_lock`, `hit_keeps_kill_wait`, and `kill_rescan_delay=0`.

With `kill_rescan_lock`, post-kill rescan occurs no earlier than
`last_fire_loop + follow_lock - 1` (using `follow_lock` from `measured.json`).

X (player 2, idle, auto-acquire) first attacks the nearest in-range enemy A (player 1 Zergling
with 1 HP, killed in one shot); player 1 Ultralisk B is also in range but farther. By default X
rescans `kill_rescan_delay` loops after A dies and engages B. When enabled, Marine engages B at
fire + 12 (`follow_lock` 13 - 1) and Stalker at fire + 8, while Marauder (`follow_lock` 1) and
Hydralisk (no `follow_lock`) are unchanged. Disabled by default, leaving behavior bit-identical.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, MARAUDER, STALKER, HYDRALISK = 48, 51, 74, 107
ZERGLING, ULTRALISK = 105, 109
X, A, B = 0, 1, 2
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,),
        'acquire_start_loop': 2}  # fmt: skip
N_LOOPS = 80


def _scenario(unit: int, gap: float) -> Scenario:
    """Place A at edge distance `gap` along +x and B at edge `range - 0.3` along -y (farther)."""
    spec = STATIC.get(unit)
    rng, r = spec.weapons[0].range, spec.radius
    ra, rb = r + STATIC.get(ZERGLING).radius, r + STATIC.get(ULTRALISK).radius
    return Scenario(
        FLAT,
        spawns=(
            Spawn(2, unit, 16.0, 16.0),
            Spawn(1, ZERGLING, 16.0 + ra + gap, 16.0),
            Spawn(1, ULTRALISK, 16.0, 16.0 - rb - (rng - 0.3)),
        ),
    )


def _run(
    scenarios: list[Scenario],
    seeds: list[int] | None = None,
    health: list[float] | None = None,
    **config: object,
) -> list[WorldState]:
    """Set A's health to `health` (default 1 per env) and step with noops; returns states from
    row 0 (post-reset). Seeds default to 0, 1, ..."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(scenarios), cfg)
    try:
        rows = [be.reset(scenarios, seeds=seeds or list(range(len(scenarios))))]
        sim = be._sim  # noqa: SLF001 - direct state setup for test
        assert sim is not None
        sim.world.health[:, A] = np.asarray(health or [1.0] * len(scenarios), np.float32)
        noop = Command.noop(len(scenarios), rows[0].x.shape[1])
        rows += [be.step(noop, 1) for _ in range(N_LOOPS)]
    finally:
        be.close()
    return rows


def _events(rows: list[WorldState], b: int = 0) -> tuple[int, int, int]:
    """Return (X's last fire row on A, A's death row, X's engage-B row); fire = cooldown jump."""
    cd = [float(s.weapon_cooldown[b, X]) for s in rows]
    dead = next(k for k, s in enumerate(rows) if not s.alive[b, A])
    fire = max(k for k in range(1, dead + 1) if cd[k] > cd[k - 1] + 0.5)
    engage = next(k for k, s in enumerate(rows) if int(s.engaged_target_slot[b, X]) == B)
    return fire, dead, engage


def _same(a: list[WorldState], b: list[WorldState]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for f in dataclasses.fields(x):
            assert np.array_equal(getattr(x, f.name), getattr(y, f.name), equal_nan=True), f.name


def test_defaults_and_validation() -> None:
    assert PyrefConfig().kill_rescan_lock is False
    with pytest.raises(ValueError, match="kill_rescan_lock.*targeting='sc2'"):
        PyrefConfig(kill_rescan_lock=True)
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'kill_rescan_lock' not in opts


@pytest.mark.parametrize(
    ('lock', 'last', 'delay', 'want'),
    [
        (13, 5, 2, 17),  # Marine: last fire 5 + 12 is later than loop 10 + 2
        (13, 9, 2, 21),
        (13, -1, 2, 12),  # never fired: default delay
        (9, 5, 2, 13),  # Stalker, Zealot: +8
        (1, 9, 2, 12),  # Marauder: +0, not delayed
        (-1, 9, 2, 12),  # no follow_lock: default delay
        (13, 0, 2, 12),  # lock already expired: default delay
        (13, 0, 1, 12),
        (13, -2, 1, 11),
    ],
)
def test_on_kill_due(lock: int, last: int, delay: int, want: int) -> None:
    """`on_kill` sets rescan loop to max(loop + kill_rescan_delay, last_fire + follow_lock - 1)."""
    cfg = PyrefConfig(**{**BASE, 'kill_rescan_lock': True, 'kill_rescan_delay': delay})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        be.reset([_scenario(MARINE, 2.0)], seeds=[0])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.game_loop[:] = 10
        sim.params.follow_lock[0, X] = lock
        sim.combat.last_fire_loop[0, X] = last
        targeting.on_kill(sim, 0, X)
        assert int(sim.combat.scan_loop[0, X]) == want
    finally:
        be.close()


def test_marine_waits_for_lock() -> None:
    """Marine engages B at dead + 2 by default, or fire + 12 with `kill_rescan_lock`; setting
    `kill_rescan_delay=1` leaves the lock-governed result unchanged."""
    sc = [_scenario(MARINE, 2.0)]
    fire, dead, engage = _events(_run(sc))
    assert dead - fire <= 1
    assert engage == dead + 2
    fire1, dead1, engage1 = _events(_run(sc, kill_rescan_lock=True))
    assert (fire1, dead1) == (fire, dead)
    assert engage1 == fire + 12
    assert _events(_run(sc, kill_rescan_lock=True, kill_rescan_delay=1)) == (fire, dead, fire + 12)


def test_stalker_waits_for_lock() -> None:
    """Stalker (follow_lock=9) engages B at fire + 8."""
    sc = [_scenario(STALKER, 2.0)]
    fire, dead, engage = _events(_run(sc))
    assert engage == dead + 2 < fire + 8
    assert _events(_run(sc, kill_rescan_lock=True)) == (fire, dead, fire + 8)


@pytest.mark.parametrize('unit', [MARAUDER, HYDRALISK])
def test_units_without_wait_unchanged(unit: int) -> None:
    """Marauder (follow_lock=1, offset 0) and Hydralisk (no lock) match disabled bit-for-bit."""
    sc = [_scenario(unit, 2.0)]
    _same(_run(sc, kill_rescan_lock=True), _run(sc))


def test_batch_matches_single() -> None:
    """Batched run (varying A gap, 1-3 shot HP, and seeds) matches single-env bit-for-bit."""
    gaps, health = [0.5, 2.0, 3.5], [1.0, 7.0, 13.0]
    scs = [_scenario(MARINE, g) for g in gaps]
    every = _run(scs, health=health, kill_rescan_lock=True)
    engages = set()
    for b, sc in enumerate(scs):
        one = _run([sc], seeds=[b], health=[health[b]], kill_rescan_lock=True)
        for s_all, s_one in zip(every, one, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True), f
        fire, _, engage = _events(every, b)
        assert engage == fire + 12
        engages.add(engage)
    assert len(engages) == len(scs), 'engage rows should differ across envs'


# ---------------------------------------------------------------- hit_keeps_kill_wait


def test_hit_keeps_kill_wait_validation() -> None:
    assert PyrefConfig().hit_keeps_kill_wait is False
    with pytest.raises(ValueError, match="hit_keeps_kill_wait.*targeting='sc2'"):
        PyrefConfig(hit_keeps_kill_wait=True)


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('scan_loop', [-1, 10, 14])
def test_on_hit_during_kill_wait(keep: bool, scan_loop: int) -> None:
    """X is hit by B on loop 10: while waiting (`scan_loop=14`), enabling keeps the wait and
    skips retaliation, whereas disabling engages B immediately. Not waiting (-1) or due (10)
    engages in both modes."""
    cfg = PyrefConfig(**{**BASE, 'kill_rescan_lock': True, 'hit_keeps_kill_wait': keep})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        be.reset([_scenario(MARINE, 2.0)], seeds=[0])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.game_loop[:] = 10
        sim.combat.scan_loop[0, X] = scan_loop
        targeting.on_hit(sim, 0, B, X)
        waiting = scan_loop > 10
        assert int(sim.world.engaged_target_slot[0, X]) == (-1 if keep and waiting else B)
        assert int(sim.combat.scan_loop[0, X]) == (scan_loop if keep and waiting else -1)
    finally:
        be.close()


def test_hit_during_wait_episode() -> None:
    """When A dies, out-of-acquire Stalker C (player 1, edge 5.8 along +y) attacks X during the
    wait: enabled still engages nearest B at fire + 12; disabled engages C on the hit row."""
    base = _scenario(MARINE, 2.0)
    d = STATIC.get(MARINE).radius + STATIC.get(STALKER).radius + 5.8
    sc = Scenario(FLAT, spawns=(*base.spawns, Spawn(1, STALKER, 16.0, 16.0 + d)))
    c = len(base.spawns)
    rows: dict[bool, list[WorldState]] = {}
    for keep in (False, True):
        cfg = PyrefConfig(**{**BASE, 'kill_rescan_lock': True, 'hit_keeps_kill_wait': keep})  # type: ignore[arg-type]
        be = PyrefBackend(STATIC, MAPS, 1, cfg)
        try:
            out = [be.reset([sc], seeds=[0])]
            sim = be._sim  # noqa: SLF001
            assert sim is not None
            sim.world.health[:, A] = np.float32(1)
            dead = N_LOOPS
            for k in range(N_LOOPS):
                cmd = Command.noop(1, out[0].x.shape[1])
                if k == 0:
                    cmd.set(0, c, CommandKind.HOLD)
                if not out[-1].alive[0, A] and dead == N_LOOPS:
                    dead = k
                    cmd.set(0, c, CommandKind.ATTACK_UNIT, target_slot=X)
                out.append(be.step(cmd, 1))
        finally:
            be.close()
        rows[keep] = out
    fire, dead, _ = _events(rows[True])
    hp = [float(s.health[0, X]) for s in rows[False]]
    hit = next(k for k in range(dead, len(hp)) if hp[k] < hp[k - 1])
    assert hit < fire + 12
    first = {k: next((r, int(s.engaged_target_slot[0, X])) for r, s in enumerate(v)
                     if r > dead and int(s.engaged_target_slot[0, X]) >= 0)
             for k, v in rows.items()}  # fmt: skip
    assert first[False] == (hit, c)
    assert first[True] == (fire + 12, B)


# ---------------------------------------------------------------- kill_rescan_delay=0


def test_delay0_validation() -> None:
    with pytest.raises(ValueError, match="kill_rescan_delay.*targeting='sc2'"):
        PyrefConfig(kill_rescan_delay=0)
    with pytest.raises(ValueError, match='kill_rescan_delay'):
        PyrefConfig(targeting='sc2', kill_rescan_delay=-1)
    assert PyrefConfig(targeting='sc2', kill_rescan_delay=0).kill_rescan_delay == 0


@pytest.mark.parametrize(('delay', 'lag'), [(2, 2), (1, 1), (0, 0)])
def test_delay0_same_row(delay: int, lag: int) -> None:
    """Without `kill_rescan_lock`, X engages B at `dead + kill_rescan_delay` (0 = same loop)."""
    _, dead, engage = _events(_run([_scenario(MARINE, 2.0)], kill_rescan_delay=delay))
    assert engage == dead + lag


def test_delay0_lock() -> None:
    """With `kill_rescan_lock` and delay 0: Marine still waits until fire + 12; Marauder (lock
    offset 0) engages on the death row."""
    fire, dead, engage = _events(_run([_scenario(MARINE, 2.0)], kill_rescan_lock=True,
                                      kill_rescan_delay=0))  # fmt: skip
    assert engage == fire + 12 > dead
    _, dead, engage = _events(_run([_scenario(MARAUDER, 2.0)], kill_rescan_lock=True,
                                   kill_rescan_delay=0))  # fmt: skip
    assert engage == dead


def test_delay0_batch_matches_single() -> None:
    """With `kill_rescan_delay=0`, batched run matches single-env bit-for-bit."""
    gaps, health = [0.5, 2.0, 3.5], [1.0, 7.0, 13.0]
    scs = [_scenario(MARINE, g) for g in gaps]
    every = _run(scs, health=health, kill_rescan_lock=True, kill_rescan_delay=0)
    for b, sc in enumerate(scs):
        one = _run([sc], seeds=[b], health=[health[b]], kill_rescan_lock=True, kill_rescan_delay=0)
        for s_all, s_one in zip(every, one, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True), f
