"""Tests for pyref `kill_rescan_order`.

With `kill_rescan_delay=0`, when another unit kills X's target, whether X rescans on the
death loop or the next loop depends on the per-loop update order between the killer and X.

X (player 2 Marine, idle, auto-acquire) engages nearest enemy A (player 1 Zergling with high
HP so X cannot kill it); player 1 Ultralisk B is also in range but farther; player 2 unit V
sits north of A and also attacks A. Right before a loop where X does not fire and V hits A,
A's HP is set to 1 so V scores the kill. When disabled, X engages B on the death loop. When
enabled, an instant-hit Marine killer V causes same-loop rescan if V updates before X (else
next loop), while a projectile Stalker killer V always causes next-loop rescan. Disabled by
default, leaving default behavior bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref import rng as pyref_rng
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import NO_SLOT, Command, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, STALKER, ZERGLING, ULTRALISK = 48, 74, 105, 109
X, A, B, V = 0, 1, 2, 3
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'damage': 'sc2', 'auto_acquire': True,
        'auto_acquire_owners': (2,), 'acquire_start_loop': 2, 'kill_rescan_delay': 0}  # fmt: skip
N_LOOPS = 60
BIG = 1.0e4


def _scenario(killer: int) -> Scenario:
    """A at edge 2 (+x from X), B at edge range - 0.3 (-y), V at edge 2 (+y from A)."""
    rx, rz = STATIC.get(MARINE).radius, STATIC.get(ZERGLING).radius
    rb, rv = STATIC.get(ULTRALISK).radius, STATIC.get(killer).radius
    ax = 16.0 + rx + rz + 2.0
    rng_x = STATIC.get(MARINE).weapons[0].range
    return Scenario(
        FLAT,
        spawns=(
            Spawn(2, MARINE, 16.0, 16.0),
            Spawn(1, ZERGLING, ax, 16.0),
            Spawn(1, ULTRALISK, 16.0, 16.0 - rx - rb - (rng_x - 0.3)),
            Spawn(2, killer, ax, 16.0 + rz + rv + 2.0),
        ),
    )


def _run(
    killer: int, seed: int, kill_row: int | None, **config: object
) -> tuple[list[WorldState], bool]:
    """Start A at BIG HP, drop to 1 before `kill_row`, and step with noops. Returns all rows and
    whether V updates before X on `kill_row`."""
    cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    first = False
    try:
        rows = [be.reset([_scenario(killer)], seeds=[seed])]
        sim = be._sim  # noqa: SLF001 - direct state setup for test
        assert sim is not None
        sim.world.health[0, A] = np.float32(BIG)
        noop = Command.noop(1, rows[0].x.shape[1])
        for k in range(1, N_LOOPS + 1):
            if k == kill_row:
                sim.world.health[0, A] = np.float32(1)
                first = targeting.updates_before(sim, 0, V, X)
            rows.append(be.step(noop, 1))
    finally:
        be.close()
    return rows, first


def _kill_row(killer: int, seed: int) -> int:
    """First step after X engages A where X does not fire and A loses HP (hit by V)."""
    rows, _ = _run(killer, seed, None)
    cd = [float(s.weapon_cooldown[0, X]) for s in rows]
    hp = [float(s.health[0, A]) for s in rows]
    eng = [int(s.engaged_target_slot[0, X]) for s in rows]
    return next(k for k in range(2, len(rows)) if eng[k - 1] == A and hp[k] < hp[k - 1]
                and cd[k] <= cd[k - 1] + 0.5)  # fmt: skip


def _events(rows: list[WorldState]) -> tuple[int, int]:
    """Return (A's death row, X's engage-B row)."""
    dead = next(k for k, s in enumerate(rows) if not s.alive[0, A])
    engage = next(k for k, s in enumerate(rows) if int(s.engaged_target_slot[0, X]) == B)
    return dead, engage


def test_defaults_and_validation() -> None:
    assert PyrefConfig().kill_rescan_order is False
    msg = "kill_rescan_order.*kill_rescan_delay=0.*damage='sc2'"
    with pytest.raises(ValueError, match=msg):
        PyrefConfig(targeting='sc2', damage='sc2', kill_rescan_order=True)
    with pytest.raises(ValueError, match=msg):
        PyrefConfig(targeting='sc2', kill_rescan_delay=0, kill_rescan_order=True)
    assert PyrefConfig(**{**BASE, 'kill_rescan_order': True}).kill_rescan_order  # type: ignore[arg-type]
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'kill_rescan_order' not in opts
    assert pyref_rng.STREAMS[:3] == ('spawn_facing', 'cooldown', 'scan_phase')


def test_updates_before_is_a_random_order() -> None:
    """Each loop defines a strict total order across units, varying across loops."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**BASE))  # type: ignore[arg-type]
    try:
        be.reset([_scenario(MARINE)], seeds=[3])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        seen = set()
        for loop in range(40):
            sim.world.game_loop[:] = loop
            before = targeting.updates_before
            for k in range(4):
                for u in range(4):
                    if k != u:
                        assert before(sim, 0, k, u) != before(sim, 0, u, k)
            seen.add(targeting.updates_before(sim, 0, V, X))
        assert seen == {True, False}
    finally:
        be.close()


@pytest.mark.parametrize(
    ('killer', 'proj', 'lock'),
    [
        (NO_SLOT, False, False),
        (X, False, False),
        (V, True, False),
        (V, False, False),
        (V, True, True),
    ],
)
def test_on_kill_due(killer: int, proj: bool, lock: bool) -> None:
    """`on_kill` rescan loop: no killer (non-damage death) or self-kill uses current loop;
    projectile kill uses next loop; hitscan kill depends on update order; `kill_rescan_lock`
    takes max with last_fire + lock - 1 (5 + 12 = 17 here)."""
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**{**BASE, 'kill_rescan_order': True,
                                                      'kill_rescan_lock': lock}))  # type: ignore[arg-type]  # fmt: skip
    try:
        be.reset([_scenario(MARINE)], seeds=[0])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.game_loop[:] = 10
        sim.buf.kill_by[0, A], sim.buf.kill_proj[0, A] = killer, proj
        sim.params.follow_lock[0, X] = 13
        sim.combat.last_fire_loop[0, X] = 5
        targeting.on_kill(sim, 0, X, A)
        late = killer == V and (proj or not targeting.updates_before(sim, 0, V, X))
        want = max(10 + late, 17) if lock else 10 + late
        assert int(sim.combat.scan_loop[0, X]) == want
        targeting.on_kill(sim, 0, X)  # no victim slot passed: default behavior
        assert int(sim.combat.scan_loop[0, X]) == (17 if lock else 10)
    finally:
        be.close()


def test_marine_killer_follows_update_order() -> None:
    """Marine killer V: disabled always engages B on the death row; enabled engages on the death
    row iff V updates before X (else next row), with both cases observed. Pre-death rows match."""
    outcomes = set()
    for seed in range(8):
        k = _kill_row(MARINE, seed)
        off, _ = _run(MARINE, seed, k)
        on, first = _run(MARINE, seed, k, kill_rescan_order=True)
        dead, engage = _events(off)
        assert dead == k
        assert engage == dead
        dead_on, engage_on = _events(on)
        assert dead_on == dead
        assert engage_on == dead + (0 if first else 1)
        for s_off, s_on in zip(off[:dead], on[:dead], strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_off, f), getattr(s_on, f), equal_nan=True), f
        outcomes.add(first)
    assert outcomes == {True, False}


def test_projectile_killer_next_row() -> None:
    """Projectile killer V (Stalker): enabled always engages B on the row after death."""
    for seed in range(4):
        k = _kill_row(STALKER, seed)
        dead, engage = _events(_run(STALKER, seed, k)[0])
        assert engage == dead
        dead_on, engage_on = _events(_run(STALKER, seed, k, kill_rescan_order=True)[0])
        assert (dead_on, engage_on) == (dead, dead + 1)


def test_batch_matches_single() -> None:
    """When enabled, batched run matches single-env bit-for-bit."""
    seeds = [0, 1, 2]
    rows_k = [_kill_row(MARINE, s) for s in seeds]
    cfg = PyrefConfig(**{**BASE, 'kill_rescan_order': True})  # type: ignore[arg-type]
    be = PyrefBackend(STATIC, MAPS, len(seeds), cfg)
    try:
        every = [be.reset([_scenario(MARINE)] * len(seeds), seeds=seeds)]
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.health[:, A] = np.float32(BIG)
        noop = Command.noop(len(seeds), every[0].x.shape[1])
        for k in range(1, N_LOOPS + 1):
            for b, kr in enumerate(rows_k):
                if k == kr:
                    sim.world.health[b, A] = np.float32(1)
            every.append(be.step(noop, 1))
    finally:
        be.close()
    for b, s in enumerate(seeds):
        one, _ = _run(MARINE, s, rows_k[b], kill_rescan_order=True)
        for s_all, s_one in zip(every, one, strict=True):
            for f in WorldState.UNIT_FIELDS:
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True), f
