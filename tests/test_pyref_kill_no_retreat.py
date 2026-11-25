"""Tests for pyref `kill_no_retreat` and `kill_no_retreat_end`.

Idle units of `retreat_owners` retreat by `retreat_distance` when hit. With
`kill_no_retreat=True`, units whose ATTACK_UNIT target dies and revert to idle do not retreat
when hit until they receive a new command.

Scenario: player 1 Marine A at (10, 16); player 2 Marine E1 ahead in range with 1 HP; player 2
Marine E2 above in range. On loop 0, A issues ATTACK_UNIT E1 and kills E1 on its first shot,
while E2 (player 2 auto-acquire) keeps attacking A.

With `kill_no_retreat_end='lock'`, no-retreat lasts only until `last_fire_loop + follow_lock`
(13 for Marine), after which hits trigger normal retreat; units that never fired retreat
immediately after the target dies.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.subsystems import targeting
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V4, _V5, _V6

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE = 48
RM = STATIC.get(MARINE).radius
A, E1, E2 = 0, 1, 2
BASE = PyrefConfig(commands='delayed', command_delay=1, move_lag=1, **_V5)  # type: ignore[arg-type]
ON = dataclasses.replace(BASE, kill_no_retreat=True)
LOCK = dataclasses.replace(ON, kill_no_retreat_end='lock')
STOP_AT = 60  # loop to issue STOP to A (new command clears kill_idle)


def _run(cfg: PyrefConfig, loops: int = 100) -> list[WorldState]:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        scen = Scenario(
            FLAT,
            spawns=(
                Spawn(1, MARINE, 10.0, 16.0),
                Spawn(2, MARINE, 10.0 + 2 * RM + 3.0, 16.0),
                Spawn(2, MARINE, 10.0, 16.0 + 2 * RM + 4.0),
            ),
        )
        rows = [be.reset([scen], seeds=[0])]
        sim = be._sim  # noqa: SLF001 - direct state setup for test
        assert sim is not None
        sim.world.health[0, E1] = 1.0
        for loop in range(loops):
            cmd = Command.noop(1, 3)
            if loop == 0:
                cmd.set(0, A, CommandKind.ATTACK_UNIT, target_slot=E1)
            elif loop == STOP_AT:
                cmd.set(0, A, CommandKind.STOP)
            rows.append(be.step(cmd, 1))
    finally:
        be.close()
    return rows


def _hits(rows: list[WorldState], u: int) -> list[int]:
    hp = [float(s.health[0, u]) for s in rows]
    return [k for k in range(1, len(hp)) if hp[k] < hp[k - 1] - 0.1]


def test_defaults() -> None:
    assert PyrefConfig().kill_no_retreat is False
    assert PyrefConfig().kill_no_retreat_end == 'command'
    for opts in (_V4, _V5, _V6):
        assert 'kill_no_retreat' not in opts
        assert 'kill_no_retreat_end' not in opts
    assert BASE.retreat_owners == (1,)
    with pytest.raises(ValueError, match='kill_no_retreat_end'):
        dataclasses.replace(ON, kill_no_retreat_end='never')
    with pytest.raises(ValueError, match='kill_no_retreat_end.*kill_no_retreat'):
        dataclasses.replace(BASE, kill_no_retreat_end='lock')


def test_no_retreat_until_next_command() -> None:
    off, on = _run(BASE), _run(ON)
    died = next(k for k, s in enumerate(off) if not s.alive[0, E1])
    assert died == next(k for k, s in enumerate(on) if not s.alive[0, E1])
    # Both configs are bit-identical before E1 dies
    for f in WorldState.UNIT_FIELDS:
        for k in range(died):
            a, b = getattr(off[k], f), getattr(on[k], f)
            assert np.array_equal(a, b, equal_nan=True), (f, k)
    after = [k for k in _hits(off, A) if died < k < STOP_AT]
    assert after, 'A should be hit after E1 dies and before STOP'
    # Off: retreats on hit (MOVE away from E2 along -y)
    k = after[0]
    assert int(off[k].order_kind[0, A]) == CommandKind.MOVE
    assert float(off[k + 5].y[0, A]) < float(off[k].y[0, A]) - 0.3
    # On: stays idle and stationary without engaging until STOP
    for j in range(died + 1, STOP_AT + 1):
        assert int(on[j].order_kind[0, A]) == CommandKind.NOOP, j
        assert float(on[j].x[0, A]) == float(on[died].x[0, A])
        assert float(on[j].y[0, A]) == float(on[died].y[0, A])
    # After STOP, unit is normally idle and retreats on hit
    later = [k for k in _hits(on, A) if k > STOP_AT + 1]
    assert later, 'A should be hit after STOP'
    assert int(on[later[0]].order_kind[0, A]) == CommandKind.MOVE


@pytest.mark.parametrize('on', [False, True])
def test_flag_state(on: bool) -> None:
    """`combat.kill_idle` is set only when enabled and cleared on a new command."""
    cfg = ON if on else BASE
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        scen = Scenario(FLAT, spawns=(Spawn(1, MARINE, 10.0, 16.0), Spawn(2, MARINE, 14.0, 16.0)))
        be.reset([scen], seeds=[0])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        sim.world.health[0, 1] = 1.0
        cmd = Command.noop(1, 2)
        cmd.set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
        be.step(cmd, 1)
        for _ in range(10):
            be.step(Command.noop(1, 2), 1)
        assert not sim.world.alive[0, 1]
        assert bool(sim.combat.kill_idle[0, 0]) is on
        cmd = Command.noop(1, 2)
        cmd.set(0, 0, CommandKind.MOVE, target_xy=(8.0, 16.0))
        be.step(cmd, 2)
        assert not sim.combat.kill_idle[0, 0]
    finally:
        be.close()


def test_lock_ends_at_follow_lock() -> None:
    """In 'lock' mode, hits <= 12 loops after the killing shot match 'command' (no retreat),
    and the first hit after that triggers retreat; states are bit-identical until then."""
    on, lk = _run(ON), _run(LOCK)
    died = next(k for k, s in enumerate(on) if not s.alive[0, E1])
    cd = [float(s.weapon_cooldown[0, A]) for s in lk]
    fire = max(k for k in range(1, died + 1) if cd[k] > cd[k - 1] + 0.5)
    hits = [k for k in _hits(lk, A) if died < k < STOP_AT]
    early = [k for k in hits if k - fire <= 12]
    late = [k for k in hits if k - fire >= 13]
    assert early and late, (fire, hits)
    for k in early:
        assert int(lk[k].order_kind[0, A]) == CommandKind.NOOP, k
    k = late[0]
    assert int(lk[k].order_kind[0, A]) == CommandKind.MOVE
    assert int(on[k].order_kind[0, A]) == CommandKind.NOOP
    for f in WorldState.UNIT_FIELDS:
        for j in range(k):
            a, b = getattr(on[j], f), getattr(lk[j], f)
            assert np.array_equal(a, b, equal_nan=True), (f, j)


@pytest.mark.parametrize(('end', 'since_fire', 'retreat'), [
    ('command', 12, False), ('command', 40, False), ('command', None, False),
    ('lock', 12, False), ('lock', 13, True), ('lock', None, True),
])  # fmt: skip
def test_lock_window(end: str, since_fire: int | None, retreat: bool) -> None:
    """Direct `on_hit`: no retreat while `game_loop - last_fire < follow_lock` (13); units that
    never fired (-1) retreat in 'lock' mode."""
    cfg = dataclasses.replace(ON, kill_no_retreat_end=end)
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    try:
        scen = Scenario(FLAT, spawns=(Spawn(1, MARINE, 10.0, 16.0), Spawn(2, MARINE, 14.0, 16.0)))
        be.reset([scen], seeds=[0])
        sim = be._sim  # noqa: SLF001
        assert sim is not None
        assert int(sim.params.follow_lock[0, 0]) == 13
        sim.world.game_loop[:] = 100
        sim.combat.kill_idle[0, 0] = True
        sim.combat.last_fire_loop[0, 0] = -1 if since_fire is None else 100 - since_fire
        targeting.on_hit(sim, 0, 1, 0)
        assert (int(sim.world.order_kind[0, 0]) == CommandKind.MOVE) is retreat
    finally:
        be.close()
