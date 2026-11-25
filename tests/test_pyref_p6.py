"""Targeting tests for `pyref` with `targeting='sc2'`.

- Acquire range is `max(range + 0.5, 5)` (edge-to-edge), preferring armed enemies then distance.
- Idle units scan every 8 loops (`2 + 8n`); commands such as STOP trigger an immediate scan.
- Taking damage triggers retaliation on the hit loop regardless of acquire range; after a kill
  with no new target, idle units MOVE back to their pre-acquire leash position.
- Idle units of `retreat_owners` retreat 5 tiles away from the attacker on hit; HOLD does not.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn

MARINE, MEDIVAC, ZEALOT, STALKER, ZERGLING = 48, 54, 73, 74, 105
MAP_ID = 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {MAP_ID: load_map_info(MAP_ID)}
# Combat timing config (without command delay / move_lag) plus targeting='sc2'; P1 passive
P6 = PyrefConfig(
    attack='sc2',
    damage='sc2',
    auto_acquire=True,
    auto_acquire_owners=(2,),
    acquire_start_loop=2,
    targeting='sc2',
)


def make(cfg: PyrefConfig = P6) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, cfg)


def gap(a: int, b: int, edge: float) -> float:
    return STATIC.get(a).radius + STATIC.get(b).radius + edge


def scenario(*spawns: Spawn) -> Scenario:
    return Scenario(map_id=MAP_ID, spawns=spawns)


def run(backend: PyrefBackend, first: Command, loops: int) -> list:
    """Issue `first` on loop 0 then noops; return post-step states (index k = loop k)."""
    n = first.kind.shape[1]
    return [backend.step(first, 1)] + [
        backend.step(Command.noop(1, n), 1) for _ in range(loops - 1)
    ]


def hold(n: int, *slots: int) -> Command:
    cmd = Command.noop(1, n)
    for s in slots:
        cmd.set(0, s, CommandKind.HOLD)
    return cmd


def engaged(states: list, slot: int) -> list[int]:
    return [int(s.engaged_target_slot[0, slot]) for s in states]


def first_engage(states: list, slot: int) -> int | None:
    return next((k for k, t in enumerate(engaged(states, slot)) if t >= 0), None)


# ---------------------------------------------------------------- Acquire range and scans


@pytest.mark.parametrize(
    ('unit', 'edge', 'acquires'),
    [(MARINE, 5.4, True), (MARINE, 5.8, False), (ZEALOT, 4.8, True), (ZEALOT, 5.3, False)],
)
def test_acquire_range(unit: int, edge: float, acquires: bool) -> None:
    """Player 2 idle unit vs player 1 HOLD Zergling: acquire range is range+0.5, min 5."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(2, unit, 10.0 + gap(ZERGLING, unit, edge), 16.0),
            )
        ]
    )
    states = run(backend, hold(2, 0), 20)
    assert (first_engage(states, 1) == 2) if acquires else first_engage(states, 1) is None


def test_basic_targeting_ignores_min_range() -> None:
    """`targeting='basic'` is unchanged: Zealot acquires only within range + 0.5."""
    backend = make(dataclasses.replace(P6, targeting='basic'))
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(2, ZEALOT, 10.0 + gap(ZERGLING, ZEALOT, 4.8), 16.0),
            )
        ]
    )
    assert first_engage(run(backend, hold(2, 0), 20), 1) is None


@pytest.mark.parametrize('start', [8.0, 9.0, 10.0])
def test_scan_period(start: float) -> None:
    """Zergling walks toward idle Marine: engages on first scan loop (2 + 8n) after entering."""
    backend = make()
    x1 = 10.0 + gap(ZERGLING, MARINE, start)
    backend.reset([scenario(Spawn(1, ZERGLING, x1, 16.0), Spawn(2, MARINE, 10.0, 16.0))])
    cmd = Command.noop(1, 2).set(
        0, 0, CommandKind.MOVE, target_xy=(10.0 + gap(ZERGLING, MARINE, 1.0), 16.0)
    )
    states = run(backend, cmd, 60)
    k = first_engage(states, 1)
    assert k is not None and (k - 2) % 8 == 0
    # states[i] is post-loop i position, visible to targeting on loop i + 1
    edge = [float(s.x[0, 0] - s.x[0, 1]) - gap(ZERGLING, MARINE, 0.0) for s in states]
    entered = next(i for i, e in enumerate(edge) if e <= 5.5)
    assert entered < k <= entered + 8


def test_stop_scans_immediately() -> None:
    """STOP triggers an immediate scan on its activation loop instead of waiting for loop 2."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(2, MARINE, 10.0 + gap(ZERGLING, MARINE, 5.3), 16.0),
            )
        ]
    )
    cmd = hold(2, 0).set(0, 1, CommandKind.STOP)
    assert first_engage(run(backend, cmd, 10), 1) == 0


def test_armed_target_first() -> None:
    """Armed Zergling is prioritized over a closer unarmed Medivac."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(2, MARINE, 10.0, 16.0),
                Spawn(1, MEDIVAC, 10.0 + gap(MARINE, MEDIVAC, 2.0), 16.0),
                Spawn(1, ZERGLING, 10.0, 16.0 + gap(MARINE, ZERGLING, 4.0)),
            )
        ]
    )
    states = run(backend, hold(3, 2), 10)
    assert engaged(states, 0)[2] == 2


# ---------------------------------------------------------------- Retaliation, kill, HOLD


def test_fight_back_on_hit_frame() -> None:
    """Stalker hits idle Marine from edge 6 (> 5.5 acquire range): Marine engages on hit loop."""
    backend = make()
    x1 = 10.0 + gap(STALKER, MARINE, 6.0)
    backend.reset([scenario(Spawn(1, STALKER, 10.0, 16.0), Spawn(2, MARINE, x1, 16.0))])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    states = run(backend, cmd, 40)
    hp = [float(s.health[0, 1]) for s in states]
    hit = next(k for k in range(1, len(hp)) if hp[k] < hp[k - 1])
    assert first_engage(states, 1) == hit
    assert engaged(states, 1)[hit] == 0
    assert float(states[-1].x[0, 1]) < x1, 'Marine should advance into range'


def test_return_after_kill() -> None:
    """Marine advances to kill HOLD Zergling, then MOVEs back to its leash spot and stops."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(2, MARINE, 10.0 + gap(ZERGLING, MARINE, 5.4), 16.0),
            )
        ]
    )
    x0 = 10.0 + gap(ZERGLING, MARINE, 5.4)
    states = run(backend, hold(2, 0), 300)
    death = next(k for k, s in enumerate(states) if not s.alive[0, 0])
    kinds = [int(s.order_kind[0, 1]) for s in states[death:]]
    assert CommandKind.MOVE in kinds
    assert kinds[-1] == CommandKind.NOOP
    assert abs(float(states[-1].x[0, 1]) - x0) < 1e-4
    assert min(float(s.x[0, 1]) for s in states) < x0 - 0.05, 'Marine should have stepped forward'


def test_hold_does_not_chase() -> None:
    """HOLD unit does not engage an enemy outside weapon range even within acquire range."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(1, ZERGLING, 10.0, 16.0),
                Spawn(2, MARINE, 10.0 + gap(ZERGLING, MARINE, 5.3), 16.0),
            )
        ]
    )
    states = run(backend, hold(2, 0, 1), 40)
    assert first_engage(states, 1) is None
    assert float(states[-1].x[0, 1]) == pytest.approx(10.0 + gap(ZERGLING, MARINE, 5.3))


def test_player1_does_not_attack_move() -> None:
    """With `auto_acquire_owners=(2,)`, player 1 ATTACK_MOVE walks past enemies unengaged."""
    backend = make()
    backend.reset([scenario(Spawn(1, MARINE, 6.0, 16.0), Spawn(2, ZERGLING, 15.0, 18.0))])
    cmd = hold(2, 1).set(0, 0, CommandKind.ATTACK_MOVE, target_xy=(24.0, 16.0))
    states = run(backend, cmd, 160)
    assert first_engage(states, 0) is None
    assert float(states[-1].x[0, 0]) == pytest.approx(24.0)


def test_attack_move_turns_within_acquire_range() -> None:
    """Player 2 ATTACK_MOVE diverts to engage an enemy outside range within range + 0.5."""
    backend = make()
    backend.reset(
        [
            scenario(
                Spawn(2, MARINE, 6.0, 16.0),
                Spawn(1, ZERGLING, 15.0, 16.0 + gap(MARINE, ZERGLING, 5.3)),
            )
        ]
    )
    cmd = hold(2, 1).set(0, 0, CommandKind.ATTACK_MOVE, target_xy=(24.0, 16.0))
    states = run(backend, cmd, 200)
    assert first_engage(states, 0) is not None
    assert not states[-1].alive[0, 1]


@pytest.mark.parametrize('hold_victim', [False, True])
def test_retreat_on_hit(hold_victim: bool) -> None:
    """With `retreat_owners=(1,)`, idle Marine retreats 5 tiles on hit; HOLD stays put."""
    backend = make(dataclasses.replace(P6, retreat_owners=(1,)))
    x1 = 12.0 + gap(MARINE, STALKER, 6.0)
    backend.reset([scenario(Spawn(1, MARINE, 12.0, 16.0), Spawn(2, STALKER, x1, 16.0))])
    cmd = Command.noop(1, 2).set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    if hold_victim:
        cmd.set(0, 0, CommandKind.HOLD)
    states = run(backend, cmd, 30)
    hp = [float(s.health[0, 0]) for s in states]
    hit = next(k for k in range(1, len(hp)) if hp[k] < hp[k - 1])
    s = states[hit]
    if hold_victim:
        assert all(int(t.order_kind[0, 0]) == CommandKind.HOLD for t in states[hit:])
        return
    assert int(s.order_kind[0, 0]) == CommandKind.MOVE
    assert float(s.order_x[0, 0]) == pytest.approx(7.0, abs=1e-4)
    assert float(s.order_y[0, 0]) == pytest.approx(16.0, abs=1e-4)
    assert int(s.engaged_target_slot[0, 0]) == -1, 'retreating unit does not retaliate'


def test_retreat_clamped_to_map() -> None:
    backend = make(dataclasses.replace(P6, retreat_owners=(1,)))
    x0, y0, _, _ = MAPS[MAP_ID].playable
    backend.reset(
        [
            scenario(
                Spawn(1, MARINE, x0 + 1.0, 16.0),
                Spawn(2, STALKER, x0 + 1.0 + gap(MARINE, STALKER, 5.5), 16.0),
            )
        ]
    )
    states = run(backend, Command.noop(1, 2).set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0), 30)
    xs = [float(s.order_x[0, 0]) for s in states if int(s.order_kind[0, 0]) == CommandKind.MOVE]
    assert xs and min(xs) >= x0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match='scan_period'):
        PyrefConfig(scan_period=0)
    with pytest.raises(ValueError, match='kill_rescan_delay'):
        PyrefConfig(kill_rescan_delay=0)
    assert np.isfinite(PyrefConfig().acquire_min_range)
