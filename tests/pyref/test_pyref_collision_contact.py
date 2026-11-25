"""Two-unit collision tests for `pyref`: contact distance, `sc2_avoid`, idle pushing, and return.

Self-contained fixtures on `32x32_flat` (all units owned by Player 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

MARINE, MARAUDER, ZERGLING, STALKER, ZEALOT = 48, 51, 105, 74, 73
STATIC = load_static_data()
FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
SLACK = 0.0625
SC2_P2: dict[str, object] = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': SLACK,
    'idle_push_share': 0.9,
    'idle_return': True,
}


def make(**config: object) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))  # type: ignore[arg-type]


def radius(unit: int) -> float:
    return STATIC.get(unit).radius


def play(
    backend: PyrefBackend, spawns: tuple[Spawn, ...], moves: dict[int, tuple[float, float]], n: int
) -> list[WorldState]:
    """Reset, issue MOVE on loop 0 for `moves`, step 1 loop `n` times; return states from reset."""
    state = backend.reset([Scenario(map_id=FLAT, spawns=spawns)])
    cmd = Command.noop(1, len(spawns))
    for u, xy in moves.items():
        cmd.set(0, u, CommandKind.MOVE, target_xy=xy)
    states = [state, backend.step(cmd, 1)]
    states += [backend.step(Command.noop(1, len(spawns)), 1) for _ in range(n - 1)]
    return states


def xy(states: list[WorldState], u: int) -> np.ndarray:
    return np.array([[s.x[0, u], s.y[0, u]] for s in states], np.float64)


def min_dist(states: list[WorldState], u: int = 0, t: int = 1) -> float:
    d = xy(states, u) - xy(states, t)
    return float(np.hypot(d[:, 0], d[:, 1]).min())


def lateral(
    states: list[WorldState], u: int, start: tuple[float, float], goal: tuple[float, float]
):
    """Lateral offset relative to the ideal line per frame; positive is left of travel."""
    p = xy(states, u) - np.asarray(start)
    d = np.asarray(goal) - np.asarray(start)
    d = d / np.hypot(*d)
    return d[0] * p[:, 1] - d[1] * p[:, 0]


def extreme(lat: np.ndarray) -> float:
    return float(lat[np.abs(lat).argmax()])


# ---------------------------------------------------------------- Defaults and validation


def test_sc2_avoid_without_neighbours_equals_none() -> None:
    spawns = (Spawn(1, MARINE, 8.0, 16.0),)
    a = play(make(collision='hard'), spawns, {0: (24.0, 20.0)}, 130)
    b = play(make(collision='hard', steering='sc2_avoid'), spawns, {0: (24.0, 20.0)}, 130)
    assert np.array_equal(xy(a, 0), xy(b, 0))
    assert (a[-1].x[0, 0], a[-1].y[0, 0]) == (24.0, 20.0)


def test_idle_push_share_validated() -> None:
    with pytest.raises(ValueError, match='idle_push_share'):
        PyrefConfig(idle_push_share=1.5)


# ---------------------------------------------------------------- Contact distance


def test_contact_slack_sets_min_center_distance() -> None:
    """Two overlapping idle Marines are pushed apart to center distance = 2*radius - slack."""
    spawns = (Spawn(1, MARINE, 16.0, 16.0), Spawn(1, MARINE, 16.3, 16.0))
    s = play(make(collision='hard', contact_slack=SLACK, collision_iterations=8), spawns, {}, 4)[-1]
    d = float(np.hypot(s.x[0, 0] - s.x[0, 1], s.y[0, 0] - s.y[0, 1]))
    assert d == pytest.approx(2 * radius(MARINE) - SLACK, abs=1e-4)


# ---------------------------------------------------------------- Head-on and crossing


@pytest.mark.parametrize(
    ('dy', 'side'),
    [(0.0, -1), (0.25, -1), (-0.25, 1), (0.5, -1)],
)
def test_head_on_detours_away_and_arrives(dy: float, side: int) -> None:
    """Head-on pair: each yields to its away side (right for dy > 0, left for dy < 0, right when
    collinear), avoids deadlock and deep overlap, returns to the line, and arrives cleanly."""
    a0, a1 = (8.0, 16.0), (24.0, 16.0)
    b0, b1 = (24.0, 16.0 + dy), (8.0, 16.0 + dy)
    spawns = (Spawn(1, MARINE, *a0), Spawn(1, MARINE, *b0))
    states = play(make(**SC2_P2), spawns, {0: a1, 1: b1}, 200)
    assert np.sign(extreme(lateral(states, 0, a0, a1))) == side
    assert np.sign(extreme(lateral(states, 1, b0, b1))) == side
    assert min_dist(states) >= 2 * radius(MARINE) - SLACK - 1e-4
    end = states[-1]
    assert (end.x[0, 0], end.y[0, 0]) == a1 and (end.x[0, 1], end.y[0, 1]) == b1
    assert np.all(end.order_kind[0] == CommandKind.NOOP)


@pytest.mark.parametrize('angle', [45, 90, 135])
def test_crossing_both_pass_right(angle: int) -> None:
    """Crossing paths meeting at (16, 16): both detour right and arrive at their destinations."""
    th = np.deg2rad(angle)
    a0, a1 = (8.0, 16.0), (24.0, 16.0)
    b0 = (16.0 - 8.0 * np.cos(th), 16.0 - 8.0 * np.sin(th))
    b1 = (16.0 + 8.0 * np.cos(th), 16.0 + 8.0 * np.sin(th))
    spawns = (Spawn(1, MARINE, *a0), Spawn(1, MARINE, float(b0[0]), float(b0[1])))
    states = play(make(**SC2_P2), spawns, {0: a1, 1: (float(b1[0]), float(b1[1]))}, 200)
    assert extreme(lateral(states, 0, a0, a1)) < 0
    assert extreme(lateral(states, 1, b0, b1)) < 0
    end = states[-1]
    assert np.hypot(end.x[0, 0] - a1[0], end.y[0, 0] - a1[1]) < 1e-5
    assert np.hypot(end.x[0, 1] - b1[0], end.y[0, 1] - b1[1]) < 1e-5


# ---------------------------------------------------------------- Rear-end following


def test_rear_end_follows_without_overtaking() -> None:
    """Collinear pursuit: faster Zergling matches Marine's speed at contact without overtaking."""
    spawns = (Spawn(1, MARINE, 5.0, 16.0), Spawn(1, ZERGLING, 3.0, 16.0))
    states = play(make(**SC2_P2), spawns, {0: (24.0, 16.0), 1: (28.0, 16.0)}, 260)
    contact = radius(MARINE) + radius(ZERGLING) - SLACK
    # Leader is pushed aside on its arrival loop, so locate arrival by cleared order
    arrived = next(
        i for i, s in enumerate(states) if i > 0 and s.order_kind[0, 0] == CommandKind.NOOP
    )
    for s in states[:arrived]:
        assert s.x[0, 1] < s.x[0, 0]
        assert abs(s.y[0, 1] - 16.0) < 1e-3, 'no lateral detour while following'
    gap = states[arrived - 1].x[0, 0] - states[arrived - 1].x[0, 1]
    assert gap == pytest.approx(contact, abs=1e-3), 'follows at contact distance'
    end = states[-1]
    assert (end.x[0, 1], end.y[0, 1]) == (28.0, 16.0)
    assert np.hypot(end.x[0, 0] - 24.0, end.y[0, 0] - 16.0) > 0.1, 'stopped leader pushed away'


def test_offset_overtake_passes_on_far_side() -> None:
    """Offset pursuit (dy=0.25, faster unit on the left): overtakes to the left (far side)."""
    spawns = (Spawn(1, MARINE, 5.0, 16.0), Spawn(1, ZERGLING, 3.0, 16.25))
    states = play(make(**SC2_P2), spawns, {0: (24.0, 16.0), 1: (28.0, 16.25)}, 260)
    assert extreme(lateral(states, 1, (3.0, 16.25), (28.0, 16.25))) > 0
    passed = next(i for i, s in enumerate(states) if s.x[0, 1] > s.x[0, 0])
    assert states[passed].x[0, 0] < 24.0, 'overtakes before slow unit arrives'
    assert (states[-1].x[0, 1], states[-1].y[0, 1]) == (28.0, 16.25)


# ---------------------------------------------------------------- Passing an idle ally


def _pass_idle(mover: int, idle: int, dy: float, **config: object) -> list[WorldState]:
    spawns = (Spawn(1, mover, 8.0, 16.0), Spawn(1, idle, 16.0, 16.0 + dy))
    return play(make(**(SC2_P2 | config)), spawns, {0: (24.0, 16.0)}, 300)


def _away(states: list[WorldState], i: int = -1) -> float:
    s = states[i]
    return float(np.hypot(s.x[0, 1] - 16.0, s.y[0, 1] - (states[0].y[0, 1])))


def test_head_on_idle_pushed_then_returns() -> None:
    """Head-on pass through idle Marine: pushed beyond `return_far`, idle unit gets a return
    MOVE and restores its original spot while mover arrives; stays put if `idle_return=False`."""
    states = _pass_idle(MARINE, MARINE, 0.0)
    kinds = [int(s.order_kind[0, 1]) for s in states]
    assert CommandKind.MOVE in kinds
    i = kinds.index(CommandKind.MOVE)
    assert _away(states, i) > PyrefConfig().return_far
    assert states[i].order_x[0, 1] == 16.0 and states[i].order_y[0, 1] == 16.0
    assert _away(states) < 1e-5 and states[-1].order_kind[0, 1] == CommandKind.NOOP
    assert (states[-1].x[0, 0], states[-1].y[0, 0]) == (24.0, 16.0)
    no_return = _pass_idle(MARINE, MARINE, 0.0, idle_return=False)
    assert _away(no_return) > 0.5
    assert all(s.order_kind[0, 1] == CommandKind.NOOP for s in no_return)


@pytest.mark.parametrize(
    ('mover', 'idle', 'returns'),
    [
        (MARAUDER, MARINE, True),
        (STALKER, ZEALOT, True),
        (MARINE, MARINE, False),
        (MARINE, MARAUDER, False),
    ],
)
def test_side_push_return_depends_on_radius(mover: int, idle: int, returns: bool) -> None:
    """Offset pass (dy=0.3) past an idle ally: side-pushed idle unit returns after `return_delay`
    loops when pushed by a larger-radius mover, and stays displaced otherwise."""
    states = _pass_idle(mover, idle, 0.3)
    assert max(_away(states, i) for i in range(len(states))) > PyrefConfig().return_threshold
    if returns:
        assert _away(states) < 1e-5
    else:
        assert _away(states) > PyrefConfig().return_threshold
    assert (states[-1].x[0, 0], states[-1].y[0, 0]) == (24.0, 16.0)


def test_commanded_unit_does_not_return() -> None:
    """A unit that received an explicit command (STOP) has no home spot and does not return."""
    backend = make(**SC2_P2)
    spawns = (Spawn(1, MARINE, 8.0, 16.0), Spawn(1, MARINE, 16.0, 16.0))
    backend.reset([Scenario(map_id=FLAT, spawns=spawns)])
    cmd = Command.noop(1, 2)
    cmd.set(0, 0, CommandKind.MOVE, target_xy=(24.0, 16.0))
    cmd.set(0, 1, CommandKind.STOP)
    states = [backend.step(cmd, 1)] + [backend.step(Command.noop(1, 2), 1) for _ in range(300)]
    assert all(s.order_kind[0, 1] == CommandKind.NOOP for s in states)
    assert np.hypot(states[-1].x[0, 1] - 16.0, states[-1].y[0, 1] - 16.0) > 0.5
