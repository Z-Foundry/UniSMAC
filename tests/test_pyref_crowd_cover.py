"""Tests for pyref `crowd_cover`: when a MOVE unit u contacts a same-owner ground unit t whose
center is within contact distance of u's goal, and t is stationary with a non-NOOP order or
moving in a non-parallel direction, clear u's order after collision on the next loop.

Scenarios: two Marauders swapping positions vertically, or a Marine walking toward a point
occupied by a HOLD Marine. Disabled by default with bit-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn

from .test_pyref_batch import _sim as _batch_sim

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, MARAUDER = 48, 51
SLACK = 0.0625
BASE = {'collision': 'hard', 'steering': 'sc2_avoid', 'contact_slack': SLACK}


def _r(unit: int) -> float:
    return STATIC.get(unit).radius


def _backend(spawns: tuple[Spawn, ...], **config: object) -> PyrefBackend:
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**{**BASE, **config}))  # type: ignore[arg-type]
    be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])
    return be


def _cmd(be: PyrefBackend, orders: dict[int, tuple[CommandKind, float, float]]) -> Command:
    sim = be._sim  # noqa: SLF001
    assert sim is not None
    n = sim.n_slots
    kind = np.zeros((1, n), np.int8)
    tx = np.full((1, n), np.nan, np.float32)
    ty = np.full((1, n), np.nan, np.float32)
    for u, (k, x, y) in orders.items():
        kind[0, u], tx[0, u], ty[0, u] = int(k), x, y
    return Command(
        kind=kind,
        target_slot=np.full((1, n), NO_SLOT, np.int32),
        target_x=tx,
        target_y=ty,
        ability=np.zeros((1, n), np.int32),
    )


def _run(be: PyrefBackend, orders: dict, loops: int) -> list[np.ndarray]:
    """Issue `orders` on loop 0 then no-op; return `order_kind` after each loop."""
    out = []
    empty = _cmd(be, {})
    for i in range(loops):
        st = be.step(_cmd(be, orders) if i == 0 else empty, 1)
        out.append(st.order_kind[0].copy())
    return out


def _swap(**config: object) -> PyrefBackend:
    """Two Marauders spawned at (16, 16 +/- 0.9666) swapping vertically toward goals shifted
    -1.187 in x and separated by 1.0 in y."""
    return _backend(
        (Spawn(1, MARAUDER, 16.0, 16.9666), Spawn(1, MARAUDER, 16.0, 15.0334)), **config
    )


SWAP = {0: (CommandKind.MOVE, 14.813, 15.5), 1: (CommandKind.MOVE, 14.813, 16.5)}


def test_defaults_and_validation() -> None:
    assert not PyrefConfig().crowd_cover
    with pytest.raises(ValueError, match="crowd_cover.*collision='hard'"):
        PyrefConfig(crowd_cover=True)


def test_swap_clears_on_contact() -> None:
    """Swapping Marauders clear their orders once they contact while covering each other's goal;
    when off, both keep MOVEing until reaching their goals."""
    on = _run(_swap(crowd_cover=True), SWAP, 30)
    off = _run(_swap(), SWAP, 30)
    end_on = [next(i for i, k in enumerate(on) if k[s] == CommandKind.NOOP) for s in (0, 1)]
    assert all(e < 12 for e in end_on)
    assert all(k[s] == CommandKind.MOVE for k in off[:12] for s in (0, 1))


def test_clear_is_one_loop_late() -> None:
    """Triggering loop only sets the pending flag; order clears after collision next loop."""
    be = _swap(crowd_cover=True)
    sim = be._sim  # noqa: SLF001
    assert sim is not None
    seen = None
    empty = _cmd(be, {})
    for i in range(30):
        st = be.step(_cmd(be, SWAP) if i == 0 else empty, 1)
        due = sim.motion.crowd_cover_due[0, :2].copy()
        if seen is None and due.any():
            seen = (i, due)
            assert all(st.order_kind[0, s] == CommandKind.MOVE for s in (0, 1))
        elif seen is not None and i == seen[0] + 1:
            for s in (0, 1):
                if seen[1][s]:
                    assert st.order_kind[0, s] == CommandKind.NOOP
            break
    assert seen is not None


def test_hold_blocker() -> None:
    """Goal occupied by a pinned HOLD ally: clears upon contact when on, pushes forever when off."""
    sp = (Spawn(1, MARINE, 14.0, 16.0), Spawn(1, MARINE, 17.35, 16.0))
    orders = {0: (CommandKind.MOVE, 17.0, 16.0), 1: (CommandKind.HOLD, np.nan, np.nan)}
    on = _run(_backend(sp, crowd_cover=True, pin_engaged=True), orders, 40)
    off = _run(_backend(sp, pin_engaged=True), orders, 40)
    assert any(k[0] == CommandKind.NOOP for k in on)
    assert all(k[0] == CommandKind.MOVE for k in off)


def test_enemy_does_not_cover() -> None:
    """An enemy covering the goal does not clear the order."""
    sp = (Spawn(1, MARINE, 14.0, 16.0), Spawn(2, MARINE, 17.35, 16.0))
    orders = {0: (CommandKind.MOVE, 17.0, 16.0), 1: (CommandKind.HOLD, np.nan, np.nan)}
    on = _run(_backend(sp, crowd_cover=True, pin_engaged=True), orders, 40)
    assert all(k[0] == CommandKind.MOVE for k in on)


def test_parallel_and_idle_do_not_cover() -> None:
    """If t covers u's goal while moving parallel (cos >= 0.9) or idle (NOOP, no waypoint), do
    not flag; flag if t moves sideways (cos < 0.9) or holds (calls `_crowd_cover` directly)."""
    sp = (Spawn(1, MARINE, 14.0, 16.0), Spawn(1, MARINE, 14.6875, 16.0))
    be = _backend(sp, crowd_cover=True)
    sim = be._sim  # noqa: SLF001
    assert sim is not None
    w = sim.world
    ground = w.alive & ~w.is_flying

    def due(t_kind: CommandKind, wp: tuple[float, float] | None) -> bool:
        w.order_kind[0, :2] = (int(CommandKind.MOVE), int(t_kind))
        w.order_x[0, 0], w.order_y[0, 0] = 15.0, 16.0
        sim.buf.waypoint_x[0, :] = np.nan
        sim.buf.waypoint_y[0, :] = np.nan
        sim.buf.waypoint_x[0, 0], sim.buf.waypoint_y[0, 0] = 15.0, 16.0
        if wp is not None:
            sim.buf.waypoint_x[0, 1], sim.buf.waypoint_y[0, 1] = wp
        sim.motion.crowd_cover_due[:] = False
        collision._crowd_cover(sim, ground)  # noqa: SLF001
        return bool(sim.motion.crowd_cover_due[0, 0])

    assert not due(CommandKind.MOVE, (20.0, 16.0))  # parallel
    assert not due(CommandKind.NOOP, None)  # idle
    assert due(CommandKind.MOVE, (14.6875, 20.0))  # perpendicular
    assert due(CommandKind.HOLD, None)


def test_new_command_resets_due() -> None:
    """A new command clears the pending flag so the new MOVE is not cancelled."""
    be = _swap(crowd_cover=True)
    sim = be._sim  # noqa: SLF001
    assert sim is not None
    empty = _cmd(be, {})
    be.step(_cmd(be, SWAP), 1)
    for _ in range(30):
        be.step(empty, 1)
        if sim.motion.crowd_cover_due[0, 0]:
            break
    assert sim.motion.crowd_cover_due[0, 0]
    st = be.step(_cmd(be, {0: (CommandKind.MOVE, 25.0, 16.0)}), 1)
    assert st.order_kind[0, 0] == CommandKind.MOVE
    assert not sim.motion.crowd_cover_due[0, 0]


@pytest.mark.parametrize('seed', range(3))
def test_order_independent(seed: int) -> None:
    """Batched random state is slot-order independent (reversing slots reverses outputs)."""
    sim = _batch_sim('27m_vs_30m', 4, seed, crowd_cover=True, **BASE)
    _scatter_orders(sim, seed)
    a = _cover_once(sim)
    rev = _batch_sim('27m_vs_30m', 4, seed, crowd_cover=True, **BASE)
    _scatter_orders(rev, seed)
    perm = np.arange(rev.n_slots)[::-1]
    w = rev.world
    for name in ('x', 'y', 'radius', 'alive', 'is_flying', 'owner', 'order_kind', 'order_x',
                 'order_y', 'order_target_slot'):  # fmt: skip
        arr = getattr(w, name)
        arr[:] = arr[:, perm]
    rev.motion.crowd_cover_due[:] = rev.motion.crowd_cover_due[:, perm]
    for name in ('waypoint_x', 'waypoint_y'):
        arr = getattr(rev.buf, name)
        arr[:] = arr[:, perm]
    b = _cover_once(rev)
    assert (a == b[:, perm]).all()
    assert a.any()


def _scatter_orders(sim: Sim, seed: int) -> None:
    rng = np.random.default_rng(seed + 100)
    w = sim.world
    shape = w.x.shape
    mv = rng.random(shape) < 0.6
    hold = ~mv & (rng.random(shape) < 0.5)
    w.order_kind[:] = np.where(
        mv, int(CommandKind.MOVE), np.where(hold, int(CommandKind.HOLD), int(CommandKind.NOOP))
    )
    w.order_x[:] = np.where(mv, w.x + rng.uniform(-1, 1, shape), np.nan).astype(np.float32)
    w.order_y[:] = np.where(mv, w.y + rng.uniform(-1, 1, shape), np.nan).astype(np.float32)
    sim.buf.waypoint_x[:] = w.order_x
    sim.buf.waypoint_y[:] = w.order_y
    sim.motion.crowd_cover_due[:] = mv & (rng.random(shape) < 0.2)


def _cover_once(sim: Sim) -> np.ndarray:
    """Run `_crowd_cover` once and return [B, U] codes: 0 unchanged, 1 cleared, 2 flagged."""
    w = sim.world
    before = w.order_kind.copy()
    collision._crowd_cover(sim, w.alive & ~w.is_flying)  # noqa: SLF001
    cleared = (before == CommandKind.MOVE) & (w.order_kind == CommandKind.NOOP)
    return np.where(cleared, 1, np.where(sim.motion.crowd_cover_due, 2, 0))
