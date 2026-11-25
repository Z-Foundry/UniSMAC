"""Tests for three optional pyref collision consistency fixes:

* `push_clearance`: post-push terrain projection enforces pathing clearance rho, not just
  center-cell passability;
* `arrival_clears_waypoint`: arriving units clear their waypoint on the same loop so collision
  treats them as idle;
* `pin_pair_share='younger'`: when two pinned units overlap, the more recently pinned unit
  takes the full separation push.

All three default to off (bit-identical baseline); batch sweep matches exact pairwise sweep.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig, navmesh
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import collision
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState
from unismac.profiles import _V4, _V5, _V6
from unismac.scenarios import spawn_record

FLAT = 'SMAC_Maps/32x32_flat'  # passable region x, y in [2, 30)
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, STALKER = 48, 74
RM = STATIC.get(MARINE).radius
SLACK = 0.0625
CONTACT = 2 * RM - SLACK
BASE = PyrefConfig(
    pathing='navmesh', collision='hard', contact_slack=SLACK, idle_push_share=0.9, pin_engaged=True
)


def _backend(cfg: PyrefConfig, *spawns: Spawn) -> PyrefBackend:
    be = PyrefBackend(STATIC, MAPS, 1, cfg)
    be.reset([Scenario(FLAT, spawns=spawns)], seeds=[0])
    return be


def _sim(be: PyrefBackend) -> Sim:
    sim = be._sim  # noqa: SLF001 - test mutates internal state directly
    assert sim is not None
    return sim


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.push_clearance is False
    assert cfg.arrival_clears_waypoint is False
    assert cfg.pin_pair_share == 'half'
    for opts in (_V4, _V5, _V6):
        assert not {'push_clearance', 'arrival_clears_waypoint', 'pin_pair_share'} & set(opts)
    with pytest.raises(ValueError, match='push_clearance'):
        PyrefConfig(push_clearance=True, collision='hard')  # pathing != 'navmesh'
    with pytest.raises(ValueError, match='push_clearance'):
        PyrefConfig(push_clearance=True, pathing='navmesh')  # collision != 'hard'
    with pytest.raises(ValueError, match='arrival_clears_waypoint'):
        PyrefConfig(arrival_clears_waypoint=True)
    with pytest.raises(ValueError, match='pin_pair_share'):
        PyrefConfig(pin_pair_share='older', pin_engaged=True)
    with pytest.raises(ValueError, match='pin_engaged'):
        PyrefConfig(pin_pair_share='younger')


# ---------------------------------------------------------------- push_clearance


def _wall_push(cfg: PyrefConfig) -> tuple[float, float]:
    """Idle Marine A sits 0.01 above the lower wall clearance (y = 2); HOLD Marine B directly
    above overlaps it by 0.3. B is pinned so A is pushed 0.3 toward the wall (still in a
    passable cell). Returns A's post-collision y and clearance to navmesh obstacle edges."""
    ya = 2.0 + 0.3125 + 0.01  # 0.01 outside minimum pathing wall clearance
    be = _backend(cfg, Spawn(1, MARINE, 16.0, ya), Spawn(1, MARINE, 16.0, ya + CONTACT - 0.3))
    try:
        sim = _sim(be)
        sim.world.order_kind[0, 1] = CommandKind.HOLD
        sim.buf.waypoint_x[:] = np.nan
        collision.hard(sim)
        mesh = navmesh.build(MAPS[FLAT].pathing, navmesh.path_radius(RM))
        y = float(sim.world.y[0, 0])
        return y, navmesh.point_clearance(float(sim.world.x[0, 0]), y, mesh.edges)
    finally:
        be.close()


def test_push_clearance_keeps_path_radius_from_wall() -> None:
    """Default center-cell check lets A be pushed within rho - 1/8 (0.3125) of the wall;
    enabled keeps clearance at rho (0.3125 from the grid wall)."""
    rho = navmesh.path_radius(RM)
    y_off, c_off = _wall_push(BASE)
    assert 2.0 < y_off < 2.0 + 0.3125 - 0.1
    assert c_off < rho - 0.1
    y_on, c_on = _wall_push(dataclasses.replace(BASE, push_clearance=True))
    assert c_on == pytest.approx(rho, abs=2e-4)
    assert c_on >= rho
    assert y_on == pytest.approx(2.0 + 0.3125, abs=2e-4)


def test_push_clearance_corner() -> None:
    """A unit pushed into a concave bottom-left corner keeps clearance >= rho from both edges."""
    rho = navmesh.path_radius(RM)
    be = _backend(dataclasses.replace(BASE, push_clearance=True), Spawn(1, MARINE, 2.1, 2.1))
    try:
        sim = _sim(be)
        sim.buf.waypoint_x[:] = np.nan
        collision.hard(sim)
        x, y = float(sim.world.x[0, 0]), float(sim.world.y[0, 0])
        mesh = navmesh.build(MAPS[FLAT].pathing, rho)
        assert navmesh.point_clearance(x, y, mesh.edges) >= rho
        assert x == pytest.approx(2.3125, abs=2e-4) and y == pytest.approx(2.3125, abs=2e-4)
    finally:
        be.close()


def test_push_clearance_leaves_far_units() -> None:
    """Units at clearance >= rho from obstacles are bit-identical on/off."""
    spawns = (Spawn(1, MARINE, 10.0, 10.0), Spawn(1, MARINE, 10.0 + CONTACT - 0.2, 10.0))
    out = []
    for cfg in (BASE, dataclasses.replace(BASE, push_clearance=True)):
        be = _backend(cfg, *spawns)
        try:
            sim = _sim(be)
            sim.buf.waypoint_x[:] = np.nan
            collision.hard(sim)
            out.append((sim.world.x.copy(), sim.world.y.copy()))
        finally:
            be.close()
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])


# ---------------------------------------------------------------- arrival_clears_waypoint


def _arrive_into(cfg: PyrefConfig, loops: int = 30) -> list[WorldState]:
    """Marine A at (10, 16) MOVEs to (12, 16); idle Marine B stands at (12 + CONTACT - 0.1, 16),
    so A overlaps B by 0.1 on the arrival loop."""
    be = _backend(
        dataclasses.replace(cfg, kinematics='accel'),
        Spawn(1, MARINE, 10.0, 16.0),
        Spawn(1, MARINE, 12.0 + CONTACT - 0.1, 16.0),
    )
    try:
        rows = [be.step(Command.noop(1, 2), 1)]
        cmd = Command.noop(1, 2)
        cmd.set(0, 0, CommandKind.MOVE, target_xy=(12.0, 16.0))
        rows.append(be.step(cmd, 1))
        for _ in range(loops):
            rows.append(be.step(Command.noop(1, 2), 1))
    finally:
        be.close()
    return rows


def test_arrival_clears_waypoint_idle_split() -> None:
    """On arrival loop: by default A still counts as moving and idle B takes 0.9; when enabled,
    both are idle and split 50/50. Earlier loops are bit-identical."""
    off, on = (
        _arrive_into(BASE),
        _arrive_into(dataclasses.replace(BASE, arrival_clears_waypoint=True)),
    )
    k = next(i for i, s in enumerate(off) if int(s.order_kind[0, 0]) == CommandKind.NOOP and i > 1)
    for f in WorldState.UNIT_FIELDS:
        for i in range(k):
            assert np.array_equal(getattr(off[i], f), getattr(on[i], f), equal_nan=True), (f, i)
    assert int(on[k].order_kind[0, 0]) == CommandKind.NOOP
    da_off = 12.0 - float(off[k].x[0, 0])
    db_off = float(off[k].x[0, 1] - off[k - 1].x[0, 1])
    da_on = 12.0 - float(on[k].x[0, 0])
    db_on = float(on[k].x[0, 1] - on[k - 1].x[0, 1])
    assert db_off == pytest.approx(9 * da_off, rel=1e-3)
    assert da_on == pytest.approx(db_on, rel=1e-3)
    assert da_on > da_off


# ---------------------------------------------------------------- pin_pair_share


def _pinned_pair(cfg: PyrefConfig) -> tuple[np.ndarray, np.ndarray]:
    """Two HOLD Marines overlapping by 0.2: A is pinned from loop 0, B from loop 3 (NOOP
    before). Returns x before and after collision on loop 3."""
    be = _backend(cfg, Spawn(1, MARINE, 16.0, 16.0), Spawn(1, MARINE, 16.0 + CONTACT - 0.2, 16.0))
    try:
        sim = _sim(be)
        w = sim.world
        w.order_kind[0, 0] = CommandKind.HOLD
        x_b = float(w.x[0, 1])
        for loop in range(4):
            w.game_loop[0] = loop
            if loop == 3:
                w.order_kind[0, 1] = CommandKind.HOLD
            sim.buf.waypoint_x[:] = np.nan
            before = w.x.copy()
            collision.hard(sim)
            if loop < 3:
                w.x[0, 1] = x_b  # restore B before loop 3 so we isolate the loop-3 push
                w.x[0, 0] = before[0, 0]
        return before[0].copy(), w.x[0].copy()
    finally:
        be.close()


def test_pin_pair_share_younger_moves() -> None:
    """Default splits 50/50 between two pinned units; 'younger' pushes only later-pinned B."""
    b0, a0 = _pinned_pair(BASE)
    assert float(b0[0] - a0[0]) == pytest.approx(0.1, abs=1e-4)
    assert float(a0[1] - b0[1]) == pytest.approx(0.1, abs=1e-4)
    b1, a1 = _pinned_pair(dataclasses.replace(BASE, pin_pair_share='younger'))
    assert a1[0] == b1[0]
    assert float(a1[1] - b1[1]) == pytest.approx(0.2, abs=1e-4)


def test_pin_since_tracks_segments() -> None:
    """`motion.pin_since` records the loop pinning started, resets to -1 when unpinned, and
    restarts when pinned again."""
    be = _backend(dataclasses.replace(BASE, pin_pair_share='younger'), Spawn(1, MARINE, 16.0, 16.0))
    try:
        sim = _sim(be)
        w, m = sim.world, sim.motion
        sim.buf.waypoint_x[:] = np.nan
        seq = []
        for loop, hold in enumerate((False, True, True, False, True)):
            w.game_loop[0] = loop
            w.order_kind[0, 0] = CommandKind.HOLD if hold else CommandKind.NOOP
            w.engaged_target_slot[0, 0] = NO_SLOT
            collision.hard(sim)
            seq.append(int(m.pin_since[0, 0]))
        assert seq == [-1, 1, 1, -1, 4]
    finally:
        be.close()


# ---------------------------------------------------------------- Batch vs exact pairwise


def _crowd(seed: int, **config: object) -> Sim:
    """Pack 27m_vs_30m spawns tightly with random HOLD, engagement, and waypoints."""
    rec = spawn_record('27m_vs_30m')
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    be = PyrefBackend(STATIC, {rec.map_id: load_map_info(rec.map_id)}, 3, PyrefConfig(**config))  # type: ignore[arg-type]
    be.reset([sc] * 3, seeds=[0, 1, 2])
    sim = be._sim  # noqa: SLF001
    assert sim is not None
    rng = np.random.default_rng(seed)
    w, buf = sim.world, sim.buf
    shape = w.x.shape
    cx, cy = np.nanmean(w.x), np.nanmean(w.y)
    span = 0.6 * np.sqrt(shape[1])
    w.x[:] = (cx + rng.uniform(-span, span, shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-span, span, shape)).astype(np.float32)
    w.alive[rng.random(shape) < 0.1] = False
    moving = rng.random(shape) < 0.4
    ang = rng.uniform(-np.pi, np.pi, shape)
    buf.waypoint_x[:] = np.where(moving, w.x + 3 * np.cos(ang), np.nan).astype(np.float32)
    buf.waypoint_y[:] = np.where(moving, w.y + 3 * np.sin(ang), np.nan).astype(np.float32)
    r = rng.random(shape)
    w.order_kind[r < 0.3] = CommandKind.HOLD
    w.engaged_target_slot[:] = np.where((r >= 0.3) & (r < 0.6), 0, NO_SLOT)
    sim.motion.pin_since[:] = np.where(rng.random(shape) < 0.5, rng.integers(0, 5, shape), -1)
    return sim


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('small', [48, 0])
def test_younger_sweep_matches_exact(seed: int, small: int, monkeypatch) -> None:
    monkeypatch.setattr(collision, 'SWEEP_SMALL', small)
    sim = _crowd(
        seed, collision='hard', contact_slack=SLACK, idle_push_share=0.9, pin_engaged=True,
        pin_pair_share='younger',
    )  # fmt: skip
    k = collision._Consts(sim.config)  # noqa: SLF001
    k.pin = collision.pinned_mask(sim)
    k.since = sim.motion.pin_since
    both = k.pin[:, :, None] & k.pin[:, None, :] & (k.since[:, :, None] != k.since[:, None, :])
    assert both.sum() > 20
    ground = sim.world.alive & ~sim.world.is_flying
    moving = ~np.isnan(sim.buf.waypoint_x)
    ref = copy.deepcopy(sim)
    collision._sweep(sim, ground, moving, k)  # noqa: SLF001
    for b in range(ref.batch_size):
        collision._sweep_exact(ref, b, np.flatnonzero(ground[b]).tolist(), moving[b], k)  # noqa: SLF001
    np.testing.assert_array_equal(sim.world.x, ref.world.x)
    np.testing.assert_array_equal(sim.world.y, ref.world.y)
