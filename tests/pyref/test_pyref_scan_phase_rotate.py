"""Tests for pyref `scan_phase='tag_rotate'`: per-episode rotating slot-ordered scan phases.

Uses the same slot-contiguous pattern as `'tag_order'` (row f is slot u's scan row iff
f == u + c (mod 8)), with offset c rotating by episode seed e:
* e != 0: c = (2e + 4) mod 8;
* e = 0: slot u's row phase is pi(u) + 2 (mod 8), where pi is the random slot permutation from
  `per_unit` (stream `'scan_phase'`, game_loop 0).

Offset c depends solely on the episode seed, independent of sequential vs batched resets.
Autocast heal shares the same scan row as auto-acquire.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info
from unismac.backends.pyref import PyrefBackend, PyrefConfig, rng
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import abilities, targeting
from unismac.contract import NO_SLOT, CommandKind, Scenario, Variant
from unismac.scenarios import spawn_record

from .test_pyref_batch import STATIC, _sim

START = 2
SC2_OPTS = {'targeting': 'sc2', 'auto_acquire': True, 'acquire_start_loop': START}


def _row_phase(sim: Sim) -> np.ndarray:
    """Row index (mod 8) on which each unit's periodic scan result appears."""
    return (sim.combat.scan_shift.astype(np.int64) + 1 + sim.config.acquire_start_loop) % 8


def _first_rows(seed: int, n: int) -> list[int]:
    """Episode 0: rank in random spawn permutation + 2 (same permutation as per_unit)."""
    draw = [rng.uniform(seed, 'scan_phase', 0, u) for u in range(n)]
    order = sorted(range(n), key=lambda u: (draw[u], u))
    out = [0] * n
    for r, u in enumerate(order):
        out[u] = (r + 2) % 8
    return out


@pytest.mark.parametrize('name', ['3m', '8m', '2s3z', '3s5z', 'MMM'])
def test_rows_rotate_with_episode(name: str) -> None:
    """e >= 1: slot u row phase is (u + 2e + 4) mod 8; e = 0: rank + 2 (seeds 0..B-1)."""
    sim = _sim(name, 17, 0, scan_phase='tag_rotate', **SC2_OPTS)
    row = _row_phase(sim)
    n = row.shape[1]
    assert row[0].tolist() == _first_rows(0, n)
    for e in range(1, 17):
        c = (2 * e + 4) % 8
        assert row[e].tolist() == [(u + c) % 8 for u in range(n)], e
    # Rotation: e = 1, 2, 3, 4 yield c = 6, 0, 2, 4 (period 4)
    assert [int(row[e, 0]) for e in range(1, 9)] == [6, 0, 2, 4, 6, 0, 2, 4]


def test_first_episode_is_shifted_per_unit() -> None:
    """Episode 0 row phase equals per_unit scan_shift + 2 (mod 8); phases are distinct
    for n <= 8, and for n > 8 each phase appears floor(n / 8) or ceil(n / 8) times."""
    for name in ('3m', '8m', '2s3z', '3s5z', 'MMM'):
        rot = _sim(name, 1, 0, scan_phase='tag_rotate', **SC2_OPTS)
        per = _sim(name, 1, 0, scan_phase='per_unit', **SC2_OPTS)
        got = _row_phase(rot)[0]
        np.testing.assert_array_equal(got, (per.combat.scan_shift[0] + 2) % 8, err_msg=name)
        counts = np.bincount(got, minlength=8)
        n = got.size
        assert counts.max() - counts.min() <= 1 or n < 8, name
        assert counts.max() == -(-n // 8), name


@pytest.mark.parametrize('start', [0, 2, 5])
def test_row_phase_independent_of_acquire_start(start: int) -> None:
    """acquire_start_loop adjusts scan_shift without changing row phase (even e = 0)."""
    sim = _sim('8m', 8, 0, scan_phase='tag_rotate', targeting='sc2', auto_acquire=True,
               acquire_start_loop=start)  # fmt: skip
    ref = _sim('8m', 8, 0, scan_phase='tag_rotate', **SC2_OPTS)
    np.testing.assert_array_equal(_row_phase(sim), _row_phase(ref))


def test_same_as_tag_order_with_same_c() -> None:
    """Episodes e >= 1 match tag_order bitwise whenever both yield the same offset c."""
    n_ep = 32
    rot = _sim('2s3z', n_ep, 0, scan_phase='tag_rotate', **SC2_OPTS)
    tag = _sim('2s3z', n_ep, 0, scan_phase='tag_order', **SC2_OPTS)
    same = 0
    for e in range(1, n_ep):
        c_tag = 2 * int(4.0 * rng.uniform(e, 'scan_phase', 1, 0))
        if c_tag == (2 * e + 4) % 8:
            np.testing.assert_array_equal(rot.combat.scan_shift[e], tag.combat.scan_shift[e])
            same += 1
        else:
            assert not np.array_equal(rot.combat.scan_shift[e], tag.combat.scan_shift[e])
    assert 0 < same < n_ep - 1


def test_independent_of_how_episodes_are_batched() -> None:
    """Sequential resets (batch_size=1, episode e with seed e) and a single batched reset
    (env b with seed b) yield identical phases: c depends only on seed, not reset count."""
    rec = spawn_record('8m')
    sc = Scenario(rec.map_id, spawns=tuple(sorted(rec.spawns, key=lambda s: s.owner)))
    maps = {rec.map_id: load_map_info(rec.map_id)}
    cfg = PyrefConfig(scan_phase='tag_rotate', **SC2_OPTS)  # type: ignore[arg-type]
    n_ep = 12
    batched = PyrefBackend(STATIC, maps, n_ep, cfg)
    batched.reset([sc] * n_ep, seeds=list(range(n_ep)))
    assert batched._sim is not None
    one = PyrefBackend(STATIC, maps, 1, cfg)
    for e in range(n_ep):
        one.reset([sc], seeds=[e])
        assert one._sim is not None
        np.testing.assert_array_equal(
            one._sim.combat.scan_shift[0], batched._sim.combat.scan_shift[e], err_msg=str(e)
        )
    # Resetting again with the same seed keeps the same phase (no reset-counter rotation)
    one.reset([sc], seeds=[5])
    assert one._sim is not None
    np.testing.assert_array_equal(one._sim.combat.scan_shift[0], batched._sim.combat.scan_shift[5])


def test_options() -> None:
    PyrefConfig(scan_phase='tag_rotate')
    with pytest.raises(ValueError, match='scan_period == 8'):
        PyrefConfig(scan_phase='tag_rotate', scan_period=4)
    with pytest.raises(ValueError, match='tag_rotate'):
        PyrefConfig(scan_phase='rotate')


def test_idle_scan_follows_rotation() -> None:
    """Idle units engage at loop == acquire_start_loop + scan_shift (row loop + 1 == row
    phase) across episode 0 (permuted) and episodes 1-3 (rotating)."""
    sim = _sim('2s3z', 4, 1, scan_phase='tag_rotate', **SC2_OPTS)
    w, c = sim.world, sim.combat
    w.alive[:] = True
    cx, cy = float(np.nanmean(w.x)), float(np.nanmean(w.y))
    g = np.random.default_rng(7)
    w.x[:] = (cx + g.uniform(-1, 1, w.x.shape)).astype(np.float32)
    w.y[:] = (cy + g.uniform(-1, 1, w.y.shape)).astype(np.float32)
    w.order_kind[:] = CommandKind.NOOP
    w.order_target_slot[:] = NO_SLOT
    row = _row_phase(sim)
    for loop in range(16):
        w.game_loop[:] = loop
        w.engaged_target_slot[:] = NO_SLOT
        c.scan_loop[:] = -1
        targeting.sc2(sim)
        got = w.engaged_target_slot != NO_SLOT
        np.testing.assert_array_equal(got, (loop + 1) % 8 == row, err_msg=f'loop {loop}')


@pytest.mark.parametrize('batch', [1, 3])
def test_autocast_scan_same_row_as_acquire(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    """Autocast heal scans on the same loop as acquire: (loop - scan_shift) % 8 == start.
    batch=1 tests episode 0 (permuted); batch=3 tests its last env e = 2 (c = 0)."""
    sim = _sim('MMM', batch, 2, scan_phase='tag_rotate', abilities='heal', heal_autocast=True,
               **SC2_OPTS)  # fmt: skip
    b = batch - 1
    w = sim.world
    w.variant[:] = Variant.STD  # autocast heal applies only to standard Medivac
    med = np.flatnonzero(np.isin(w.unit_type[b], list(abilities.HEAL_CASTERS)) & w.alive[b])
    assert med.size > 0
    calls: list[tuple[int, int, int]] = []

    def record(sim: Sim, bb: int, u: int) -> None:
        calls.append((bb, int(sim.world.game_loop[bb]), u))

    monkeypatch.setattr(abilities, '_autocast', record)
    for loop in range(24):
        w.game_loop[:] = loop
        w.order_kind[:, :] = np.where(
            np.isin(w.unit_type, list(abilities.HEAL_CASTERS)), CommandKind.NOOP, w.order_kind
        )
        abilities.heal(sim)
    row = _row_phase(sim)[b]
    for u in med:
        loops = [lp for bb, lp, v in calls if bb == b and v == u]
        assert loops == [lp for lp in range(24) if (lp + 1) % 8 == row[u]]
