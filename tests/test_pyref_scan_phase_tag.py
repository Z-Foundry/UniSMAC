"""Tests for pyref `scan_phase='tag_order'`: slot-contiguous scan phases with random even offset.

Row f is slot u's scan row iff f == u + c (mod 8), where c is drawn per episode from {0, 2, 4, 6}
using stream `'scan_phase'` (game_loop 1, slot 0). Periodic scans run at
loop == acquire_start_loop + scan_shift (mod 8) with results visible on the next row. Autocast
heal shares the same scan row as auto-acquire; `'global'` and `'per_unit'` are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.backends.pyref import PyrefConfig, rng
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import abilities, targeting
from unismac.contract import NO_SLOT, CommandKind, Variant

from .test_pyref_batch import _sim

START = 2
SC2_OPTS = {'targeting': 'sc2', 'auto_acquire': True, 'acquire_start_loop': START}


def _row_phase(sim: Sim) -> np.ndarray:
    """Row index (mod 8) on which each unit's periodic scan result appears."""
    return (sim.combat.scan_shift.astype(np.int64) + 1 + sim.config.acquire_start_loop) % 8


@pytest.mark.parametrize('name', ['3m', '8m', '2s3z', '3s5z'])
def test_rows_contiguous_in_slot_order(name: str) -> None:
    """Within an episode, slot u has row phase u + c with even c from scan_phase stream."""
    sim = _sim(name, 16, 0, scan_phase='tag_order', **SC2_OPTS)
    row = _row_phase(sim)
    n = row.shape[1]
    cs = []
    for b in range(16):
        c = 2 * int(4.0 * rng.uniform(b, 'scan_phase', 1, 0))  # _sim uses seeds 0..B-1
        assert row[b].tolist() == [(u + c) % 8 for u in range(n)]
        cs.append(c)
    assert set(cs) <= {0, 2, 4, 6}
    assert len(set(cs)) > 1


@pytest.mark.parametrize('start', [0, 2, 5])
def test_row_phase_independent_of_acquire_start(start: int) -> None:
    """acquire_start_loop adjusts scan_shift without changing row phase."""
    sim = _sim('8m', 8, 0, scan_phase='tag_order', targeting='sc2', auto_acquire=True,
               acquire_start_loop=start)  # fmt: skip
    ref = _sim('8m', 8, 0, scan_phase='tag_order', **SC2_OPTS)
    np.testing.assert_array_equal(_row_phase(sim), _row_phase(ref))


def test_first_enemy_parity_follows_agent_count() -> None:
    """Player 2's first unit has row phase k = n1 + c, matching ally count parity (odd for
    3m, even for 8m)."""
    for name, n1 in (('3m', 3), ('8m', 8), ('2s3z', 5)):
        sim = _sim(name, 16, 0, scan_phase='tag_order', **SC2_OPTS)
        k = _row_phase(sim)[:, n1]
        assert (k % 2 == n1 % 2).all(), name


def test_options() -> None:
    PyrefConfig(scan_phase='tag_order')
    with pytest.raises(ValueError, match='scan_period == 8'):
        PyrefConfig(scan_phase='tag_order', scan_period=4)
    with pytest.raises(ValueError, match='tag_order'):
        PyrefConfig(scan_phase='tag')


def test_per_unit_and_global_unchanged() -> None:
    """'per_unit' remains a random permutation keyed at game_loop 0; 'global' is zeros."""
    sim = _sim('3s5z', 8, 0, scan_phase='per_unit')
    n = sim.combat.scan_shift.shape[1]
    for b in range(8):
        draw = [rng.uniform(b, 'scan_phase', 0, u) for u in range(n)]
        order = sorted(range(n), key=lambda u: (draw[u], u))
        want = [0] * n
        for r, u in enumerate(order):
            want[u] = r % 8
        assert sim.combat.scan_shift[b].tolist() == want
    assert not _sim('3s5z', 4, 0, **SC2_OPTS).combat.scan_shift.any()


def test_idle_scan_follows_tag_order() -> None:
    """Idle units engage at loop == acquire_start_loop + scan_shift (row == slot + c)."""
    sim = _sim('2s3z', 4, 1, scan_phase='tag_order', **SC2_OPTS)
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


def test_autocast_scan_same_row_as_acquire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autocast heal scans on the same loop as acquire: (loop - scan_shift) % 8 == start."""
    sim = _sim('MMM', 1, 2, scan_phase='tag_order', abilities='heal', heal_autocast=True,
               **SC2_OPTS)  # fmt: skip
    w = sim.world
    w.variant[:] = Variant.STD  # autocast heal applies only to standard Medivac
    med = np.flatnonzero(np.isin(w.unit_type[0], list(abilities.HEAL_CASTERS)) & w.alive[0])
    assert med.size > 0
    calls: list[tuple[int, int]] = []

    def record(sim: Sim, b: int, u: int) -> None:
        calls.append((int(sim.world.game_loop[b]), u))

    monkeypatch.setattr(abilities, '_autocast', record)
    for loop in range(24):
        w.game_loop[:] = loop
        w.order_kind[0, med] = CommandKind.NOOP
        abilities.heal(sim)
    row = _row_phase(sim)[0]
    for u in med:
        loops = [lp for lp, v in calls if v == u]
        assert loops == [lp for lp in range(24) if (lp + 1) % 8 == row[u]]
