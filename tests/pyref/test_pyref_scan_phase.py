"""Tests for pyref `scan_phase` (per-unit periodic scan offsets) and `retarget` / `retarget_ready`.

With `scan_phase='per_unit'`, each reset draws a random permutation over slots and uses
`rank % scan_period` as each unit's phase offset for periodic scans (auto-acquire and autocast
heal). Default `'global'` keeps all units in lockstep with bitwise-identical behavior.
`retarget_ready` allows switching to a closer out-of-range enemy in acquire range when cooldown is
zero.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.backends.pyref import PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import abilities, targeting
from unismac.contract import NO_SLOT, CommandKind, Variant

from .test_pyref_batch import _sim

SC2_OPTS = {'targeting': 'sc2', 'auto_acquire': True, 'acquire_start_loop': 2}


def test_shift_is_permutation_and_varies_with_seed() -> None:
    sim = _sim('3s5z', 16, 0, scan_phase='per_unit')
    shift = sim.combat.scan_shift
    n = shift.shape[1]
    for b in range(16):
        assert sorted(shift[b].tolist()) == sorted(r % 8 for r in range(n))
    assert len({tuple(r) for r in shift.tolist()}) > 8  # most seeds 0..15 differ
    assert not _sim('3s5z', 4, 0).combat.scan_shift.any()


def test_default_is_global() -> None:
    assert PyrefConfig().scan_phase == 'global'
    with pytest.raises(ValueError, match='scan_phase'):
        PyrefConfig(scan_phase='tag')


def _idle_in_range(name: str, phase: str):
    """All units idle and clustered within acquire range, with engaged targets cleared."""
    sim = _sim(name, 4, 1, scan_phase=phase, **SC2_OPTS)
    w = sim.world
    w.alive[:] = True
    cx, cy = float(np.nanmean(w.x)), float(np.nanmean(w.y))
    rng = np.random.default_rng(7)
    w.x[:] = (cx + rng.uniform(-1, 1, w.x.shape)).astype(np.float32)
    w.y[:] = (cy + rng.uniform(-1, 1, w.y.shape)).astype(np.float32)
    w.order_kind[:] = CommandKind.NOOP
    w.order_target_slot[:] = NO_SLOT
    return sim


@pytest.mark.parametrize('phase', ['global', 'per_unit'])
def test_idle_scan_follows_unit_phase(phase: str) -> None:
    """Idle units engage only when (loop - acquire_start_loop - scan_shift) % 8 == 0."""
    sim = _idle_in_range('2s3z', phase)
    w, c = sim.world, sim.combat
    for loop in range(16):
        w.game_loop[:] = loop
        w.engaged_target_slot[:] = NO_SLOT
        c.scan_loop[:] = -1
        targeting.sc2(sim)
        due = (loop - 2 - c.scan_shift) % 8 == 0
        if phase == 'global':
            due &= loop >= 2
        got = w.engaged_target_slot != NO_SLOT
        np.testing.assert_array_equal(got, due, err_msg=f'loop {loop}')


def test_autocast_scan_follows_unit_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autocast heal shares scan_shift: scans at (loop - scan_shift) % 8 == phase."""
    sim = _sim('MMM', 1, 2, scan_phase='per_unit', abilities='heal', heal_autocast=True)
    w = sim.world
    w.variant[:] = Variant.STD  # autocast heal applies only to standard Medivac
    med = np.flatnonzero(np.isin(w.unit_type[0], list(abilities.HEAL_CASTERS)) & w.alive[0])
    assert med.size > 0
    seen: dict[int, list[int]] = {int(u): [] for u in med}
    calls: list[tuple[int, int]] = []

    def record(sim: Sim, b: int, u: int) -> None:
        calls.append((int(sim.world.game_loop[b]), u))

    monkeypatch.setattr(abilities, '_autocast', record)
    for loop in range(24):
        w.game_loop[:] = loop
        w.order_kind[0, med] = CommandKind.NOOP
        abilities.heal(sim)
    for loop, u in calls:
        seen[u].append(loop)
    for u, loops in seen.items():
        s = int(sim.combat.scan_shift[0, u])
        assert loops == [lp for lp in range(24) if (lp - s) % 8 == abilities.AUTOCAST_PHASE]


def _chase(retarget: bool, lock: str = 'new'):
    """2s3z: enemy unit on ATTACK_MOVE engaged with out-of-range far ally; near is closer."""
    sim = _idle_in_range('2s3z', 'global')
    sim.config = dataclasses.replace(sim.config, retarget=retarget, retarget_lock=lock)
    w = sim.world
    own = w.owner[0]
    ally, enemy = np.flatnonzero(own == 1), np.flatnonzero(own == 2)
    s, far, near = int(enemy[0]), int(ally[0]), int(ally[1])
    w.x[:], w.y[:] = np.float32(0), np.float32(0)
    w.x[:, ally] = np.float32(40)  # keep remaining allies far away
    w.x[:, s], w.y[:, s] = np.float32(10), np.float32(10)
    w.x[:, near], w.y[:, near] = np.float32(13), np.float32(10)
    # Center dist 9: inside sight (Zealot 9, Stalker 10) but outside weapon range (<= 6)
    w.x[:, far], w.y[:, far] = np.float32(18.9), np.float32(10)
    others = [u for u in enemy if u != s]
    w.x[:, others] = np.float32(5)
    w.order_kind[:, s] = CommandKind.ATTACK_MOVE
    w.order_x[:, s], w.order_y[:, s] = np.float32(20), np.float32(10)
    w.engaged_target_slot[:, s] = far
    return sim, s, far, near


@pytest.mark.parametrize('retarget', [False, True])
def test_retarget_on_scan_when_target_out_of_range(retarget: bool) -> None:
    sim, s, far, near = _chase(retarget)
    w = sim.world
    w.game_loop[:] = 3  # non-scan loop (acquire_start_loop=2, period=8)
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()
    w.game_loop[:] = 10  # scan loop
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == (near if retarget else far)).all()


def test_retarget_deferred_after_fire() -> None:
    """Scan frame within `retarget_busy` (8) loops of last fire defers the retarget
    check to last_fire + 8."""
    sim, s, far, near = _chase(True)
    w, c = sim.world, sim.combat
    c.last_fire_loop[:, s] = 5
    w.game_loop[:] = 10  # scan frame, 5 loops after fire: defers to 13
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()
    assert (c.rt_due[:, s] == 13).all()
    w.game_loop[:] = 12
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()
    w.game_loop[:] = 13
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == near).all()


@pytest.mark.parametrize('lock', ['new', 'previous'])
def test_retarget_needs_closer_in_range_and_never_switches_back(lock: str) -> None:
    """Out-of-range closer enemy does not switch; once switched, never switches back."""
    sim, s, far, near = _chase(True, lock)
    w = sim.world
    w.x[:, near] = np.float32(17.5)  # closer than far, but outside weapon range
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()
    w.x[:, near] = np.float32(13)
    w.game_loop[:] = 18
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == near).all()
    w.x[:, far] = np.float32(12.5)  # original target returns in range and closer
    w.x[:, near] = np.float32(17.5)  # new target leaves range
    w.game_loop[:] = 26
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == near).all()


def test_retarget_keeps_target_in_range() -> None:
    sim, s, far, near = _chase(True)
    w = sim.world
    w.x[:, far] = w.x[:, s] + w.radius[:, s] + w.radius[:, far] + np.float32(0.05)  # close
    w.x[:, near] = w.x[:, far]
    w.y[:, near] = w.y[:, s] + np.float32(0.5)
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()


def _ready_chase(ready: bool, margin: float = 0.2):
    """Variant of `_chase` with nearer ally Zealot outside weapon range but inside acquire
    range (Stalker range 6, acquire 6.5, edge dist 6.375) vs far at 7.65 (diff 1.275)."""
    sim, s, far, near = _chase(True)
    sim.config = dataclasses.replace(sim.config, retarget_ready=ready, retarget_ready_margin=margin)
    sim.world.x[:, near] = np.float32(17.5)
    return sim, s, far, near


@pytest.mark.parametrize('ready', [False, True])
def test_retarget_ready_switches_to_closer_out_of_range(ready: bool) -> None:
    """At zero cooldown, an out-of-range enemy in acquire range closer by > 0.2 switches
    only when `retarget_ready` is enabled."""
    sim, s, far, near = _ready_chase(ready)
    w = sim.world
    assert (w.weapon_cooldown[:, s] == 0).all()
    w.game_loop[:] = 3  # non-scan loop
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == (near if ready else far)).all()
    if ready:
        assert (sim.combat.rt_lock[:, s] == near).all()


def test_retarget_ready_needs_zero_cooldown() -> None:
    """While on cooldown, closer out-of-range enemies still do not trigger a switch."""
    sim, s, far, near = _ready_chase(True)
    w = sim.world
    w.weapon_cooldown[:, s] = np.float32(6)
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()


def test_retarget_ready_needs_margin() -> None:
    """Edge difference 0.075 < 0.2 does not switch; lowering margin to 0.05 switches."""
    for margin, want in ((0.2, 'far'), (0.05, 'near')):
        sim, s, far, near = _ready_chase(True, margin)
        w = sim.world
        w.x[:, far] = np.float32(17.7)  # edge dist 6.45, 0.075 farther than near (6.375)
        w.game_loop[:] = 10
        targeting.sc2(sim)
        assert (w.engaged_target_slot[:, s] == (far if want == 'far' else near)).all()


def test_retarget_ready_needs_acquire_range() -> None:
    """Closer enemy outside acquire range (edge dist > 6.5) does not trigger a switch."""
    sim, s, far, near = _ready_chase(True)
    w = sim.world
    w.x[:, near] = np.float32(17.8)  # edge dist 6.675
    w.x[:, far] = np.float32(20)
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()


def test_retarget_ready_options() -> None:
    cfg = PyrefConfig()
    assert not cfg.retarget_ready
    assert cfg.retarget_ready_margin == pytest.approx(0.2)
    with pytest.raises(ValueError, match='retarget_ready.*retarget'):
        PyrefConfig(retarget_ready=True)
    with pytest.raises(ValueError, match='retarget_ready_margin'):
        PyrefConfig(retarget=True, retarget_ready=True, retarget_ready_margin=-0.1)


@pytest.mark.parametrize('lock', ['new', 'previous', 'none'])
def test_retarget_lock_third_target(lock: str) -> None:
    """After switching to near, near leaves range and a third ally enters closer in range:
    'previous' and 'none' switch to third while 'new' stays locked on near. 'new' records
    near in rt_lock, 'previous' records replaced target near, 'none' leaves NO_SLOT."""
    sim, s, far, near = _chase(True, lock)
    w = sim.world
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == near).all()
    third = int(np.flatnonzero(w.owner[0] == 1)[2])
    w.x[:, near] = np.float32(18.9)  # out of range
    w.x[:, third], w.y[:, third] = np.float32(13), np.float32(10)
    w.game_loop[:] = 18
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == (near if lock == 'new' else third)).all()
    assert (sim.combat.rt_lock[:, s] == (NO_SLOT if lock == 'none' else near)).all()


def test_retarget_lock_none_switches_back() -> None:
    """With 'none', after switching to near, if the original target returns closer in
    range while near leaves range, the unit switches back to the original target."""
    sim, s, far, near = _chase(True, 'none')
    w = sim.world
    w.game_loop[:] = 10
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == near).all()
    w.x[:, far] = np.float32(12.5)
    w.x[:, near] = np.float32(17.5)
    w.game_loop[:] = 18
    targeting.sc2(sim)
    assert (w.engaged_target_slot[:, s] == far).all()


def test_retarget_lock_option() -> None:
    assert PyrefConfig().retarget_lock == 'new'
    PyrefConfig(retarget=True, retarget_lock='none')
    with pytest.raises(ValueError, match='retarget_lock'):
        PyrefConfig(retarget_lock='old')
