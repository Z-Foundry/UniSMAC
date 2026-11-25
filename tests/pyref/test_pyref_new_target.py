"""Tests for pyref `chase_lock_target` and `slop_on_arrival`.

* `chase_lock_target`: `chase_lock` only holds units still engaged on their last-fired target;
  units that switched to a new target chase normally.
* `slop_on_arrival`: reaching strict weapon range grants range tolerance even before firing,
  so the unit holds and fires in place while the target stays within `range + follow_slop`;
  switching targets or receiving a breaking command restarts from strict range.

F (player 2 Marine) is at (16, 16); A and B are player 1 Ultralisks (player 1 does not
auto-acquire). Both flags are disabled by default, leaving default behavior bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, ULTRALISK = 48, 109
F, A, B = 0, 1, 2
RNG = STATIC.get(MARINE).weapons[0].range
RR = STATIC.get(MARINE).radius + STATIC.get(ULTRALISK).radius
BASE = {'targeting': 'sc2', 'attack': 'sc2', 'auto_acquire': True, 'auto_acquire_owners': (2,)}
LOCK = ((MARINE, 12),)


class Trio:
    """A is at edge `range - 0.5` (+x from F); B is at edge 9 (-x, outside acquire range)."""

    def __init__(self, source: str = 'amove', **config: object) -> None:
        cfg = PyrefConfig(**{**BASE, **config})  # type: ignore[arg-type]
        self.be = PyrefBackend(STATIC, MAPS, 1, cfg)
        sc = Scenario(
            FLAT,
            spawns=(
                Spawn(2, MARINE, 16.0, 16.0),
                Spawn(1, ULTRALISK, 16.0 + RR + RNG - 0.5, 16.0),
                Spawn(1, ULTRALISK, 16.0 - RR - 9.0, 16.0),
            ),
        )
        self.be.reset([sc], seeds=[0])
        sim = self.be._sim  # noqa: SLF001 - direct position and cooldown setup
        assert sim is not None
        self.sim = sim
        self.first = self.cmd(CommandKind.ATTACK_MOVE, target_xy=(16.0, 4.0))
        if source == 'attack_unit':
            self.first = self.cmd(CommandKind.ATTACK_UNIT, target_slot=A)
        self.states: list[WorldState] = []

    def cmd(self, kind: CommandKind, **kw: object) -> Command:
        c = Command.noop(1, 3)
        c.set(0, F, kind, **kw)  # type: ignore[arg-type]
        return c

    def put(self, slot: int, d: float, side: float = 1.0) -> None:
        """Teleport `slot` to edge `range + d` along +x (side=1) or -x (side=-1) from F."""
        w = self.sim.world
        w.x[0, slot] = np.float32(float(w.x[0, F]) + side * (RR + RNG + d))
        w.y[0, slot] = w.y[0, F]

    def step(self, cmd: Command | None = None) -> WorldState:
        if cmd is None:
            cmd = self.first if not self.states else Command.noop(1, 3)
        s = self.be.step(cmd, 1)
        self.states.append(s)
        return s

    def fired(self) -> bool:
        s = self.states
        if len(s) < 2:
            return len(s) == 1 and float(s[0].weapon_cooldown[0, F]) > 0.5
        return float(s[-1].weapon_cooldown[0, F]) > float(s[-2].weapon_cooldown[0, F]) + 0.5

    def run_to_fire(self, limit: int = 80) -> None:
        for _ in range(limit):
            self.step()
            if self.fired():
                return
        raise AssertionError('F should fire')

    def dx(self) -> float:
        s = self.states
        return float(s[-1].x[0, F]) - float(s[-2].x[0, F])

    def eng(self) -> int:
        return int(self.states[-1].engaged_target_slot[0, F])


def test_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert not cfg.chase_lock_target and not cfg.slop_on_arrival and not cfg.reissue_rescan
    with pytest.raises(ValueError, match='slop_on_arrival'):
        PyrefConfig(slop_on_arrival=True, attack='sc2')  # targeting='basic'
    with pytest.raises(ValueError, match='slop_on_arrival'):
        PyrefConfig(slop_on_arrival=True, targeting='sc2')  # attack='instant'
    with pytest.raises(ValueError, match='reissue_rescan'):
        PyrefConfig(reissue_rescan=True)
    assert PyrefConfig(slop_on_arrival=True, targeting='sc2', attack='sc2').slop_on_arrival


# ---------------------------------------------------------------- chase_lock_target


def _switch_in_lock(**config: object) -> list[float]:
    """After F fires on A, teleport A out of acquire range and B to edge `range + 0.3` on -x,
    then reissue ATTACK_MOVE to a new point so F immediately acquires B; returns F's next 13
    x-displacements (index 0 = first step after fire)."""
    t = Trio('amove', **config)
    t.run_to_fire()
    t.sim.world.x[0, A], t.sim.world.y[0, A] = np.float32(16.0), np.float32(29.0)
    t.put(B, 0.3, side=-1.0)
    out = []
    t.step(t.cmd(CommandKind.ATTACK_MOVE, target_xy=(16.0, 5.0)))
    assert t.eng() == B
    out.append(t.dx())
    for _ in range(12):
        t.step()
        out.append(t.dx())
    return out


def test_chase_lock_blocks_new_target_by_default() -> None:
    """With `chase_lock` alone, F holds 12 loops even for a new target, chasing (-x) on step 13."""
    dx = _switch_in_lock(chase_lock=LOCK)
    assert all(abs(v) < 1e-6 for v in dx[:12])
    assert dx[12] < -1e-3


def test_chase_lock_target_chases_new_target() -> None:
    """With `chase_lock_target`, switching to a new target chases immediately on that step."""
    dx = _switch_in_lock(chase_lock=LOCK, chase_lock_target=True)
    assert dx[0] < -1e-3
    assert np.array_equal(dx, _switch_in_lock(chase_lock=()))


def test_chase_lock_target_keeps_lock_on_same_target() -> None:
    """Staying on the last-fired target matches `chase_lock` alone step-for-step (no chase)."""

    def run(**config: object) -> list[float]:
        t = Trio('amove', chase_lock=LOCK, **config)
        t.run_to_fire()
        t.put(A, 0.3)
        out = []
        for _ in range(13):
            t.step()
            out.append(t.dx())
        return out

    on, off = run(chase_lock_target=True), run()
    assert on == off
    assert all(abs(v) < 1e-6 for v in on[:12]) and on[12] > 1e-3


def test_chase_lock_target_without_chase_lock_is_noop() -> None:
    assert _switch_in_lock(chase_lock_target=True) == _switch_in_lock()


# ---------------------------------------------------------------- slop_on_arrival


def _arrived(source: str = 'attack_unit', **config: object) -> Trio:
    """Engage F on in-range A with cooldown set to 30 so F cannot fire, then step 3 loops."""
    t = Trio(source, **config)
    t.sim.combat.cooldown[0, F] = 30.0
    for _ in range(3):
        t.step()
    assert t.eng() == A and not t.fired()
    return t


@pytest.mark.parametrize('follow_lock', [False, True])
@pytest.mark.parametrize('source', ['attack_unit', 'amove'])
def test_tolerance_after_arrival(source: str, follow_lock: bool) -> None:
    """Unfired target backing to range+0.9 holds, range+1.1 chases; disabled chases past range."""
    t = _arrived(source, slop_on_arrival=True, follow_lock=follow_lock)
    t.put(A, 0.9)
    for _ in range(5):
        t.step()
        assert abs(t.dx()) < 1e-6 and not t.fired()
    t.put(A, 1.1)
    t.step()
    assert t.dx() > 1e-3

    off = _arrived(source, follow_lock=follow_lock)
    off.put(A, 0.5)
    off.step()
    assert off.dx() > 1e-3


@pytest.mark.parametrize('follow_lock', [False, True])
def test_fire_in_place_from_tolerance(follow_lock: bool) -> None:
    """When cooldown reaches 0 inside tolerance, F fires in place beyond strict weapon range."""
    t = _arrived(slop_on_arrival=True, follow_lock=follow_lock)
    t.put(A, 0.5)
    for _ in range(40):
        t.step()
        assert abs(t.dx()) < 1e-6
        if t.fired():
            break
    else:
        raise AssertionError('should fire within range tolerance')
    w = t.sim.world
    edge = float(np.hypot(w.x[0, A] - w.x[0, F], w.y[0, A] - w.y[0, F])) - RR
    assert edge > RNG + 0.4


def test_new_target_not_carried_over() -> None:
    """Tolerance on A does not carry over to B: switching to ATTACK_UNIT B at range+0.5 chases."""
    t = _arrived(slop_on_arrival=True, follow_lock=True)
    t.put(A, 0.5)
    t.step()
    assert abs(t.dx()) < 1e-6
    t.put(B, 0.5, side=-1.0)
    t.step(t.cmd(CommandKind.ATTACK_UNIT, target_slot=B))
    assert t.eng() == B and t.dx() < -1e-3


def test_breaking_command_restarts_from_strict_range() -> None:
    """After ATTACK_MOVE engages A in tolerance, issuing ATTACK_UNIT A (breaking command) resets
    to strict range so range+0.3 chases; reissuing the same ATTACK_MOVE does not break it."""
    t = _arrived('amove', slop_on_arrival=True, follow_lock=True)
    t.put(A, 0.3)  # in acquire range: default reissue clears and reacquires A
    t.step()
    t.step(t.first)
    assert abs(t.dx()) < 1e-6
    t.step(t.cmd(CommandKind.ATTACK_UNIT, target_slot=A))
    assert t.dx() > 1e-3


def test_hold_has_no_tolerance() -> None:
    """HOLD never enters range tolerance (as with `follow_lock`) and does not chase."""
    t = _arrived(slop_on_arrival=True)
    t.step(t.cmd(CommandKind.HOLD))
    t.put(A, 0.5)
    t.step()
    assert not bool(t.sim.combat.follow_tol[0, F])
