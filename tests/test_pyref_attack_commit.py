"""Tests for pyref `attack_commit` and `attack_on_arrival`.

attack_commit (only shots that actually fire consume cooldown):
1. A target leaving range mid-windup does not cancel the shot; the shooter holds position
   during windup and completes the shot.
2. Switching targets mid-windup cancels the shot and refunds the cooldown.
3. If an earlier slot kills the target on the same loop, the later shot is skipped, cooldown
   is refunded, and it does not count as fired.

attack_on_arrival: a unit chasing the same target into range (windup >= 1) fires 1 loop
after entering range instead of 2; newly acquired targets, zero-windup units, and stationary
shooters are unaffected.

Both default to off with bit-identical behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import NO_SLOT, Command, CommandKind, Scenario, Spawn, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, MARAUDER, STALKER, ULTRALISK = 48, 51, 74, 109
BASE = {
    'targeting': 'sc2',
    'attack': 'sc2',
    'damage': 'sc2',
    'auto_acquire': True,
    'auto_acquire_owners': (2,),  # Player 1 targets do not auto-acquire or retaliate
}


def _range(unit: int) -> float:
    return STATIC.get(unit).weapons[0].range


def _gap(a: int, b: int) -> float:
    """Sum of radii: center distance equals this value when edge-to-edge gap is 0."""
    return STATIC.get(a).radius + STATIC.get(b).radius


class Fight:
    """Player 2 shooters (from slot 0) vs Player 1 Ultralisk (last slot) near y = 16.

    shooters: (unit_type, x, y); target at (tx, 16); extra: additional Player 1 units;
    config overrides BASE. Advances 1 loop per step."""

    def __init__(
        self,
        shooters: list[tuple[int, float, float]],
        tx: float,
        extra: list[tuple[int, float, float]] | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
        cfg = PyrefConfig(**{**BASE, **(config or {})})  # type: ignore[arg-type]
        self.be = PyrefBackend(STATIC, MAPS, 1, cfg)
        spawns = [Spawn(2, u, x, y) for u, x, y in shooters]
        spawns.append(Spawn(1, ULTRALISK, tx, 16.0))
        spawns += [Spawn(1, u, x, y) for u, x, y in extra or []]
        self.n = len(spawns)
        self.t = len(shooters)
        self.be.reset([Scenario(FLAT, spawns=tuple(spawns))], seeds=[0])
        sim = self.be._sim  # noqa: SLF001 - test edits positions/HP and inspects internal state
        assert sim is not None
        self.sim = sim
        self.states: list[WorldState] = []

    def attack(self, target: int, shooters: list[int] | None = None) -> Command:
        cmd = Command.noop(1, self.n)
        for u in range(self.t) if shooters is None else shooters:
            cmd.set(0, u, CommandKind.ATTACK_UNIT, target_slot=target)
        return cmd

    def step(self, cmd: Command | None = None) -> WorldState:
        s = self.be.step(cmd if cmd is not None else Command.noop(1, self.n), 1)
        self.states.append(s)
        return s

    def hp(self, slot: int) -> float:
        return float(self.sim.world.health[0, slot])

    def until_windup(self, u: int = 0, limit: int = 40) -> None:
        """Advance until unit u starts an attack windup (windup_left >= 1)."""
        for _ in range(limit):
            if int(self.sim.combat.windup_left[0, u]) >= 1:
                return
            self.step()
        raise AssertionError('unit should have started attack windup')


def _duel(**config: object) -> Fight:
    """Marine at (10, 16), Ultralisk on +x at edge distance (range - 0.05)."""
    tx = 10.0 + _gap(MARINE, ULTRALISK) + _range(MARINE) - 0.05
    return Fight([(MARINE, 10.0, 16.0)], tx, config=config)


def test_config_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.attack_commit is False and cfg.attack_on_arrival is False
    with pytest.raises(ValueError, match='attack_commit'):
        PyrefConfig(attack_commit=True, attack='sc2')  # damage='standard'
    with pytest.raises(ValueError, match='attack_commit'):
        PyrefConfig(attack_commit=True, damage='sc2')  # attack='instant'
    with pytest.raises(ValueError, match='attack_on_arrival'):
        PyrefConfig(attack_on_arrival=True)
    assert PyrefConfig(attack_commit=True, attack='sc2', damage='sc2').attack_commit
    assert PyrefConfig(attack_on_arrival=True, attack='sc2').attack_on_arrival


@pytest.mark.parametrize('commit', [False, True])
def test_windup_keeps_when_target_leaves_range(commit: bool) -> None:
    """Move target 0.1 outside range mid-windup: when on, shot still lands and shooter holds
    position; when off, windup cancels (spending cooldown with no damage) and shooter chases."""
    f = _duel(attack_commit=commit)
    f.step(f.attack(1))
    f.until_windup()
    hp, x0 = f.hp(1), float(f.sim.world.x[0, 0])
    w = f.sim.world
    w.x[0, 1] = np.float32(float(w.x[0, 0]) + _gap(MARINE, ULTRALISK) + _range(MARINE) + 0.1)
    s = f.step()
    assert int(f.sim.combat.windup_left[0, 0]) == -1
    assert float(s.weapon_cooldown[0, 0]) > 5
    if commit:
        assert f.hp(1) < hp
        assert float(s.x[0, 0]) == x0
    else:
        assert f.hp(1) == hp
        assert float(s.x[0, 0]) > x0


@pytest.mark.parametrize('commit', [False, True])
def test_cancelled_windup_refunds_cooldown(commit: bool) -> None:
    """Switch targets mid-windup (also in range): when on, cooldown is refunded and the new
    target is hit shortly after turning; when off, cooldown stays spent."""
    rng, g = _range(MARINE), _gap(MARINE, ULTRALISK)
    behind = [(ULTRALISK, 10.0 - g - rng + 0.05, 16.0)]
    f = Fight([(MARINE, 10.0, 16.0)], 10.0 + g + rng - 0.05, behind, {'attack_commit': commit})
    f.step(f.attack(1))
    f.until_windup()
    hp1, hp2 = f.hp(1), f.hp(2)
    f.step(f.attack(2))
    assert int(f.sim.combat.windup_left[0, 0]) == -1 and f.hp(1) == hp1
    cd = float(f.sim.combat.cooldown[0, 0])
    assert (cd == 0) if commit else (cd > 5)
    for _ in range(5):  # 180 deg turn takes 3 loops (facing_rate 1.09) plus 1 loop windup
        f.step()
    assert (f.hp(2) < hp2) == commit


@pytest.mark.parametrize('commit', [False, True])
def test_same_loop_overkill_skipped(commit: bool) -> None:
    """Two Marines fire on the same loop and the first kills the target: when on, the later
    slot skips its shot (cooldown 0, not counted as fired); when off, both spend cooldown."""
    tx = 10.0 + _gap(MARINE, ULTRALISK) + _range(MARINE) - 0.5
    f = Fight([(MARINE, 10.0, 15.5), (MARINE, 10.0, 16.5)], tx, config={'attack_commit': commit})
    f.step(f.attack(2))
    f.until_windup(0)
    assert int(f.sim.combat.windup_left[0, 1]) == 1
    f.sim.world.health[0, 2] = 1.0
    s = f.step()
    c = f.sim.combat
    loop = int(c.last_fire_loop[0, 0])
    assert not s.alive[0, 2]
    assert float(s.weapon_cooldown[0, 0]) > 5 and loop >= 0
    if commit:
        assert float(s.weapon_cooldown[0, 1]) == 0 and float(c.cooldown[0, 1]) == 0
        assert int(c.last_fire_loop[0, 1]) == -1
    else:
        assert float(s.weapon_cooldown[0, 1]) > 5 and int(c.last_fire_loop[0, 1]) == loop


def _chase(unit: int, **config: object) -> tuple[int, int, float]:
    """Chase Ultralisk from 1.5 outside range; returns (first in-range step, first damage step,
    observed cooldown on that step)."""
    g = _gap(unit, ULTRALISK)
    f = Fight([(unit, 10.0, 16.0)], 10.0 + g + _range(unit) + 1.5, config=config)
    hp = f.hp(1)
    inside = hit = -1
    for k in range(60):
        s = f.step(f.attack(1) if k == 0 else None)
        edge = float(s.x[0, 1]) - float(s.x[0, 0]) - g
        if inside < 0 and edge <= _range(unit) + 1e-6:
            inside = k
        if f.hp(1) < hp:
            hit = k
            break
    assert 0 < inside < hit
    return inside, hit, float(f.states[hit].weapon_cooldown[0, 0])


@pytest.mark.parametrize('unit', [MARINE, STALKER])
def test_arrival_fires_one_loop_earlier(unit: int) -> None:
    """Chasing into range: fires 2 loops after entry when off (Marine windup 1; Stalker has
    projectile travel), 1 loop earlier when on; entry loop and firing cooldown match."""
    i0, h0, cd0 = _chase(unit)
    i1, h1, cd1 = _chase(unit, attack_on_arrival=True)
    assert i1 == i0 and h1 == h0 - 1
    assert cd1 == pytest.approx(cd0)
    if unit == MARINE:
        assert h0 - i0 == 2


def test_arrival_ignores_zero_windup_and_standing_units() -> None:
    """Zero-windup units (Marauder) chasing in and stationary shooters are identical on/off."""
    assert _chase(MARAUDER, attack_on_arrival=True) == _chase(MARAUDER)
    a, b = _duel(), _duel(attack_on_arrival=True)
    _same_run(a, b)


def test_arrival_needs_same_target() -> None:
    """MOVE toward the target for 4 loops (moving, ready, facing), then ATTACK_UNIT in range:
    no engaged target on the previous loop, so not treated as chase arrival (hits 2 loops later)."""

    def run(**config: object) -> int:
        f = _duel(**config)
        move = Command.noop(1, f.n)
        move.set(0, 0, CommandKind.MOVE, target_xy=(28.0, 16.0))
        hp = f.hp(1)
        for k in range(30):
            f.step(move if k == 0 else f.attack(1) if k == 4 else None)
            if k == 3:
                assert float(f.sim.motion.speed[0, 0]) > 0
            if f.hp(1) < hp:
                return k
        raise AssertionError('unit should have hit target')

    assert run(attack_on_arrival=True) == run() == 5


def _same_run(a: Fight, b: Fight, steps: int = 80) -> None:
    for k in range(steps):
        sa = a.step(a.attack(1) if k == 0 else None)
        sb = b.step(b.attack(1) if k == 0 else None)
        for name in ('x', 'y', 'health', 'weapon_cooldown', 'alive', 'engaged_target_slot'):
            assert np.array_equal(getattr(sa, name), getattr(sb, name)), (k, name)


def test_commit_no_op_without_trigger() -> None:
    """Single shooter vs stationary target (no range exit, retarget, or overkill): identical."""
    _same_run(_duel(), _duel(attack_commit=True))


def test_state_untouched_when_off() -> None:
    """When off, new state buffers remain untouched (zero/False/initial values)."""
    f = _duel()
    f.step(f.attack(1))
    for _ in range(60):
        f.step()
    assert not f.sim.combat.windup_cost.any()
    assert not f.sim.buf.windup_hold.any()
    assert (f.sim.buf.fire_prev_loop == -1).all()
    assert (f.sim.combat.arrive_target == NO_SLOT).all()
