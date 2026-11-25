"""Tests for pyref engaged follow-delay rule `follow_lock`.

F (Player 2) attacks T (Player 1, non-retaliating). After F fires (weapon_cooldown jumps),
teleport T to edge distance `range + d` and inspect F's per-step displacement. Letting k be
the firing step and j = game_loop - fire_loop at step k + j:

1. For j < L, hold position; on step L, chase if T is strictly outside weapon range.
2. If T is inside range when the lock ends, enter tolerance and only chase past `range + 1.0`.
3. If cooldown reaches 0 while within tolerance, fire in place and re-lock for L loops.

Disabled by default with bit-identical behavior; `follow_lock` in static measured data is
inert when disabled.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, CommandKind, Scenario, Spawn, StaticData, Variant, WorldState

FLAT = 'SMAC_Maps/32x32_flat'
MAPS = {FLAT: load_map_info(FLAT)}
STATIC = load_static_data()
MARINE, MARAUDER, ZEALOT, STALKER, HYDRALISK, ULTRALISK = 48, 51, 73, 74, 107, 109
LOCK = {STALKER: 9, ZEALOT: 9, MARINE: 13, MARAUDER: 1}
SOURCES = ('attack_unit', 'amove', 'idle')
F, T = 0, 1
BASE = {
    'targeting': 'sc2',
    'attack': 'sc2',
    'auto_acquire': True,
    'auto_acquire_owners': (2,),  # Player 1 target T does not auto-acquire or retaliate
}


def _range(unit: int) -> float:
    return STATIC.get(unit).weapons[0].range


class Duel:
    """F at (10, 16), T on +x at edge distance `range - 0.5` (0.05 for Zealot)."""

    def __init__(
        self,
        unit: int = STALKER,
        source: str = 'attack_unit',
        batch: int = 1,
        seeds: tuple[int, ...] | None = None,
        static: StaticData = STATIC,
        **config: object,
    ) -> None:
        cfg = PyrefConfig(**{**BASE, 'follow_lock': True, **config})  # type: ignore[arg-type]
        self.be = PyrefBackend(static, MAPS, batch, cfg)
        self.unit, self.batch = unit, batch
        self.rng = _range(unit)
        self.r = STATIC.get(unit).radius + STATIC.get(ULTRALISK).radius
        start = self.rng - 0.5 if self.rng > 1 else 0.05
        sc = Scenario(
            FLAT,
            spawns=(Spawn(2, unit, 10.0, 16.0), Spawn(1, ULTRALISK, 10.0 + self.r + start, 16.0)),
        )
        self.be.reset([sc] * batch, seeds=list(seeds or range(batch)))
        sim = self.be._sim  # noqa: SLF001 - test edits positions and cooldown directly
        assert sim is not None
        self.sim = sim
        self.first = Command.noop(batch, 2)
        for b in range(batch):
            if source == 'attack_unit':
                self.first.set(b, F, CommandKind.ATTACK_UNIT, target_slot=T)
            elif source == 'amove':
                self.first.set(b, F, CommandKind.ATTACK_MOVE, target_xy=(10.0, 4.0))
            elif source == 'hold':
                self.first.set(b, F, CommandKind.HOLD)
        self.states: list[WorldState] = []

    def put(self, d: float | list[float]) -> None:
        """Teleport T along +x from F to edge distance `range + d` (per-env d)."""
        w = self.sim.world
        dd = np.broadcast_to(np.asarray(d, np.float64), (self.batch,))
        w.x[:, T] = (w.x[:, F].astype(np.float64) + self.r + self.rng + dd).astype(np.float32)
        w.y[:, T] = w.y[:, F]

    def step(self, cmd: Command | None = None) -> WorldState:
        if cmd is None:
            cmd = self.first if not self.states else Command.noop(self.batch, 2)
        s = self.be.step(cmd, 1)
        self.states.append(s)
        return s

    def fired(self, b: int = 0) -> bool:
        s = self.states
        if len(s) < 2:
            return len(s) == 1 and float(s[0].weapon_cooldown[b, F]) > 0.5
        return float(s[-1].weapon_cooldown[b, F]) > float(s[-2].weapon_cooldown[b, F]) + 0.5

    def run_to_fire(self, limit: int = 80) -> int:
        """Advance until F fires and return that state index."""
        for _ in range(limit):
            self.step()
            if self.fired():
                return len(self.states) - 1
        raise AssertionError('F should have fired')

    def moved(self, b: int = 0) -> bool:
        s = self.states
        return bool(abs(float(s[-1].x[b, F]) - float(s[-2].x[b, F])) > 1e-6)


def _moves_after(duel: Duel, d: float, steps: int) -> list[bool]:
    """Teleport T to `range + d` after firing; return whether F moved on each step (idx 0: j=1)."""
    duel.run_to_fire()
    duel.put(d)
    out = []
    for _ in range(steps):
        duel.step()
        out.append(duel.moved())
    return out


def test_config_defaults_and_validation() -> None:
    cfg = PyrefConfig()
    assert cfg.follow_lock is False and cfg.follow_slop == 1.0
    with pytest.raises(ValueError, match='follow_lock'):
        PyrefConfig(follow_lock=True, attack='sc2')  # targeting='basic'
    with pytest.raises(ValueError, match='follow_lock'):
        PyrefConfig(follow_lock=True, targeting='sc2')  # attack='instant'
    with pytest.raises(ValueError, match='follow_slop'):
        PyrefConfig(follow_slop=-0.1)
    assert PyrefConfig(follow_lock=True, targeting='sc2', attack='sc2').follow_lock


def test_measured_lock() -> None:
    for unit, lock in LOCK.items():
        assert STATIC.get(unit).measured['follow_lock'] == lock
        assert int(Duel(unit).sim.params.follow_lock[0, F]) == lock
    assert int(Duel(HYDRALISK).sim.params.follow_lock[0, F]) == -1
    # *_RL variants merge measured data by (unit_type, variant) and do not inherit standard keys
    assert 'follow_lock' not in STATIC.get(STALKER, Variant.RL).measured


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('unit', list(LOCK))
def test_lock_then_strict_check(unit: int, source: str) -> None:
    """Target at range + 0.5 (in tolerance): stays still during lock, chases on strict range
    check at step L. When off, chases on the very next step."""
    lock = LOCK[unit]
    moved = _moves_after(Duel(unit, source), 0.5, lock + 2)
    assert moved[: lock - 1] == [False] * (lock - 1)
    assert moved[lock - 1]
    assert _moves_after(Duel(unit, source, follow_lock=False), 0.5, 1) == [True]


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('unit', [STALKER, ZEALOT, MARAUDER])
def test_tolerance_wait_fire_and_relock(unit: int, source: str) -> None:
    """Target still in range at lock expiry: later moving to range + 0.5 does not trigger chase;
    fires in place when cooldown hits 0, re-locks for L, then chases on strict check at expiry."""
    lock = LOCK[unit]
    duel = Duel(unit, source)
    k = duel.run_to_fire()
    for _ in range(lock):  # steps k + 1 .. k + L: target remains in range
        duel.step()
        assert not duel.moved() and not duel.fired()
    duel.put(0.5)
    for _ in range(60):
        duel.step()
        assert not duel.moved()
        if duel.fired():
            break
    else:
        raise AssertionError('should have fired within tolerance')
    k2 = len(duel.states) - 1
    assert k2 > k + lock
    for j in range(1, lock + 1):
        duel.step()
        assert duel.moved() == (j == lock), j

    off = Duel(unit, source, follow_lock=False)
    off.run_to_fire()
    for _ in range(lock):
        off.step()
    off.put(0.5)
    off.step()
    assert off.moved()


@pytest.mark.parametrize('source', SOURCES)
def test_follow_only_beyond_range_plus_slop(source: str) -> None:
    """In tolerance mode, holds position at range + 0.9 and chases once target hits range + 1.1."""
    duel = Duel(STALKER, source)
    duel.run_to_fire()
    for _ in range(LOCK[STALKER]):
        duel.step()
    duel.put(0.9)
    for _ in range(5):  # Stalker cooldown is ~30 loops, so no shot during these steps
        duel.step()
        assert not duel.moved() and not duel.fired()
    duel.put(1.1)
    duel.step()
    assert duel.moved()


def test_slop_option() -> None:
    """follow_slop=0.5: chases at range + 0.6."""
    duel = Duel(STALKER, follow_slop=0.5)
    duel.run_to_fire()
    for _ in range(LOCK[STALKER]):
        duel.step()
    duel.put(0.4)
    duel.step()
    assert not duel.moved()
    duel.put(0.6)
    duel.step()
    assert duel.moved()


def test_no_slop_fire_inside_lock() -> None:
    """Cooldown hits 0 during lock while target is in slop but outside strict range: does not
    fire, then chases when the lock ends."""
    duel = Duel(MARINE)
    duel.run_to_fire()
    duel.sim.combat.cooldown[0, F] = 3.0  # cooldown reaches 0 at step 3 (Marine lock is 13)
    duel.put(0.4)
    for j in range(1, LOCK[MARINE] + 1):
        duel.step()
        assert not duel.fired(), j
        assert duel.moved() == (j == LOCK[MARINE]), j


@pytest.mark.parametrize('reissue_unit', [False, True])
def test_reissued_attack_unit_keeps_lock(reissue_unit: bool) -> None:
    """Reissuing the same ATTACK_UNIT every step does not reset the lock."""
    duel = Duel(STALKER, reissue_unit=reissue_unit)
    duel.run_to_fire()
    duel.put(0.5)
    for j in range(1, LOCK[STALKER] + 1):
        duel.step(duel.first)
        assert duel.moved() == (j == LOCK[STALKER]), j


# commands='delayed' with move_lag: commands take effect next loop, and the activation loop
# still displaces using the previous loop's velocity.
DELAY = {'commands': 'delayed', 'command_delay': 1, 'move_lag': 1}


def _cmd(duel: Duel, kind: CommandKind, **kw: object) -> Command:
    cmd = Command.noop(duel.batch, 2)
    for b in range(duel.batch):
        cmd.set(b, F, kind, **kw)  # type: ignore[arg-type]
    return cmd


def _west(duel: Duel) -> Command:
    return _cmd(duel, CommandKind.MOVE, target_xy=(float(duel.sim.world.x[0, F]) - 3.0, 16.0))


def _dx(duel: Duel, b: int = 0) -> float:
    s = duel.states
    return float(s[-1].x[b, F]) - float(s[-2].x[b, F])


@pytest.mark.parametrize('kind', [CommandKind.MOVE, CommandKind.HOLD, CommandKind.STOP])
def test_command_breaks_lock(kind: CommandKind) -> None:
    """Issuing another order (MOVE/HOLD/STOP) after firing breaks the lock: a subsequent
    ATTACK_UNIT checks strict range on activation and chases immediately, then re-locks for L
    after firing again."""
    lock = LOCK[STALKER]
    duel = Duel(STALKER)
    duel.run_to_fire()
    duel.put(0.5)
    other = _west(duel) if kind == CommandKind.MOVE else _cmd(duel, kind)
    duel.step(other)  # j = 1
    if kind == CommandKind.MOVE:
        assert _dx(duel) < -1e-3
    duel.step(duel.first)  # j = 2 < L: ATTACK_UNIT activates
    assert _dx(duel) > 1e-3
    # After chasing into range and firing, normal lock of L applies
    duel.run_to_fire()
    duel.put(0.5)
    for j in range(1, lock + 1):
        duel.step()
        assert duel.moved() == (j == lock), j


def test_command_in_range_enters_tolerance() -> None:
    """ATTACK_UNIT after MOVE with target still in range enters tolerance immediately: retreating
    to range + 0.5 does not trigger chase at fire + L, only past range + 1.0."""
    lock = LOCK[STALKER]
    duel = Duel(STALKER)
    duel.run_to_fire()
    duel.step(_west(duel))  # j = 1: step back 0.26, target still in range (range - 0.24)
    duel.step(duel.first)  # j = 2
    assert not duel.moved()
    duel.put(0.5)
    for j in range(3, lock + 4):  # Stalker cooldown is ~30 loops, no shot here
        duel.step()
        assert not duel.moved() and not duel.fired(), j
    duel.put(1.1)
    duel.step()
    assert _dx(duel) > 1e-3


def test_command_break_timing_with_move_lag() -> None:
    """command_delay=1 + move_lag=1: ATTACK_UNIT issued on step i activates on step i + 1
    (still moving along MOVE velocity) and starts chasing on step i + 2."""
    duel = Duel(STALKER, **DELAY)
    duel.run_to_fire()
    duel.put(0.5)
    duel.step(_west(duel))  # j = 1: issue MOVE
    duel.step()  # j = 2: MOVE activates, uses previous loop's speed (0)
    assert not duel.moved()
    duel.step(duel.first)  # j = 3: issue ATTACK_UNIT while MOVE steps west
    assert _dx(duel) < -1e-3
    duel.step()  # j = 4: ATTACK_UNIT activates, still uses MOVE velocity
    assert _dx(duel) < -1e-3
    duel.step()  # j = 5 < L: chases east
    assert _dx(duel) > 1e-3


@pytest.mark.parametrize('reissue', [False, True])
def test_same_point_attack_move_keeps_lock(reissue: bool) -> None:
    """Reissuing ATTACK_MOVE to the same point keeps the lock; a new point breaks it."""
    lock = LOCK[STALKER]
    duel = Duel(STALKER, 'amove', reissue=reissue)
    duel.run_to_fire()
    duel.put(0.3)  # within acquire range (range + 0.5) so immediate scan picks T again
    for j in range(1, lock + 1):
        duel.step(duel.first)
        assert duel.moved() == (j == lock), j
    other = Duel(STALKER, 'amove', reissue=reissue)
    other.run_to_fire()
    other.put(0.3)
    other.step(_cmd(other, CommandKind.ATTACK_MOVE, target_xy=(10.0, 5.0)))
    assert _dx(other) > 1e-3


@pytest.mark.parametrize('reissue_unit', [False, True])
def test_reissue_unit_lag_at_follow_frame(reissue_unit: bool) -> None:
    """move_lag=1 with reissue activating on lock-expiry step: default delays chase by 1 loop
    due to lag; `reissue_unit` skips the lag mark and chases on time."""
    lock = LOCK[STALKER]
    duel = Duel(STALKER, reissue_unit=reissue_unit, **DELAY)
    duel.run_to_fire()
    duel.put(0.5)
    first = lock if reissue_unit else lock + 1
    for j in range(1, lock + 2):
        duel.step(duel.first if j == lock - 1 else None)
        assert duel.moved() == (j >= first), j


@pytest.mark.parametrize('follow_lock', [False, True])
def test_reissue_unit_skips_lag_mark(follow_lock: bool) -> None:
    """`reissue_unit` skips setting `motion.lag_loop` when reissuing ATTACK_UNIT on the same
    target; ATTACK_UNIT after MOVE still sets it. Does not require `follow_lock`."""
    for on in (False, True):
        duel = Duel(STALKER, follow_lock=follow_lock, reissue_unit=on, **DELAY)
        duel.step()  # loop 0: issue ATTACK_UNIT; loop 1: activates
        duel.step()
        assert int(duel.sim.motion.lag_loop[0, F]) == 1
        duel.step(duel.first)
        duel.step()  # loop 3: activates
        assert int(duel.sim.motion.lag_loop[0, F]) == (1 if on else 3)
        duel.step(_west(duel))
        duel.step(duel.first)  # loop 5: MOVE activates; loop 6: ATTACK_UNIT activates
        duel.step()
        assert int(duel.sim.motion.lag_loop[0, F]) == 6


def test_reissue_unit_only_skips_lag() -> None:
    """With move_lag=0, `reissue_unit` matches default bit-for-bit even on per-step reissues."""
    for unit in (STALKER, MARINE):
        traces = []
        for on in (False, True):
            duel = Duel(unit, reissue_unit=on)
            duel.run_to_fire()
            duel.put(0.5)
            for _ in range(40):
                duel.step(duel.first)
            traces.append(duel.states)
        _same(*traces)


def test_follow_cmd_inert_when_off() -> None:
    """When `follow_lock` is off, `combat.follow_cmd` stays -1 and post-MOVE ATTACK_UNIT chases."""
    duel = Duel(STALKER, follow_lock=False, **DELAY)
    duel.run_to_fire()
    duel.put(0.5)
    duel.step(_west(duel))
    for _ in range(6):
        duel.step(duel.first)
    assert (duel.sim.combat.follow_cmd == -1).all()


def test_reissue_unit_defaults() -> None:
    assert PyrefConfig().reissue_unit is False
    assert PyrefConfig(reissue_unit=True).reissue_unit  # works without follow_lock/targeting='sc2'
    from unismac.profiles import _V4, _V5, _V6

    for opts in (_V4, _V5, _V6):
        assert 'reissue_unit' not in opts and 'follow_lock' not in opts


def test_batch_command_break_matches_single() -> None:
    """Batched MOVE -> ATTACK_UNIT (varied d per env) matches single-env runs bit-for-bit."""
    ds = [0.5, -0.4, 1.5]

    def run(duel: Duel, d: float | list[float]) -> list[WorldState]:
        duel.run_to_fire()
        duel.put(d)
        duel.step(_west(duel))
        duel.step(duel.first)
        for _ in range(30):
            duel.step()
        return duel.states

    every = run(Duel(STALKER, batch=3, **DELAY), ds)
    for b, d in enumerate(ds):
        one = run(Duel(STALKER, seeds=(b,), **DELAY), d)
        for s_all, s_one in zip(every, one, strict=True):
            for f in ('x', 'y', 'weapon_cooldown', 'engaged_target_slot', 'health'):
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True)


def _trace(duel: Duel, d: float, steps: int) -> list[WorldState]:
    duel.run_to_fire()
    duel.put(d)
    for _ in range(steps):
        duel.step()
    return duel.states


def _same(a: list[WorldState], b: list[WorldState]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for f in dataclasses.fields(x):
            assert np.array_equal(getattr(x, f.name), getattr(y, f.name), equal_nan=True), f.name


def test_hold_unaffected() -> None:
    """HOLD is unaffected: disengages outside strict range, bit-identical to disabled."""
    on = _trace(Duel(STALKER, 'hold'), 0.5, 60)
    off = _trace(Duel(STALKER, 'hold', follow_lock=False), 0.5, 60)
    _same(on, off)
    cd = [0.0] + [float(s.weapon_cooldown[0, F]) for s in on]
    assert sum(b > a + 0.5 for a, b in zip(cd, cd[1:], strict=False)) == 1  # only before teleport
    assert not any(abs(float(s.x[0, F]) - 10.0) > 1e-6 for s in on)


@pytest.mark.parametrize('source', SOURCES)
def test_unit_without_lock_unaffected(source: str) -> None:
    """Units without a `follow_lock` measured entry (Hydralisk) are bit-identical to disabled."""
    on = _trace(Duel(HYDRALISK, source), 0.5, 40)
    off = _trace(Duel(HYDRALISK, source, follow_lock=False), 0.5, 40)
    _same(on, off)


def test_measured_key_inert_when_off() -> None:
    """When disabled, stripping `follow_lock` from static measured data is bit-identical."""

    def strip(m: dict[str, float]) -> dict[str, float]:
        return {n: v for n, v in m.items() if n != 'follow_lock'}

    units = {
        k: dataclasses.replace(s, measured=strip(dict(s.measured))) for k, s in STATIC.units.items()
    }
    bare = StaticData(units=units, source=STATIC.source)
    for unit in LOCK:
        a = _trace(Duel(unit, 'amove', follow_lock=False), 0.5, 30)
        b = _trace(Duel(unit, 'amove', static=bare, follow_lock=False), 0.5, 30)
        _same(a, b)


@pytest.mark.parametrize('source', ['amove', 'idle'])
def test_coexists_with_chase_lock(source: str) -> None:
    """Enabling `chase_lock` (Stalker 8 = L - 1) alongside `follow_lock` is bit-identical."""
    a = _trace(Duel(STALKER, source), 0.5, 40)
    b = _trace(Duel(STALKER, source, chase_lock=((STALKER, 8),)), 0.5, 40)
    _same(a, b)


def test_batch_matches_single() -> None:
    """Batched run (varied d and seeds) matches single-env runs bit-for-bit."""
    ds = [0.5, 0.9, 1.5]
    duel = Duel(STALKER, 'amove', batch=3)
    duel.run_to_fire()
    duel.put(ds)
    for _ in range(40):
        duel.step()
    for b, d in enumerate(ds):
        one = _trace(Duel(STALKER, 'amove', seeds=(b,)), d, 40)
        for s_all, s_one in zip(duel.states, one, strict=True):
            for f in ('x', 'y', 'weapon_cooldown', 'engaged_target_slot', 'health'):
                assert np.array_equal(getattr(s_all, f)[b], getattr(s_one, f)[0], equal_nan=True)
