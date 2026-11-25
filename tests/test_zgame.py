"""Tests for the zgame Rust backend: contract conformance, step-by-step parity with pyref across
default and extended configs, thread/batch invariance, config validation, error paths, and events.

Extension is installed via ``pip install zfoundry-zgame``. Fixtures live inline because pytest
importlib mode cannot import sibling helper modules.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

pytest.importorskip('zgame', reason="requires the optional 'zgame' extension")
from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.zgame import CAPABILITIES, ZgameBackend, capabilities_for, config_options
from unismac.contract import (
    NO_SLOT,
    AgentPlayer,
    Backend,
    CapabilityError,
    Command,
    CommandKind,
    ComputerPlayer,
    ContractError,
    MapInfo,
    Scenario,
    Spawn,
    UnitOverride,
    WorldState,
    conformance,
)

_M1_FIELDS: frozenset[str] = frozenset(
    {
        'commands',
        'command_delay',
        'move_lag',
        'spawn_facing',
        'kinematics',
        'rl_turn',
        'arrival_radius',
        'collision',
        'collision_iterations',
        'contact_slack',
        'idle_push_share',
        'mover_push_share',
        'mover_push_power',
        'mover_push_cap',
        'mover_speed_cap',
        'push_clearance',
        'arrival_clears_waypoint',
        'pin_engaged',
        'pin_after_fire',
        'pin_lag',
        'pin_pair_share',
        'idle_return',
        'return_threshold',
        'return_far',
        'return_delay',
        'idle_return_rule',
        'return_ratio',
        'crowd_arrival',
        'crowd_goal_tol',
        'crowd_progress',
        'crowd_loops',
        'crowd_chain',
        'crowd_cover',
        'native_relation',
        'native_enemy_relation',
        'native_idle_share',
        'native_speed_rule',
        'native_second_pass',
        'reissue_unit',
    }
)

_M2_FIELDS: frozenset[str] = _M1_FIELDS | frozenset(
    {
        'targeting',
        'auto_acquire',
        'auto_acquire_owners',
        'acquire_start_loop',
        'acquire_margin',
        'acquire_min_range',
        'scan_period',
        'scan_phase',
        'retarget',
        'retarget_busy',
        'retarget_busy_lock',
        'retarget_defer',
        'retarget_defer_memory',
        'retarget_ready',
        'retarget_ready_margin',
        'retarget_acquire',
        'retarget_lock',
        'retarget_armed_only',
        'kill_rescan_delay',
        'kill_rescan_lock',
        'kill_rescan_order',
        'scan_after_move',
        'retarget_after_move',
        'hit_assist',
        'hit_assist_range',
        'hit_assist_pick',
        'hit_assist_tie',
        'hit_assist_cap',
        'hit_keeps_kill_wait',
        'acquire_center',
        'kill_hold',
        'report_engaged_as_attack_unit',
        'retreat_owners',
        'retreat_distance',
        'kill_no_retreat',
        'kill_no_retreat_end',
        'passive_from_map',
        'attack',
        'damage',
        'chase_lock',
        'chase_lock_target',
        'follow_lock',
        'follow_slop',
        'slop_on_arrival',
        'attack_commit',
        'attack_on_arrival',
        'charge',
        'amove_needs_weapon',
        'attack_unit_needs_weapon',
        'dead_target_move',
        'slow_arrival',
        'hit_slow_owner1_only',
        'projectile_3d',
        'min_damage',
        'regen_race_fallback',
        'range_radius',
        'visibility',
        'vision_levels',
        'reveal_loops',
        'lost_vision',
        'lost_vision_giveup',
        'abilities',
        'stop_keeps_heal',
        'heal_counts_armed',
        'spawn_energy',
        'heal_autocast',
        'heal_on_arrival',
        'heal_facing',
    }
)

_SUB_PROFILES: dict[str, str] = {'v4': 'aligned-v4', 'axo': 'acc-ram-cov-axo'}


def _profile_subset_options(name: str) -> dict[str, Any]:
    """Return non-default PyrefConfig options for a named subset ('v4', 'axo', '*-m1', '*-m2')."""
    from unismac.profiles import profile_overrides

    if name.endswith('-m1'):
        prof, keep = name[:-3], _M1_FIELDS
    elif name.endswith('-m2'):
        prof, keep = name[:-3], _M2_FIELDS
    else:
        prof, keep = name, None
    _, over = profile_overrides(_SUB_PROFILES[prof])
    out: dict[str, Any] = {}
    for key, value in over.items():
        if key == 'backend.pyref_options':
            out.update(value)
        elif key.startswith('backend.pyref_'):
            out[key.removeprefix('backend.pyref_')] = value
    if keep is not None:
        out = {k: v for k, v in out.items() if k in keep}
    return out


MARINE, MARAUDER, MEDIVAC, ZEALOT, STALKER = 48, 51, 54, 73, 74
MAP_ID = 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {MAP_ID: load_map_info(MAP_ID)}
# Historical tolerance constants (once pyref kinematics.py switched float32 np.arctan2 to C libm
# atan2f, all 24 WorldState fields including facing match bitwise in-process even with AVX-512;
# kept for signature compatibility while primary parity tests use exact=True / facing_atol=0.0).
FACING_ATOL = 1e-5
M1_POS_ATOL = 1e-4
M1_FACING_ATOL = 1e-3


def _walled(info: MapInfo) -> MapInfo:
    """Add a vertical wall (x in [15, 16), y in [6, 22) unpathable) for hard-collision tests."""
    pathing = np.array(info.pathing, copy=True)
    pathing[6:22, 15] = False
    return dataclasses.replace(info, pathing=pathing)


WALL_MAPS = {MAP_ID: _walled(MAPS[MAP_ID])}


def zgame(
    batch_size: int = 1,
    config: object | None = None,
    maps: Mapping[str, MapInfo] | None = None,
    **kw: object,
) -> ZgameBackend:
    return ZgameBackend(STATIC, maps or MAPS, batch_size, config, **kw)  # type: ignore[arg-type]


def skirmish(rng: np.random.Generator, n: int, types: tuple[int, ...] = (MARINE,)) -> Scenario:
    """Spawn n random units per side: Player 1 on the left, Player 2 on the right."""
    spawns = []
    for owner in (1, 2):
        for i in range(n):
            x = (8.0 if owner == 1 else 20.0) + 3.0 * rng.random()
            y = 10.0 + 1.5 * i + rng.random()
            spawns.append(Spawn(owner, int(rng.choice(types)), float(x), float(y)))
    return Scenario(map_id=MAP_ID, spawns=tuple(spawns))


def random_command(rng: np.random.Generator, world: WorldState, p_act: float = 0.3) -> Command:
    """Issue random commands (STOP, MOVE, ATTACK_UNIT, ATTACK_MOVE, HOLD) with prob p_act;
    targets may lie outside the playable area (testing clamping) or hit allies/dead units."""
    b_size, n = world.alive.shape
    cmd = Command.noop(b_size, n)
    for b in range(b_size):
        for u in range(n):
            if not world.alive[b, u] or rng.random() > p_act:
                continue
            kind = CommandKind(int(rng.choice([1, 2, 3, 4, 5, 4, 3])))
            if kind == CommandKind.ATTACK_UNIT:
                t = int(rng.integers(n - 1))
                cmd.set(b, u, kind, target_slot=t + (t >= u))
            elif kind in (CommandKind.MOVE, CommandKind.ATTACK_MOVE):
                xy = (float(rng.uniform(-2.0, 34.0)), float(rng.uniform(-2.0, 34.0)))
                cmd.set(b, u, kind, target_xy=xy)
            else:
                cmd.set(b, u, kind)
    return cmd


def assert_same(a: WorldState, b: WorldState, *, facing_atol: float = 0.0) -> None:
    for f in dataclasses.fields(WorldState):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f.name
        if f.name == 'facing' and facing_atol > 0:
            np.testing.assert_allclose(x, y, rtol=0, atol=facing_atol, err_msg=f.name)
        elif x.dtype.kind == 'f':
            assert np.array_equal(x, y, equal_nan=True), f.name
        else:
            assert np.array_equal(x, y), f.name


# ---------------------------------------------------------------- Contract conformance


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('check', conformance.CHECKS, ids=lambda c: c.__name__)
def test_conformance(batch_size: int, check) -> None:
    backend = zgame(batch_size)
    assert isinstance(backend, Backend)
    check(backend)


def test_capabilities_are_truthful() -> None:
    caps = zgame().capabilities
    assert caps == CAPABILITIES
    assert caps.name == 'zgame' and caps.deterministic and caps.unit_overrides
    assert (caps.pathing, caps.collision, caps.abilities, caps.fog) == (
        'none',
        'none',
        False,
        False,
    )
    assert not (caps.engine_ai or caps.map_trigger or caps.terrain_height or caps.ability_query)
    # Capabilities follow config, matching pyref: collision='hard' reports 'hard'
    hard = PyrefConfig(collision='hard')
    assert zgame(1, hard).capabilities == capabilities_for({'collision': 'hard'})
    assert zgame(1, hard).capabilities.collision == 'hard'
    assert PyrefBackend(STATIC, MAPS, 1, hard).capabilities.collision == 'hard'
    # abilities='heal' enables ABILITY (MedivacHeal), matching pyref
    heal = PyrefConfig(abilities='heal')
    assert zgame(1, heal).capabilities.abilities
    assert PyrefBackend(STATIC, MAPS, 1, heal).capabilities.abilities


# ---------------------------------------------------------------- Default pyref parity


@pytest.mark.parametrize('seed', [0, 1])
def test_matches_pyref_default(seed: int) -> None:
    """Bitwise parity across all fields (including facing) with default pyref under random
    commands. Mixes shields, armor, and attribute bonuses covering chase, focus fire, and death."""
    rng = np.random.default_rng(seed)
    b_size = 3
    scenarios = [
        skirmish(rng, 4, (MARINE, MARINE, MARAUDER, ZEALOT, STALKER)) for _ in range(b_size)
    ]
    pyref = PyrefBackend(STATIC, MAPS, b_size)
    zg = zgame(b_size)
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    for _ in range(40):
        assert_same(wp, wz)
        cmd = random_command(rng, wp)
        wp = pyref.step(cmd, 8)
        wz = zg.step(cmd, 8)
    assert_same(wp, wz)
    assert not wz.alive.all(), 'expected at least one unit death'


def test_matches_pyref_with_smac_portable_config_and_overrides() -> None:
    """smac-portable heal/spawn_energy/heal_autocast options do not change behavior without
    Medivacs; unit_overrides set max stats and spawn units at full health/shield."""
    config = PyrefConfig(abilities='heal', spawn_energy=50.0, heal_autocast=True)
    rng = np.random.default_rng(7)
    base = skirmish(rng, 3, (MARINE, STALKER))
    scenario = dataclasses.replace(
        base,
        unit_overrides=(UnitOverride(owner=2, unit_type=STALKER, health_max=60.0),),
    )
    pyref = PyrefBackend(STATIC, MAPS, 2, config)
    zg = zgame(2, config)
    wp = pyref.reset([scenario, base])
    wz = zg.reset([scenario, base])
    for _ in range(30):
        assert_same(wp, wz)
        cmd = random_command(rng, wp)
        wp, wz = pyref.step(cmd, 8), zg.step(cmd, 8)
    assert_same(wp, wz)


def test_partial_reset_matches_pyref() -> None:
    rng = np.random.default_rng(3)
    scenarios = [skirmish(rng, 3) for _ in range(3)]
    pyref, zg = PyrefBackend(STATIC, MAPS, 3), zgame(3)
    wp, wz = pyref.reset(scenarios), zg.reset(scenarios)
    for t in range(12):
        cmd = random_command(rng, wp)
        wp, wz = pyref.step(cmd, 8), zg.step(cmd, 8)
        if t % 4 == 3:
            ids = [t % 3]
            fresh = [skirmish(rng, 3)]
            wp, wz = pyref.reset(fresh, env_ids=ids), zg.reset(fresh, env_ids=ids)
        assert_same(wp, wz)


# ---------------------------------------------------------------- Movement & collision parity

ALL_TYPES = (MARINE, MARAUDER, ZEALOT, STALKER, MEDIVAC)
# name -> (config, maps, unit_types). v4-m1 / axo-m1 test profile movement/collision subsets;
# the remaining cases exercise non-default option combinations (including ones unused by v4/axo).
M1_CASES: dict[str, tuple[PyrefConfig, dict[str, MapInfo], tuple[int, ...]]] = {
    'v4-m1': (PyrefConfig(**_profile_subset_options('v4-m1')), MAPS, ALL_TYPES),
    'axo-m1': (PyrefConfig(**_profile_subset_options('axo-m1')), MAPS, ALL_TYPES),
    'axo-m1-wall': (PyrefConfig(**_profile_subset_options('axo-m1')), WALL_MAPS, ALL_TYPES),
    'delay2': (PyrefConfig(commands='delayed', command_delay=2), MAPS, (MARINE, MEDIVAC)),
    'accel-radius': (
        PyrefConfig(kinematics='accel', arrival_radius=0.05, spawn_facing='random'),
        MAPS,
        (MARINE, MEDIVAC),
    ),
    'hard-params': (
        PyrefConfig(
            collision='hard',
            collision_iterations=2,
            contact_slack=0.03,
            idle_push_share=0.7,
            mover_push_share='approach',
            mover_push_cap=0.6,
            mover_speed_cap=True,
            pin_engaged=True,
            idle_return=True,
            return_threshold=0.4,
            return_far=2.0,
            return_delay=10,
            crowd_arrival=True,
            crowd_goal_tol=0.8,
            crowd_progress=0.1,
            crowd_loops=3,
            crowd_chain=False,
            crowd_cover=True,
            reissue_unit=True,
        ),
        WALL_MAPS,
        ALL_TYPES,
    ),
}


def assert_close(a: WorldState, b: WorldState, *, pos_fields: tuple[str, ...] = ('x', 'y')) -> None:
    """Approximate check: x/y within M1_POS_ATOL, facing angle within M1_FACING_ATOL, rest
    bitwise. pos_fields lists position-derived fields (extended with order_x/order_y later)."""
    for f in dataclasses.fields(WorldState):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f.name
        if f.name in pos_fields:
            np.testing.assert_allclose(
                x, y, rtol=0, atol=M1_POS_ATOL, equal_nan=True, err_msg=f.name
            )
        elif f.name == 'facing':
            d = np.remainder(x.astype(np.float64) - y + math.pi, 2 * math.pi) - math.pi
            assert np.nanmax(np.abs(d), initial=0.0) <= M1_FACING_ATOL, f.name
        elif x.dtype.kind == 'f':
            assert np.array_equal(x, y, equal_nan=True), f.name
        else:
            assert np.array_equal(x, y), f.name


def _compare_m1(name: str, *, exact: bool, steps: int = 30, n_loops: int = 4) -> None:
    """M1_CASES[name]: 3 envs of 4v4 mixed units compared each step; bitwise when exact=True."""
    config, maps, types = M1_CASES[name]
    rng = np.random.default_rng(sorted(M1_CASES).index(name))
    b_size = 3
    scenarios = [skirmish(rng, 4, types) for _ in range(b_size)]
    pyref = PyrefBackend(STATIC, maps, b_size, config)
    zg = zgame(b_size, config, maps)
    check = assert_same if exact else assert_close
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    for _ in range(steps):
        check(wp, wz)
        cmd = random_command(rng, wp)
        wp, wz = pyref.step(cmd, n_loops), zg.step(cmd, n_loops)
    check(wp, wz)


@pytest.mark.parametrize('name', sorted(M1_CASES))
def test_matches_pyref_m1(name: str) -> None:
    """Movement/collision options match pyref bitwise on all fields (including facing)."""
    _compare_m1(name, exact=True)


def test_m1_wall_projects_like_pyref() -> None:
    """Hard collision near walls projects units out of unpathable cells identically to pyref."""
    config = PyrefConfig(collision='hard')
    spawns = (
        Spawn(1, MARINE, 13.0, 12.0),
        Spawn(1, MARINE, 13.0, 12.6),
        Spawn(1, ZEALOT, 13.4, 13.2),
        Spawn(2, MARINE, 25.0, 28.0),
    )
    scenario = Scenario(map_id=MAP_ID, spawns=spawns)
    pyref = PyrefBackend(STATIC, WALL_MAPS, 1, config)
    zg = zgame(1, config, WALL_MAPS)
    wp, wz = pyref.reset([scenario]), zg.reset([scenario])
    cmd = Command.noop(1, 4)
    for u in range(3):
        cmd.set(0, u, CommandKind.MOVE, target_xy=(20.0, 12.5 + 0.3 * u))
    wp, wz = pyref.step(cmd, 1), zg.step(cmd, 1)
    for _ in range(40):
        assert_same(wp, wz)
        wp, wz = pyref.step(Command.noop(1, 4), 2), zg.step(Command.noop(1, 4), 2)
    assert_same(wp, wz)
    # Straight-line movement into wall is projected back, stopping along the left edge
    assert (wz.x[0, :3] < 15.0).all() and (wz.x[0, :3] > 14.0).all()


def test_m1_crowd_cover_and_reissue_like_pyref() -> None:
    """crowd_cover clears MOVE when blocked by a HOLD ally; reissue_unit preserves state when
    re-issuing ATTACK_UNIT to the same target. Both match pyref bitwise."""
    config = PyrefConfig(**_profile_subset_options('axo-m1'))
    spawns = (
        Spawn(1, MARINE, 16.0, 16.0),
        Spawn(1, MARINE, 15.3, 16.0),
        Spawn(1, STALKER, 12.0, 20.0),
        Spawn(2, ZEALOT, 19.0, 21.0),
    )
    scenario = Scenario(map_id=MAP_ID, spawns=spawns)
    pyref = PyrefBackend(STATIC, MAPS, 1, config)
    zg = zgame(1, config)
    wp, wz = pyref.reset([scenario], seeds=[3]), zg.reset([scenario], seeds=[3])
    cmd = Command.noop(1, 4).set(0, 0, CommandKind.HOLD)
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(16.05, 16.0))
    cmd.set(0, 2, CommandKind.ATTACK_UNIT, target_slot=3)
    covered = False
    for t in range(30):
        wp, wz = pyref.step(cmd, 1), zg.step(cmd, 1)
        assert_same(wp, wz)
        covered |= bool(wp.order_kind[0, 1] == CommandKind.NOOP and wp.x[0, 1] < 16.0)
        # Re-issue ATTACK_UNIT to the same target every loop
        cmd = Command.noop(1, 4).set(0, 2, CommandKind.ATTACK_UNIT, target_slot=3)
        if t == 10:
            cmd.set(0, 3, CommandKind.MOVE, target_xy=(10.0, 21.0))
    assert covered, 'MOVE should clear when target point is occupied'


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_m1_bitwise_without_avx512() -> None:
    """With NumPy AVX-512 disabled (set in a subprocess before import so float32 arctan2 uses
    scalar libm), all M1_CASES match pyref bitwise."""
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        'for n in sorted(m["M1_CASES"]):\n'
        '    m["_compare_m1"](n, exact=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- Targeting & combat parity

# Targeting, retargeting, and attack fields (damage and heal tested separately)
SC2_TARGETING_FIELDS = frozenset({
    'targeting', 'attack', 'auto_acquire', 'auto_acquire_owners', 'acquire_start_loop',
    'scan_phase', 'retarget', 'retarget_busy_lock', 'retarget_acquire', 'retarget_lock',
    'retarget_armed_only', 'kill_rescan_lock', 'scan_after_move', 'retarget_after_move',
    'kill_hold', 'report_engaged_as_attack_unit', 'retreat_owners', 'kill_no_retreat',
    'kill_no_retreat_end', 'chase_lock', 'chase_lock_target', 'follow_lock', 'slop_on_arrival',
    'attack_on_arrival', 'amove_needs_weapon', 'attack_unit_needs_weapon', 'dead_target_move',
})  # fmt: skip


# Combat fields for damage='sc2' and its dependents
SC2_DAMAGE_FIELDS = frozenset({'damage', 'attack_commit', 'slow_arrival', 'hit_slow_owner1_only'})

M2_POS_FIELDS = ('x', 'y', 'order_x', 'order_y')


def _m2_sub(profile: str, fields: frozenset[str]) -> PyrefConfig:
    """Combine a profile's movement/collision subset with its non-default options in fields."""
    opts = _profile_subset_options(f'{profile}-m1')
    opts.update({k: v for k, v in _profile_subset_options(profile).items() if k in fields})
    return PyrefConfig(**opts)


def _sc2_targeting(profile: str) -> PyrefConfig:
    """Targeting and attack options (with standard damage)."""
    return _m2_sub(profile, SC2_TARGETING_FIELDS)


def _sc2_damage(profile: str) -> PyrefConfig:
    """Add damage='sc2' and its dependent combat options."""
    return _m2_sub(profile, SC2_TARGETING_FIELDS | SC2_DAMAGE_FIELDS)


# name -> (config, maps, unit_types); sc2-variants tests supported options unused by v4/axo
M2_CASES: dict[str, tuple[PyrefConfig, dict[str, MapInfo], tuple[int, ...]]] = {
    'sc2-v4': (_sc2_targeting('v4'), MAPS, ALL_TYPES),
    'sc2-axo': (_sc2_targeting('axo'), MAPS, ALL_TYPES),
    'sc2-variants': (
        PyrefConfig(
            targeting='sc2',
            attack='sc2',
            auto_acquire=True,
            acquire_start_loop=3,
            acquire_margin=1.0,
            acquire_min_range=3.0,
            scan_period=4,
            retarget=True,
            retarget_busy=5,
            kill_rescan_lock=True,
            kill_hold=True,
            retreat_owners=(1,),
            retreat_distance=3.0,
            kill_no_retreat=True,
            chase_lock=((MARINE, 6), (ZEALOT, 4)),
            follow_lock=True,
            follow_slop=0.5,
            attack_on_arrival=True,
            dead_target_move=True,
            collision='hard',
        ),
        MAPS,
        ALL_TYPES,
    ),
    'dmg-v4': (_sc2_damage('v4'), MAPS, ALL_TYPES),
    'dmg-axo': (_sc2_damage('axo'), MAPS, ALL_TYPES),
    'dmg-variants': (
        PyrefConfig(attack='sc2', damage='sc2', attack_commit=True, slow_arrival=True),
        MAPS,
        ALL_TYPES,
    ),
}


@contextmanager
def _pyref_events() -> Iterator[tuple[list[tuple], list[tuple]]]:
    """Wrap pyref `damage._hit` and `damage._fire` (without modifying pyref) to record each hit
    (env, loop, attacker, target, shield_loss, hp_loss, lethal) and fire (env, loop, u, t)."""
    from unismac.backends.pyref.subsystems import damage

    hits: list[tuple] = []
    fires: list[tuple] = []
    hit0, fire0 = damage._hit, damage._fire

    def hit(sim, b, u, t, wi, n, min_damage):  # noqa: ANN001, ANN202
        w = sim.world
        s0, h0 = w.shield[b, t], w.health[b, t]
        hit0(sim, b, u, t, wi, n, min_damage)
        s1, h1 = w.shield[b, t], w.health[b, t]
        loop = int(w.game_loop[b])
        hits.append((b, loop, u, t, float(s0 - s1), float(h0 - h1), bool(h0 > 0 and h1 <= 0)))

    def fire(sim, b, u, t, loop):  # noqa: ANN001, ANN202
        fires.append((b, loop, u, t))
        fire0(sim, b, u, t, loop)

    damage._hit, damage._fire = hit, fire
    try:
        yield hits, fires
    finally:
        damage._hit, damage._fire = hit0, fire0


def _check_events(
    ev: dict[str, np.ndarray], hits: list[tuple], fires: list[tuple], sc2: bool, b_size: int
) -> None:
    """Verify zgame damage, death (with killer), and fire events match pyref per step."""
    kind = ev['kind']
    for b in range(b_size):
        m = ev['env'] == b
        dmg = [
            (int(lp), int(a), int(v), float(s), float(h))
            for lp, a, v, s, h in zip(
                ev['game_loop'][m & (kind == 2)],
                ev['other'][m & (kind == 2)],
                ev['unit'][m & (kind == 2)],
                ev['f0'][m & (kind == 2)],
                ev['f1'][m & (kind == 2)],
                strict=True,
            )
        ]
        want = [(lp, u, t, s, h) for bb, lp, u, t, s, h, _ in hits if bb == b]
        assert dmg == want, f'env {b} damage events'
        deaths = [
            (int(lp), int(v), int(k))
            for lp, v, k in zip(
                ev['game_loop'][m & (kind == 3)],
                ev['unit'][m & (kind == 3)],
                ev['other'][m & (kind == 3)],
                strict=True,
            )
        ]
        # First attacker reducing HP from > 0 to <= 0 on the same loop; deaths ordered by slot
        killers: dict[tuple[int, int], int] = {}
        for bb, lp, u, t, _, _, killed in hits:
            if bb == b and killed:
                killers.setdefault((lp, t), u)
        assert deaths == sorted((lp, t, u) for (lp, t), u in killers.items()), f'env {b} deaths'
        fired = [
            (int(lp), int(a), int(t))
            for lp, a, t in zip(
                ev['game_loop'][m & (kind == 1)],
                ev['unit'][m & (kind == 1)],
                ev['other'][m & (kind == 1)],
                strict=True,
            )
        ]
        src = fires if sc2 else [(bb, lp, u, t) for bb, lp, u, t, *_ in hits]
        assert fired == [(lp, u, t) for bb, lp, u, t in src if bb == b], f'env {b} fires'


def _compare_m2(name: str, *, exact: bool, steps: int = 40, n_loops: int = 4) -> None:
    """M2_CASES[name]: 3 envs of 4v4 mixed units. Player 1 opens with ATTACK_MOVE while Player 2
    idles (exercising auto-acquire, retaliation, and retreat), then random commands (p=0.1);
    compares state and event streams (damage, deaths with killers, fires) each step."""
    config, maps, types = M2_CASES[name]
    rng = np.random.default_rng(100 + sorted(M2_CASES).index(name))
    b_size = 3
    scenarios = [skirmish(rng, 4, types) for _ in range(b_size)]
    pyref = PyrefBackend(STATIC, maps, b_size, config)
    zg = zgame(b_size, config, maps)

    def check(a: WorldState, b: WorldState) -> None:
        # In-process: leash and retreat targets derive from positions (ulp diffs enter order_x/y)
        if exact:
            assert_same(a, b)
        else:
            assert_close(a, b, pos_fields=M2_POS_FIELDS)

    sc2 = config.damage == 'sc2'
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    cmd = Command.noop(b_size, 8)
    for b in range(b_size):
        for u in range(4):
            cmd.set(b, u, CommandKind.ATTACK_MOVE, target_xy=(24.0, 13.0))
    with _pyref_events() as (hits, fires):
        for _ in range(steps):
            check(wp, wz)
            hits.clear()
            fires.clear()
            wp, wz = pyref.step(cmd, n_loops), zg.step(cmd, n_loops)
            _check_events(zg.events(), hits, fires, sc2, b_size)
            cmd = random_command(rng, wp, 0.1)
    check(wp, wz)
    assert not wz.alive.all(), 'expected at least one unit death'


@pytest.mark.parametrize('name', sorted(M2_CASES))
def test_matches_pyref_m2(name: str) -> None:
    """Targeting and combat options match pyref bitwise on all fields (including facing)."""
    _compare_m2(name, exact=True)


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_m2_bitwise_without_avx512() -> None:
    """With NumPy AVX-512 disabled in a subprocess, all M2_CASES match pyref bitwise."""
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        'for n in sorted(m["M2_CASES"]):\n'
        '    m["_compare_m2"](n, exact=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- Thread & batch invariance


def _trajectory(
    scenarios: list[Scenario],
    seeds: list[int],
    commands: list[Command],
    threads: int,
    config: object | None = None,
    maps: Mapping[str, MapInfo] | None = None,
) -> list[WorldState]:
    zg = zgame(len(scenarios), config, maps, threads=threads, min_parallel=1)
    out = [zg.reset(scenarios, seeds=seeds)]
    out += [zg.step(cmd, 8) for cmd in commands]
    return out


def _rows(x: object, i: int) -> object:
    """Slice env i from WorldState / Command while keeping the batch dimension."""
    return type(x)(**{f.name: getattr(x, f.name)[i : i + 1] for f in dataclasses.fields(x)})  # type: ignore[arg-type]


@pytest.mark.parametrize('case', ['default', 'axo-m1-wall', 'hard-params', 'dmg-axo'])
def test_r1_threads_and_batch_size(case: str) -> None:
    """Same scenarios and commands match bitwise across 1/2/4 threads and batched vs single-env
    execution (default, movement/collision, and targeting/combat configs)."""
    config, maps, types = (
        (None, MAPS, (MARINE, ZEALOT, STALKER))
        if case == 'default'
        else {**M1_CASES, **M2_CASES}[case]
    )
    rng = np.random.default_rng(11)
    b_size = 6
    scenarios = [skirmish(rng, 4, types) for _ in range(b_size)]
    seeds = list(range(b_size))
    # Generate command trace once on 1 thread, then replay under all thread/batch setups
    zg = zgame(b_size, config, maps)
    world = zg.reset(scenarios, seeds=seeds)
    commands = []
    for _ in range(25):
        commands.append(random_command(rng, world))
        world = zg.step(commands[-1], 8)
    base = _trajectory(scenarios, seeds, commands, 1, config, maps)
    assert_same(base[-1], world)
    for threads in (2, 4):
        traj = _trajectory(scenarios, seeds, commands, threads, config, maps)
        for a, b in zip(base, traj, strict=True):
            assert_same(a, b)
    for i in range(b_size):
        single = [_rows(c, i) for c in commands]
        one = _trajectory([scenarios[i]], [seeds[i]], single, 1, config, maps)  # type: ignore[arg-type]
        for a, b in zip(base, one, strict=True):
            assert_same(_rows(a, i), b)  # type: ignore[arg-type]


def test_same_seed_same_result() -> None:
    rng = np.random.default_rng(5)
    scenarios = [skirmish(rng, 3) for _ in range(2)]
    cmd_rng = np.random.default_rng(9)
    first = zgame(2)
    w = first.reset(scenarios, seeds=[1, 2])
    cmds = []
    for _ in range(10):
        cmds.append(random_command(cmd_rng, w))
        w = first.step(cmds[-1], 8)
    second = zgame(2, threads=2, min_parallel=1)
    v = second.reset(scenarios, seeds=[1, 2])
    for c in cmds:
        v = second.step(c, 8)
    assert_same(w, v)


# ---------------------------------------------------------------- Config validation


@pytest.mark.parametrize(
    'config',
    [
        {'attack': 'melee'},
        PyrefConfig(auto_acquire=True),
        PyrefConfig(targeting='sc2', retarget=True, retarget_ready=True),
        PyrefConfig(targeting='sc2', retarget=True, retarget_defer='switch'),
        PyrefConfig(
            targeting='sc2', retarget=True, retarget_defer='switch', retarget_defer_memory=True
        ),
        PyrefConfig(pathing='grid'),
        PyrefConfig(steering='separation'),
        PyrefConfig(steering='sc2_avoid', avoid_closing_only=True),
        PyrefConfig(steering='sc2_avoid', avoid_share='speed', avoid_share_power_headon=2.0),
        PyrefConfig(collision='hard', pin_engaged=True, pin_after_fire=True),
        PyrefConfig(collision='hard', pin_engaged=True, pin_slide=True, pin_slide_mode='pinned'),
        PyrefConfig(
            steering='sc2_avoid', pin_engaged=True, contact_orbit=True, orbit_movers='right'
        ),
        PyrefConfig(steering='sc2_avoid', pin_engaged=True, contact_orbit=True, orbit_from='prev'),
        PyrefConfig(
            steering='sc2_avoid', pin_engaged=True, pin_sidestep=True, pin_sidestep_lat=0.3
        ),
        PyrefConfig(collision='hard', mover_push_share='speed'),
        PyrefConfig(collision='hard', mover_push_power=8.0),
        PyrefConfig(collision='hard', idle_return=True, idle_return_rule='diameter'),
        # Unported heuristic options (not implemented in zgame)
        PyrefConfig(collision='hard', pathing='navmesh', push_clearance=True),
        PyrefConfig(collision='hard', arrival_clears_waypoint=True),
        PyrefConfig(collision='hard', pin_engaged=True, pin_pair_share='younger'),
        PyrefConfig(kinematics='accel', rl_turn='curve'),
        PyrefConfig(min_damage=1.0),
        {'no_such_field': 1},
    ],
    ids=lambda c: str(config_options(c))[:50],
)
def test_non_default_config_raises(config: object) -> None:
    with pytest.raises(CapabilityError):
        zgame(1, config)


def test_m12_config_validation() -> None:
    """Native combat options are accepted or raise ValueError per PyrefConfig validation rules."""
    with pytest.raises(ValueError, match='targeting'):
        zgame(1, {'kill_rescan_delay': 0})
    with pytest.raises(ValueError, match='scan_period'):
        zgame(1, {'targeting': 'sc2', 'scan_phase': 'tag_order', 'scan_period': 16})
    with pytest.raises(ValueError, match='attack'):
        zgame(1, {'targeting': 'sc2', 'damage': 'sc2', 'native_backswing': True})
    assert zgame(1, PyrefConfig(targeting='sc2', scan_phase='tag_order', scan_period=8)) is not None


def test_config_options_diff_against_defaults() -> None:
    assert config_options(None) == {}
    assert config_options(PyrefConfig()) == {}
    assert config_options(PyrefConfig(abilities='heal', spawn_energy=50.0)) == {
        'abilities': 'heal',
        'spawn_energy': 50.0,
    }
    assert config_options({'attack': 'instant'}) == {'attack': 'instant'}
    zgame(1, {'attack': 'instant'})  # explicit default impl name is accepted
    with pytest.raises(ValueError, match='spawn_energy'):
        zgame(1, {'spawn_energy': -1.0})
    with pytest.raises(TypeError):
        zgame(1, 'immediate')
    assert zgame(1).impls['kinematics'] == 'constant_speed'


# Heal & energy fields (abilities='heal' enabled by smac-portable preset)
HEAL_FIELDS = frozenset({'stop_keeps_heal', 'heal_counts_armed', 'heal_on_arrival', 'heal_facing'})
MMM = (MARINE, MARAUDER, MEDIVAC)


def _heal_config(profile: str, autocast: bool = True, heal_pick: str = 'life') -> PyrefConfig:
    """Combine profile's movement, targeting, combat, and heal fields plus smac-portable heal
    settings; heal_pick selects the autocast target criterion."""
    sub = _m2_sub(profile, SC2_TARGETING_FIELDS | SC2_DAMAGE_FIELDS | HEAL_FIELDS)
    return dataclasses.replace(
        sub, abilities='heal', spawn_energy=50.0, heal_autocast=autocast, heal_pick=heal_pick
    )


def _heal_command(rng: np.random.Generator, world: WorldState, p_heal: float) -> Command:
    """In addition to random_command (p=0.2), alive Medivacs issue heal with prob p_heal to a
    random target (enemy, full-HP, or air targets are ignored as in SC2)."""
    cmd = random_command(rng, world, 0.2)
    b_size, n = world.alive.shape
    for b in range(b_size):
        for u in range(n):
            if world.alive[b, u] and world.unit_type[b, u] == MEDIVAC and rng.random() < p_heal:
                t = int(rng.integers(n - 1))
                cmd.set(b, u, CommandKind.ABILITY, target_slot=t + (t >= u), ability=386)
    return cmd


def _compare_heal(
    profile: str, *, exact: bool, autocast: bool = True, steps: int = 50, heal_pick: str = 'life'
) -> None:
    """3 envs of MMM: Player 1 opens with ATTACK_MOVE, then random + heal commands; compares
    state and event stream each step and requires at least one unit to actually be healed."""
    config = _heal_config(profile, autocast, heal_pick)
    rng = np.random.default_rng(200 + ('v4', 'axo').index(profile) + 2 * autocast)
    b_size = 3
    scenarios = [skirmish(rng, 4, MMM) for _ in range(b_size)]
    pyref = PyrefBackend(STATIC, MAPS, b_size, config)
    zg = zgame(b_size, config)
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = zg.reset(scenarios, seeds=list(range(b_size)))
    cmd = Command.noop(b_size, 8)
    for b in range(b_size):
        for u in range(4):
            cmd.set(b, u, CommandKind.ATTACK_MOVE, target_xy=(24.0, 13.0))
    healed = False
    with _pyref_events() as (hits, fires):
        for _ in range(steps):
            if exact:
                assert_same(wp, wz)
            else:
                assert_close(wp, wz, pos_fields=M2_POS_FIELDS)
            hits.clear()
            fires.clear()
            before = wz.health.copy()
            wp, wz = pyref.step(cmd, 4), zg.step(cmd, 4)
            _check_events(zg.events(), hits, fires, True, b_size)
            healed |= bool((wz.alive & (wz.health > before)).any())
            cmd = _heal_command(rng, wp, 0.3)
    assert healed, 'expected at least one unit to be healed'


def _compare_heal_pick(heal_pick: str, *, exact: bool) -> int:
    """Autocast target-selection test: Player 1 Marine A (slot 0), Marauder B (slot 1), distant
    Medivac (slot 2); Player 2 Marine (slot 3) shoots A once and B four times then retreats, and
    Medivac flies between them to autocast. A has lower HP (39 vs 105) while B has lower HP
    fraction (0.84 vs 0.867). Compares state/events per loop and returns first healed slot."""
    config = _heal_config('axo', True, heal_pick)
    spawns = (
        Spawn(1, MARINE, 10.0, 16.0),
        Spawn(1, MARAUDER, 10.0, 18.0),
        Spawn(1, MEDIVAC, 2.0, 30.0),
        Spawn(2, MARINE, 15.0, 17.0),
    )
    scenario = Scenario(map_id=MAP_ID, spawns=spawns)
    pyref = PyrefBackend(STATIC, MAPS, 1, config)
    zg = zgame(1, config)
    wp = pyref.reset([scenario], seeds=[3])
    wz = zg.reset([scenario], seeds=[3])
    phase, pick = 0, None
    with _pyref_events() as (hits, fires):
        for _ in range(400):
            if exact:
                assert_same(wp, wz)
            else:
                assert_close(wp, wz, pos_fields=M2_POS_FIELDS)
            if wp.order_kind[0, 2] == CommandKind.ABILITY:
                pick = int(wp.order_target_slot[0, 2])
                h = wp.health[0, :2]
                assert h[0] < h[1] and h[0] / 45.0 > h[1] / 125.0, h
                break
            cmd = Command.noop(1, 4)
            if phase == 0 and wp.health[0, 0] < 45.0:
                phase = 1
            if phase == 1 and wp.health[0, 1] <= 105.0:
                phase = 2
                cmd.set(0, 3, CommandKind.MOVE, target_xy=(30.0, 2.0))
                mid = (float(wp.x[0, :2].mean()), float(wp.y[0, :2].mean()))
                cmd.set(0, 2, CommandKind.MOVE, target_xy=mid)
            elif phase < 2:
                cmd.set(0, 3, CommandKind.ATTACK_UNIT, target_slot=phase)
            hits.clear()
            fires.clear()
            wp, wz = pyref.step(cmd, 1), zg.step(cmd, 1)
            _check_events(zg.events(), hits, fires, True, 1)
    assert pick is not None, 'Medivac did not autocast heal'
    return pick


@pytest.mark.parametrize('autocast', [True, False])
@pytest.mark.parametrize('profile', ['v4', 'axo'])
def test_heal_matches_pyref(profile: str, autocast: bool) -> None:
    """Medivac heal, autocast, energy regen, and heal options match pyref bitwise on MMM."""
    _compare_heal(profile, exact=True, autocast=autocast)


def test_heal_pick_matches_pyref() -> None:
    """Both heal_pick modes match pyref bitwise and select different targets ('life' picks
    lower-HP A, 'fraction' picks lower-fraction B); random MMM with 'fraction' also matches."""
    assert _compare_heal_pick('life', exact=True) == 0
    assert _compare_heal_pick('fraction', exact=True) == 1
    _compare_heal('axo', exact=True, heal_pick='fraction')


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_heal_bitwise_without_avx512() -> None:
    """With NumPy AVX-512 disabled in a subprocess, all heal tests (incl. heal_pick='fraction')
    and race-fallback regen match pyref bitwise."""
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        'for p in ("v4", "axo"):\n'
        '    for a in (True, False):\n'
        '        m["_compare_heal"](p, exact=True, autocast=a)\n'
        'm["_compare_heal"]("axo", exact=True, heal_pick="fraction")\n'
        'for k in ("life", "fraction"):\n'
        '    m["_compare_heal_pick"](k, exact=True)\n'
        'm["_compare_regen"](exact=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- Race-fallback regen

HYDRALISK, ZERGLING = 107, 105
RL = 1  # SMAC _RL variants (Zealot_RL, Zergling_RL) also lack measured regen entries


def _regen_scenarios() -> list[Scenario]:
    """Player 1 has 4 Marines; Player 2 has Stalker_RL, Zealot_RL, Hydralisk, and Zergling_RL
    without measured regen entries (reversed order in env 1)."""
    p2 = [(STALKER, RL), (ZEALOT, RL), (HYDRALISK, 0), (ZERGLING, RL)]
    out = []
    for order in (p2, p2[::-1]):
        spawns = [Spawn(1, MARINE, 8.0, 11.0 + 2.0 * i) for i in range(4)]
        spawns += [Spawn(2, t, 14.0, 11.0 + 2.0 * i, v) for i, (t, v) in enumerate(order)]
        out.append(Scenario(map_id=MAP_ID, spawns=tuple(spawns)))
    return out


def _compare_regen(*, exact: bool, fallback: bool = True, other: str = 'zgame') -> WorldState:
    """dmg-axo with `regen_race_fallback`: Player 1 Marines attack for 40 loops then retreat
    left while Player 2 holds; then 240 idle loops (shields wait 160 loops, Zerg HP regens
    continuously). Compares state and events each step and returns final WorldState (or final
    fallback=False pyref state when other='pyref')."""
    config = dataclasses.replace(_sc2_damage('axo'), regen_race_fallback=fallback)
    scenarios = _regen_scenarios()
    b_size = len(scenarios)
    pyref = PyrefBackend(STATIC, MAPS, b_size, config)
    if other == 'pyref':
        cmp: Backend = PyrefBackend(
            STATIC, MAPS, b_size, dataclasses.replace(config, regen_race_fallback=False)
        )
    else:
        cmp = zgame(b_size, config)
    wp = pyref.reset(scenarios, seeds=list(range(b_size)))
    wz = cmp.reset(scenarios, seeds=list(range(b_size)))
    attack = Command.noop(b_size, 8)
    retreat = Command.noop(b_size, 8)
    for b in range(b_size):
        for u in range(4):
            attack.set(b, u, CommandKind.ATTACK_UNIT, target_slot=4 + u)
            retreat.set(b, u, CommandKind.MOVE, target_xy=(1.0, 11.0 + 2.0 * u))
            retreat.set(b, 4 + u, CommandKind.HOLD)
    plan = [attack] + [Command.noop(b_size, 8)] * 9 + [retreat] + [Command.noop(b_size, 8)] * 60
    with _pyref_events() as (hits, fires):
        for cmd in plan:
            if other == 'zgame':
                if exact:
                    assert_same(wp, wz)
                else:
                    assert_close(wp, wz, pos_fields=M2_POS_FIELDS)
            hits.clear()
            fires.clear()
            wp, wz = pyref.step(cmd, 4), cmp.step(cmd, 4)
            if other == 'zgame':
                _check_events(cmp.events(), hits, fires, True, b_size)  # type: ignore[attr-defined]
    if other == 'zgame':
        (assert_same if exact else assert_close)(wp, wz)  # type: ignore[operator]
        return wp
    return wz


@pytest.mark.parametrize('backend_cls', [PyrefBackend, ZgameBackend])
@pytest.mark.parametrize('enabled', [False, True])
def test_stalker_shield_regen(backend_cls, enabled: bool) -> None:
    """After taking a shot and ceasing fire, both Stalkers regen shield only when enabled."""
    config = PyrefConfig(attack='sc2', damage='sc2', stalker_shield_regen=enabled)
    backend = backend_cls(STATIC, MAPS, 3, config)
    targets = [(STALKER, 0), (STALKER, RL), (ZEALOT, RL)]
    backend.reset(
        [
            Scenario(MAP_ID, spawns=(Spawn(1, MARINE, 10, 16), Spawn(2, unit, 12, 16, variant)))
            for unit, variant in targets
        ]
    )
    attack, stop, noop = (Command.noop(3, 2) for _ in range(3))
    for b in range(3):
        attack.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=1)
        stop.set(b, 0, CommandKind.STOP)
    hit = backend.step(attack, 2)
    shield = hit.shield[:, 1]
    np.testing.assert_array_equal(shield, hit.shield_max[:, 1] - 6)
    np.testing.assert_array_equal(backend.step(stop, 159).shield[:, 1], shield)
    rate = np.array([0.125, 0.125, 0.0]) * enabled
    np.testing.assert_array_equal(backend.step(noop, 1).shield[:, 1], shield + rate)
    np.testing.assert_array_equal(backend.step(noop, 1).shield[:, 1], shield + 2 * rate)


def test_regen_race_fallback_matches_pyref() -> None:
    """`regen_race_fallback` matches pyref bitwise; compared to disabled, Player 2 units show
    higher shield and Zerg HP (verifying both shield and Zerg HP fallback paths)."""
    on = _compare_regen(exact=True)
    off = _compare_regen(exact=False, other='pyref')
    alive = on.alive[:, 4:] & off.alive[:, 4:]
    assert (alive & (on.shield[:, 4:] > off.shield[:, 4:])).any(), 'expected shield regen'
    assert (alive & (on.health[:, 4:] > off.health[:, 4:])).any(), 'expected Zerg HP regen'


def test_abilities_none_keeps_energy() -> None:
    """With abilities='none', Medivac skips heal and energy regen while keeping spawn_energy."""
    medivac = Scenario(
        map_id=MAP_ID,
        spawns=(
            Spawn(1, MARINE, 10.0, 16.0),
            Spawn(1, MEDIVAC, 9.0, 16.0),
            Spawn(2, MARINE, 18.0, 16.0),
        ),
    )
    zg = zgame(1, PyrefConfig(spawn_energy=50.0))
    world = zg.reset([medivac])
    assert world.energy[0, 1] == 50.0
    assert zg.step(Command.noop(1, 3), 16).energy[0, 1] == 50.0
    world = zgame(1, PyrefConfig(abilities='heal', spawn_energy=50.0)).reset([medivac])
    assert world.energy[0, 1] == 50.0


# ---------------------------------------------------------------- Error paths


def test_contract_errors() -> None:
    zg = zgame(2)
    with pytest.raises(ContractError, match='reset'):
        zg.step(Command.noop(2, 3), 8)
    with pytest.raises(ContractError):
        ZgameBackend(STATIC, MAPS, 0)
    zg.reset([conformance.SCENARIO] * 2)
    with pytest.raises(ContractError, match='n_loops'):
        zg.step(Command.noop(2, 3), 0)
    with pytest.raises(ContractError):
        zg.reset([conformance.SCENARIO] * 2, env_ids=[0, 0])
    with pytest.raises(ContractError):
        zg.reset([conformance.SCENARIO], env_ids=[2])
    with pytest.raises(ContractError, match='Unknown map'):
        zg.reset([dataclasses.replace(conformance.SCENARIO, map_id='nope')] * 2)
    far = dataclasses.replace(conformance.SCENARIO, spawns=(Spawn(1, MARINE, 99.0, 16.0),))
    with pytest.raises(ContractError):
        zg.reset([far, far])
    four = dataclasses.replace(conformance.SCENARIO, n_slots=4)
    with pytest.raises(ContractError, match='Partial reset|partial reset'):
        zg.reset([four], env_ids=[1])
    assert zg.reset([four, four]).n_slots == 4  # full reset may change slot count
    cmd = Command.noop(2, 4).set(1, 0, CommandKind.ATTACK_UNIT, target_slot=0)
    with pytest.raises(ContractError, match='env 1'):
        zg.step(cmd, 1)
    cmd = Command.noop(2, 4).set(0, 3, CommandKind.MOVE, target_xy=(1.0, 1.0))
    with pytest.raises(ContractError):
        zg.step(cmd, 1)


def test_capability_errors() -> None:
    zg = zgame(1)
    computer = dataclasses.replace(
        conformance.SCENARIO, players=(AgentPlayer(), ComputerPlayer('terran'))
    )
    with pytest.raises(CapabilityError, match='ComputerPlayer'):
        zg.reset([computer])
    trigger = Scenario(map_id=MAP_ID, spawn_mode='map_trigger', n_slots=4)
    with pytest.raises(CapabilityError, match='map_trigger'):
        zg.reset([trigger])
    zg.reset([conformance.SCENARIO])
    cmd = Command.noop(1, 3).set(0, 0, CommandKind.ABILITY, target_slot=1, ability=386)
    with pytest.raises(CapabilityError, match='ABILITY'):
        zg.step(cmd, 1)
    # abilities='heal' supports only MedivacHeal (386)
    zg = zgame(1, PyrefConfig(abilities='heal'))
    zg.reset([conformance.SCENARIO])
    zg.step(cmd, 1)  # non-Medivac caster is ignored as in SC2
    cmd = Command.noop(1, 3).set(0, 0, CommandKind.ABILITY, target_slot=1, ability=999)
    with pytest.raises(CapabilityError, match='999'):
        zg.step(cmd, 1)


# ---------------------------------------------------------------- Events


def test_kill_produces_death_event_with_killer() -> None:
    zg = zgame(1)
    zg.reset([conformance.SCENARIO])
    cmd = Command.noop(1, 3)
    cmd.set(0, 0, CommandKind.ATTACK_UNIT, target_slot=2)
    cmd.set(0, 1, CommandKind.ATTACK_UNIT, target_slot=2)
    world = zg.step(cmd, 8)
    ev = zg.events()
    assert ev['kind'].tolist() == [0, 0]  # both commands took effect
    assert ev['code'].tolist() == [int(CommandKind.ATTACK_UNIT)] * 2
    deaths: list[tuple[int, int]] = []
    fires, damage = 0, 0.0
    while True:
        # Columnar events: kind 0=command, 1=fire, 2=damage (unit=victim, other=source, f1=hp
        # loss), 3=death (unit=victim, other=killer)
        fires += int((ev['kind'] == 1).sum())
        damage += float(ev['f1'][ev['kind'] == 2].sum())
        deaths += [
            (int(ev['unit'][i]), int(ev['other'][i])) for i in np.flatnonzero(ev['kind'] == 3)
        ]
        if not world.alive[0, 2] or world.game_loop[0] > 40 * 8:
            break
        world = zg.step(Command.noop(1, 3), 8)
        ev = zg.events()
    assert not world.alive[0, 2]
    assert len(deaths) == 1 and deaths[0][0] == 2 and deaths[0][1] in (0, 1)
    assert fires >= 8 and damage >= 45.0
    assert world.order_target_slot[0, :2].tolist() == [NO_SLOT, NO_SLOT]


def test_events_can_be_disabled() -> None:
    zg = zgame(1, events=False)
    zg.reset([conformance.SCENARIO])
    cmd = Command.noop(1, 3).set(0, 0, CommandKind.MOVE, target_xy=(12.0, 16.0))
    zg.step(cmd, 8)
    assert zg.events()['kind'].size == 0


# ---------------------------------------------------------------- Native movement & combat parity

# rghf-nat-cbt (SC2_NATIVE): acc-ram-cov-axo-rghf base + native movement + native combat timing
_M12_SC2_NATIVE: dict[str, object] = {
    'commands': 'delayed',
    'command_delay': 1,
    'move_lag': 1,
    'pathing': 'navmesh',
    'steering': 'native',
    'kinematics': 'accel',
    'collision': 'native',
    'targeting': 'sc2',
    'attack': 'sc2',
    'damage': 'sc2',
    'abilities': 'heal',
    'spawn_facing': 'random',
    'auto_acquire': True,
    'auto_acquire_owners': (2,),
    'acquire_start_loop': 0,
    'scan_phase': 'per_unit',
    'retarget': True,
    'retarget_busy_lock': True,
    'retarget_acquire': True,
    'retarget_lock': 'none',
    'retarget_armed_only': True,
    'reissue_unit': True,
    'kill_rescan_delay': 0,
    'kill_rescan_lock': True,
    'kill_rescan_order': True,
    'scan_after_move': True,
    'retarget_after_move': True,
    'hit_assist': True,
    'kill_hold': True,
    'report_engaged_as_attack_unit': True,
    'retreat_owners': (1,),
    'kill_no_retreat': True,
    'kill_no_retreat_end': 'lock',
    'chase_lock': ((48, 12), (73, 8), (74, 8)),
    'chase_lock_target': True,
    'follow_lock': True,
    'slop_on_arrival': True,
    'attack_commit': True,
    'attack_on_arrival': True,
    'stop_keeps_heal': True,
    'amove_needs_weapon': True,
    'attack_unit_needs_weapon': True,
    'dead_target_move': True,
    'slow_arrival': True,
    'hit_slow_owner1_only': True,
    'regen_race_fallback': True,
    'spawn_energy': 50.0,
    'heal_autocast': True,
    'heal_on_arrival': True,
    'heal_facing': True,
    'heal_pick': 'fraction',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'pin_engaged': True,
    'idle_return': True,
    'native_share_rule': 'priority_tree',
    'native_batch_flags': True,
    'native_nbr_order': 'quantized',
    'native_steer_avoid': True,
    'native_move_reissue': True,
    'native_stop': True,
    'native_air_sep': True,
    'native_wall_clearance': True,
    'native_assist_on_hit': True,
    'native_backswing': True,
    'native_retarget': True,
    'native_scan_cadence': True,
    'native_loop_order': True,
}

_M12_NAT_ONLY = {
    k: v
    for k, v in _M12_SC2_NATIVE.items()
    if k
    not in (
        'native_assist_on_hit',
        'native_backswing',
        'native_retarget',
        'native_scan_cadence',
        'native_loop_order',
        'hit_assist',
        'kill_rescan_delay',
        'kill_rescan_order',
    )
} | {'contact_slack': 0.0, 'idle_push_share': 0.5, 'acquire_start_loop': 2, 'kill_rescan_delay': 4}


def _m12_cfg(opts: Mapping[str, Any]) -> PyrefConfig:
    return PyrefConfig(**opts)


_BASE_CBT_CFG = _m12_cfg(_M12_SC2_NATIVE)

M12_CASES: dict[str, tuple[PyrefConfig, bool]] = {
    'rghf-nat': (_m12_cfg(_M12_NAT_ONLY), False),
    'rghf-nat-cbt': (_BASE_CBT_CFG, False),
    'rghf-nat-cbt-rot': (
        dataclasses.replace(_BASE_CBT_CFG, scan_phase='tag_rotate', scan_period=8),
        False,
    ),
    'rghf-nat-cbt-sym': (
        dataclasses.replace(
            _BASE_CBT_CFG,
            auto_acquire_owners=(),
            retreat_owners=(1, 2),
            hit_slow_owner1_only=False,
        ),
        True,
    ),
    'rghf-nat-cbt-measured': (
        dataclasses.replace(_BASE_CBT_CFG, range_radius='measured'),
        False,
    ),
}


def _assert_m12_world(a: WorldState, b: WorldState, *, exact: bool) -> None:
    if exact:
        assert_same(a, b)
        return
    # Under native mode, all 23 non-facing fields are bitwise equal; facing allows +-pi wrap tol
    for f in dataclasses.fields(WorldState):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x.dtype == y.dtype and x.shape == y.shape, f.name
        if f.name == 'facing':
            d = np.remainder(x.astype(np.float64) - y + math.pi, 2 * math.pi) - math.pi
            assert np.nanmax(np.abs(d), initial=0.0) <= M1_FACING_ATOL, f.name
        elif x.dtype.kind == 'f':
            assert np.array_equal(x, y, equal_nan=True), f.name
        else:
            assert np.array_equal(x, y), f.name


def _m12_scenarios(name: str, b_size: int, mirror: bool) -> list[Scenario]:
    rng = np.random.default_rng(700 + sorted(M12_CASES).index(name))
    out: list[Scenario] = []
    for _ in range(b_size):
        if not mirror:
            out.append(skirmish(rng, 4, ALL_TYPES))
            continue
        spawns: list[Spawn] = []
        for i in range(4):
            t = int(rng.choice(ALL_TYPES))
            x = float(9.0 + 2.5 * rng.random())
            y = float(11.0 + 1.5 * i + 0.5 * rng.random())
            spawns.append(Spawn(1, t, x, y))
        for i in range(4):
            s = spawns[i]
            spawns.append(Spawn(2, s.unit_type, 32.0 - s.x, s.y))
        out.append(Scenario(map_id=MAP_ID, spawns=tuple(spawns), mirror='x'))
    return out


def _compare_m12(name: str, *, exact: bool, steps: int = 35) -> None:
    cfg, mirror = M12_CASES[name]
    b_size = 3
    scenarios = _m12_scenarios(name, b_size, mirror)
    pyref = PyrefBackend(STATIC, MAPS, b_size, cfg)
    zg1 = zgame(b_size, cfg, MAPS, threads=1, min_parallel=1)
    zg4 = zgame(b_size, cfg, MAPS, threads=4, min_parallel=1)
    seeds = list(range(b_size))
    wp = pyref.reset(scenarios, seeds=seeds)
    wz1 = zg1.reset(scenarios, seeds=seeds)
    wz4 = zg4.reset(scenarios, seeds=seeds)
    _assert_m12_world(wp, wz1, exact=exact)
    assert_same(wz1, wz4)
    rng = np.random.default_rng(800 + sorted(M12_CASES).index(name))
    cmd = Command.noop(b_size, 8)
    for b in range(b_size):
        for u in range(4):
            cmd.set(b, u, CommandKind.ATTACK_MOVE, target_xy=(22.0, 14.0))
    with _pyref_events() as (hits, fires):
        for st in range(steps):
            hits.clear()
            fires.clear()
            wp = pyref.step(cmd, 8)
            wz1 = zg1.step(cmd, 8)
            wz4 = zg4.step(cmd, 8)
            _assert_m12_world(wp, wz1, exact=exact)
            assert_same(wz1, wz4)
            _check_events(zg1.events(), hits, fires, True, b_size)
            if st == 14:
                fresh = _m12_scenarios(name, 1, mirror)
                wp = pyref.reset(fresh, env_ids=[1], seeds=[101])
                wz1 = zg1.reset(fresh, env_ids=[1], seeds=[101])
                wz4 = zg4.reset(fresh, env_ids=[1], seeds=[101])
                _assert_m12_world(wp, wz1, exact=exact)
                assert_same(wz1, wz4)
            cmd = random_command(rng, wp, 0.2)


@pytest.mark.parametrize('name', sorted(M12_CASES))
def test_matches_pyref_m12(name: str) -> None:
    """Native movement/combat configs match pyref bitwise in-process (AVX-512 ON) across all
    24 fields (incl. facing), events, threads=1 vs 4, and partial resets."""
    _compare_m12(name, exact=True)


def _compare_m12_edge_cases(*, exact: bool) -> None:
    """Targeted edge cases: dead_target_move, assist, backswing+defer, and native_loop_order."""
    cfg = dataclasses.replace(
        _BASE_CBT_CFG,
        auto_acquire_owners=(1, 2),
        retarget_lock='previous',
        hit_assist_range=4.0,
    )
    # 0: P1 Marine (kills 2 for backswing+defer), 1: P1 Zealot (triggers assist),
    # 2: P2 low-HP Marine, 3/4: P2 bystanders calling for help
    spawns = (
        Spawn(1, MARINE, 10.0, 16.0),
        Spawn(1, ZEALOT, 10.5, 17.5),
        Spawn(2, MARINE, 14.0, 16.0),
        Spawn(2, MARINE, 14.8, 17.5),
        Spawn(2, MARINE, 14.8, 14.5),
    )
    scen = Scenario(
        map_id=MAP_ID,
        spawns=spawns,
        unit_overrides=(UnitOverride(owner=2, unit_type=MARINE, health_max=6.0),),
    )
    pyref = PyrefBackend(STATIC, MAPS, 2, cfg)
    zg = zgame(2, cfg, MAPS)
    wp = pyref.reset([scen, scen], seeds=[3, 7])
    wz = zg.reset([scen, scen], seeds=[3, 7])
    _assert_m12_world(wp, wz, exact=exact)
    cmd = Command.noop(2, 5)
    for b in range(2):
        # ATTACK_UNIT on dying target followed by MOVE to exercise dead_target and backswing hold
        cmd.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=2)
        cmd.set(b, 1, CommandKind.ATTACK_MOVE, target_xy=(16.0, 16.0))
        cmd.set(b, 3, CommandKind.ATTACK_UNIT, target_slot=1)
        cmd.set(b, 4, CommandKind.ATTACK_MOVE, target_xy=(10.0, 16.0))
    rng = np.random.default_rng(12)
    with _pyref_events() as (hits, fires):
        for st in range(25):
            hits.clear()
            fires.clear()
            wp, wz = pyref.step(cmd, 1 if st < 12 else 8), zg.step(cmd, 1 if st < 12 else 8)
            _assert_m12_world(wp, wz, exact=exact)
            _check_events(zg.events(), hits, fires, True, 2)
            cmd = random_command(rng, wp, 0.25) if st >= 6 else Command.noop(2, 5)


def test_m12_edge_cases_match_pyref() -> None:
    _compare_m12_edge_cases(exact=True)


@pytest.mark.skipif(
    platform.machine() not in ('x86_64', 'AMD64'),
    reason='AVX-512 feature toggle is x86_64 only',
)
def test_m12_bitwise_without_avx512() -> None:
    """With NumPy AVX-512 disabled, all native cases and edge cases match bitwise incl. facing."""
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        'import runpy\n'
        f'm = runpy.run_path({__file__!r})\n'
        'for n in sorted(m["M12_CASES"]):\n'
        '    m["_compare_m12"](n, exact=True)\n'
        'm["_compare_m12_edge_cases"](exact=True)\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800
    )
    assert r.returncode == 0, r.stderr[-4000:]


# ------------------------------------------------------------ libm atan2f bitwise parity

_B04_PARITY_SCENARIOS: tuple[tuple[str, str], ...] = (
    ('smac-portable', '3m'),
    ('smac-portable', '8m'),
    ('smac-portable', '2s3z'),
    ('smac-portable', '3s5z'),
    ('smac-portable', 'MMM'),
    ('smac-hard-portable', 'corridor'),
)


def test_b04_libm_atan2f_loaded_and_edge_values() -> None:
    """Platform C libm atan2f is loaded; scalar and array helpers match bitwise on random/edges."""
    from unismac.backends.pyref.subsystems import kinematics as kin

    assert kin._ATAN2F is not None  # noqa: SLF001
    rng = np.random.default_rng(42)
    vx = rng.uniform(-1.0, 1.0, size=1024).astype(np.float32)
    vy = rng.uniform(-1.0, 1.0, size=1024).astype(np.float32)
    vec = kin._atan2f_array(vy, vx)  # noqa: SLF001
    assert vec.dtype == np.float32 and vec.shape == vx.shape
    for i in range(64):
        assert vec[i].tobytes() == kin._atan2f(vy[i], vx[i]).tobytes()  # noqa: SLF001
    # Quadrant boundaries, negative zero, and NaN
    edge_y = np.array([0.0, -0.0, 1.0, -1.0, np.nan], dtype=np.float32)
    edge_x = np.array([-1.0, -1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    edge_out = kin._atan2f_array(edge_y, edge_x)  # noqa: SLF001
    for i in range(len(edge_y)):
        sc = kin._atan2f(edge_y[i], edge_x[i])  # noqa: SLF001
        if np.isnan(edge_y[i]):
            assert np.isnan(edge_out[i]) and np.isnan(sc)
        else:
            assert edge_out[i].tobytes() == sc.tobytes()


@pytest.mark.parametrize(
    ('preset', 'scenario'),
    _B04_PARITY_SCENARIOS,
    ids=[f'{p}/{s}' for p, s in _B04_PARITY_SCENARIOS],
)
def test_b04_all_24_world_fields_bitwise_in_process(preset: str, scenario: str) -> None:
    """In-process (AVX-512 ON, without NPY_DISABLE_CPU_FEATURES), 6 representative scenarios
    under rghf-nat-cbt match bitwise across all 24 WorldState fields (including facing)
    over 60 steps."""
    from unismac import profiles
    from unismac.env import make_vec_env

    _, std_over = profiles.profile_overrides(profiles.STANDARD_PROFILE)
    ep = make_vec_env(
        preset, scenario, backend='pyref', batch_size=4, seed=0, overrides=std_over, accel='auto'
    )
    ez = make_vec_env(
        preset, scenario, backend='zgame', batch_size=4, seed=0, overrides=std_over, accel='auto'
    )
    try:
        assert ep.accel == 'rust' and ez.accel == 'rust'
        ep.reset()
        ez.reset()
        assert len(dataclasses.fields(ep.world)) == 24
        assert_same(ep.world, ez.world)
        rng = np.random.default_rng(123)
        for _ in range(60):
            avail = ep.get_avail_actions()
            acts = np.zeros((4, avail.shape[1]), dtype=np.int64)
            for b in range(4):
                for a in range(avail.shape[1]):
                    valid = np.flatnonzero(avail[b, a])
                    acts[b, a] = int(valid[int(rng.integers(len(valid)))])
            ep.step(acts)
            ez.step(acts)
            assert_same(ep.world, ez.world)
    finally:
        ep.close()
        ez.close()


# ------------------------------------------------------------ ZgameBackend passive_from_map

_B02_CORRIDOR, _B02_HARD_3M, _B02_SMAC_3M = 'new_maps/corridor', 'new_maps/3m', 'SMAC_Maps/3m'
_B02_DUEL = (Spawn(1, MARINE, 19.0, 19.0), Spawn(2, MARINE, 21.5, 21.5))


def _b02_maps() -> dict[str, MapInfo]:
    return {m: load_map_info(m) for m in (_B02_CORRIDOR, _B02_HARD_3M, _B02_SMAC_3M)}


def test_b02_zgame_passive_from_map_owners_and_behavior() -> None:
    """ZgameBackend passive_from_map matches PyrefBackend bitwise per frame; on corridor,
    Player 1 retaliates when enabled and stays passive when disabled."""
    maps = _b02_maps()
    base = dataclasses.replace(_BASE_CBT_CFG, passive_from_map=False)
    on = dataclasses.replace(_BASE_CBT_CFG, passive_from_map=True)

    for cfg, expect_fire in ((base, False), (on, True)):
        single = {_B02_CORRIDOR: maps[_B02_CORRIDOR]}
        pyref = PyrefBackend(STATIC, single, 1, cfg)
        zg = ZgameBackend(STATIC, single, 1, cfg)
        try:
            scen = [Scenario(map_id=_B02_CORRIDOR, spawns=_B02_DUEL)]
            wp = pyref.reset(scen, seeds=[0])
            wz = zg.reset(scen, seeds=[0])
            assert_same(wp, wz)
            noop = Command.noop(1, len(_B02_DUEL))
            hp0 = float(wz.health[0, 1])
            for _ in range(40):
                wp, wz = pyref.step(noop, 1), zg.step(noop, 1)
                assert_same(wp, wz)
            fired = float(wz.health[0, 1]) < hp0
            assert fired is expect_fire
        finally:
            pyref.close()
            zg.close()

    # Single-map init resolves owner rules directly (no rebuild); non-corridor keeps ((2,), (1,))
    for mid, want in (
        (_B02_CORRIDOR, (None, ())),
        (_B02_HARD_3M, ((2,), (1,))),
        (_B02_SMAC_3M, ((2,), (1,))),
    ):
        zg = ZgameBackend(STATIC, {mid: maps[mid]}, 1, on)
        try:
            assert zg._engine_owners == want  # noqa: SLF001
            assert zg.config == on  # keeps original config for dual.check_backend
        finally:
            zg.close()


def test_b02_zgame_passive_from_map_multi_map_and_errors() -> None:
    """Multi-map switching, partial-reset/mixed-map errors, 3-player rules, and dual checks."""
    from unismac.env import dual

    maps = _b02_maps()
    on = dataclasses.replace(_BASE_CBT_CFG, passive_from_map=True)
    pyref = PyrefBackend(STATIC, maps, 2, on)
    zg = ZgameBackend(STATIC, maps, 2, on)
    try:
        hard_batch = [Scenario(map_id=_B02_HARD_3M, spawns=_B02_DUEL)] * 2
        corr_batch = [Scenario(map_id=_B02_CORRIDOR, spawns=_B02_DUEL)] * 2
        smac_batch = [Scenario(map_id=_B02_SMAC_3M, spawns=_B02_DUEL)] * 2
        # 1. Full-batch switches HARD_3M -> CORRIDOR -> SMAC_3M match PyrefBackend bitwise
        for batch, want_owners in (
            (hard_batch, ((2,), (1,))),
            (corr_batch, (None, ())),
            (smac_batch, ((2,), (1,))),
            (corr_batch, (None, ())),
        ):
            wp = pyref.reset(batch, seeds=[0, 1])
            wz = zg.reset(batch, seeds=[0, 1])
            assert zg._engine_owners == want_owners  # noqa: SLF001
            assert_same(wp, wz)
            noop = Command.noop(2, len(_B02_DUEL))
            for _ in range(15):
                wp, wz = pyref.step(noop, 1), zg.step(noop, 1)
                assert_same(wp, wz)

        # 2. Same-rule partial reset succeeds; different-rule partial reset raises ContractError
        wp = pyref.reset([corr_batch[0]], env_ids=[1], seeds=[2])
        wz = zg.reset([corr_batch[0]], env_ids=[1], seeds=[2])
        assert_same(wp, wz)
        with pytest.raises(
            ContractError,
            match='passive_from_map: partial reset',
        ):
            zg.reset([hard_batch[0]], env_ids=[1])

        # 3. Mixing maps with different passive rules in one reset raises ContractError
        with pytest.raises(
            ContractError,
            match='passive_from_map: maps in one reset',
        ):
            zg.reset([corr_batch[0], hard_batch[0]])

        # 4. Bare ZgameBackend preserves self._config so dual.check_backend rejects passive_from_map
        with pytest.raises(ValueError, match='passive_from_map'):
            dual.check_backend(zg, _B02_HARD_3M)
    finally:
        pyref.close()
        zg.close()

    # 5. Non-bool passive_from_map raises ValueError
    with pytest.raises(ValueError, match='passive_from_map'):
        ZgameBackend(STATIC, {_B02_HARD_3M: maps[_B02_HARD_3M]}, 1, {'passive_from_map': 'yes'})

    # 6. Single-map init reset with n_players=3 resolves 3-player owners to ((2, 3), (1,))
    spawns3 = (
        Spawn(1, MARINE, 18.0, 18.0),
        Spawn(2, MARINE, 20.5, 18.0),
        Spawn(3, MARINE, 20.5, 20.5),
    )
    scen3 = Scenario(
        map_id=_B02_HARD_3M,
        players=(AgentPlayer(), AgentPlayer(), AgentPlayer()),
        spawns=spawns3,
    )
    single_hard = {_B02_HARD_3M: maps[_B02_HARD_3M]}
    p3 = PyrefBackend(STATIC, single_hard, 1, on)
    z3 = ZgameBackend(STATIC, single_hard, 1, on)
    try:
        wp = p3.reset([scen3], seeds=[0])
        wz = z3.reset([scen3], seeds=[0])
        assert z3._engine_owners == ((2, 3), (1,))  # noqa: SLF001
        assert_same(wp, wz)
        noop3 = Command.noop(1, 3)
        for _ in range(15):
            wp, wz = p3.step(noop3, 1), z3.step(noop3, 1)
            assert_same(wp, wz)
    finally:
        p3.close()
        z3.close()


# ---------------------------------------- Multi-level vision/3D projectiles & Zealot Charge

_M13_B01_OPTS: dict[str, object] = {
    'visibility': 'sc2',
    'lost_vision': 'sc2',
    'projectile_3d': True,
}
_M13_B03_OPTS: dict[str, object] = {'charge': True}
_M13_BOTH_OPTS: dict[str, object] = {**_M13_B01_OPTS, **_M13_B03_OPTS}

_M13_B01_SCENARIOS: tuple[tuple[str, str, bool], ...] = (
    ('smac-hard-portable', 'mmmt', True),
    ('smac-hard-portable', '3st_vs_5zl', False),
    ('smac-hard-portable', '3rp_vs_5zl', False),
    ('smac-hard-portable', '6m_vs_10m', False),
    ('smac-hard-portable', 'corridor', False),
    ('smac-portable', '2c_vs_64zg', True),
    ('smac-portable', 'MMM2', True),
)

_M13_B03_ACTIVE_SCENARIOS: tuple[tuple[str, str], ...] = (
    ('smac-hard-portable', 'corridor'),
    ('smac-hard-portable', '3rp_vs_5zl'),
    ('smac-hard-portable', '3st_vs_5zl'),
    ('smac-hard-portable', 'mmmt_vs_zspi'),
)

_M13_B03_GATED_SCENARIOS: tuple[tuple[str, str], ...] = (
    ('smac-hard-portable', '2s3z'),
    ('smac-hard-portable', '3s5z'),
    ('smac-hard-portable', '3s_vs_5z'),
    ('smac-portable', '2s3z'),
)


def _m13_env_overrides(profile_name: str, extra_opts: Mapping[str, object]) -> dict[str, object]:
    from unismac import profiles

    _, over = profiles.profile_overrides(profile_name)
    out = dict(over)
    raw = out.get('backend.pyref_options', {})
    assert isinstance(raw, Mapping)
    out['backend.pyref_options'] = {**dict(raw), **dict(extra_opts)}
    return out


def _m13_sample_actions(rng: np.random.Generator, avail: np.ndarray) -> np.ndarray:
    acts = np.zeros(avail.shape[:2], dtype=np.int64)
    for b in range(avail.shape[0]):
        for a in range(avail.shape[1]):
            valid = np.flatnonzero(avail[b, a])
            acts[b, a] = int(valid[int(rng.integers(len(valid)))])
    return acts


def test_m13_capabilities_validation_and_config_coverage() -> None:
    """capabilities_for matches PyrefBackend exactly; vision/3D-projectile (6 opts) and Charge
    (1 opt) parsing and PyrefConfig.__post_init__ cross-validation are fully aligned."""
    from unismac.backends.pyref.backend import capabilities_for as pyref_capabilities_for

    for cfg in (
        PyrefConfig(),
        PyrefConfig(targeting='sc2', visibility='sc2'),
        PyrefConfig(damage='sc2', projectile_3d=True),
        PyrefConfig(
            targeting='sc2',
            damage='sc2',
            visibility='sc2',
            vision_levels='interp',
            reveal_loops=30,
            lost_vision='sc2',
            lost_vision_giveup=24,
            projectile_3d=True,
            charge=True,
        ),
        dataclasses.replace(_BASE_CBT_CFG, **_M13_BOTH_OPTS),  # type: ignore[arg-type]
    ):
        want = pyref_capabilities_for(cfg)
        got = capabilities_for(config_options(cfg))
        assert dataclasses.replace(got, name=want.name) == want
        zg = zgame(1, cfg)
        try:
            assert dataclasses.replace(zg.capabilities, name=want.name) == want
        finally:
            zg.close()

    assert not capabilities_for(config_options(PyrefConfig())).fog
    assert not capabilities_for(config_options(PyrefConfig())).terrain_height
    assert capabilities_for(config_options(PyrefConfig(targeting='sc2', visibility='sc2'))).fog
    assert capabilities_for(
        config_options(PyrefConfig(targeting='sc2', visibility='sc2'))
    ).terrain_height
    assert not capabilities_for(config_options(PyrefConfig(damage='sc2', projectile_3d=True))).fog
    assert capabilities_for(
        config_options(PyrefConfig(damage='sc2', projectile_3d=True))
    ).terrain_height

    # PyrefConfig.__post_init__ cross-validation and invalid-value checks
    for bad, exc_ty, pat in (
        ({'visibility': 'sc2'}, ValueError, "targeting='sc2'"),
        ({'visibility': 'bad'}, CapabilityError, 'visibility'),
        ({'vision_levels': 'bad'}, ValueError, 'vision_levels'),
        ({'reveal_loops': -1}, ValueError, 'reveal_loops'),
        ({'lost_vision': 'sc2'}, ValueError, "visibility='sc2'"),
        ({'lost_vision': 'bad'}, ValueError, 'lost_vision'),
        (
            {'targeting': 'sc2', 'visibility': 'sc2', 'lost_vision_giveup': 0},
            ValueError,
            'lost_vision_giveup',
        ),
        ({'charge': 'yes'}, ValueError, 'charge'),
        ({'projectile_3d': 'yes'}, ValueError, 'projectile_3d'),
    ):
        with pytest.raises(exc_ty, match=pat):
            zgame(1, bad)


@pytest.mark.parametrize(
    ('preset', 'scenario', 'expect_fog'),
    _M13_B01_SCENARIOS,
    ids=[f'{p}/{s}' for p, s, _ in _M13_B01_SCENARIOS],
)
def test_m13_b01_all_24_world_fields_bitwise_in_process(
    preset: str, scenario: str, expect_fog: bool
) -> None:
    """In-process (AVX-512 ON), rghf-nat-cbt + visibility='sc2' / lost_vision='sc2' /
    projectile_3d=True across 7 maps for 85 steps matches pyref bitwise on all 24 WorldState
    fields (incl. visible and facing), with VecEnv(accel='auto') selecting accel='rust'."""
    from unismac import profiles
    from unismac.env import make_vec_env

    over = _m13_env_overrides(profiles.STANDARD_PROFILE, _M13_B01_OPTS)
    batch = 1 if scenario == '2c_vs_64zg' else 2
    ep = make_vec_env(
        preset, scenario, backend='pyref', batch_size=batch, seed=0, overrides=over, accel='auto'
    )
    ez = make_vec_env(
        preset, scenario, backend='zgame', batch_size=batch, seed=0, overrides=over, accel='auto'
    )
    try:
        assert ep.accel == 'rust' and ez.accel == 'rust'
        ep.reset()
        ez.reset()
        assert len(dataclasses.fields(ep.world)) == 24
        assert_same(ep.world, ez.world)
        rng = np.random.default_rng(2026)
        saw_hidden = False
        for _ in range(85):
            if ep.done.all():
                ep.reset()
                ez.reset()
                assert_same(ep.world, ez.world)
            avail = ep.get_avail_actions()
            np.testing.assert_array_equal(avail, ez.get_avail_actions())
            acts = _m13_sample_actions(rng, avail)
            ep.step(acts)
            ez.step(acts)
            assert_same(ep.world, ez.world)
            alive = ez.world.alive
            if (~ez.world.visible[:, :, 0] & alive).any() or (
                ~ez.world.visible[:, :, 1] & alive
            ).any():
                saw_hidden = True
        if expect_fog:
            assert saw_hidden, f'{preset}/{scenario} expected true-vision occlusion'
    finally:
        ep.close()
        ez.close()


@pytest.mark.parametrize(
    ('profile_name', 'extra'),
    [
        ('aligned-v5', _M13_B01_OPTS),
        ('rghf-nat-cbt', {**_M13_B01_OPTS, 'vision_levels': 'interp'}),
        ('rghf-nat-cbt', {**_M13_B01_OPTS, 'lost_vision': 'immediate'}),
    ],
    ids=['aligned-v5+b01', 'rghf-nat-cbt+interp', 'rghf-nat-cbt+lost_immediate'],
)
def test_m13_b01_variants_and_aligned_v5_bitwise(
    profile_name: str, extra: Mapping[str, object]
) -> None:
    """Variants (aligned-v5, vision_levels='interp', lost_vision='immediate') on mmmt over 85
    steps match pyref bitwise across all 24/24 fields."""
    from unismac.env import make_vec_env

    over = _m13_env_overrides(profile_name, extra)
    ep = make_vec_env(
        'smac-hard-portable', 'mmmt', backend='pyref', batch_size=2, seed=1, overrides=over
    )
    ez = make_vec_env(
        'smac-hard-portable', 'mmmt', backend='zgame', batch_size=2, seed=1, overrides=over
    )
    try:
        assert ep.accel == 'rust' and ez.accel == 'rust'
        ep.reset()
        ez.reset()
        assert_same(ep.world, ez.world)
        rng = np.random.default_rng(77)
        for _ in range(85):
            if ep.done.all():
                ep.reset()
                ez.reset()
                assert_same(ep.world, ez.world)
            avail = ep.get_avail_actions()
            acts = _m13_sample_actions(rng, avail)
            ep.step(acts)
            ez.step(acts)
            assert_same(ep.world, ez.world)
    finally:
        ep.close()
        ez.close()


def test_m13_b01_cliff_reveal_3d_lost_vision_events_and_r1() -> None:
    """Verify high-ground occlusion, attack reveal, 3D projectile flight delay, lost-vision
    chase/giveup, event parity, and multi-thread determinism on new_maps/mmmt."""
    mmmt_id = 'new_maps/mmmt'
    mmmt_maps = {mmmt_id: load_map_info(mmmt_id)}
    cfg = dataclasses.replace(
        _BASE_CBT_CFG,
        visibility='sc2',
        lost_vision='sc2',
        projectile_3d=True,
        reveal_loops=47,
        lost_vision_giveup=48,
        passive_from_map=False,
    )
    cfg_2d = dataclasses.replace(cfg, projectile_3d=False)

    # Central high ground (x in [10,23), y in [11,21)) and south low ground on new_maps/mmmt
    low, high = (16.5, 7.5), (16.5, 11.5)
    spawns = (
        Spawn(1, STALKER, *high),
        Spawn(2, ZEALOT, *low),
        Spawn(2, MARINE, low[0] + 1.5, low[1]),
    )
    scen = Scenario(map_id=mmmt_id, spawns=spawns)

    pyref = PyrefBackend(STATIC, mmmt_maps, 2, cfg)
    zg1 = ZgameBackend(STATIC, mmmt_maps, 2, cfg, threads=1)
    zg4 = ZgameBackend(STATIC, mmmt_maps, 2, cfg, threads=4, min_parallel=1)
    zg2d = ZgameBackend(STATIC, mmmt_maps, 1, cfg_2d, threads=1)
    try:
        wp = pyref.reset([scen, scen], seeds=[0, 5])
        wz1 = zg1.reset([scen, scen], seeds=[0, 5])
        wz4 = zg4.reset([scen, scen], seeds=[0, 5])
        assert_same(wp, wz1)
        assert_same(wz1, wz4)
        # Initial vision is computed on reset: low-ground P2 cannot see high-ground P1 Stalker
        assert not bool(wz1.visible[0, 0, 1]) and bool(wz1.visible[0, 1, 0])

        cmd = Command.noop(2, 3)
        for b in range(2):
            cmd.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=1)
            cmd.set(b, 1, CommandKind.HOLD)
            cmd.set(b, 2, CommandKind.ATTACK_UNIT, target_slot=0)

        # Also verify 3D projectile lands later than 2D (single-env comparison)
        zg2d.reset([scen], seeds=[0])
        c2d = Command.noop(1, 3).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
        c2d.set(0, 1, CommandKind.HOLD)
        hit_3d, hit_2d = None, None

        with _pyref_events() as (hits, fires):
            for lp in range(1, 90):
                hits.clear()
                fires.clear()
                step_cmd = cmd if lp == 1 else Command.noop(2, 3)
                if lp == 25:
                    # Stalker retreats out of vision to trigger P2 lost-vision chase and giveup
                    step_cmd = Command.noop(2, 3)
                    for b in range(2):
                        step_cmd.set(b, 0, CommandKind.MOVE, target_xy=(16.5, 20.0))
                wp = pyref.step(step_cmd, 1)
                wz1 = zg1.step(step_cmd, 1)
                wz4 = zg4.step(step_cmd, 1)
                assert_same(wp, wz1)
                assert_same(wz1, wz4)
                _check_events(zg1.events(), hits, fires, True, 2)
                if hit_3d is None and wz1.shield[0, 1] < 50.0:
                    hit_3d = lp
                w2 = zg2d.step(c2d if lp == 1 else Command.noop(1, 3), 1)
                if hit_2d is None and w2.shield[0, 1] < 50.0:
                    hit_2d = lp
        assert hit_3d is not None and hit_2d is not None and hit_3d > hit_2d
    finally:
        pyref.close()
        zg1.close()
        zg4.close()
        zg2d.close()


@pytest.mark.parametrize(
    ('preset', 'scenario'),
    _M13_B03_ACTIVE_SCENARIOS,
    ids=[f'{p}/{s}' for p, s in _M13_B03_ACTIVE_SCENARIOS],
)
@pytest.mark.parametrize(
    ('tag', 'extra'),
    [('charge', _M13_B03_OPTS), ('b01+charge', _M13_BOTH_OPTS)],
    ids=['charge', 'b01+charge'],
)
def test_m13_b03_active_maps_bitwise_and_effect(
    preset: str, scenario: str, tag: str, extra: Mapping[str, object]
) -> None:
    """4 charge-enabled maps under rghf-nat-cbt + charge=True (and with vision/3D) match pyref
    bitwise on all 24/24 fields and diverge from charge=False (confirming Charge triggers)."""
    from unismac import profiles
    from unismac.env import make_vec_env

    over_on = _m13_env_overrides(profiles.STANDARD_PROFILE, extra)
    over_off = _m13_env_overrides(profiles.STANDARD_PROFILE, {**dict(extra), 'charge': False})
    ep = make_vec_env(
        preset, scenario, backend='pyref', batch_size=2, seed=0, overrides=over_on, accel='auto'
    )
    ez = make_vec_env(
        preset, scenario, backend='zgame', batch_size=2, seed=0, overrides=over_on, accel='auto'
    )
    ez_off = make_vec_env(
        preset, scenario, backend='zgame', batch_size=2, seed=0, overrides=over_off, accel='auto'
    )
    try:
        assert ep.accel == 'rust' and ez.accel == 'rust' and ez_off.accel == 'rust'
        ep.reset()
        ez.reset()
        ez_off.reset()
        assert_same(ep.world, ez.world)
        rng = np.random.default_rng(314)
        diverged = False
        for _ in range(85):
            if ep.done.all():
                ep.reset()
                ez.reset()
                assert_same(ep.world, ez.world)
            avail = ep.get_avail_actions()
            np.testing.assert_array_equal(avail, ez.get_avail_actions())
            acts = _m13_sample_actions(rng, avail)
            ep.step(acts)
            ez.step(acts)
            assert_same(ep.world, ez.world)
            if not diverged:
                if ez_off.done.all():
                    ez_off.reset()
                av_off = ez_off.get_avail_actions()
                if np.array_equal(av_off, avail):
                    ez_off.step(acts)
                    if not (
                        np.array_equal(ez.world.x, ez_off.world.x, equal_nan=True)
                        and np.array_equal(ez.world.health, ez_off.world.health, equal_nan=True)
                        and np.array_equal(ez.world.shield, ez_off.world.shield, equal_nan=True)
                    ):
                        diverged = True
                else:
                    diverged = True
        assert diverged, f'{preset}/{scenario} ({tag}) expected Charge to trigger'
    finally:
        ep.close()
        ez.close()
        ez_off.close()


@pytest.mark.parametrize(
    ('preset', 'scenario'),
    _M13_B03_GATED_SCENARIOS,
    ids=[f'{p}/{s}' for p, s in _M13_B03_GATED_SCENARIOS],
)
def test_m13_b03_gated_maps_zero_diff(preset: str, scenario: str) -> None:
    """Per-map Charge gating: on 4 maps without Charge in map_abilities.json, charge=True and
    charge=False have zero diff and match pyref bitwise."""
    from unismac import profiles
    from unismac.env import make_vec_env

    over_on = _m13_env_overrides(profiles.STANDARD_PROFILE, _M13_B03_OPTS)
    over_off = _m13_env_overrides(profiles.STANDARD_PROFILE, {'charge': False})
    ep = make_vec_env(
        preset, scenario, backend='pyref', batch_size=2, seed=2, overrides=over_on, accel='auto'
    )
    ez_on = make_vec_env(
        preset, scenario, backend='zgame', batch_size=2, seed=2, overrides=over_on, accel='auto'
    )
    ez_off = make_vec_env(
        preset, scenario, backend='zgame', batch_size=2, seed=2, overrides=over_off, accel='auto'
    )
    try:
        assert ep.accel == 'rust' and ez_on.accel == 'rust' and ez_off.accel == 'rust'
        ep.reset()
        ez_on.reset()
        ez_off.reset()
        assert_same(ep.world, ez_on.world)
        assert_same(ez_on.world, ez_off.world)
        rng = np.random.default_rng(99)
        for _ in range(85):
            if ep.done.all():
                ep.reset()
                ez_on.reset()
                ez_off.reset()
                assert_same(ep.world, ez_on.world)
                assert_same(ez_on.world, ez_off.world)
            avail = ep.get_avail_actions()
            acts = _m13_sample_actions(rng, avail)
            ep.step(acts)
            ez_on.step(acts)
            ez_off.step(acts)
            assert_same(ep.world, ez_on.world)
            assert_same(ez_on.world, ez_off.world)
    finally:
        ep.close()
        ez_on.close()
        ez_off.close()


def test_m13_b03_multi_map_partial_reset_and_slow_immunity() -> None:
    """Fine-grained Charge check: multi-map mixed batch + partial reset without engine rebuild,
    Marauder slow immunity, +8 charge impact damage, and event parity with pyref."""
    corr_id, gated_id = 'new_maps/corridor', 'new_maps/2s3z'
    maps = {corr_id: load_map_info(corr_id), gated_id: load_map_info(gated_id)}
    cfg = dataclasses.replace(
        _BASE_CBT_CFG, charge=True, passive_from_map=False, hit_slow_owner1_only=False
    )

    # Zealot (1) vs Marauder (2): test slow immunity on hit and +8 bonus damage on first hit
    duel_spawns = (Spawn(1, ZEALOT, 15.0, 16.0), Spawn(2, MARAUDER, 20.0, 16.0))
    sc_corr = Scenario(map_id=corr_id, spawns=duel_spawns)
    sc_gated = Scenario(map_id=gated_id, spawns=duel_spawns)

    pyref = PyrefBackend(STATIC, maps, 2, cfg)
    zg = ZgameBackend(STATIC, maps, 2, cfg, threads=2, min_parallel=1)
    try:
        eng0 = zg._engine  # noqa: SLF001
        wp = pyref.reset([sc_corr, sc_gated], seeds=[0, 0])
        wz = zg.reset([sc_corr, sc_gated], seeds=[0, 0])
        assert zg._engine is eng0  # noqa: SLF001
        assert_same(wp, wz)

        cmd = Command.noop(2, 2)
        for b in range(2):
            cmd.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=1)
            cmd.set(b, 1, CommandKind.HOLD)

        hp0 = float(wz.health[0, 1])
        first_drop_corr = 0.0
        saw_bonus_event = False
        mid_lead = 0.0
        for lp in range(50):
            step_cmd = cmd if lp == 0 else Command.noop(2, 2)
            wp = pyref.step(step_cmd, 1)
            wz = zg.step(step_cmd, 1)
            assert_same(wp, wz)
            ev = zg.events()
            if lp == 15:
                mid_lead = float(wz.x[0, 0]) - float(wz.x[1, 0])
            if first_drop_corr == 0.0 and float(wz.health[0, 1]) < hp0:
                first_drop_corr = hp0 - float(wz.health[0, 1])
                # First charge strike emits weapon hit (7.0) followed by bonus charge damage (8.0)
                m_dmg = (ev['env'] == 0) & (ev['kind'] == 2)
                dmg_h = [float(x) for x in ev['f1'][m_dmg]]
                assert dmg_h == [7.0, 8.0], dmg_h
                saw_bonus_event = True
        # On corridor Zealot charges (immune to Marauder slow, 8-1+8=15); on 2s3z gated & slowed
        assert first_drop_corr == 15.0 and saw_bonus_event
        assert mid_lead > 1.0, f'loop 15 charge lead should be > 1.0, got {mid_lead}'

        # Partial reset swapping maps between envs (no engine rebuild); continue bitwise check
        wp = pyref.reset([sc_gated], env_ids=[0], seeds=[3])
        wz = zg.reset([sc_gated], env_ids=[0], seeds=[3])
        assert zg._engine is eng0  # noqa: SLF001
        assert_same(wp, wz)
        wp = pyref.reset([sc_corr], env_ids=[1], seeds=[4])
        wz = zg.reset([sc_corr], env_ids=[1], seeds=[4])
        assert zg._engine is eng0  # noqa: SLF001
        assert_same(wp, wz)

        c2 = Command.noop(2, 2)
        for b in range(2):
            c2.set(b, 0, CommandKind.ATTACK_UNIT, target_slot=1)
            c2.set(b, 1, CommandKind.HOLD)
        mid_lead_swapped = 0.0
        for lp in range(35):
            step_cmd = c2 if lp == 0 else Command.noop(2, 2)
            wp = pyref.step(step_cmd, 1)
            wz = zg.step(step_cmd, 1)
            assert_same(wp, wz)
            if lp == 15:
                mid_lead_swapped = float(wz.x[1, 0]) - float(wz.x[0, 0])
        # After swap, env 1 (corridor) leads env 0 (2s3z) via Charge
        assert mid_lead_swapped > 1.0, f'post-swap loop 15 lead <= 1.0: {mid_lead_swapped}'
    finally:
        pyref.close()
        zg.close()
