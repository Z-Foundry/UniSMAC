"""Full internal simulation state `Sim` for `pyref`, plus per-env reset and unit spawning.

`Sim` holds all mutable state of a backend instance: contract `WorldState`, expanded static
`UnitParams`, persistent `MotionState`, per-loop `LoopBuffers`, `CombatState`, `VisionState`,
`AbilityState`, `pending` commands, and the delayed command `queue`. All arrays have fixed
shapes `[B, U]` (`[B, U, W]` for weapons, `[B, U, P]` for visibility) so they map cleanly to
native structs. Per-env `env_map` and pure memoization caches (`path_cache`, `nav_cache`,
`wall_cache`) do not affect simulation behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np

from unismac.contract import (
    LOOPS_PER_GAME_SECOND,
    NO_SLOT,
    AgentPlayer,
    CapabilityError,
    Command,
    ContractError,
    MapInfo,
    Scenario,
    StaticData,
    UnitTypeSpec,
    WorldState,
    mirror_facing,
)

from . import rng
from .config import PyrefConfig
from .navmesh import NavMesh

f32 = np.float32

# Up to two weapon slots per unit: 0 = ground, 1 = air ('any' weapons populate both).
WEAPON_GROUND = 0
WEAPON_AIR = 1
N_WEAPON_SLOTS = 2
SPLASH_RINGS = 3  # up to 3 radial splash rings (measured.json splash_radial_radius_1..3)
# Default edge trigger range for suicide units vs structures when not provided in measured.json.
SUICIDE_RANGE_STRUCTURE = 0.17


@dataclass
class UnitParams:
    """Per-slot static unit parameters populated from `StaticData` on reset."""

    speed: np.ndarray  # float32[B, U], tiles / loop
    sight: np.ndarray  # float32[B, U], sight radius (tiles)
    # Kinematics parameters from `UnitTypeSpec.measured` (assets/measured.json); inf = unlimited.
    accel: np.ndarray  # float32[B, U], tiles / loop^2; default inf
    decel: np.ndarray  # float32[B, U], tiles / loop^2; default inf
    turn_rate: np.ndarray  # float32[B, U], max heading turn rate (rad / loop); default inf
    heading_blend: np.ndarray  # float32[B, U], exponential heading blend factor; default NaN
    facing_rate: np.ndarray  # float32[B, U], max observed facing turn rate (rad/loop); def inf
    armor: np.ndarray  # float32[B, U]
    attr_mask: np.ndarray  # int32[B, U], attribute bitmask (see `PyrefBackend.attributes`)
    agent: np.ndarray  # bool[B, U], True if owner is an AgentPlayer
    biological: np.ndarray  # bool[B, U], biological attribute (Medivac Heal target eligibility)
    structure: np.ndarray  # bool[B, U], structure attribute (suicide vs structure bonus)
    # Radius used for weapon/acquire range checks (`measured.json` range_radius, fallback radius).
    range_radius: np.ndarray  # float32[B, U]
    has_weapon: np.ndarray  # bool[B, U, W]
    weapon_damage: np.ndarray  # float32[B, U, W], base damage per strike
    weapon_attacks: np.ndarray  # int32[B, U, W], strikes per attack
    weapon_range: np.ndarray  # float32[B, U, W], edge-to-edge range
    weapon_period: np.ndarray  # float32[B, U, W], cooldown in loops
    weapon_bonus: np.ndarray  # float32[B, U, W, A], extra damage per attribute bit
    # Attack / damage timing (`attack='sc2'`, `damage='sc2'`) from measured.json:
    attack_windup: np.ndarray  # int32[B, U], loops from attack start to fire; default 0
    attack_arc: np.ndarray  # float32[B, U], max residual angle (rad) to fire; default inf
    cooldown_jitter_min: np.ndarray  # float32[B, U], U(min, max) loop cooldown jitter; default 0
    cooldown_jitter_max: np.ndarray
    projectile_speed: np.ndarray  # float32[B, U], tiles / loop; default NaN (instant hit)
    projectile_delay: np.ndarray  # int32[B, U], extra loops after arrival before impact
    projectile_offset: np.ndarray  # float32[B, U], launch offset from attacker center
    attack_spacing: np.ndarray  # int32[B, U], loops between multi-attack strikes; default 0
    hit_slow_factor: np.ndarray  # float32[B, U], speed multiplier applied on hit; default 1
    hit_slow_loops: np.ndarray  # int32[B, U], hit-slow duration in loops
    shield_regen_delay: np.ndarray  # int32[B, U], loops after last hit before shield regen
    shield_regen_rate: np.ndarray  # float32[B, U], shield / loop; default 0
    health_regen_rate: np.ndarray  # float32[B, U], health / loop (continuous); default 0
    # Air-weapon projectile parameters (`projectile_*_air`); default inherits ground weapon:
    projectile_speed_air: np.ndarray  # float32[B, U]
    projectile_delay_air: np.ndarray  # int32[B, U]
    projectile_offset_air: np.ndarray  # float32[B, U]
    # Ground-weapon only (not inherited by air): target center instead of edge, and instant-hit
    # threshold (e.g., Queen ground hits instantaneously within 3.0 edge distance).
    projectile_to_center: np.ndarray  # bool[B, U]
    projectile_instant_edge: np.ndarray  # float32[B, U]
    turret_rate: np.ndarray  # float32[B, U], turret turn rate (replaces facing_rate); def NaN
    min_range: np.ndarray  # float32[B, U], minimum edge-to-edge firing range; default 0
    # Radial splash rings sorted ascending by radius (NaN = unused):
    splash_radius: np.ndarray  # float32[B, U, K]
    splash_fraction: np.ndarray  # float32[B, U, K]
    splash_perp: np.ndarray  # float32[B, U, 2], perpendicular beam (half_len, half_width)
    splash_forward: np.ndarray  # float32[B, U, 2], forward line (length, half_width)
    splash_friendly: np.ndarray  # bool[B, U], True if splash damages allies
    suicide: np.ndarray  # bool[B, U], suicide unit (synthesizes ground weapon in `_suicide_weapon`)
    suicide_damage_structure: np.ndarray  # float32[B, U], unarmored suicide damage vs structures
    suicide_range_structure: np.ndarray  # float32[B, U], edge trigger range vs structures
    barrier_absorb: np.ndarray  # float32[B, U], Barrier absorption capacity; default 0 (none)
    barrier_duration: np.ndarray  # int32[B, U], Barrier duration in loops
    barrier_cooldown: np.ndarray  # int32[B, U], Barrier cooldown in loops (0 = once per life)
    # Zealot Charge (`config.charge`, granted only to owners listed in map_abilities.json):
    # `charge_speed = NaN` disables Charge for that unit.
    charge_speed: np.ndarray  # float32[B, U], tiles / loop
    charge_range_small: np.ndarray  # float32[B, U], trigger dist for radius <= small_radius
    charge_range_large: np.ndarray  # float32[B, U], trigger dist for larger targets
    charge_small_radius: np.ndarray  # float32[B, U]
    charge_max_loops: np.ndarray  # int32[B, U]
    charge_cooldown: np.ndarray  # int32[B, U]
    charge_bonus: np.ndarray  # float32[B, U], unarmored bonus damage vs stationary target
    # Post-fire follow lock (`config.follow_lock`, from `measured.json`); -1 disables.
    follow_lock: np.ndarray  # int32[B, U]

    @classmethod
    def empty(cls, batch_size: int, n_slots: int, n_attrs: int) -> UnitParams:
        s = (batch_size, n_slots)
        w = (*s, N_WEAPON_SLOTS)
        return cls(
            speed=np.zeros(s, np.float32),
            sight=np.zeros(s, np.float32),
            accel=np.full(s, np.inf, np.float32),
            decel=np.full(s, np.inf, np.float32),
            turn_rate=np.full(s, np.inf, np.float32),
            heading_blend=np.full(s, np.nan, np.float32),
            facing_rate=np.full(s, np.inf, np.float32),
            armor=np.zeros(s, np.float32),
            attr_mask=np.zeros(s, np.int32),
            agent=np.zeros(s, np.bool_),
            biological=np.zeros(s, np.bool_),
            structure=np.zeros(s, np.bool_),
            range_radius=np.full(s, np.nan, np.float32),
            has_weapon=np.zeros(w, np.bool_),
            weapon_damage=np.zeros(w, np.float32),
            weapon_attacks=np.zeros(w, np.int32),
            weapon_range=np.zeros(w, np.float32),
            weapon_period=np.zeros(w, np.float32),
            weapon_bonus=np.zeros((*w, n_attrs), np.float32),
            attack_windup=np.zeros(s, np.int32),
            attack_arc=np.full(s, np.inf, np.float32),
            cooldown_jitter_min=np.zeros(s, np.float32),
            cooldown_jitter_max=np.zeros(s, np.float32),
            projectile_speed=np.full(s, np.nan, np.float32),
            projectile_delay=np.zeros(s, np.int32),
            projectile_offset=np.zeros(s, np.float32),
            attack_spacing=np.zeros(s, np.int32),
            hit_slow_factor=np.ones(s, np.float32),
            hit_slow_loops=np.zeros(s, np.int32),
            shield_regen_delay=np.zeros(s, np.int32),
            shield_regen_rate=np.zeros(s, np.float32),
            health_regen_rate=np.zeros(s, np.float32),
            projectile_speed_air=np.full(s, np.nan, np.float32),
            projectile_delay_air=np.zeros(s, np.int32),
            projectile_offset_air=np.zeros(s, np.float32),
            projectile_to_center=np.zeros(s, np.bool_),
            projectile_instant_edge=np.full(s, np.nan, np.float32),
            turret_rate=np.full(s, np.nan, np.float32),
            min_range=np.zeros(s, np.float32),
            splash_radius=np.full((*s, SPLASH_RINGS), np.nan, np.float32),
            splash_fraction=np.zeros((*s, SPLASH_RINGS), np.float32),
            splash_perp=np.full((*s, 2), np.nan, np.float32),
            splash_forward=np.full((*s, 2), np.nan, np.float32),
            splash_friendly=np.zeros(s, np.bool_),
            suicide=np.zeros(s, np.bool_),
            suicide_damage_structure=np.zeros(s, np.float32),
            suicide_range_structure=np.zeros(s, np.float32),
            barrier_absorb=np.zeros(s, np.float32),
            barrier_duration=np.zeros(s, np.int32),
            barrier_cooldown=np.zeros(s, np.int32),
            charge_speed=np.full(s, np.nan, np.float32),
            charge_range_small=np.zeros(s, np.float32),
            charge_range_large=np.zeros(s, np.float32),
            charge_small_radius=np.zeros(s, np.float32),
            charge_max_loops=np.zeros(s, np.int32),
            charge_cooldown=np.zeros(s, np.int32),
            charge_bonus=np.zeros(s, np.float32),
            follow_lock=np.full(s, -1, np.int32),
        )


@dataclass
class MotionState:
    """Cross-loop motion state outside contract `WorldState`, zeroed on reset."""

    speed: np.ndarray  # float32[B, U], current speed (tiles / loop)
    # float32[B, U], movement heading (rad); observed `world.facing` rotates toward it.
    heading: np.ndarray
    # Idle return (`collision='hard'` and `idle_return`):
    # float32[B, U], spawn anchor; cleared to NaN once any command is issued.
    home_x: np.ndarray
    home_y: np.ndarray
    pushed_loop: np.ndarray  # int64[B, U], last game_loop pushed by a mover (-1 if none)
    pusher_radius: np.ndarray  # float32[B, U], max radius of pushing mover on `pushed_loop`
    pushed_max: np.ndarray  # float32[B, U], max displacement from home (`rule='diameter'`)
    # Crowd arrival (`collision='hard'` and `crowd_arrival`; see `collision._crowd_arrival`):
    # float32[B, U], most recent MOVE goal (retained after stop for chain matching; NaN if none).
    crowd_goal_x: np.ndarray
    crowd_goal_y: np.ndarray
    crowd_dist: np.ndarray  # float32[B, U], post-collision distance to MOVE goal last loop
    crowd_blocked: np.ndarray  # int32[B, U], consecutive loops blocked by same-goal allies
    # bool[B, U], goal was covered by a contacting ally last loop (`crowd_cover`); clears next.
    crowd_cover_due: np.ndarray
    # Movement lag (`commands='delayed'` and `move_lag=1`): holds previous-loop velocity on the
    # command activation loop (`lag_loop`, -1 if none).
    lag_loop: np.ndarray  # int64[B, U]
    lag_vx: np.ndarray  # float32[B, U]
    lag_vy: np.ndarray
    # Contact orbit (`config.contact_orbit`; see `steering._orbit`):
    orbit_side: np.ndarray  # int8[B, U], held turn side: +1 CCW (left), -1 CW (right), 0 none
    orbit_loop: np.ndarray  # int64[B, U], last game_loop blocked (-1 if none)
    orbit_target: np.ndarray  # int32[B, U], target slot when blocked (NO_SLOT if none)
    # float32[B, U], position at previous loop steering (for `orbit_from='prev'/'prev_free'`).
    orbit_px: np.ndarray
    orbit_py: np.ndarray
    # bool[B, U], met pinned criteria during previous loop collision (`config.pin_lag`).
    pinned_prev: np.ndarray
    # float32[B, U], position at previous loop start (`config.pin_sidestep_drift`).
    drift_px: np.ndarray
    drift_py: np.ndarray
    # int64[B, U], first loop of current continuous pinned streak (`pin_pair_share='younger'`).
    pin_since: np.ndarray
    # int64[B, U], previous-loop committed displacement (Q12 integer, 1/4096 tile; used when
    # steering/collision='native'), updated in subsystems/native.py.
    native_vx: np.ndarray
    native_vy: np.ndarray
    # int64[B, U], game_loop when the active command took effect (-1 if none).
    cmd_loop: np.ndarray
    # Internal per-unit state for steering/collision='native':
    native_state: np.ndarray  # int8[B, U], fixed-point motion state (initial 0)
    native_flags: np.ndarray  # uint8[B, U], batch/wall contact bitflags (0x04/0x08/0x10)
    native_hold_still: np.ndarray  # bool[B, U], conditional stationary HOLD flag (initial False)
    # int32[B, U], optional tie-break slot override (<0 falls back to slot u)
    native_slot: np.ndarray
    native_bx: np.ndarray  # int64[B, U], blocked-progress anchor B.x (Q12)
    native_by: np.ndarray  # int64[B, U], blocked-progress anchor B.y (Q12)
    native_k: np.ndarray  # int32[B, U], blocked-progress counter k
    native_f: np.ndarray  # bool[B, U], blocked-progress first-frame flag f
    native_tol: np.ndarray  # int32[B, U], dynamic stop tolerance F (Q12)
    native_gx: np.ndarray  # int64[B, U], adjusted move destination G.x (Q12)
    native_gy: np.ndarray  # int64[B, U], adjusted move destination G.y (Q12)
    native_has_g: np.ndarray  # bool[B, U], whether adjusted destination G is active
    native_air_cd: np.ndarray  # int32[B, U], idle air-unit separation throttle counter

    @classmethod
    def empty(cls, batch_size: int, n_slots: int) -> MotionState:
        s = (batch_size, n_slots)
        return cls(
            speed=np.zeros(s, np.float32),
            heading=np.zeros(s, np.float32),
            home_x=np.full(s, np.nan, np.float32),
            home_y=np.full(s, np.nan, np.float32),
            pushed_loop=np.full(s, -1, np.int64),
            pusher_radius=np.zeros(s, np.float32),
            pushed_max=np.zeros(s, np.float32),
            crowd_goal_x=np.full(s, np.nan, np.float32),
            crowd_goal_y=np.full(s, np.nan, np.float32),
            crowd_dist=np.full(s, np.nan, np.float32),
            crowd_blocked=np.zeros(s, np.int32),
            crowd_cover_due=np.zeros(s, np.bool_),
            lag_loop=np.full(s, -1, np.int64),
            lag_vx=np.zeros(s, np.float32),
            lag_vy=np.zeros(s, np.float32),
            orbit_side=np.zeros(s, np.int8),
            orbit_loop=np.full(s, -1, np.int64),
            orbit_target=np.full(s, NO_SLOT, np.int32),
            orbit_px=np.full(s, np.nan, np.float32),
            orbit_py=np.full(s, np.nan, np.float32),
            pinned_prev=np.zeros(s, np.bool_),
            drift_px=np.full(s, np.nan, np.float32),
            drift_py=np.full(s, np.nan, np.float32),
            pin_since=np.full(s, -1, np.int64),
            native_vx=np.zeros(s, np.int64),
            native_vy=np.zeros(s, np.int64),
            cmd_loop=np.full(s, -1, np.int64),
            native_state=np.zeros(s, np.int8),
            native_flags=np.zeros(s, np.uint8),
            native_hold_still=np.zeros(s, np.bool_),
            native_slot=np.full(s, -1, np.int32),
            native_bx=np.zeros(s, np.int64),
            native_by=np.zeros(s, np.int64),
            native_k=np.zeros(s, np.int32),
            native_f=np.zeros(s, np.bool_),
            native_tol=np.zeros(s, np.int32),
            native_gx=np.zeros(s, np.int64),
            native_gy=np.zeros(s, np.int64),
            native_has_g=np.zeros(s, np.bool_),
            native_air_cd=np.zeros(s, np.int32),
        )


@dataclass
class CombatState:
    """Cross-loop attack and damage timing state (`attack='sc2'`, `damage='sc2'`).

    Tracks up to one in-flight projectile and one pending multi-attack follow-up chain per unit
    (any still-in-flight projectile resolves immediately if a new shot launches).
    """

    # float32[B, U], internal continuous cooldown (loops; may go negative).
    cooldown: np.ndarray
    windup_left: np.ndarray  # int32[B, U], remaining windup loops (-1 = not winding up)
    windup_target: np.ndarray  # int32[B, U], windup target slot (NO_SLOT if none)
    # In-flight projectile state: target, position, post-arrival wait (-1 = flying), hits, weapon.
    proj_target: np.ndarray  # int32[B, U], NO_SLOT if none
    proj_x: np.ndarray  # float32[B, U]
    proj_y: np.ndarray
    proj_wait: np.ndarray  # int32[B, U]
    proj_hits: np.ndarray  # int32[B, U]
    proj_weapon: np.ndarray  # int32[B, U]
    # Pending follow-up strikes for multi-attack weapons (`attack_spacing > 0`):
    follow_target: np.ndarray  # int32[B, U], NO_SLOT if none
    follow_loop: np.ndarray  # int64[B, U], game_loop of next strike
    follow_left: np.ndarray  # int32[B, U], remaining strikes
    follow_weapon: np.ndarray  # int32[B, U]
    last_hit_loop: np.ndarray  # int64[B, U], last game_loop damaged (-1 if never)
    # Active hit-slow window: scale speed by `slow_factor` while `slow_from <= loop <= slow_to`.
    slow_from: np.ndarray  # int64[B, U]
    slow_to: np.ndarray  # int64[B, U], -1 if none
    slow_factor: np.ndarray  # float32[B, U]
    # `targeting='sc2'`: trigger immediate scan once `game_loop >= scan_loop` (-1 if none).
    scan_loop: np.ndarray  # int64[B, U]
    # Per-unit periodic scan phase shift set on reset (`per_unit` / `tag_order` / `tag_rotate`).
    scan_shift: np.ndarray  # int64[B, U]
    # `targeting='sc2'`: leash position of an idle unit before auto-acquiring (NaN if none).
    leash_x: np.ndarray  # float32[B, U]
    leash_y: np.ndarray
    # `visibility='sc2'`: original target of an ATTACK_UNIT downgraded to ATTACK_MOVE on fog loss.
    lost_target: np.ndarray  # int32[B, U]
    # `retarget`: deferred evaluation loop (-1 if none) and locked target slot (NO_SLOT if none).
    rt_due: np.ndarray  # int64[B, U]
    rt_lock: np.ndarray  # int32[B, U]
    # `retarget_defer_memory`: remembered candidate target at defer time (NO_SLOT if none).
    rt_alt: np.ndarray  # int32[B, U]
    # `lost_vision='sc2'`: downgrade loop (-1), whether idle-acquired, and first unseen position.
    lost_loop: np.ndarray  # int64[B, U]
    lost_idle: np.ndarray  # bool[B, U]
    unseen_x: np.ndarray  # float32[B, U]
    unseen_y: np.ndarray
    proj_z: np.ndarray  # float32[B, U], projectile elevation (`projectile_3d`)
    last_fire_loop: np.ndarray  # int64[B, U], last loop fired (-1 if never)
    # Turret angle for units with `turret_rate` (`attack='sc2'`), initialized to spawn facing.
    turret: np.ndarray  # float32[B, U]
    # Barrier state (`damage='sc2'`): remaining shield, expiry loop (inclusive), last trigger loop.
    barrier_left: np.ndarray  # float32[B, U]
    barrier_until: np.ndarray  # int64[B, U], -1 if inactive
    barrier_from: np.ndarray  # int64[B, U], -1 if never triggered
    # Zealot Charge (`config.charge`): start loop (-1), active target, and impact-bonus target.
    charge_from: np.ndarray  # int64[B, U]
    charge_target: np.ndarray  # int32[B, U]
    charge_bonus_target: np.ndarray  # int32[B, U]
    # Follow lock (`config.follow_lock`; see `targeting._follow_lock`): last fired target, active
    # range-slop flag, and last loop a lock-breaking command took effect (-1 if none).
    last_fire_target: np.ndarray  # int32[B, U]
    follow_tol: np.ndarray  # bool[B, U]
    follow_cmd: np.ndarray  # int64[B, U]
    # float32[B, U], cooldown added at windup start (`attack_commit`, refunded on cancel/abort).
    windup_cost: np.ndarray
    # int32[B, U], engaged target seen by attack stage last loop (`attack_on_arrival`).
    arrive_target: np.ndarray
    # int32[B, U], engaged target associated with `follow_tol` (`slop_on_arrival`).
    tol_target: np.ndarray
    # bool[B, U], idle after ATTACK_UNIT target death awaiting next command (`kill_no_retreat`).
    kill_idle: np.ndarray
    # int32[B, U] / int64[B, U], assist-acquired attacker and hit loop (`hit_assist_pick`).
    assist_target: np.ndarray
    assist_loop: np.ndarray
    # bool[B, U], whether current engaged target comes from transient auto-acquire / assist-on-hit
    engaged_transient: np.ndarray
    # bool[B, U], whether idle unit is returning to its leash position
    leash_return: np.ndarray
    # int32[B, U, 2] / int32[B, U, 2], native_assist_on_hit: 2-slot 32-loop victim assist dedup
    # table
    assist_dedup_slot: np.ndarray
    assist_dedup_ttl: np.ndarray
    # int32[B, U, 4] / int32[B, U, 4], native_assist_on_hit: 4-slot 48-loop weapon threat memory
    threat_slot: np.ndarray
    threat_ttl: np.ndarray
    # int32[B, U], native_backswing: remaining uninterruptible backswing loops (max(follow_lock
    # - 1, 0))
    backswing_left: np.ndarray
    # int32[B, U], native_backswing / native_retarget: deferred candidate target while weapon
    # is busy
    defer_target: np.ndarray
    # bool[B, U], set when `defer_target` is written during a transient engagement order
    defer_abort: np.ndarray
    # int32[B, U], precomputed spawn/slot tie-break rank (0..U-1)
    spawn_rank: np.ndarray
    # int32[B, U], `native_loop_order`: shuffled unit processing order for Pass 1 / death timing
    unit_order: np.ndarray

    @classmethod
    def empty(cls, batch_size: int, n_slots: int) -> CombatState:
        s = (batch_size, n_slots)
        order0 = np.broadcast_to(np.arange(n_slots, dtype=np.int32), s).copy()
        return cls(
            cooldown=np.zeros(s, np.float32),
            windup_left=np.full(s, -1, np.int32),
            windup_target=np.full(s, NO_SLOT, np.int32),
            proj_target=np.full(s, NO_SLOT, np.int32),
            proj_x=np.zeros(s, np.float32),
            proj_y=np.zeros(s, np.float32),
            proj_wait=np.full(s, -1, np.int32),
            proj_hits=np.zeros(s, np.int32),
            proj_weapon=np.zeros(s, np.int32),
            follow_target=np.full(s, NO_SLOT, np.int32),
            follow_loop=np.full(s, -1, np.int64),
            follow_left=np.zeros(s, np.int32),
            follow_weapon=np.zeros(s, np.int32),
            last_hit_loop=np.full(s, -1, np.int64),
            slow_from=np.full(s, -1, np.int64),
            slow_to=np.full(s, -1, np.int64),
            slow_factor=np.ones(s, np.float32),
            scan_loop=np.full(s, -1, np.int64),
            scan_shift=np.zeros(s, np.int64),
            leash_x=np.full(s, np.nan, np.float32),
            leash_y=np.full(s, np.nan, np.float32),
            lost_target=np.full(s, NO_SLOT, np.int32),
            rt_due=np.full(s, -1, np.int64),
            rt_lock=np.full(s, NO_SLOT, np.int32),
            rt_alt=np.full(s, NO_SLOT, np.int32),
            lost_loop=np.full(s, -1, np.int64),
            lost_idle=np.zeros(s, np.bool_),
            unseen_x=np.full(s, np.nan, np.float32),
            unseen_y=np.full(s, np.nan, np.float32),
            proj_z=np.zeros(s, np.float32),
            last_fire_loop=np.full(s, -1, np.int64),
            turret=np.zeros(s, np.float32),
            barrier_left=np.zeros(s, np.float32),
            barrier_until=np.full(s, -1, np.int64),
            barrier_from=np.full(s, -1, np.int64),
            charge_from=np.full(s, -1, np.int64),
            charge_target=np.full(s, NO_SLOT, np.int32),
            charge_bonus_target=np.full(s, NO_SLOT, np.int32),
            last_fire_target=np.full(s, NO_SLOT, np.int32),
            follow_tol=np.zeros(s, np.bool_),
            follow_cmd=np.full(s, -1, np.int64),
            windup_cost=np.zeros(s, np.float32),
            arrive_target=np.full(s, NO_SLOT, np.int32),
            tol_target=np.full(s, NO_SLOT, np.int32),
            kill_idle=np.zeros(s, np.bool_),
            assist_target=np.full(s, NO_SLOT, np.int32),
            assist_loop=np.full(s, -1, np.int64),
            engaged_transient=np.zeros(s, np.bool_),
            leash_return=np.zeros(s, np.bool_),
            assist_dedup_slot=np.full((batch_size, n_slots, 2), NO_SLOT, np.int32),
            assist_dedup_ttl=np.zeros((batch_size, n_slots, 2), np.int32),
            threat_slot=np.full((batch_size, n_slots, 4), NO_SLOT, np.int32),
            threat_ttl=np.zeros((batch_size, n_slots, 4), np.int32),
            backswing_left=np.zeros(s, np.int32),
            defer_target=np.full(s, NO_SLOT, np.int32),
            defer_abort=np.zeros(s, np.bool_),
            spawn_rank=order0.copy(),
            unit_order=order0,
        )


@dataclass
class AbilityState:
    """Cross-loop ability state (`abilities='heal'`), cleared on reset and new commands."""

    channel: np.ndarray  # bool[B, U], actively channeling heal
    ready_loop: np.ndarray  # int64[B, U], earliest loop to start channeling (-1 if none)
    full_loops: np.ndarray  # int32[B, U], consecutive loops target has been at max health
    hold_after: np.ndarray  # bool[B, U], return to HOLD on completion (autocast from HOLD)
    face: np.ndarray  # bool[B, U], `heal_facing`: autocast order gated on facing alignment

    @classmethod
    def empty(cls, batch_size: int, n_slots: int) -> AbilityState:
        s = (batch_size, n_slots)
        return cls(
            channel=np.zeros(s, np.bool_),
            ready_loop=np.full(s, -1, np.int64),
            full_loops=np.zeros(s, np.int32),
            hold_after=np.zeros(s, np.bool_),
            face=np.zeros(s, np.bool_),
        )


@dataclass
class VisionState:
    """Fog-of-war state (`visibility='sc2'`), zeroed on reset (`visible = true_sight | reveal`)."""

    # bool[B, U, P]: true sight of player p+1 by sight radius and height level (read by targeting).
    true_sight: np.ndarray
    # int64[B, U, P]: game_loop (inclusive) through which unit is revealed to player p+1 (-1 none).
    reveal_to: np.ndarray

    @classmethod
    def empty(cls, batch_size: int, n_slots: int, n_players: int) -> VisionState:
        s = (batch_size, n_slots, n_players)
        return cls(true_sight=np.zeros(s, np.bool_), reveal_to=np.full(s, -1, np.int64))


@dataclass
class LoopBuffers:
    """Per-loop scratch buffers passed between pipeline stages, rewritten each loop."""

    # bool[B, U]: engaged target is in weapon range (written by targeting; overwritten by heal).
    in_range: np.ndarray
    # int32[B, U]: slot fired upon this loop, or NO_SLOT (written by attack; read next loop by
    # targeting for hit retaliation).
    fire_target: np.ndarray
    waypoint_x: np.ndarray  # float32[B, U], movement waypoint for this loop (NaN if stationary)
    waypoint_y: np.ndarray
    # float32[B, U], effective destination for MOVE / ATTACK_MOVE arrival checks (NaN if none).
    dest_x: np.ndarray
    dest_y: np.ndarray
    vel_x: np.ndarray  # float32[B, U], desired velocity in tiles / loop (steering -> kinematics)
    vel_y: np.ndarray
    # `config.follow_lock` / `slop_on_arrival` buffers (written by targeting, else False):
    # bool[B, U], hold position even though target is outside strict range
    follow_hold: np.ndarray
    # bool[B, U], allow firing within range slop (merged into in-range check in attack)
    fire_slop: np.ndarray
    # `config.attack_commit` buffers (written by attack, else False/defaults):
    # bool[B, U], hold position to finish committed windup whose target stepped out of range
    windup_hold: np.ndarray
    # Pre-fire snapshots restored if `damage.sc2` aborts the shot via `_void`:
    fire_prev_loop: np.ndarray  # int64[B, U]
    fire_prev_target: np.ndarray  # int32[B, U]
    fire_prev_backswing: np.ndarray  # int32[B, U]
    # `config.kill_rescan_order` buffers (written by `damage='sc2'`):
    # int32[B, U] killer slot (NO_SLOT if none) and bool[B, U] whether lethal hit was a projectile.
    kill_by: np.ndarray
    kill_proj: np.ndarray
    # `config.mover_speed_cap` / `collision='native'`: pre-kinematics positions read by collision.
    move_x0: np.ndarray
    move_y0: np.ndarray
    # `config.hit_assist_cap > 0`: `[b, victim, attacker]` pending hits for `assist_flush`.
    assist_hits: np.ndarray
    # `config.retarget_after_move`: units whose periodic retarget check runs in `late_scan`.
    late_retarget: np.ndarray

    @classmethod
    def empty(cls, batch_size: int, n_slots: int) -> LoopBuffers:
        s = (batch_size, n_slots)
        return cls(
            in_range=np.zeros(s, np.bool_),
            fire_target=np.full(s, NO_SLOT, np.int32),
            waypoint_x=np.full(s, np.nan, np.float32),
            waypoint_y=np.full(s, np.nan, np.float32),
            dest_x=np.full(s, np.nan, np.float32),
            dest_y=np.full(s, np.nan, np.float32),
            vel_x=np.zeros(s, np.float32),
            vel_y=np.zeros(s, np.float32),
            follow_hold=np.zeros(s, np.bool_),
            fire_slop=np.zeros(s, np.bool_),
            windup_hold=np.zeros(s, np.bool_),
            fire_prev_loop=np.full(s, -1, np.int64),
            fire_prev_target=np.full(s, NO_SLOT, np.int32),
            fire_prev_backswing=np.zeros(s, np.int32),
            kill_by=np.full(s, NO_SLOT, np.int32),
            kill_proj=np.zeros(s, np.bool_),
            move_x0=np.full(s, np.nan, np.float32),
            move_y0=np.full(s, np.nan, np.float32),
            assist_hits=np.zeros((batch_size, n_slots, n_slots), np.bool_),
            late_retarget=np.zeros(s, np.bool_),
        )


@dataclass
class Sim:
    """Complete mutable state of a backend instance; subsystems only read and mutate `Sim`."""

    config: PyrefConfig
    world: WorldState
    params: UnitParams
    motion: MotionState
    ability: AbilityState
    combat: CombatState
    vision: VisionState
    buf: LoopBuffers
    pending: Command  # command written by `step()` and consumed by the command stage
    # Circular delay queue (`commands='delayed'`), length `command_delay + 1`, indexed mod loop:
    # `queue[(L + k) % n]` holds commands received at loop L and activated at loop L + k.
    queue: list[Command]
    playable: np.ndarray  # float32[B, 4], (x0, y0, x1, y1)
    seed: np.ndarray  # int64[B], per-env seed for counter-based RNG (see rng.py)
    env_map: list[MapInfo | None]  # map per environment, set on reset
    # Memoized grid A* paths: `(map_id, start_cell, goal_cell) -> tuple[Cell, ...]`.
    path_cache: dict[tuple[str, tuple[int, int], tuple[int, int]], tuple[tuple[int, int], ...]] = (
        field(default_factory=dict)
    )
    # Memoized continuous navmeshes (`pathing='navmesh'`): `(map_id, rho) -> NavMesh`.
    nav_cache: dict[tuple[str, float], NavMesh] = field(default_factory=dict)
    # Cached unshrunk blocking-edge map (`native_wall_clearance`): `map_id -> (pathing, _WallMap)`.
    wall_cache: dict[str, tuple[np.ndarray, object]] = field(default_factory=dict)

    @classmethod
    def empty(
        cls, config: PyrefConfig, batch_size: int, n_slots: int, n_players: int, n_attrs: int
    ) -> Sim:
        return cls(
            config=config,
            world=WorldState.empty(batch_size, n_slots, n_players),
            params=UnitParams.empty(batch_size, n_slots, n_attrs),
            motion=MotionState.empty(batch_size, n_slots),
            ability=AbilityState.empty(batch_size, n_slots),
            combat=CombatState.empty(batch_size, n_slots),
            vision=VisionState.empty(batch_size, n_slots, n_players),
            buf=LoopBuffers.empty(batch_size, n_slots),
            pending=Command.noop(batch_size, n_slots),
            queue=[Command.noop(batch_size, n_slots) for _ in range(config.command_delay + 1)],
            playable=np.zeros((batch_size, 4), np.float32),
            seed=np.zeros(batch_size, np.int64),
            env_map=[None] * batch_size,
        )

    @property
    def batch_size(self) -> int:
        return self.world.batch_size

    @property
    def n_slots(self) -> int:
        return self.world.n_slots


# ---------------------------------------------------------------- Geometry & weapon selection


def edge_distance(world: WorldState, b: int, u: int, t: int) -> np.float32:
    """Edge-to-edge distance between slots `u` and `t` (center distance minus both radii)."""
    dx = world.x[b, t] - world.x[b, u]
    dy = world.y[b, t] - world.y[b, u]
    return f32(np.sqrt(dx * dx + dy * dy) - world.radius[b, u] - world.radius[b, t])


def range_radii(sim: Sim) -> np.ndarray:
    """Return `[B, U]` radii used for weapon/acquire range checks (`config.range_radius`)."""
    if sim.config.range_radius == 'measured':
        return sim.params.range_radius
    return sim.world.radius


def range_edge(sim: Sim, b: int, u: int, t: int) -> np.float32:
    """Edge-to-edge distance for range checks (same as `edge_distance` using `range_radii`)."""
    w, r = sim.world, range_radii(sim)
    dx = w.x[b, t] - w.x[b, u]
    dy = w.y[b, t] - w.y[b, u]
    return f32(np.sqrt(dx * dx + dy * dy) - r[b, u] - r[b, t])


def weapon_for(sim: Sim, b: int, u: int, t: int) -> int:
    """Return weapon slot index for unit `u` attacking `t`, or -1 if unable to attack."""
    w = WEAPON_AIR if sim.world.is_flying[b, t] else WEAPON_GROUND
    return w if sim.params.has_weapon[b, u, w] else -1


def height_z(byte: np.ndarray | float) -> np.ndarray | float:
    """Convert `terrain_height` byte to world elevation (e.g., 191 -> 7.97, 207 -> 9.97)."""
    return byte * (32.0 / 255.0) - 16.0


def cell_index(info: MapInfo, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clamped grid cell `(iy, ix)` for coordinates `(x, y)` (NaN maps to `(0, 0)`)."""
    with np.errstate(invalid='ignore'):
        ix = np.clip(np.nan_to_num(np.floor(x), nan=0.0), 0, info.width - 1).astype(np.int64)
        iy = np.clip(np.nan_to_num(np.floor(y), nan=0.0), 0, info.height - 1).astype(np.int64)
    return iy, ix


def is_enemy(world: WorldState, b: int, u: int, t: int) -> bool:
    """Return True if slot `t` is a living enemy unit of `u` (`owner == 0` is empty)."""
    return bool(world.alive[b, t] and world.owner[b, t] not in (0, world.owner[b, u]))


# ---------------------------------------------------------------- Reset & unit spawning


def _clear_env(
    dst: WorldState
    | UnitParams
    | MotionState
    | AbilityState
    | CombatState
    | VisionState
    | LoopBuffers
    | Command,
    src: object,
    b: int,
) -> None:
    """Copy env 0 from an empty batch-1 template `src` into env `b` of `dst`."""
    for f in fields(dst):
        getattr(dst, f.name)[b] = getattr(src, f.name)[0]


def check_scenario(
    scenario: Scenario, maps: Mapping[str, MapInfo], static_data: StaticData
) -> MapInfo:
    """Validate that `pyref` can run `scenario` without mutating state; return its `MapInfo`."""
    scenario.validate()
    if scenario.spawn_mode == 'map_trigger':
        raise CapabilityError('pyref does not support spawn_mode=map_trigger')
    for i, player in enumerate(scenario.players):
        if not isinstance(player, AgentPlayer):
            raise CapabilityError(f'pyref does not support {type(player).__name__} (players[{i}])')
    try:
        info = maps[scenario.map_id]
    except KeyError:
        raise ContractError(f'Unknown map {scenario.map_id!r}') from None
    x0, y0, x1, y1 = info.playable
    for i, s in enumerate(scenario.spawns):
        static_data.get(s.unit_type, s.variant)
        if not (x0 <= s.x <= x1 and y0 <= s.y <= y1):
            raise ContractError(
                f'spawns[{i}] position ({s.x}, {s.y}) outside playable area {info.playable}'
            )
    return info


def reset_env(
    sim: Sim,
    b: int,
    scenario: Scenario,
    info: MapInfo,
    static_data: StaticData,
    attributes: tuple[str, ...],
    seed: int,
    charge_owners: tuple[int, ...] = (),
) -> None:
    """Reset env `b` to the initial state of `scenario` (validated beforehand via `check_scenario`).

    `charge_owners`: player IDs whose Zealots receive Charge (from `assets/map_abilities.json`).
    """
    n_players = sim.world.n_players
    template = Sim.empty(sim.config, 1, sim.n_slots, n_players, len(attributes))
    for name in ('world', 'params', 'motion', 'ability', 'combat', 'vision', 'buf', 'pending'):
        _clear_env(getattr(sim, name), getattr(template, name), b)
    for q in sim.queue:
        _clear_env(q, template.pending, b)
    sim.playable[b] = info.playable
    sim.seed[b] = seed
    sim.env_map[b] = info

    agent_owner = [isinstance(p, AgentPlayer) for p in scenario.players]
    # `Scenario.unit_overrides`: units matching `(owner, unit_type, variant)` spawn at overridden
    # max health/shield; regen and healing read `world.health_max` / `shield_max`.
    overrides = {(o.owner, o.unit_type, o.variant): o for o in scenario.unit_overrides}
    w = sim.world
    for slot, s in enumerate(scenario.spawns):
        spec = static_data.get(s.unit_type, s.variant)
        o = overrides.get((s.owner, s.unit_type, s.variant))
        health_max = spec.health_max if o is None or o.health_max is None else o.health_max
        shield_max = spec.shield_max if o is None or o.shield_max is None else o.shield_max
        w.alive[b, slot] = True
        w.owner[b, slot] = s.owner
        w.unit_type[b, slot] = s.unit_type
        w.variant[b, slot] = s.variant
        w.x[b, slot] = s.x
        w.y[b, slot] = s.y
        sim.motion.home_x[b, slot] = s.x
        sim.motion.home_y[b, slot] = s.y
        # Spawn carries no facing: default 0.0, or draw from `spawn_facing` stream when `'random'`.
        # Initial movement heading matches spawn facing so stationary units do not rotate.
        if sim.config.spawn_facing == 'random':
            angle = 2.0 * np.pi * rng.uniform(seed, 'spawn_facing', 0, slot)
            w.facing[b, slot] = f32(angle - 2.0 * np.pi if angle >= np.pi else angle)
        else:
            w.facing[b, slot] = 0.0
        sim.motion.heading[b, slot] = w.facing[b, slot]
        w.radius[b, slot] = spec.radius
        w.health[b, slot] = w.health_max[b, slot] = health_max
        w.shield[b, slot] = w.shield_max[b, slot] = shield_max
        # Initial energy defaults to `energy_max`, or `min(spawn_energy, energy_max)` if specified
        # (e.g., 50 for explicitly spawned Medivacs; see `subsystems/abilities.py`).
        w.energy_max[b, slot] = spec.energy_max
        w.energy[b, slot] = (
            spec.energy_max
            if sim.config.spawn_energy is None
            else min(sim.config.spawn_energy, spec.energy_max)
        )
        w.is_flying[b, slot] = spec.is_flying
        _fill_params(sim.params, b, slot, spec, attributes)
        if sim.config.damage == 'sc2' and 'suicide_damage' in spec.measured:
            _suicide_weapon(sim.params, b, slot, spec, attributes)
        sim.combat.turret[b, slot] = w.facing[b, slot]
        if sim.config.regen_race_fallback:
            _regen_fallback(sim.params, b, slot, spec)
        if sim.config.charge and s.owner in charge_owners and 'charge_speed' in spec.measured:
            _charge_params(sim.params, b, slot, spec)
        sim.params.agent[b, slot] = agent_owner[s.owner - 1]
    if scenario.mirror is not None:
        # Mirror spawn: mirror Player 1 facings (and heading/turret) onto Player 2 counterparts.
        n = len(scenario.spawns) // 2
        for j in range(n):
            f = mirror_facing(w.facing[b, j], scenario.mirror)
            w.facing[b, n + j] = sim.motion.heading[b, n + j] = sim.combat.turret[b, n + j] = f
    if sim.config.scan_phase == 'per_unit':
        # Random spawn permutation from `scan_phase` stream (loop 0); ties break by lowest slot.
        n = len(scenario.spawns)
        draw = [rng.uniform(seed, 'scan_phase', 0, slot) for slot in range(n)]
        order = sorted(range(n), key=lambda slot: (draw[slot], slot))
        for rank, slot in enumerate(order):
            sim.combat.scan_shift[b, slot] = rank % sim.config.scan_period
    elif sim.config.scan_phase == 'tag_order':
        # Per-slot staggered scan phase: loop f scans slot u iff f == u + c (mod 8), with
        # c in {0, 2, 4, 6} drawn from (`scan_phase`, loop=1, slot=0). Scan outputs appear on the
        # next row, so `scan_shift = (row_phase - 1 - start) % 8`.
        c = 2 * int(4.0 * rng.uniform(seed, 'scan_phase', 1, 0))
        start = 0 if sim.config.native_scan_cadence else sim.config.acquire_start_loop
        for slot in range(len(scenario.spawns)):
            sim.combat.scan_shift[b, slot] = (slot + c - 1 - start) % 8
    elif sim.config.scan_phase == 'tag_rotate':
        # Staggered slot phase with deterministic episode rotation from `seed`:
        # `c = (2*seed + 4) mod 8` for `seed != 0`; for `seed == 0`, random rank + 2 mod 8.
        n = len(scenario.spawns)
        if seed != 0:
            c = (2 * seed + 4) % 8
            rows = [(slot + c) % 8 for slot in range(n)]
        else:
            draw = [rng.uniform(seed, 'scan_phase', 0, slot) for slot in range(n)]
            order = sorted(range(n), key=lambda slot: (draw[slot], slot))
            rows = [0] * n
            for rank, slot in enumerate(order):
                rows[slot] = (rank + 2) % 8
        start = 0 if sim.config.native_scan_cadence else sim.config.acquire_start_loop
        for slot in range(n):
            sim.combat.scan_shift[b, slot] = (rows[slot] - 1 - start) % 8
    n_spawns = len(scenario.spawns)
    sim.combat.unit_order[b] = np.arange(sim.n_slots, dtype=np.int32)
    if (
        sim.config.hit_assist_tie == 'spawn_order'
        or sim.config.scan_phase == 'per_unit'
        or (sim.config.scan_phase == 'tag_rotate' and seed == 0)
    ):
        draw_rank = [rng.uniform(seed, 'scan_phase', 0, slot) for slot in range(n_spawns)]
        rank_order = sorted(range(n_spawns), key=lambda slot: (draw_rank[slot], slot))
        for rank, slot in enumerate(rank_order):
            sim.combat.spawn_rank[b, slot] = rank
    else:
        sim.combat.spawn_rank[b, :n_spawns] = np.arange(n_spawns, dtype=np.int32)
    if n_spawns < sim.n_slots:
        sim.combat.spawn_rank[b, n_spawns:] = np.arange(n_spawns, sim.n_slots, dtype=np.int32)


def _suicide_weapon(
    p: UnitParams, b: int, slot: int, spec: UnitTypeSpec, attributes: tuple[str, ...]
) -> None:
    """Synthesize a ground trigger weapon for suicide units (e.g., Baneling, which has no static
    weapon) so targeting and pathing bring it into detonation range (`damage._burst`).
    """
    m = spec.measured
    w = WEAPON_GROUND
    p.suicide[b, slot] = True
    p.has_weapon[b, slot, w] = True
    p.weapon_damage[b, slot, w] = m['suicide_damage']
    p.weapon_attacks[b, slot, w] = 1
    p.weapon_range[b, slot, w] = m.get('suicide_trigger_range', 0.0)
    p.weapon_period[b, slot, w] = 0.0  # detonation has no cooldown
    if 'light' in attributes:
        p.weapon_bonus[b, slot, w, attributes.index('light')] = m.get('suicide_bonus_light', 0.0)
    p.suicide_damage_structure[b, slot] = m.get('suicide_damage_structure', m['suicide_damage'])
    p.suicide_range_structure[b, slot] = m.get(
        'suicide_trigger_range_structure', SUICIDE_RANGE_STRUCTURE
    )
    p.attack_windup[b, slot] = 0
    p.attack_arc[b, slot] = np.inf
    p.cooldown_jitter_min[b, slot] = p.cooldown_jitter_max[b, slot] = 0.0


# `regen_race_fallback`: race defaults when measured regen keys are absent (Protoss / Zerg).
FALLBACK_SHIELD_REGEN = (160, 0.125)
FALLBACK_HEALTH_REGEN = 70 / 4096


def _charge_params(p: UnitParams, b: int, slot: int, spec: UnitTypeSpec) -> None:
    """Populate Charge parameters from `measured.json`'s `charge_*` keys."""
    m = spec.measured
    p.charge_speed[b, slot] = m['charge_speed']
    p.charge_range_small[b, slot] = m['charge_range_small']
    p.charge_range_large[b, slot] = m['charge_range_large']
    p.charge_small_radius[b, slot] = m['charge_small_radius']
    p.charge_max_loops[b, slot] = int(m['charge_max_loops'])
    p.charge_cooldown[b, slot] = int(m['charge_cooldown'])
    p.charge_bonus[b, slot] = m.get('charge_bonus', 0.0)


def _regen_fallback(p: UnitParams, b: int, slot: int, spec: UnitTypeSpec) -> None:
    m = spec.measured
    if spec.shield_max > 0 and 'shield_regen_rate' not in m:
        p.shield_regen_delay[b, slot], p.shield_regen_rate[b, slot] = FALLBACK_SHIELD_REGEN
    if spec.race == 'zerg' and 'health_regen_rate' not in m:
        p.health_regen_rate[b, slot] = FALLBACK_HEALTH_REGEN


def _fill_params(
    p: UnitParams, b: int, slot: int, spec: UnitTypeSpec, attributes: tuple[str, ...]
) -> None:
    p.speed[b, slot] = f32(spec.movement_speed) / f32(LOOPS_PER_GAME_SECOND)
    p.sight[b, slot] = spec.sight_range
    # Parameter names/units follow `assets/measured.json`; defaults degrade to constant speed.
    p.accel[b, slot] = spec.measured.get('acceleration', np.inf)
    p.decel[b, slot] = spec.measured.get('deceleration', np.inf)
    p.turn_rate[b, slot] = spec.measured.get('turn_rate', np.inf)
    p.heading_blend[b, slot] = spec.measured.get('heading_blend', np.nan)
    p.facing_rate[b, slot] = spec.measured.get('facing_rate', np.inf)
    m = spec.measured
    p.range_radius[b, slot] = m.get('range_radius', spec.radius)
    p.attack_windup[b, slot] = int(m.get('attack_windup', 0))
    p.attack_arc[b, slot] = m.get('attack_arc', np.inf)
    p.cooldown_jitter_min[b, slot] = m.get('cooldown_jitter_min', 0.0)
    p.cooldown_jitter_max[b, slot] = m.get('cooldown_jitter_max', 0.0)
    p.projectile_speed[b, slot] = m.get('projectile_speed', np.nan)
    p.projectile_delay[b, slot] = int(m.get('projectile_delay', 0))
    p.projectile_offset[b, slot] = m.get('projectile_offset', 0.0)
    p.attack_spacing[b, slot] = int(m.get('attack_spacing', 0))
    p.hit_slow_factor[b, slot] = m.get('hit_slow_factor', 1.0)
    p.hit_slow_loops[b, slot] = int(m.get('hit_slow_loops', 0))
    p.shield_regen_delay[b, slot] = int(m.get('shield_regen_delay', 0))
    p.shield_regen_rate[b, slot] = m.get('shield_regen_rate', 0.0)
    p.health_regen_rate[b, slot] = m.get('health_regen_rate', 0.0)
    p.projectile_speed_air[b, slot] = m.get('projectile_speed_air', p.projectile_speed[b, slot])
    p.projectile_delay_air[b, slot] = int(
        m.get('projectile_delay_air', p.projectile_delay[b, slot])
    )
    p.projectile_offset_air[b, slot] = m.get('projectile_offset_air', p.projectile_offset[b, slot])
    p.projectile_to_center[b, slot] = bool(m.get('projectile_to_center', 0))
    p.projectile_instant_edge[b, slot] = m.get('projectile_instant_edge', np.nan)
    p.turret_rate[b, slot] = m.get('turret_rate', np.nan)
    p.min_range[b, slot] = m.get('min_range', 0.0)
    rings = sorted(
        (m[f'splash_radial_radius_{k}'], m.get(f'splash_radial_fraction_{k}', 1.0))
        for k in range(1, SPLASH_RINGS + 1)
        if f'splash_radial_radius_{k}' in m
    )
    for k, (radius, fraction) in enumerate(rings):
        p.splash_radius[b, slot, k] = radius
        p.splash_fraction[b, slot, k] = fraction
    if 'splash_perp_half_length' in m:
        p.splash_perp[b, slot] = (m['splash_perp_half_length'], m['splash_perp_half_width'])
    if 'splash_forward_length' in m:
        p.splash_forward[b, slot] = (m['splash_forward_length'], m['splash_forward_half_width'])
    p.splash_friendly[b, slot] = bool(m.get('splash_friendly', 0))
    p.barrier_absorb[b, slot] = m.get('barrier_absorb', 0.0)
    p.barrier_duration[b, slot] = int(m.get('barrier_duration', 0))
    p.barrier_cooldown[b, slot] = int(m.get('barrier_cooldown', 0))
    p.follow_lock[b, slot] = int(m.get('follow_lock', -1))
    p.armor[b, slot] = spec.armor
    mask = 0
    for a in spec.attributes:
        mask |= 1 << attributes.index(a)
    p.attr_mask[b, slot] = mask
    p.biological[b, slot] = 'biological' in spec.attributes
    p.structure[b, slot] = 'structure' in spec.attributes
    for weapon in spec.weapons:
        targets = {'ground': (WEAPON_GROUND,), 'air': (WEAPON_AIR,)}.get(
            weapon.target, (WEAPON_GROUND, WEAPON_AIR)
        )
        for w in targets:
            if p.has_weapon[b, slot, w]:
                continue  # keep first weapon per target domain
            p.has_weapon[b, slot, w] = True
            p.weapon_damage[b, slot, w] = weapon.damage
            p.weapon_attacks[b, slot, w] = weapon.attacks
            p.weapon_range[b, slot, w] = weapon.range
            p.weapon_period[b, slot, w] = f32(weapon.cooldown) * f32(LOOPS_PER_GAME_SECOND)
            for name, extra in weapon.bonus:
                p.weapon_bonus[b, slot, w, attributes.index(name)] = extra


def collect_attributes(static_data: StaticData) -> tuple[str, ...]:
    """Return sorted tuple of all unit attributes and weapon bonus names (bitmask order)."""
    names: set[str] = set()
    for spec in static_data.units.values():
        names.update(spec.attributes)
        for weapon in spec.weapons:
            names.update(name for name, _ in weapon.bonus)
    if len(names) > 31:
        raise ContractError(f'Attribute count {len(names)} exceeds int32 bitmask capacity')
    return tuple(sorted(names))


__all__ = [
    'AbilityState',
    'CombatState',
    'LoopBuffers',
    'MotionState',
    'Sim',
    'UnitParams',
    'check_scenario',
    'collect_attributes',
    'edge_distance',
    'is_enemy',
    'range_edge',
    'range_radii',
    'reset_env',
    'weapon_for',
]
