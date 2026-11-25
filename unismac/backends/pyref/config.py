"""Configuration dataclass for the pure-Python reference simulator (`PyrefBackend`)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PyrefConfig:
    """Subsystem selection and simulation parameters for `pyref` and `zgame`.

    Subsystem names map to implementations in `pipeline.REGISTRY`.
    Defaults provide a minimal constant-speed, collision-free, instant-hit baseline.
    """

    # ---- Subsystem implementations (in pipeline order)
    commands: str = 'immediate'  # immediate | delayed
    targeting: str = 'basic'  # basic | sc2 (scans, acquire range, hit response; see targeting.sc2)
    attack: str = 'instant'  # instant | sc2 (facing, windup, jittered cooldown; see attack.sc2)
    abilities: str = 'none'  # none | heal (Medivac Heal & energy regen; see abilities.py)
    pathing: str = 'straight'  # straight | grid | navmesh (see navmesh.py)
    steering: str = 'none'  # none | separation | sc2_avoid | native
    kinematics: str = 'constant_speed'  # constant_speed | accel
    # kinematics='accel': turning mode for units with both turn_rate and heading_blend
    # (Medivac_RL). 'cap' turns by min(blend * diff, turn_rate); 'curve' interpolates the
    # empirical per-loop turn angle table from remaining angle error. Default 'cap'.
    rl_turn: str = 'cap'
    collision: str = 'none'  # none | hard | native
    damage: str = 'standard'  # standard | sc2 (projectiles, multi-hit spacing, slow, regen)
    visibility: str = 'full'  # full | sc2 (height-layered vision, fire reveal; see visibility.sc2)
    # Height-layer rule for visibility='sc2': 'cell' quantizes each unit's cell height; 'interp'
    # linearly interpolates terrain height at x + 0.6 with distinct observer/target thresholds.
    vision_levels: str = 'cell'  # cell | interp

    # ---- Model parameters
    # commands='delayed': command takes effect k loops after arrival (see commands.delayed).
    # Single-process SC2 uses k=0 (immediate); lockstep two-player alignment uses k=1 + move_lag=1.
    command_delay: int = 0
    # commands='delayed': loops (0 or 1) during which displacement keeps prior velocity on the
    # activation loop (see kinematics). Pair command_delay=1 with move_lag=1 for lockstep SC2.
    move_lag: int = 0
    # Initial facing on reset: 'zero' (0.0, default) or 'random' (uniform in [0, 2*pi) from the
    # counter-based RNG 'spawn_facing' stream; see rng.py).
    spawn_facing: str = 'zero'
    # Whether idle (NOOP) and HOLD units auto-acquire enemies and retaliate on hit.
    auto_acquire: bool = False
    # Restrict idle (NOOP) auto-acquire to these player owners (None = all players). Does not
    # affect HOLD retaliation.
    auto_acquire_owners: tuple[int, ...] | None = None
    # Minimum game_loop at which idle (NOOP) units begin auto-acquiring (HOLD unaffected).
    acquire_start_loop: int = 0
    # Auto-acquire margin added to weapon range (edge-to-edge tiles).
    acquire_margin: float = 0.5
    # targeting='sc2': minimum auto-acquire range floor (edge-to-edge tiles, e.g. Zealot = 5.0).
    acquire_min_range: float = 5.0
    # targeting='sc2': periodic scan interval (loops) for idle and ATTACK_MOVE units.
    scan_period: int = 8
    # Periodic scan phase mode:
    # * 'global': all units scan on the same beat (`acquire_start_loop`).
    # * 'per_unit': each unit receives a per-episode random permutation rank mod `scan_period`
    #   (`combat.scan_shift`, drawn from RNG `'scan_phase'` stream), applied to targeting & heal.
    # * 'tag_order' (requires `scan_period == 8`): staggered slot phase `f = u + c (mod 8)` with
    #   random even offset `c in {0, 2, 4, 6}` per episode; autocast heal shares the same beat.
    # * 'tag_rotate' (requires `scan_period == 8`): deterministic episode rotation using `seed`
    #   (`c = (2*seed + 4) mod 8` for `seed != 0`, random permutation + 2 for `seed == 0`).
    scan_phase: str = 'global'
    # targeting='sc2': allow idle / ATTACK_MOVE units whose engaged target moved out of weapon
    # range to switch on scan beats to a closer in-range enemy (armed first, nearest edge).
    # Scans inside `retarget_busy` loops after firing are deferred to `last_fire_loop + busy`.
    retarget: bool = False
    retarget_busy: int = 8
    # Defer retarget using per-unit `measured.json` `follow_lock - 1` when present, falling back
    # to `retarget_busy`. Requires `retarget`.
    retarget_busy_lock: bool = False
    # Which retarget checks are deferred inside the busy window:
    # * 'scan' (default): defer any scan beat in the window and evaluate at expiry.
    # * 'switch': defer only scans that already want to switch, re-checking at expiry.
    retarget_defer: str = 'scan'  # scan | switch
    # Extension to `retarget_defer='switch'`: remember the candidate chosen at defer time
    # (`combat.rt_alt`); if expiry finds no fresh candidate and the current target is outside
    # acquire range, switch to `rt_alt` (or drop engagement if `rt_alt` died).
    retarget_defer_memory: bool = False
    # Extension to `retarget`: when weapon cooldown is 0 and target is out of weapon range,
    # also allow switching to enemies in acquire range whose edge distance is closer by more than
    # `retarget_ready_margin`. Requires `retarget`.
    retarget_ready: bool = False
    retarget_ready_margin: float = 0.2
    # Extension to `retarget`: when current target is out of weapon range, also allow switching to
    # any enemy within acquire range that has strictly smaller center distance (regardless of
    # cooldown). Requires `retarget`.
    retarget_acquire: bool = False
    # `rt_lock` policy after switching targets:
    # * 'new' (default): lock onto the new target until death or new command.
    # * 'previous': only prevent switching back to the immediately previous target.
    # * 'none': no lock; every decision beat may switch freely.
    retarget_lock: str = 'new'
    # When retargeting away from an armed target, restrict candidates to armed enemies (avoiding
    # switching to unarmed units such as Medivacs). Requires `retarget`.
    retarget_armed_only: bool = False
    # targeting='sc2': reissuing ATTACK_MOVE to the same clamped point preserves unit state
    # (cooldown, windup, lag, scan timer) and only drops an engaged target that left acquire range
    # or became unattackable (without triggering an immediate scan).
    reissue: bool = False
    # Extension to `reissue`: units whose target is dropped by a same-point ATTACK_MOVE reissue
    # immediately rescan this loop instead of waiting for the next scan beat.
    reissue_rescan: bool = False
    # Extension to `reissue`: units without an engaged target waiting in post-kill rescan delay
    # immediately rescan when receiving a same-point ATTACK_MOVE reissue.
    reissue_ends_wait: bool = False
    # Reissuing ATTACK_UNIT on the currently ordered target skips command activation entirely
    # (preserving lag, engagement, and timers without resetting state).
    reissue_unit: bool = False
    # targeting='sc2': loops after target death before rescanning (idle units with no new target
    # return to their leash point). `0` rescans at the end of the death loop (`targeting.kill_scan`
    # invoked by `pipeline.run_loop`). Default 2.
    kill_rescan_delay: int = 2
    # targeting='sc2': postpone post-kill rescan until at least `last_fire_loop + follow_lock - 1`
    # for units with a `measured.json` `follow_lock` entry.
    kill_rescan_lock: bool = False
    # With `kill_rescan_delay=0` and `damage='sc2'`, decide whether another unit's kill rescans on
    # the death loop or the next loop using randomized per-loop update order (`'update_order'` RNG
    # stream): instant kills by an earlier-updated killer rescan same-loop; later or projectile
    # kills rescan next loop.
    kill_rescan_order: bool = False
    # targeting='sc2': run periodic idle/ATTACK_MOVE scans after movement, damage, and visibility
    # (`targeting.late_scan`) using post-move positions rather than start-of-loop positions.
    scan_after_move: bool = False
    # Extension to `scan_after_move`: also defer non-busy periodic retarget checks to `late_scan`
    # using post-move positions. Requires `retarget` and `scan_after_move`.
    retarget_after_move: bool = False
    # targeting='sc2': when a surviving unit takes damage, idle (`auto_acquire`) and ATTACK_MOVE
    # allies within `hit_assist_range` (edge-to-edge) without an engaged target or active kill
    # wait engage the attacker on the hit loop (`targeting._assist`).
    hit_assist: bool = False
    hit_assist_range: float = 4.0
    # Attacker selection rule for `hit_assist`:
    # * 'first' (default): lock onto the first resolved hit; do not switch.
    # * 'nearest': among same-loop hits pick nearest center distance; if an assist-acquired unit
    #   has not fired yet, switch to closer attackers on later hits; periodic scans take priority.
    hit_assist_pick: str = 'first'
    # Tie-breaker for equal center distances under `hit_assist_pick='nearest'`:
    # * 'first' (default): first resolved hit (slot order).
    # * 'spawn_order': earliest unit in the per-episode random spawn permutation (`scan_phase`
    #   stream, game_loop 0).
    hit_assist_tie: str = 'first'
    # With `hit_assist` and `damage='sc2'`, cap per-victim assist triggers per loop to the first
    # `hit_assist_cap` attackers in per-loop update order (`targeting.assist_flush`). `0` = no cap.
    hit_assist_cap: int = 0
    # targeting='sc2': taking damage during a post-kill scan wait does not end the wait early.
    hit_keeps_kill_wait: bool = False
    # targeting='sc2': rank idle/ATTACK_MOVE scan candidates by center distance (after armed
    # priority) rather than edge distance. HOLD and retarget still use edge distance.
    acquire_center: bool = False
    # targeting='sc2': idle and ATTACK_MOVE units waiting for post-kill rescan hold position
    # instead of moving toward the order point.
    kill_hold: bool = False
    # Output-only view flag: report auto-engaging idle/ATTACK_MOVE units in returned `WorldState`
    # as `ATTACK_UNIT` on `engaged_target_slot` without changing internal simulation orders.
    report_engaged_as_attack_unit: bool = False
    # targeting='sc2': idle units of these owners flee away from attackers by `retreat_distance`
    # instead of retaliating when hit.
    retreat_owners: tuple[int, ...] = ()
    retreat_distance: float = 5.0
    # targeting='sc2': after an `ATTACK_UNIT` target dies, suppress hit-retreat for
    # `retreat_owners` units until the next command (or until lock expiry under `'lock'`).
    kill_no_retreat: bool = False
    # Expiry condition for `kill_no_retreat`: 'command' (until next command) or 'lock' (also ends
    # at `last_fire_loop + follow_lock`).
    kill_no_retreat_end: str = 'command'
    # Post-fire chase lock table `((unit_type, loops), ...)`: auto-engaging units hold position
    # for `loops` loops after firing even if their target leaves range.
    chase_lock: tuple[tuple[int, int], ...] = ()
    # Restrict `chase_lock` to when `engaged_target_slot == combat.last_fire_target`.
    chase_lock_target: bool = False
    # Post-fire follow lock and range slop (`targeting._follow_lock`) for units with a
    # `measured.json` `follow_lock` duration L on ATTACK_UNIT / ATTACK_MOVE / idle orders:
    # 1. Hold position for L loops after firing (firing only within strict range); chase at loop L
    #    if outside strict range.
    # 2. Otherwise enter range slop (`range + follow_slop`), holding and firing within slop until
    #    the target exceeds slop.
    # Non-reissue commands reset the lock window to the first movement loop of the new command.
    # Requires `targeting='sc2'` and `attack='sc2'`.
    follow_lock: bool = False
    # Range slop margin (edge-to-edge tiles) for `follow_lock` and `slop_on_arrival`.
    follow_slop: float = 1.0
    # Enter range slop as soon as the engaged target enters strict weapon range without requiring
    # a prior shot (applies when target != `last_fire_target`). Requires `targeting='sc2'` and
    # `attack='sc2'`.
    slop_on_arrival: bool = False
    # `attack='sc2'` & `damage='sc2'`: only committed shots spend cooldown.
    # 1. Windup cancels only if target changes or dies (holding position via `buf.windup_hold`
    #    if target leaves range).
    # 2. Canceled windups refund `combat.windup_cost`.
    # 3. Shots whose target already dropped to `<= 0` HP earlier in the loop abort (`_void`),
    #    refunding cooldown and restoring `last_fire_loop` / `last_fire_target`.
    attack_commit: bool = False
    # `attack='sc2'`: units that were moving toward the same engaged target last loop with ready
    # cooldown and `attack_windup >= 1` reduce windup and cooldown by 1 loop on attack start
    # (equivalent to starting the attack on the arrival loop).
    attack_on_arrival: bool = False
    # Enable Zealot Charge for `(map_id, owner)` entries in `assets/map_abilities.json` and units
    # with `charge_*` keys in `measured.json`. Boosts movement speed toward out-of-range targets
    # within threshold and adds unarmored `charge_bonus` on contact with stationary targets.
    charge: bool = False
    # Derive `auto_acquire_owners` and `retreat_owners` on reset from map Passive behavior in
    # `assets/map_abilities.json` (unchanged if the map has no entry).
    passive_from_map: bool = False
    # Ignore STOP commands on units actively channeling MedivacHeal.
    stop_keeps_heal: bool = False
    # Treat units actively channeling heal as armed during `targeting='sc2'` target selection.
    heal_counts_armed: bool = False
    # Ignore `ATTACK_MOVE` commands on units with no weapons.
    amove_needs_weapon: bool = False
    # Ignore `ATTACK_UNIT` commands when the unit lacks a weapon capable of hitting the target.
    attack_unit_needs_weapon: bool = False
    # When an `ATTACK_UNIT` target died during command delay, convert the command to `MOVE`
    # toward the target's last position instead of ignoring it.
    dead_target_move: bool = False
    # Under `damage='sc2'` hit slow, cap final waypoint step by `slow_factor * speed` instead of
    # scaling remaining distance, allowing slowed MOVE orders to arrive cleanly.
    slow_arrival: bool = False
    # Under `damage='sc2'`, apply hit slow only from Player 1 attackers (e.g., when only Player 1
    # Marauders have Concussive Shells).
    hit_slow_owner1_only: bool = False
    # `visibility='sc2'`: loops an attacker remains revealed to the victim's owner after a hit.
    reveal_loops: int = 47
    # `visibility='sc2'` handling when an engaged target leaves true sight:
    # * 'immediate' (default): downgrade ATTACK_UNIT to ATTACK_MOVE on the next loop; idle units
    #   drop via normal scan rules.
    # * 'sc2': downgrade after 2 consecutive unseen loops (recording first unseen position),
    #   restore on re-sighting, and give up after `lost_vision_giveup` loops.
    lost_vision: str = 'immediate'
    lost_vision_giveup: int = 48
    # `damage='sc2'`: compute projectile flight in 3D using cell `terrain_height` elevation.
    projectile_3d: bool = False
    # Arrival threshold radius (tiles) for MOVE / ATTACK_MOVE (0.0 snaps on final step).
    arrival_radius: float = 0.0
    # Minimum health damage per attack after armor reduction.
    min_damage: float = 0.5
    # `damage='sc2'`: apply race-default shield/health regen to units lacking measured regen keys
    # (Protoss shield 160-loop delay / 0.125/loop; Zerg health 70/4096 per loop).
    regen_race_fallback: bool = False
    # Initial unit energy on reset, clamped to `energy_max` (None = start at `energy_max`).
    spawn_energy: float | None = None
    # `abilities='heal'`: enable autocast healing for idle/HOLD standard Medivacs (`Variant.STD`).
    heal_autocast: bool = False
    # `abilities='heal'` + `heal_autocast`: run an immediate autocast scan (ignoring phase) when a
    # standard Medivac moves and arrives at a MOVE destination.
    heal_on_arrival: bool = False
    # `abilities='heal'` + `heal_autocast`: gate autocast heal channeling on facing alignment
    # (`<= abilities.HEAL_ARC`) instead of fixed 2-loop startup delay.
    heal_facing: bool = False
    # `abilities='heal'` + `heal_autocast`: target tie-order ('life' = lowest HP; 'fraction' =
    # lowest HP ratio; ties broken by nearest edge then lowest slot).
    heal_pick: str = 'life'
    # `collision='hard'`: Gauss-Seidel position projection iterations per loop.
    collision_iterations: int = 4
    # `collision='hard'`: minimum center distance = sum of radii - `contact_slack` (tiles).
    contact_slack: float = 0.0
    # `collision='hard'`: fraction of overlap pushed onto the idle unit in mover-vs-idle pairs.
    idle_push_share: float = 0.5
    # `collision='hard'`: overlap split rule for two moving ground units:
    # * 'half' (default): equal 50/50 split.
    # * 'speed': proportional to `|buf.vel|^mover_push_power` (faster mover yields more).
    # * 'approach': proportional to positive closing velocity along the contact normal, clamped to
    #   `[1 - mover_push_cap, mover_push_cap]` (50/50 if neither approaches).
    # * 'axis': independent per-axis closing share weighted by speed power and normal^2.
    # * 'approach_same_half': 50/50 for equal-speed pairs (within 2%), else `'approach'`.
    # * 'approach_lone_half': 50/50 only for equal-speed pairs not touching a different-speed unit.
    mover_push_share: str = 'half'  # half | speed | approach | axis | approach_{same,lone}_half
    mover_push_power: float = 1.0
    mover_push_cap: float = 0.8
    # `collision='hard'`: cap total per-loop displacement (kinematics + collision push) of moving
    # ground units to `max(params.speed, kinematics_step)`, re-projecting terrain if shortened.
    mover_speed_cap: bool = False
    # `collision='hard'` + `pathing='navmesh'`: push ground units back out to navmesh clearance
    # `rho` after cell terrain projection (`collision._clear`).
    push_clearance: bool = False
    # `collision='hard'`: clear `buf.waypoint_*` when MOVE/ATTACK_MOVE arrives in kinematics so
    # collision treats arriving units as idle on that loop.
    arrival_clears_waypoint: bool = False
    # `pin_engaged`: overlap split when two pinned units overlap: 'half' (50/50) or 'younger'
    # (the unit whose continuous pinned streak started later absorbs the full overlap).
    pin_pair_share: str = 'half'  # half | younger
    # `collision='hard'`: ground units with no waypoint that are on HOLD or have an engaged target
    # are immovable against pushable units (`collision.pinned_mask`) and treated as static
    # obstacles by `sc2_avoid`.
    pin_engaged: bool = False
    # `pin_engaged`: engaged units only become pinned once they have fired at their current target
    # (`combat.last_fire_target == engaged_target_slot`); HOLD remains pinned immediately.
    pin_after_fire: bool = False
    # `pin_engaged`: require two consecutive loops of pinned status (current and previous loop)
    # before pinning takes effect.
    pin_lag: bool = False
    # `pin_engaged` + `sc2_avoid`: treat gaps between two pinned units narrower than
    # `pin_gap_closed * unit_diameter` as closed and steer around the whole cluster.
    pin_gap_closed: float = 0.9
    # `pin_engaged` + `sc2_avoid`: exclude a unit's own engaged/ordered target from local
    # avoidance so blocked units do not stall trying to avoid their target.
    pin_slide: bool = False
    # Scope of `pin_slide`: 'any' (always exclude target) or 'pinned' (only if target is pinned).
    pin_slide_mode: str = 'any'  # any | pinned
    # Orders affected by `pin_slide`: 'any' (all orders) or 'attack_unit' (ATTACK_UNIT only).
    pin_slide_orders: str = 'any'  # any | attack_unit
    # `pin_engaged` + `sc2_avoid`: angular contact-orbit solver (`steering._orbit`) that rotates
    # desired velocity out of forbidden contact half-spaces around pinned blockers while holding
    # a consistent turn side for `orbit_hold` loops (stopping if total turn >= `orbit_max_turn`).
    contact_orbit: bool = False
    orbit_hold: int = 8
    orbit_iters: int = 12
    orbit_max_turn: float = 6.28
    orbit_movers: str = 'none'  # none | right | same
    orbit_from: str = 'steer'  # steer | prev | prev_free
    orbit_lock: bool = False
    # `pin_engaged` + `sc2_avoid`: when a MOVE ground unit is directly behind a pinned blocker
    # within `contact + SIDESTEP_RANGE` and cone `< pin_sidestep_cone` deg, sidestep laterally
    # at `speed * pin_sidestep_speed` instead of sliding tightly along the blocker's rim.
    pin_sidestep: bool = False
    pin_sidestep_cone: float = 45.0
    pin_sidestep_speed: float = 0.75
    # Optional lateral offset bound (tiles) for `pin_sidestep`: blockers with lateral distance
    # `>= pin_sidestep_lat` do not trigger sidestepping (None disables this filter).
    pin_sidestep_lat: float | None = None
    # Pairs to which `pin_sidestep_lat` applies: 'all' or 'marauder' (at least one Marauder).
    pin_sidestep_lat_scope: str = 'all'  # all | marauder
    # Keep sidestep direction consistent with the previous loop's lateral drift if > 0.01 tiles.
    pin_sidestep_drift: bool = False
    # Pick the closest pinned blocker (rather than lowest slot) and steer tangent to its rim.
    pin_sidestep_tangent: bool = False
    # `pin_engaged` + `sc2_avoid`: deterministic cluster-outline tie-breaking for aligned pinned
    # blockers, falling back to blocker grid parity when outlines tie asymmetrically.
    pin_tie_grid: bool = False
    # Radius source for weapon range, acquire range, and hit retaliation: 'observed' uses
    # `WorldState.radius`; 'measured' uses `measured.json`'s `range_radius` (fallback observed).
    range_radius: str = 'observed'  # observed | measured
    # `collision='hard'`: automatically issue a MOVE back to spawn (`home_*`) for idle units
    # displaced by movers (applies only to units that have never received a command).
    idle_return: bool = False
    # Thresholds for `idle_return`: minimum offset, immediate-return distance, and wait delay.
    return_threshold: float = 0.25
    return_far: float = 1.5
    return_delay: int = 22
    # Return rule: 'legacy' (larger pusher radius + delay) or 'diameter' (max displacement >
    # `return_ratio * 2 * radius`).
    idle_return_rule: str = 'legacy'  # legacy | diameter
    return_ratio: float = 1.0
    # `collision='hard'`: clear MOVE orders when crowded around a shared goal (`_crowd_arrival`).
    crowd_arrival: bool = False
    # Maximum goal separation (tiles) considered the same destination for `crowd_arrival`.
    crowd_goal_tol: float = 0.5
    # Minimum per-loop progress fraction and consecutive blocked loops before clearing MOVE.
    crowd_progress: float = 0.25
    crowd_loops: int = 2
    # Immediately clear MOVE upon contacting an already-stopped ally sharing the same goal.
    crowd_chain: bool = True
    # `collision='hard'`: clear MOVE on the next loop when a contacting non-idle/non-parallel ally
    # covers the unit's goal point (`collision._crowd_cover`).
    crowd_cover: bool = False
    # `steering='sc2_avoid'`: max edge distance (tiles) to consider a neighbor for avoidance.
    avoid_range: float = 1.25
    # `steering='sc2_avoid'`: trigger avoidance when predicted closest distance < contact + margin.
    avoid_margin: float = 0.05
    # `steering='sc2_avoid'`: lookahead horizon (loops) for closest-approach avoidance.
    avoid_horizon: float = 16.0
    # `steering='sc2_avoid'`: skip closest-approach avoidance for moving neighbors that are not
    # closing (`tau == 0`); pinned obstacles are unaffected.
    avoid_closing_only: bool = False
    # `steering='sc2_avoid'`: skip closest-approach avoidance for parallel moving neighbors unless
    # this unit has strictly higher max speed (`params.speed`).
    avoid_parallel_faster_only: bool = False
    # `steering='sc2_avoid'`: rightward lateral nudge (tiles) when heading straight into an idle.
    idle_nudge: float = 0.2
    # `steering='sc2_avoid'`: flip head-on avoidance side if an adjacent parallel ally occupies
    # the preferred side within `radii_sum + avoid_side_gap` while the opposite side is clear.
    avoid_side_check: bool = False
    avoid_side_gap: float = 1.0
    # `steering='sc2_avoid'`: edge gap threshold (tiles) for matching speed behind a parallel unit.
    follow_gap: float = 0.1
    # `steering='sc2_avoid'`: upper cosine threshold of the crossing-path branch (lower is -0.9);
    # values below 0.9 route small-angle convergences to closest-approach avoidance.
    avoid_cross_cos: float = 0.9
    # `steering='sc2_avoid'`: avoidance turn sharing between two movers: 'full' (both turn fully)
    # or 'speed' (scale turn angle by `min(1, 2*v_u^p / (v_u^p + v_t^p))` so faster yields more).
    avoid_share: str = 'full'  # full | speed
    avoid_share_power: float = 1.0
    # Optional per-branch overrides of `avoid_share_power` (None falls back to `avoid_share_power`).
    avoid_share_power_headon: float | None = None
    avoid_share_power_cross: float | None = None
    # `steering='separation'`: separation velocity gain per tile of overlap (in max-speed units).
    separation_gain: float = 1.0
    # `steering='separation'`: edge distance (tiles) below which separation repulsion activates.
    separation_margin: float = 0.25
    # `damage='sc2'`: fill missing shield regen (160-loop delay, 0.125/loop) for Stalker/_RL only.
    stalker_shield_regen: bool = False
    # ---- Q12 fixed-point ground movement, steering, collision, and combat timing
    # steering='native': blend waypoint direction with previous committed step direction
    # N(16 n + 15 u); set False for direction-blending ablations.
    native_blend: bool = True
    # collision='native': same-priority contact push-share relation bit for allied vs enemy pairs
    # (1: shorter displacement yields 7/8; 0: longer displacement yields 7/8; equal splits 1/2).
    native_relation: int = 1
    native_enemy_relation: int = 0
    # collision='native': Q12 yield share (out of 4096) assigned to a zero-displacement unit
    # pushed by a moving unit (default 3584 = 7/8).
    native_idle_share: int = 3584
    # collision='native': share rule when two moving units have different max speeds.
    native_speed_rule: str = 'table'  # table | faster_yields | slower_yields
    # collision='native': second contact pass selection ('contacted' or 'none').
    native_second_pass: str = 'contacted'  # contacted | none
    # steering='native': compute desired heading with sc2_avoid before Q12 blending/quantization.
    native_avoid: bool = False
    # collision='native': contact push-share resolution rule ('legacy' or 'priority_tree').
    # 'priority_tree' resolves allied/enemy priority, movement state, max-speed, and displacement
    # ties deterministically in Q12 fixed-point arithmetic.
    native_share_rule: str = 'legacy'  # legacy | priority_tree
    # collision='native': enable 4-pass live contact/wall flag tracking and cross-frame state.
    native_batch_flags: bool = False
    # collision='native': neighbor enumeration order ('slot' or 0.25-cell 'quantized' distance).
    native_nbr_order: str = 'slot'  # slot | quantized
    # steering='native': enable native ground local steering avoidance with lateral accel clamping
    # (mutually exclusive with native_avoid).
    native_steer_avoid: bool = False
    # Zero-tolerance Q12 same-point MOVE reissue skipping and progress state retention.
    native_move_reissue: bool = False
    # steering='native' & collision='native': native arrival deceleration, two-stage blocked
    # stop tolerance, and early stop clearing for ground MOVE orders.
    native_stop: bool = False
    # steering='native' & collision='native': air-unit stationary throttled separation and
    # moving flock aggregation (excluding ground-walking tall units such as Colossus).
    native_air_sep: bool = False
    # collision='native': axis-aligned blocking-edge wall clearance, bisection, and wall sliding
    # (mutually exclusive with push_clearance).
    native_wall_clearance: bool = False
    # Native combat, targeting, and loop timing options (all default False for compatibility).
    # Victim dual-slot 32-loop hit deduplication, lethal exemption, Q12 edge-distance <= 4.0
    # ally wake-up, 4-slot 48-loop threat memory, and Q12 center-distance scoring
    # (requires targeting='sc2', damage='sc2', hit_assist=True).
    native_assist_on_hit: bool = False
    # Weapon backswing window (follow_lock - 1) loaded on fire, dead-target order retention,
    # and deferred retarget slot promotion on order release
    # (requires targeting='sc2', attack='sc2', damage='sc2').
    native_backswing: bool = False
    # Unified Q12 target-scoring retarget with strict-range hysteresis, weapon-busy deferral,
    # and odd-beat spatial scan throttling while target remains in range
    # (requires targeting='sc2', retarget=True).
    native_retarget: bool = False
    # Remove acquire_start_loop gate from periodic scans and share the same 8-loop scan cadence
    # between target acquisition and Medivac autocast healing (requires targeting='sc2').
    native_scan_cadence: bool = False
    # Execute damage='sc2' in two passes (Pass 1: follow-up hits, instant fire, regen;
    # Pass 2: projectile hits and death cleanup) with per-loop update-order shuffling
    # (requires damage='sc2').
    native_loop_order: bool = False

    def __post_init__(self) -> None:
        if self.collision_iterations < 1:
            raise ValueError(f'collision_iterations must be >= 1, got {self.collision_iterations}')
        if not 0.0 <= self.idle_push_share <= 1.0:
            raise ValueError(f'idle_push_share must be in [0, 1], got {self.idle_push_share}')
        if self.mover_push_share not in (
            'half', 'speed', 'approach', 'axis', 'approach_same_half', 'approach_lone_half'
        ):  # fmt: skip
            raise ValueError(
                "mover_push_share must be 'half', 'speed', 'approach', 'axis', "
                f"'approach_same_half', or 'approach_lone_half', got {self.mover_push_share!r}"
            )
        if not 0.5 <= self.mover_push_cap <= 1.0:
            raise ValueError(f'mover_push_cap must be in [0.5, 1], got {self.mover_push_cap}')
        if self.avoid_share not in ('full', 'speed'):
            raise ValueError(f"avoid_share must be 'full' or 'speed', got {self.avoid_share!r}")
        if not 0.0 < self.avoid_share_power < math.inf:
            raise ValueError(
                f'avoid_share_power must be positive and finite, got {self.avoid_share_power}'
            )
        if not 0.0 <= self.avoid_cross_cos <= 0.9:
            raise ValueError(f'avoid_cross_cos must be in [0, 0.9], got {self.avoid_cross_cos}')
        for name in ('avoid_share_power_headon', 'avoid_share_power_cross'):
            p = getattr(self, name)
            if p is not None and not 0.0 < p < math.inf:
                raise ValueError(f'{name} must be None or positive and finite, got {p}')
        if not 0.0 < self.mover_push_power < math.inf:
            raise ValueError(
                f'mover_push_power must be positive and finite, got {self.mover_push_power}'
            )
        if self.spawn_energy is not None and self.spawn_energy < 0:
            raise ValueError(f'spawn_energy must be >= 0, got {self.spawn_energy}')
        if self.crowd_loops < 1:
            raise ValueError(f'crowd_loops must be >= 1, got {self.crowd_loops}')
        contact = self.collision in ('hard', 'native')
        if self.crowd_cover and not contact:
            raise ValueError("crowd_cover requires collision='hard' or 'native'")
        if self.push_clearance and (not contact or self.pathing != 'navmesh'):
            raise ValueError(
                "push_clearance requires collision='hard' or 'native' and pathing='navmesh'"
            )
        if self.arrival_clears_waypoint and not contact:
            raise ValueError("arrival_clears_waypoint requires collision='hard' or 'native'")
        for name in ('native_relation', 'native_enemy_relation'):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f'{name} must be 0 or 1, got {getattr(self, name)!r}')
        if not (isinstance(self.native_idle_share, int) and 0 <= self.native_idle_share <= 4096):
            raise ValueError(
                f'native_idle_share must be an int in [0, 4096], got {self.native_idle_share!r}'
            )
        if self.native_speed_rule not in ('table', 'faster_yields', 'slower_yields'):
            raise ValueError(
                "native_speed_rule must be 'table', 'faster_yields', or 'slower_yields', "
                f'got {self.native_speed_rule!r}'
            )
        if self.native_avoid and self.steering != 'native':
            raise ValueError("native_avoid requires steering='native'")
        if self.native_second_pass not in ('contacted', 'none'):
            raise ValueError(
                f"native_second_pass must be 'contacted' or 'none', got {self.native_second_pass!r}"
            )
        if self.native_share_rule not in ('legacy', 'priority_tree'):
            raise ValueError(
                "native_share_rule must be 'legacy' or 'priority_tree', "
                f'got {self.native_share_rule!r}'
            )
        if self.native_share_rule != 'legacy' and self.collision != 'native':
            raise ValueError("native_share_rule='priority_tree' requires collision='native'")
        if self.native_batch_flags and self.collision != 'native':
            raise ValueError("native_batch_flags requires collision='native'")
        if self.native_nbr_order not in ('slot', 'quantized'):
            raise ValueError(
                f"native_nbr_order must be 'slot' or 'quantized', got {self.native_nbr_order!r}"
            )
        if self.native_nbr_order != 'slot' and self.collision != 'native':
            raise ValueError("native_nbr_order='quantized' requires collision='native'")
        if self.native_steer_avoid and self.steering != 'native':
            raise ValueError("native_steer_avoid requires steering='native'")
        if self.native_steer_avoid and self.native_avoid:
            raise ValueError('native_steer_avoid cannot be combined with native_avoid')
        if self.native_stop and (self.steering != 'native' or self.collision != 'native'):
            raise ValueError("native_stop requires steering='native' and collision='native'")
        if self.native_air_sep and (self.steering != 'native' or self.collision != 'native'):
            raise ValueError("native_air_sep requires steering='native' and collision='native'")
        if self.native_wall_clearance and self.collision != 'native':
            raise ValueError("native_wall_clearance requires collision='native'")
        if self.native_wall_clearance and self.push_clearance:
            raise ValueError('native_wall_clearance cannot be combined with push_clearance')
        if self.pin_pair_share not in ('half', 'younger'):
            raise ValueError(
                f"pin_pair_share must be 'half' or 'younger', got {self.pin_pair_share!r}"
            )
        if self.pin_pair_share != 'half' and not self.pin_engaged:
            raise ValueError("pin_pair_share='younger' requires pin_engaged")
        if self.command_delay < 0:
            raise ValueError(f'command_delay must be >= 0, got {self.command_delay}')
        if self.acquire_start_loop < 0:
            raise ValueError(f'acquire_start_loop must be >= 0, got {self.acquire_start_loop}')
        if self.scan_period < 1:
            raise ValueError(f'scan_period must be >= 1, got {self.scan_period}')
        if self.rl_turn not in ('cap', 'curve'):
            raise ValueError(f"rl_turn must be 'cap' or 'curve', got {self.rl_turn!r}")
        if self.range_radius not in ('observed', 'measured'):
            raise ValueError(
                f"range_radius must be 'observed' or 'measured', got {self.range_radius!r}"
            )
        if self.pin_slide_mode not in ('any', 'pinned'):
            raise ValueError(
                f"pin_slide_mode must be 'any' or 'pinned', got {self.pin_slide_mode!r}"
            )
        if self.pin_slide_orders not in ('any', 'attack_unit'):
            raise ValueError(
                f"pin_slide_orders must be 'any' or 'attack_unit', got {self.pin_slide_orders!r}"
            )
        if self.orbit_movers not in ('none', 'right', 'same'):
            raise ValueError(
                f"orbit_movers must be 'none', 'right', or 'same', got {self.orbit_movers!r}"
            )
        if self.orbit_from not in ('steer', 'prev', 'prev_free'):
            raise ValueError(
                f"orbit_from must be 'steer', 'prev', or 'prev_free', got {self.orbit_from!r}"
            )
        if self.orbit_hold < 0:
            raise ValueError(f'orbit_hold must be >= 0, got {self.orbit_hold}')
        if self.orbit_iters < 1:
            raise ValueError(f'orbit_iters must be >= 1, got {self.orbit_iters}')
        if not self.orbit_max_turn > 0:
            raise ValueError(f'orbit_max_turn must be > 0, got {self.orbit_max_turn}')
        if self.pin_after_fire and not self.pin_engaged:
            raise ValueError('pin_after_fire requires pin_engaged')
        if self.pin_lag and not self.pin_engaged:
            raise ValueError('pin_lag requires pin_engaged')
        avoid = self.steering == 'sc2_avoid' or (self.steering == 'native' and self.native_avoid)
        if self.contact_orbit and (not avoid or not self.pin_engaged):
            raise ValueError("contact_orbit requires steering='sc2_avoid' and pin_engaged")
        if self.pin_sidestep and (not avoid or not self.pin_engaged):
            raise ValueError("pin_sidestep requires steering='sc2_avoid' and pin_engaged")
        if not 0 < self.pin_sidestep_cone <= 90:
            raise ValueError(f'pin_sidestep_cone must be in (0, 90], got {self.pin_sidestep_cone}')
        if not 0 < self.pin_sidestep_speed <= 1:
            raise ValueError(f'pin_sidestep_speed must be in (0, 1], got {self.pin_sidestep_speed}')
        if self.pin_sidestep_lat is not None and not (
            math.isfinite(self.pin_sidestep_lat) and self.pin_sidestep_lat > 0
        ):
            raise ValueError(
                f'pin_sidestep_lat must be None or positive and finite, got {self.pin_sidestep_lat}'
            )
        if self.pin_sidestep_lat_scope not in ('all', 'marauder'):
            raise ValueError(
                "pin_sidestep_lat_scope must be 'all' or 'marauder', "
                f'got {self.pin_sidestep_lat_scope!r}'
            )
        if self.pin_sidestep_lat_scope == 'marauder' and self.pin_sidestep_lat is None:
            raise ValueError("pin_sidestep_lat_scope='marauder' requires pin_sidestep_lat")
        if self.pin_sidestep_drift and not self.pin_sidestep:
            raise ValueError('pin_sidestep_drift requires pin_sidestep')
        if self.pin_sidestep_tangent and not self.pin_sidestep:
            raise ValueError('pin_sidestep_tangent requires pin_sidestep')
        if self.pin_tie_grid and (not avoid or not self.pin_engaged):
            raise ValueError("pin_tie_grid requires steering='sc2_avoid' and pin_engaged")
        if self.scan_phase not in ('global', 'per_unit', 'tag_order', 'tag_rotate'):
            raise ValueError(
                "scan_phase must be 'global', 'per_unit', 'tag_order', or 'tag_rotate', "
                f'got {self.scan_phase!r}'
            )
        if self.scan_phase in ('tag_order', 'tag_rotate') and self.scan_period != 8:
            raise ValueError(
                f'scan_phase={self.scan_phase!r} requires scan_period == 8, got {self.scan_period}'
            )
        if self.reveal_loops < 0:
            raise ValueError(f'reveal_loops must be >= 0, got {self.reveal_loops}')
        if self.lost_vision not in ('immediate', 'sc2'):
            raise ValueError(f"lost_vision must be 'immediate' or 'sc2', got {self.lost_vision!r}")
        if self.lost_vision == 'sc2' and self.visibility != 'sc2':
            raise ValueError("lost_vision='sc2' requires visibility='sc2'")
        if self.retarget_busy < 0:
            raise ValueError(f'retarget_busy must be >= 0, got {self.retarget_busy}')
        if self.retarget_ready_margin < 0:
            raise ValueError(
                f'retarget_ready_margin must be >= 0, got {self.retarget_ready_margin}'
            )
        if self.reissue_rescan and not self.reissue:
            raise ValueError('reissue_rescan requires reissue')
        if self.retarget_ready and not self.retarget:
            raise ValueError('retarget_ready requires retarget')
        if self.retarget_armed_only and not self.retarget:
            raise ValueError('retarget_armed_only requires retarget')
        if self.retarget_acquire and not self.retarget:
            raise ValueError('retarget_acquire requires retarget')
        if self.retarget_busy_lock and not self.retarget:
            raise ValueError('retarget_busy_lock requires retarget')
        if self.retarget_defer not in ('scan', 'switch'):
            raise ValueError(
                f"retarget_defer must be 'scan' or 'switch', got {self.retarget_defer!r}"
            )
        if self.retarget_defer != 'scan' and not self.retarget:
            raise ValueError("retarget_defer='switch' requires retarget")
        if self.retarget_defer_memory and self.retarget_defer != 'switch':
            raise ValueError("retarget_defer_memory requires retarget_defer='switch'")
        if self.retarget_lock not in ('new', 'previous', 'none'):
            raise ValueError(
                f"retarget_lock must be 'new', 'previous', or 'none', got {self.retarget_lock!r}"
            )
        if self.lost_vision_giveup < 1:
            raise ValueError(f'lost_vision_giveup must be >= 1, got {self.lost_vision_giveup}')
        if self.visibility == 'sc2' and self.targeting != 'sc2':
            raise ValueError("visibility='sc2' requires targeting='sc2'")
        if self.kill_rescan_delay < 0:
            raise ValueError(f'kill_rescan_delay must be >= 0, got {self.kill_rescan_delay}')
        if self.kill_rescan_delay == 0 and self.targeting != 'sc2':
            raise ValueError("kill_rescan_delay=0 requires targeting='sc2'")
        if self.kill_rescan_lock and self.targeting != 'sc2':
            raise ValueError("kill_rescan_lock requires targeting='sc2'")
        if self.kill_rescan_order and (self.kill_rescan_delay != 0 or self.damage != 'sc2'):
            raise ValueError("kill_rescan_order requires kill_rescan_delay=0 and damage='sc2'")
        if self.scan_after_move and self.targeting != 'sc2':
            raise ValueError("scan_after_move requires targeting='sc2'")
        if self.retarget_after_move and not (self.retarget and self.scan_after_move):
            raise ValueError('retarget_after_move requires retarget and scan_after_move')
        if self.hit_assist and self.targeting != 'sc2':
            raise ValueError("hit_assist requires targeting='sc2'")
        if self.hit_assist_pick not in ('first', 'nearest'):
            raise ValueError(
                f"hit_assist_pick must be 'first' or 'nearest', got {self.hit_assist_pick!r}"
            )
        if self.hit_assist_pick != 'first' and not self.hit_assist:
            raise ValueError("hit_assist_pick='nearest' requires hit_assist")
        if self.heal_pick not in ('life', 'fraction'):
            raise ValueError(f"heal_pick must be 'life' or 'fraction', got {self.heal_pick!r}")
        if self.hit_assist_tie not in ('first', 'spawn_order'):
            raise ValueError(
                f"hit_assist_tie must be 'first' or 'spawn_order', got {self.hit_assist_tie!r}"
            )
        if self.hit_assist_tie != 'first' and self.hit_assist_pick != 'nearest':
            raise ValueError("hit_assist_tie='spawn_order' requires hit_assist_pick='nearest'")
        if not isinstance(self.hit_assist_cap, int) or self.hit_assist_cap < 0:
            raise ValueError(
                f'hit_assist_cap must be a non-negative integer, got {self.hit_assist_cap!r}'
            )
        if self.hit_assist_cap > 0 and not (self.hit_assist and self.damage == 'sc2'):
            raise ValueError("hit_assist_cap > 0 requires hit_assist and damage='sc2'")
        if self.kill_no_retreat_end not in ('command', 'lock'):
            raise ValueError(
                f"kill_no_retreat_end must be 'command' or 'lock', got {self.kill_no_retreat_end!r}"
            )
        if self.kill_no_retreat_end != 'command' and not self.kill_no_retreat:
            raise ValueError("kill_no_retreat_end='lock' requires kill_no_retreat")
        if self.hit_keeps_kill_wait and self.targeting != 'sc2':
            raise ValueError("hit_keeps_kill_wait requires targeting='sc2'")
        if self.acquire_center and self.targeting != 'sc2':
            raise ValueError("acquire_center requires targeting='sc2'")
        if self.reissue_ends_wait and not self.reissue:
            raise ValueError('reissue_ends_wait requires reissue')
        lock = tuple((int(t), int(n)) for t, n in self.chase_lock)
        if any(n < 0 for _, n in lock):
            raise ValueError(f'chase_lock loop counts must be >= 0, got {self.chase_lock}')
        object.__setattr__(self, 'chase_lock', lock)
        if self.follow_slop < 0:
            raise ValueError(f'follow_slop must be >= 0, got {self.follow_slop}')
        if self.follow_lock and (self.targeting != 'sc2' or self.attack != 'sc2'):
            raise ValueError("follow_lock requires targeting='sc2' and attack='sc2'")
        if self.slop_on_arrival and (self.targeting != 'sc2' or self.attack != 'sc2'):
            raise ValueError("slop_on_arrival requires targeting='sc2' and attack='sc2'")
        if self.attack_commit and (self.attack != 'sc2' or self.damage != 'sc2'):
            raise ValueError("attack_commit requires attack='sc2' and damage='sc2'")
        if self.attack_on_arrival and self.attack != 'sc2':
            raise ValueError("attack_on_arrival requires attack='sc2'")
        if self.move_lag not in (0, 1):
            raise ValueError(f'move_lag must be 0 or 1, got {self.move_lag}')
        if self.move_lag and (self.commands != 'delayed' or self.command_delay < 1):
            raise ValueError("move_lag=1 requires commands='delayed' and command_delay >= 1")
        if self.spawn_facing not in ('zero', 'random'):
            raise ValueError(f"spawn_facing must be 'zero' or 'random', got {self.spawn_facing!r}")
        if self.commands == 'immediate' and self.command_delay != 0:
            raise ValueError("command_delay > 0 requires commands='delayed'")
        if self.native_assist_on_hit and (
            self.targeting != 'sc2' or self.damage != 'sc2' or not self.hit_assist
        ):
            raise ValueError(
                "native_assist_on_hit requires targeting='sc2', damage='sc2', and hit_assist=True"
            )
        if self.native_backswing and (
            self.targeting != 'sc2' or self.attack != 'sc2' or self.damage != 'sc2'
        ):
            raise ValueError(
                "native_backswing requires targeting='sc2', attack='sc2', and damage='sc2'"
            )
        if self.native_retarget and (self.targeting != 'sc2' or not self.retarget):
            raise ValueError("native_retarget requires targeting='sc2' and retarget=True")
        if self.native_scan_cadence and self.targeting != 'sc2':
            raise ValueError("native_scan_cadence requires targeting='sc2'")
        if self.native_loop_order and self.damage != 'sc2':
            raise ValueError("native_loop_order requires damage='sc2'")
