"""Read-only metadata catalog and query helpers for all 164 `PyrefConfig` fields.

Provides thematic grouping, tier classification (`core`, `native`, `legacy_heuristic`),
and `zgame` backend support status without altering `PyrefConfig`'s flat dataclass layout.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .config import PyrefConfig

OptionTier = Literal['core', 'native', 'legacy_heuristic']

OPTION_GROUPS: tuple[str, ...] = (
    'stages',
    'commands',
    'targeting',
    'attack',
    'abilities',
    'steering',
    'collision',
    'damage',
    'visibility',
)

OPTION_TIERS: tuple[str, ...] = ('core', 'native', 'legacy_heuristic')


@dataclass(frozen=True)
class OptionMeta:
    """Metadata descriptor for a single `PyrefConfig` option field."""

    name: str
    group: str
    tier: OptionTier
    default: Any
    zgame_supported: bool
    summary: str


_DEFAULTS = PyrefConfig()

# (name, group, tier, zgame_supported, summary) in `PyrefConfig` field definition order.
_SPECS: tuple[tuple[str, str, OptionTier, bool, str], ...] = (
    # ---- Pipeline stage selectors
    ('commands', 'stages', 'core', True, 'Command stage implementation (immediate or delayed).'),
    ('targeting', 'stages', 'core', True, 'Targeting stage implementation (basic or sc2).'),
    ('attack', 'stages', 'core', True, 'Attack timing stage implementation (instant or sc2).'),
    ('abilities', 'stages', 'core', True, 'Abilities stage implementation (none or heal).'),
    (
        'pathing',
        'stages',
        'core',
        True,
        'Global pathing stage (straight, grid, or navmesh; zgame supports straight/navmesh).',
    ),
    (
        'steering',
        'stages',
        'core',
        True,
        'Local steering stage (none, separation, sc2_avoid, or native; zgame omits separation).',
    ),
    (
        'kinematics',
        'stages',
        'core',
        True,
        'Kinematics integration implementation (constant_speed or accel).',
    ),
    ('rl_turn', 'steering', 'legacy_heuristic', False, 'Empirical turning curve for Medivac_RL.'),
    (
        'collision',
        'stages',
        'core',
        True,
        'Collision solver implementation (none, hard, or native).',
    ),
    ('damage', 'stages', 'core', True, 'Damage settlement implementation (standard or sc2).'),
    ('visibility', 'stages', 'core', True, 'Visibility stage implementation (full or sc2).'),
    (
        'vision_levels',
        'visibility',
        'core',
        True,
        'Terrain elevation quantization rule for visibility=sc2 (cell or interp).',
    ),
    # ---- Commands & spawn
    (
        'command_delay',
        'commands',
        'core',
        True,
        'Delay in loops before commands activate under commands=delayed.',
    ),
    (
        'move_lag',
        'commands',
        'core',
        True,
        'Keep previous velocity for 1 loop on command activation under commands=delayed.',
    ),
    (
        'spawn_facing',
        'commands',
        'core',
        True,
        'Initial unit facing mode on reset (zero or deterministic RNG random).',
    ),
    # ---- Targeting & acquisition
    (
        'auto_acquire',
        'targeting',
        'core',
        True,
        'Allow idle and HOLD units to auto-acquire enemies and retaliate on hit.',
    ),
    (
        'auto_acquire_owners',
        'targeting',
        'core',
        True,
        'Restrict idle auto-acquire to specific player IDs (None = all players).',
    ),
    (
        'acquire_start_loop',
        'targeting',
        'core',
        True,
        'Earliest game loop at which idle units begin auto-acquiring enemies.',
    ),
    (
        'acquire_margin',
        'targeting',
        'core',
        True,
        'Extra margin in tiles added to weapon range for auto-acquire checks.',
    ),
    (
        'acquire_min_range',
        'targeting',
        'core',
        True,
        'Minimum edge-to-edge auto-acquire range floor in tiles for targeting=sc2.',
    ),
    (
        'scan_period',
        'targeting',
        'core',
        True,
        'Periodic target scan interval in loops for idle and ATTACK_MOVE units.',
    ),
    (
        'scan_phase',
        'targeting',
        'core',
        True,
        'Periodic scan staggering mode (global, per_unit, tag_order, or tag_rotate).',
    ),
    (
        'retarget',
        'targeting',
        'core',
        True,
        'Allow idle/ATTACK_MOVE units to switch to closer in-range enemies on scan beats.',
    ),
    (
        'retarget_busy',
        'targeting',
        'core',
        True,
        'Post-fire busy window in loops during which retarget scans are deferred.',
    ),
    (
        'retarget_busy_lock',
        'targeting',
        'core',
        True,
        'Use per-unit measured follow_lock - 1 as the retarget busy window.',
    ),
    (
        'retarget_defer',
        'targeting',
        'legacy_heuristic',
        False,
        'Defer all busy-window scans (scan) or only scans wanting a switch (switch).',
    ),
    (
        'retarget_defer_memory',
        'targeting',
        'legacy_heuristic',
        False,
        'Remember deferred switch candidate until busy-window expiry.',
    ),
    (
        'retarget_ready',
        'targeting',
        'legacy_heuristic',
        False,
        'Allow retargeting within acquire range when weapon cooldown is ready.',
    ),
    (
        'retarget_ready_margin',
        'targeting',
        'legacy_heuristic',
        False,
        'Minimum edge-distance advantage required for retarget_ready switches.',
    ),
    (
        'retarget_acquire',
        'targeting',
        'core',
        True,
        'Allow switching to strictly closer enemies in acquire range when out of weapon range.',
    ),
    (
        'retarget_lock',
        'targeting',
        'core',
        True,
        'Target lock policy after retargeting (new, previous, or none).',
    ),
    (
        'retarget_armed_only',
        'targeting',
        'core',
        True,
        'Restrict retarget candidates to armed enemies when leaving an armed target.',
    ),
    (
        'reissue',
        'commands',
        'legacy_heuristic',
        False,
        'Preserve unit timers on same-point ATTACK_MOVE reissue without immediate scan.',
    ),
    (
        'reissue_rescan',
        'commands',
        'legacy_heuristic',
        False,
        'Trigger immediate rescan when same-point ATTACK_MOVE reissue drops a target.',
    ),
    (
        'reissue_ends_wait',
        'commands',
        'legacy_heuristic',
        False,
        'End post-kill rescan wait immediately on same-point ATTACK_MOVE reissue.',
    ),
    (
        'reissue_unit',
        'commands',
        'core',
        True,
        'Skip command activation when reissuing ATTACK_UNIT on the active target.',
    ),
    (
        'kill_rescan_delay',
        'targeting',
        'core',
        True,
        'Loops after target death before rescanning (0 rescans on the death loop).',
    ),
    (
        'kill_rescan_lock',
        'targeting',
        'core',
        True,
        'Defer post-kill rescan until at least last_fire_loop + follow_lock - 1.',
    ),
    (
        'kill_rescan_order',
        'targeting',
        'core',
        True,
        'Decide same-loop vs next-loop post-kill rescan from per-loop update order.',
    ),
    (
        'scan_after_move',
        'targeting',
        'core',
        True,
        'Run periodic idle/ATTACK_MOVE scans after movement and visibility updates.',
    ),
    (
        'retarget_after_move',
        'targeting',
        'core',
        True,
        'Also defer non-busy periodic retarget checks to post-move late_scan.',
    ),
    (
        'hit_assist',
        'targeting',
        'core',
        True,
        'Alert nearby idle and ATTACK_MOVE allies to engage an attacker when an ally is hit.',
    ),
    (
        'hit_assist_range',
        'targeting',
        'core',
        True,
        'Edge-to-edge radius in tiles for alerting allies under hit_assist.',
    ),
    (
        'hit_assist_pick',
        'targeting',
        'core',
        True,
        'Attacker selection rule among hits under hit_assist (first or nearest).',
    ),
    (
        'hit_assist_tie',
        'targeting',
        'core',
        True,
        'Tie-breaker for equidistant hit_assist attackers (first or spawn_order).',
    ),
    (
        'hit_assist_cap',
        'targeting',
        'core',
        True,
        'Maximum hit_assist triggers per victim per loop in update order (0 = uncapped).',
    ),
    (
        'hit_keeps_kill_wait',
        'targeting',
        'core',
        True,
        'Keep post-kill rescan wait active when taking damage during the wait.',
    ),
    (
        'acquire_center',
        'targeting',
        'core',
        True,
        'Rank idle and ATTACK_MOVE scan candidates by center distance instead of edge.',
    ),
    (
        'kill_hold',
        'targeting',
        'core',
        True,
        'Hold position during post-kill rescan wait instead of moving toward order point.',
    ),
    (
        'report_engaged_as_attack_unit',
        'targeting',
        'core',
        True,
        'Output-only view flag reporting auto-engaging units as ATTACK_UNIT in WorldState.',
    ),
    (
        'retreat_owners',
        'targeting',
        'core',
        True,
        'Player IDs whose idle units flee away from attackers instead of retaliating.',
    ),
    (
        'retreat_distance',
        'targeting',
        'core',
        True,
        'Distance in tiles that idle retreat_owners units flee when hit.',
    ),
    (
        'kill_no_retreat',
        'targeting',
        'core',
        True,
        'Suppress idle hit-retreat after an ATTACK_UNIT target dies.',
    ),
    (
        'kill_no_retreat_end',
        'targeting',
        'core',
        True,
        'Expiry rule for kill_no_retreat suppression (command or lock).',
    ),
    # ---- Attack & combat
    (
        'chase_lock',
        'attack',
        'core',
        True,
        'Per-unit-type post-fire hold duration table ((unit_type, loops), ...).',
    ),
    (
        'chase_lock_target',
        'attack',
        'core',
        True,
        'Restrict chase_lock holding to when engaged target matches last_fire_target.',
    ),
    (
        'follow_lock',
        'attack',
        'core',
        True,
        'Enable post-fire follow lock and range-slop holding from measured follow_lock.',
    ),
    (
        'follow_slop',
        'attack',
        'core',
        True,
        'Extra edge-to-edge weapon range slop in tiles for follow_lock and slop_on_arrival.',
    ),
    (
        'slop_on_arrival',
        'attack',
        'core',
        True,
        'Enter range slop immediately on reaching strict weapon range without a prior shot.',
    ),
    (
        'attack_commit',
        'attack',
        'core',
        True,
        'Hold position during committed windups and refund cooldown on canceled/voided shots.',
    ),
    (
        'attack_on_arrival',
        'attack',
        'core',
        True,
        'Reduce windup and cooldown by 1 loop when arriving in range with ready weapon.',
    ),
    (
        'charge',
        'attack',
        'core',
        True,
        'Enable Zealot Charge speed boost and impact bonus from map_abilities.json.',
    ),
    (
        'passive_from_map',
        'targeting',
        'core',
        True,
        'Derive auto_acquire_owners and retreat_owners per map from map_abilities.json.',
    ),
    # ---- Abilities & commands filters
    (
        'stop_keeps_heal',
        'abilities',
        'core',
        True,
        'Ignore STOP commands on units actively channeling MedivacHeal.',
    ),
    (
        'heal_counts_armed',
        'abilities',
        'core',
        True,
        'Treat units actively channeling heal as armed during targeting=sc2 priority.',
    ),
    (
        'amove_needs_weapon',
        'commands',
        'core',
        True,
        'Ignore ATTACK_MOVE commands issued to units without any weapon.',
    ),
    (
        'attack_unit_needs_weapon',
        'commands',
        'core',
        True,
        'Ignore ATTACK_UNIT commands when the unit cannot attack the target domain.',
    ),
    (
        'dead_target_move',
        'commands',
        'core',
        True,
        'Convert ATTACK_UNIT to MOVE toward last position if target dies during delay.',
    ),
    # ---- Damage, vision & energy
    (
        'slow_arrival',
        'damage',
        'core',
        True,
        'Cap final waypoint step by slowed speed so hit-slowed MOVE orders arrive cleanly.',
    ),
    (
        'hit_slow_owner1_only',
        'damage',
        'core',
        True,
        'Apply Concussive Shells hit slow only from Player 1 attackers.',
    ),
    (
        'reveal_loops',
        'visibility',
        'core',
        True,
        'Duration in loops an attacker stays revealed after hitting under visibility=sc2.',
    ),
    (
        'lost_vision',
        'visibility',
        'core',
        True,
        'Handling when an engaged target leaves true sight (immediate or sc2).',
    ),
    (
        'lost_vision_giveup',
        'visibility',
        'core',
        True,
        'Loops before abandoning a lost-vision target under lost_vision=sc2.',
    ),
    (
        'projectile_3d',
        'damage',
        'core',
        True,
        'Compute 3D projectile flight distance using terrain_height elevation.',
    ),
    (
        'arrival_radius',
        'commands',
        'core',
        True,
        'Arrival threshold radius in tiles for MOVE and ATTACK_MOVE orders.',
    ),
    (
        'min_damage',
        'damage',
        'core',
        False,
        'Minimum post-armor health damage per hit (fixed at 0.5 in zgame).',
    ),
    (
        'regen_race_fallback',
        'damage',
        'core',
        True,
        'Apply Protoss shield and Zerg health regen defaults when measured keys are absent.',
    ),
    (
        'spawn_energy',
        'abilities',
        'core',
        True,
        'Initial energy on reset clamped to energy_max (None starts at energy_max).',
    ),
    (
        'heal_autocast',
        'abilities',
        'core',
        True,
        'Enable autocast MedivacHeal for idle and HOLD standard Medivacs.',
    ),
    (
        'heal_on_arrival',
        'abilities',
        'core',
        True,
        'Run an immediate autocast heal scan when a standard Medivac arrives at MOVE.',
    ),
    (
        'heal_facing',
        'abilities',
        'core',
        True,
        'Gate autocast heal channeling on facing alignment instead of fixed startup delay.',
    ),
    (
        'heal_pick',
        'abilities',
        'core',
        True,
        'Autocast heal target priority rule (life = lowest HP; fraction = lowest HP ratio).',
    ),
    # ---- Collision & pinning
    (
        'collision_iterations',
        'collision',
        'core',
        True,
        'Gauss-Seidel overlap projection iterations per loop for collision=hard.',
    ),
    (
        'contact_slack',
        'collision',
        'core',
        True,
        'Allowed radius overlap slack in tiles before collision separation pushes units.',
    ),
    (
        'idle_push_share',
        'collision',
        'core',
        True,
        'Fraction of overlap displacement applied to the idle unit in mover-vs-idle pairs.',
    ),
    (
        'mover_push_share',
        'collision',
        'legacy_heuristic',
        True,
        'Overlap split rule for two moving ground units (zgame supports half and approach).',
    ),
    (
        'mover_push_power',
        'collision',
        'legacy_heuristic',
        False,
        'Speed exponent used by speed and axis mover_push_share modes.',
    ),
    (
        'mover_push_cap',
        'collision',
        'legacy_heuristic',
        True,
        'Upper clamp bound on approach-based overlap split share.',
    ),
    (
        'mover_speed_cap',
        'collision',
        'legacy_heuristic',
        True,
        'Cap combined kinematics plus collision displacement of moving ground units.',
    ),
    (
        'push_clearance',
        'collision',
        'legacy_heuristic',
        False,
        'Re-project ground units out to navmesh clearance after hard collision push.',
    ),
    (
        'arrival_clears_waypoint',
        'collision',
        'legacy_heuristic',
        False,
        'Clear waypoint on kinematics arrival so collision treats arriving unit as idle.',
    ),
    (
        'pin_pair_share',
        'collision',
        'legacy_heuristic',
        False,
        'Overlap split rule between two pinned units (half or younger).',
    ),
    (
        'pin_engaged',
        'collision',
        'core',
        True,
        'Treat stationary HOLD and engaged ground units as immovable pinned obstacles.',
    ),
    (
        'pin_after_fire',
        'collision',
        'legacy_heuristic',
        False,
        'Delay pinning of engaged units until they have fired at their current target.',
    ),
    (
        'pin_lag',
        'collision',
        'legacy_heuristic',
        False,
        'Require two consecutive loops of pinned status before pinning activates.',
    ),
    # ---- Steering avoidance & orbit/sidestep heuristics
    (
        'pin_gap_closed',
        'steering',
        'legacy_heuristic',
        True,
        'Gap diameter multiplier below which sc2_avoid treats a pinned gap as closed.',
    ),
    (
        'pin_slide',
        'steering',
        'legacy_heuristic',
        True,
        'Exclude a unit own engaged/ordered target from sc2_avoid local avoidance.',
    ),
    (
        'pin_slide_mode',
        'steering',
        'legacy_heuristic',
        False,
        'Target exclusion scope for pin_slide (any or pinned).',
    ),
    (
        'pin_slide_orders',
        'steering',
        'legacy_heuristic',
        False,
        'Orders affected by pin_slide exclusion (any or attack_unit).',
    ),
    (
        'contact_orbit',
        'steering',
        'legacy_heuristic',
        True,
        'Enable angular contact-orbit solver around pinned blockers under sc2_avoid.',
    ),
    (
        'orbit_hold',
        'steering',
        'legacy_heuristic',
        True,
        'Loops to hold a chosen contact_orbit turn side before re-evaluating.',
    ),
    (
        'orbit_iters',
        'steering',
        'legacy_heuristic',
        True,
        'Maximum projection iterations for the contact_orbit angular solver.',
    ),
    (
        'orbit_max_turn',
        'steering',
        'legacy_heuristic',
        True,
        'Maximum cumulative orbit turn angle in radians before stopping.',
    ),
    (
        'orbit_movers',
        'steering',
        'legacy_heuristic',
        False,
        'Include moving blockers in contact_orbit (none, right, or same).',
    ),
    (
        'orbit_from',
        'steering',
        'legacy_heuristic',
        False,
        'Reference direction for contact_orbit rotation (steer, prev, or prev_free).',
    ),
    (
        'orbit_lock',
        'steering',
        'legacy_heuristic',
        False,
        'Lock contact_orbit turn direction across consecutive blocked loops.',
    ),
    (
        'pin_sidestep',
        'steering',
        'legacy_heuristic',
        True,
        'Sidestep laterally when a MOVE ground unit is directly behind a pinned blocker.',
    ),
    (
        'pin_sidestep_cone',
        'steering',
        'legacy_heuristic',
        True,
        'Half-cone angle in degrees triggering pin_sidestep behind a blocker.',
    ),
    (
        'pin_sidestep_speed',
        'steering',
        'legacy_heuristic',
        True,
        'Speed multiplier applied while sidestepping around a pinned blocker.',
    ),
    (
        'pin_sidestep_lat',
        'steering',
        'legacy_heuristic',
        False,
        'Optional maximum lateral offset in tiles that triggers pin_sidestep.',
    ),
    (
        'pin_sidestep_lat_scope',
        'steering',
        'legacy_heuristic',
        False,
        'Unit pair filter for pin_sidestep_lat (all or marauder).',
    ),
    (
        'pin_sidestep_drift',
        'steering',
        'legacy_heuristic',
        True,
        'Keep pin_sidestep side consistent with previous-loop lateral drift.',
    ),
    (
        'pin_sidestep_tangent',
        'steering',
        'legacy_heuristic',
        False,
        'Steer tangent to the closest pinned blocker rim under pin_sidestep.',
    ),
    (
        'pin_tie_grid',
        'steering',
        'legacy_heuristic',
        True,
        'Deterministic cluster-outline and grid-parity tie-breaking for pinned blockers.',
    ),
    (
        'range_radius',
        'targeting',
        'core',
        True,
        'Radius source for weapon/acquire range checks (observed or measured).',
    ),
    # ---- Idle return & crowd arrival
    (
        'idle_return',
        'collision',
        'core',
        True,
        'Return never-commanded idle units to their spawn anchor after being pushed.',
    ),
    (
        'return_threshold',
        'collision',
        'core',
        True,
        'Minimum offset in tiles from home anchor to trigger idle_return.',
    ),
    (
        'return_far',
        'collision',
        'core',
        True,
        'Distance in tiles from home anchor that triggers immediate idle_return.',
    ),
    (
        'return_delay',
        'collision',
        'core',
        True,
        'Wait delay in loops after a push before initiating idle_return.',
    ),
    (
        'idle_return_rule',
        'collision',
        'legacy_heuristic',
        False,
        'Displacement trigger rule for idle_return (legacy or diameter).',
    ),
    (
        'return_ratio',
        'collision',
        'legacy_heuristic',
        False,
        'Unit diameter multiplier threshold for idle_return_rule=diameter.',
    ),
    (
        'crowd_arrival',
        'collision',
        'legacy_heuristic',
        True,
        'Clear MOVE orders when units are blocked around a shared destination.',
    ),
    (
        'crowd_goal_tol',
        'collision',
        'legacy_heuristic',
        True,
        'Maximum distance in tiles between goals treated as the same crowd destination.',
    ),
    (
        'crowd_progress',
        'collision',
        'legacy_heuristic',
        True,
        'Minimum per-loop progress fraction before counting a loop as crowd-blocked.',
    ),
    (
        'crowd_loops',
        'collision',
        'legacy_heuristic',
        True,
        'Consecutive blocked loops required to clear MOVE under crowd_arrival.',
    ),
    (
        'crowd_chain',
        'collision',
        'legacy_heuristic',
        True,
        'Immediately clear MOVE when touching a stopped ally with the same goal.',
    ),
    (
        'crowd_cover',
        'collision',
        'legacy_heuristic',
        True,
        'Clear MOVE next loop when a contacting non-idle ally covers the goal point.',
    ),
    # ---- Avoidance & separation parameters
    (
        'avoid_range',
        'steering',
        'legacy_heuristic',
        True,
        'Maximum edge distance in tiles to consider a neighbor in sc2_avoid.',
    ),
    (
        'avoid_margin',
        'steering',
        'legacy_heuristic',
        True,
        'Extra clearance margin in tiles for closest-approach avoidance triggers.',
    ),
    (
        'avoid_horizon',
        'steering',
        'legacy_heuristic',
        True,
        'Lookahead horizon in loops for sc2_avoid closest-approach prediction.',
    ),
    (
        'avoid_closing_only',
        'steering',
        'legacy_heuristic',
        False,
        'Skip closest-approach avoidance for non-closing moving neighbors.',
    ),
    (
        'avoid_parallel_faster_only',
        'steering',
        'legacy_heuristic',
        True,
        'Skip avoidance for parallel movers unless this unit has higher max speed.',
    ),
    (
        'idle_nudge',
        'steering',
        'legacy_heuristic',
        True,
        'Rightward lateral nudge in tiles when heading straight into an idle unit.',
    ),
    (
        'avoid_side_check',
        'steering',
        'legacy_heuristic',
        True,
        'Flip head-on avoidance side when a parallel ally occupies the preferred side.',
    ),
    (
        'avoid_side_gap',
        'steering',
        'legacy_heuristic',
        True,
        'Gap distance in tiles checked for adjacent allies under avoid_side_check.',
    ),
    (
        'follow_gap',
        'steering',
        'legacy_heuristic',
        True,
        'Edge gap threshold in tiles for speed-matching behind a parallel mover.',
    ),
    (
        'avoid_cross_cos',
        'steering',
        'legacy_heuristic',
        True,
        'Upper cosine bound routing small-angle convergences out of crossing branch.',
    ),
    (
        'avoid_share',
        'steering',
        'legacy_heuristic',
        True,
        'Avoidance turn sharing rule between two movers (full or speed).',
    ),
    (
        'avoid_share_power',
        'steering',
        'legacy_heuristic',
        True,
        'Speed exponent scaling turn share under avoid_share=speed.',
    ),
    (
        'avoid_share_power_headon',
        'steering',
        'legacy_heuristic',
        False,
        'Optional head-on branch override for avoid_share_power.',
    ),
    (
        'avoid_share_power_cross',
        'steering',
        'legacy_heuristic',
        False,
        'Optional crossing branch override for avoid_share_power.',
    ),
    (
        'separation_gain',
        'steering',
        'legacy_heuristic',
        False,
        'Repulsion velocity gain per tile of overlap for steering=separation.',
    ),
    (
        'separation_margin',
        'steering',
        'legacy_heuristic',
        False,
        'Edge distance threshold in tiles activating steering=separation repulsion.',
    ),
    (
        'stalker_shield_regen',
        'damage',
        'core',
        True,
        'Populate default shield regen parameters for Stalker and Stalker_RL units.',
    ),
    # ---- Q12 native fixed-point movement, steering, collision & combat timing
    (
        'native_blend',
        'steering',
        'native',
        True,
        'Blend waypoint direction with previous committed Q12 step under steering=native.',
    ),
    (
        'native_relation',
        'collision',
        'native',
        True,
        'Same-priority Q12 contact yield relation bit (0 or 1) for allied pairs.',
    ),
    (
        'native_enemy_relation',
        'collision',
        'native',
        True,
        'Same-priority Q12 contact yield relation bit (0 or 1) for enemy pairs.',
    ),
    (
        'native_idle_share',
        'collision',
        'native',
        True,
        'Q12 yield share out of 4096 applied to stationary units pushed by movers.',
    ),
    (
        'native_speed_rule',
        'collision',
        'native',
        True,
        'Q12 push-share rule for movers with different max speeds (table/faster/slower).',
    ),
    (
        'native_second_pass',
        'collision',
        'native',
        True,
        'Second Q12 contact resolution pass selection (contacted or none).',
    ),
    (
        'native_avoid',
        'steering',
        'native',
        True,
        'Compute desired heading via sc2_avoid before Q12 direction blending.',
    ),
    (
        'native_share_rule',
        'collision',
        'native',
        True,
        'Q12 contact push-share resolution rule (legacy or priority_tree).',
    ),
    (
        'native_batch_flags',
        'collision',
        'native',
        True,
        'Enable 4-pass live contact/wall flag tracking and cross-frame contact state.',
    ),
    (
        'native_nbr_order',
        'collision',
        'native',
        True,
        'Neighbor enumeration order in native collision batches (slot or quantized).',
    ),
    (
        'native_steer_avoid',
        'steering',
        'native',
        True,
        'Enable native ground local steering avoidance with lateral accel clamping.',
    ),
    (
        'native_move_reissue',
        'commands',
        'native',
        True,
        'Skip zero-tolerance Q12 same-point MOVE reissues and preserve stop progress.',
    ),
    (
        'native_stop',
        'steering',
        'native',
        True,
        'Enable native arrival deceleration, blocked stop tolerance, and early stop.',
    ),
    (
        'native_air_sep',
        'steering',
        'native',
        True,
        'Enable native air-unit stationary separation and moving flock aggregation.',
    ),
    (
        'native_wall_clearance',
        'collision',
        'native',
        True,
        'Enable axis-aligned blocking-edge wall clearance, bisection, and wall sliding.',
    ),
    (
        'native_assist_on_hit',
        'targeting',
        'native',
        True,
        'Enable Q12 hit deduplication, ally wake-up, threat memory, and center scoring.',
    ),
    (
        'native_backswing',
        'attack',
        'native',
        True,
        'Enable weapon backswing lock, dead-target retention, and deferred slot promotion.',
    ),
    (
        'native_retarget',
        'targeting',
        'native',
        True,
        'Enable unified Q12 target-scoring retarget with hysteresis and odd-beat throttling.',
    ),
    (
        'native_scan_cadence',
        'targeting',
        'native',
        True,
        'Unify 8-loop scan cadence across target acquisition and Medivac autocast healing.',
    ),
    (
        'native_loop_order',
        'damage',
        'native',
        True,
        'Run two-pass damage settlement with per-loop update-order shuffling.',
    ),
)

OPTIONS_META: Mapping[str, OptionMeta] = MappingProxyType(
    {
        name: OptionMeta(
            name=name,
            group=group,
            tier=tier,
            default=getattr(_DEFAULTS, name),
            zgame_supported=zgame_supported,
            summary=summary,
        )
        for name, group, tier, zgame_supported, summary in _SPECS
    }
)


def _validate_choice(kind: str, value: str, choices: tuple[str, ...]) -> None:
    if value in choices:
        return
    hint = difflib.get_close_matches(value, choices, n=3, cutoff=0.5)
    suggestion = f' Did you mean {", ".join(repr(h) for h in hint)}?' if hint else ''
    raise ValueError(f'Unknown {kind} {value!r}; valid choices: {list(choices)}.{suggestion}')


def describe_pyref_options(
    *,
    group: str | None = None,
    tier: str | None = None,
    zgame_only: bool = False,
) -> tuple[OptionMeta, ...]:
    """Return `OptionMeta` entries filtered by `group`, `tier`, and/or `zgame_only`.

    Raises `ValueError` with close-match suggestions if `group` or `tier` is unknown.
    """
    if group is not None:
        _validate_choice('group', group, OPTION_GROUPS)
    if tier is not None:
        _validate_choice('tier', tier, OPTION_TIERS)
    return tuple(
        meta
        for meta in OPTIONS_META.values()
        if (group is None or meta.group == group)
        and (tier is None or meta.tier == tier)
        and (not zgame_only or meta.zgame_supported)
    )


__all__ = [
    'OPTIONS_META',
    'OPTION_GROUPS',
    'OPTION_TIERS',
    'OptionMeta',
    'describe_pyref_options',
]
