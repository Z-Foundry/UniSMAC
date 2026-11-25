"""Shared profile-subset helpers for zgame parity tests."""

from __future__ import annotations

from typing import Any

M1_FIELDS: frozenset[str] = frozenset(
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

M2_FIELDS: frozenset[str] = M1_FIELDS | frozenset(
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

SUB_PROFILES: dict[str, str] = {'v4': 'aligned-v4', 'axo': 'acc-ram-cov-axo'}


def profile_subset_options(name: str) -> dict[str, Any]:
    """Return non-default PyrefConfig options for a named subset ('v4', 'axo', '*-m1', '*-m2')."""
    from unismac.profiles import profile_overrides

    if name.endswith('-m1'):
        prof, keep = name[:-3], M1_FIELDS
    elif name.endswith('-m2'):
        prof, keep = name[:-3], M2_FIELDS
    else:
        prof, keep = name, None
    _, over = profile_overrides(SUB_PROFILES[prof])
    out: dict[str, Any] = {}
    for key, value in over.items():
        if key == 'backend.pyref_options':
            out.update(value)
        elif key.startswith('backend.pyref_'):
            out[key.removeprefix('backend.pyref_')] = value
    if keep is not None:
        out = {k: v for k, v in out.items() if k in keep}
    return out
