"""Named simulator configuration profiles for UniSMAC.

Each profile defines a default backend (``'pyref'`` or ``'sc2'``) and a dictionary
of environment overrides applied on top of a preset. All ``pyref``-family profiles
execute identically on both ``backend='zgame'`` (recommended for speed) and
``backend='pyref'``.

Public profiles (`PUBLIC_PROFILES`, returned by `list_profiles()`):

* ``'sc2-native'`` (`STANDARD_PROFILE`): Recommended default profile. Enables
  Q12 fixed-point movement, steering, collision, stop dynamics, combat/targeting
  timing, assist-on-hit, race-fallback regeneration, health-fraction autocast
  healing, and map-driven passive behavior.
* ``'sc2-explore'`` (`STANDARD_EXPLORE_PROFILE`): Fast short-budget exploration
  profile using the same Q12 movement and map-driven passive behavior with
  simplified combat/targeting (no early assist-on-hit barrier).
* ``'sc2-symmetric'`` (`STANDARD_SYMMETRIC_PROFILE`): Symmetric two-sided edition
  of ``'sc2-native'`` for ``opponent.kind='external'`` self-play
  (``auto_acquire_owners=()``, ``retreat_owners=(1, 2)``,
  ``hit_slow_owner1_only=False``, ``passive_from_map=False``).
* ``'sc2-live'``: Live StarCraft II backend with no simulator overrides.
* ``'minimal'``: Minimal baseline using only preset-provided backend defaults.

For reproducibility of historical alignment and ablation runs, `LEGACY_PROFILES`
preserves 17 backward-compatible profile names (available in `PROFILES` and via
`list_profiles(include_legacy=True)`).

Command-latency fields are routed through ``backend.pyref_commands``,
``backend.pyref_command_delay``, and ``backend.pyref_move_lag``; all other
simulator parameters are routed through ``backend.pyref_options`` and can be
deep-merged with custom overrides via `profile_overrides` or `merge_overrides`.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    'LEGACY_PROFILES',
    'PROFILES',
    'PUBLIC_PROFILES',
    'Profile',
    'STANDARD_EXPLORE_PROFILE',
    'STANDARD_PROFILE',
    'STANDARD_PROFILES',
    'STANDARD_SYMMETRIC_PROFILE',
    'backend_label',
    'list_profiles',
    'merge_overrides',
    'parse_pyref_options',
    'profile_overrides',
    'pyref_extra_overrides',
]

_DELAY = {
    'backend.pyref_commands': 'delayed',
    'backend.pyref_command_delay': 1,
    'backend.pyref_move_lag': 1,
}

# ---------------------------------------------------------------- Public net profile definitions

# Core non-default options shared by sc2-explore, sc2-native, and sc2-symmetric.
_SC2_NATIVE_CORE: dict[str, Any] = {
    # Stages
    'targeting': 'sc2',
    'attack': 'sc2',
    'pathing': 'navmesh',
    'steering': 'native',
    'kinematics': 'accel',
    'collision': 'native',
    'damage': 'sc2',
    # Q12 fixed-point movement, steering, collision, and wall clearance
    'spawn_facing': 'random',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'idle_return': True,
    'pin_engaged': True,
    'native_share_rule': 'priority_tree',
    'native_batch_flags': True,
    'native_nbr_order': 'quantized',
    'native_steer_avoid': True,
    'native_move_reissue': True,
    'native_stop': True,
    'native_air_sep': True,
    'native_wall_clearance': True,
    # Targeting, retargeting, and command reissue
    'auto_acquire': True,
    'auto_acquire_owners': (2,),
    'retreat_owners': (1,),
    'passive_from_map': True,
    'scan_phase': 'per_unit',
    'scan_after_move': True,
    'retarget': True,
    'retarget_lock': 'none',
    'retarget_armed_only': True,
    'retarget_busy_lock': True,
    'retarget_acquire': True,
    'retarget_after_move': True,
    'reissue_unit': True,
    'kill_hold': True,
    'kill_rescan_lock': True,
    'kill_no_retreat': True,
    'kill_no_retreat_end': 'lock',
    'report_engaged_as_attack_unit': True,
    'chase_lock': ((48, 12), (73, 8), (74, 8)),
    'chase_lock_target': True,
    'follow_lock': True,
    'slop_on_arrival': True,
    # Attack, damage, regen, and Medivac heal
    'attack_commit': True,
    'attack_on_arrival': True,
    'amove_needs_weapon': True,
    'attack_unit_needs_weapon': True,
    'dead_target_move': True,
    'slow_arrival': True,
    'hit_slow_owner1_only': True,
    'regen_race_fallback': True,
    'stop_keeps_heal': True,
    'heal_on_arrival': True,
    'heal_facing': True,
    'heal_pick': 'fraction',
}

_NAT_COMBAT_OPTIONS: dict[str, Any] = {
    'native_backswing': True,
    'native_retarget': True,
    'hit_assist': True,
    'native_assist_on_hit': True,
    'native_scan_cadence': True,
    'native_loop_order': True,
    'kill_rescan_delay': 0,
    'kill_rescan_order': True,
}

_SYM_OPTIONS: dict[str, Any] = {'auto_acquire_owners': (), 'retreat_owners': (1, 2)}

_SC2_EXPLORE: dict[str, Any] = {**_SC2_NATIVE_CORE, 'acquire_start_loop': 2}
_SC2_NATIVE: dict[str, Any] = {**_SC2_NATIVE_CORE, **_NAT_COMBAT_OPTIONS}
_SC2_SYMMETRIC: dict[str, Any] = {
    **_SC2_NATIVE,
    **_SYM_OPTIONS,
    'hit_slow_owner1_only': False,
    'passive_from_map': False,
}

# ---------------------------------------------------------------- Legacy historical profiles

_ALIGNED_OPTIONS = {
    'collision': 'hard',
    'steering': 'sc2_avoid',
    'contact_slack': 0.0625,
    'idle_push_share': 0.9,
    'idle_return': True,
    'crowd_arrival': True,
    'avoid_side_check': True,
    'pathing': 'navmesh',
    'attack': 'sc2',
    'damage': 'sc2',
    'spawn_facing': 'random',
}
_ACQ_OPTIONS = {'auto_acquire': True, 'auto_acquire_owners': (2,), 'acquire_start_loop': 2}
_FULL_OPTIONS = {'targeting': 'sc2', 'retreat_owners': (1,)}
_V3_OPTIONS = {
    'chase_lock': ((48, 12), (73, 8), (74, 8)),
    'stop_keeps_heal': True,
    'heal_counts_armed': True,
    'amove_needs_weapon': True,
}
_V3 = {**_ALIGNED_OPTIONS, **_ACQ_OPTIONS, **_FULL_OPTIONS, **_V3_OPTIONS}
_V4_OPTIONS = {'pin_engaged': True, 'kill_hold': True, 'report_engaged_as_attack_unit': True}
_V4 = {**_V3, **_V4_OPTIONS}
_V5_OPTIONS = {'retarget': True, 'pin_slide': True, 'scan_phase': 'per_unit'}
_V5 = {**_V4, **_V5_OPTIONS}
_V6_OPTIONS = {'pin_slide_orders': 'attack_unit'}
_V6 = {**_V5, **_V6_OPTIONS}
_AXO_OPTIONS = {
    'retarget_ready': False,
    'retarget_lock': 'none',
    'follow_lock': True,
    'reissue_unit': True,
    'attack_commit': True,
    'attack_on_arrival': True,
    'avoid_share': 'speed',
    'avoid_share_power': 4,
    'chase_lock_target': True,
    'slop_on_arrival': True,
    'attack_unit_needs_weapon': True,
    'slow_arrival': True,
    'retarget_armed_only': True,
    'hit_slow_owner1_only': True,
    'heal_counts_armed': False,
    'retarget_busy_lock': True,
    'kill_rescan_lock': True,
    'retarget_acquire': True,
    'scan_after_move': True,
    'dead_target_move': True,
    'mover_speed_cap': True,
    'kill_no_retreat': True,
    'kill_no_retreat_end': 'lock',
    'pin_sidestep': True,
    'pin_sidestep_drift': True,
    'pin_tie_grid': True,
    'avoid_parallel_faster_only': True,
    'kinematics': 'accel',
    'heal_on_arrival': True,
    'heal_facing': True,
    'retarget_after_move': True,
    'crowd_cover': True,
    'avoid_cross_cos': 0.8,
    'contact_orbit': True,
    'mover_push_share': 'approach',
}
_AXO = {**_V5, **_AXO_OPTIONS}
_RGHF_OPTIONS = {'regen_race_fallback': True, 'heal_pick': 'fraction'}
_AXO_RGHF = {**_AXO, **_RGHF_OPTIONS}
_V4_SYM = {**_V4, **_SYM_OPTIONS}
_AXO_SYM = {**_AXO, **_SYM_OPTIONS, 'hit_slow_owner1_only': False}


BackendName = Literal['sc2', 'pyref']


class Profile:
    """Named simulator configuration bundle."""

    def __init__(
        self, name: str, backend: BackendName, overrides: dict[str, Any], doc: str
    ) -> None:
        self.name = name
        self.backend = backend
        self.overrides = overrides
        self.doc = doc


PUBLIC_PROFILES: dict[str, Profile] = {
    'sc2-native': Profile(
        'sc2-native',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_NATIVE)},
        'Recommended default profile: full Q12 fixed-point movement, collision, '
        'combat/targeting timing, assist-on-hit, and map-driven passive behavior',
    ),
    'sc2-explore': Profile(
        'sc2-explore',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_EXPLORE)},
        'Short-budget exploration profile: Q12 fixed-point movement/collision '
        'with simplified combat targeting (no early assist-on-hit barrier)',
    ),
    'sc2-symmetric': Profile(
        'sc2-symmetric',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_SYMMETRIC)},
        'Symmetric two-sided edition of sc2-native for opponent.kind=external '
        'self-play and head-to-head evaluation',
    ),
    'sc2-live': Profile(
        'sc2-live',
        'sc2',
        {},
        'Live StarCraft II backend with no simulator overrides',
    ),
    'minimal': Profile(
        'minimal',
        'pyref',
        {},
        'Minimal baseline using only preset-provided backend defaults',
    ),
}

LEGACY_PROFILES: dict[str, Profile] = {
    'sc2': Profile('sc2', 'sc2', {}, 'Legacy alias of sc2-live (live StarCraft II backend)'),
    'preset': Profile('preset', 'pyref', {}, 'Legacy alias of minimal (preset defaults only)'),
    'aligned': Profile(
        'aligned',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_ALIGNED_OPTIONS)},
        'Legacy baseline: delayed commands, hard collision/avoidance, navmesh, combat timing',
    ),
    'aligned-acq': Profile(
        'aligned-acq',
        'pyref',
        {**_DELAY, 'backend.pyref_options': {**_ALIGNED_OPTIONS, **_ACQ_OPTIONS}},
        'Legacy baseline: aligned plus Player 2 idle auto-acquire',
    ),
    'aligned-full': Profile(
        'aligned-full',
        'pyref',
        {
            **_DELAY,
            'backend.pyref_options': {**_ALIGNED_OPTIONS, **_ACQ_OPTIONS, **_FULL_OPTIONS},
        },
        'Legacy baseline: aligned-acq plus SC2 targeting rules and Player 1 hit retreat',
    ),
    'aligned-v3': Profile(
        'aligned-v3',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_V3)},
        'Legacy baseline: aligned-full plus chase lock and heal/attack-move adjustments',
    ),
    'aligned-v3-noretreat': Profile(
        'aligned-v3-noretreat',
        'pyref',
        {
            **_DELAY,
            'backend.pyref_options': {k: v for k, v in _V3.items() if k != 'retreat_owners'},
        },
        'Legacy baseline: aligned-v3 without Player 1 idle retreat',
    ),
    'aligned-v4': Profile(
        'aligned-v4',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_V4)},
        'Legacy transfer baseline: aligned-v3 plus pinned engaged units, kill hold, '
        'and ATTACK_UNIT engagement reporting',
    ),
    'aligned-v5': Profile(
        'aligned-v5',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_V5)},
        'Legacy baseline: aligned-v4 plus retargeting, pin slide, and per-unit scan phase',
    ),
    'aligned-v6': Profile(
        'aligned-v6',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_V6)},
        'Legacy baseline: aligned-v5 restricting pin slide to ATTACK_UNIT orders',
    ),
    'acc-ram-cov-axo': Profile(
        'acc-ram-cov-axo',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_AXO)},
        'Legacy floating-point heuristic baseline: aligned-v5 plus 35 combat, '
        'avoidance, pushing, and acceleration options',
    ),
    'acc-ram-cov-axo-rghf': Profile(
        'acc-ram-cov-axo-rghf',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_AXO_RGHF)},
        'Legacy floating-point heuristic baseline: acc-ram-cov-axo plus race-fallback '
        'regeneration and health-fraction autocast healing',
    ),
    'rghf-nat': Profile(
        'rghf-nat',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_EXPLORE)},
        'Legacy alias of sc2-explore',
    ),
    'rghf-nat-cbt': Profile(
        'rghf-nat-cbt',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_NATIVE)},
        'Legacy alias of sc2-native',
    ),
    'aligned-v4-sym': Profile(
        'aligned-v4-sym',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_V4_SYM)},
        'Legacy two-sided symmetric edition of aligned-v4',
    ),
    'acc-ram-cov-axo-sym': Profile(
        'acc-ram-cov-axo-sym',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_AXO_SYM)},
        'Legacy two-sided symmetric edition of acc-ram-cov-axo',
    ),
    'rghf-nat-cbt-sym': Profile(
        'rghf-nat-cbt-sym',
        'pyref',
        {**_DELAY, 'backend.pyref_options': dict(_SC2_SYMMETRIC)},
        'Legacy alias of sc2-symmetric',
    ),
}

PROFILES: dict[str, Profile] = {**PUBLIC_PROFILES, **LEGACY_PROFILES}

STANDARD_PROFILE: str = 'sc2-native'
STANDARD_EXPLORE_PROFILE: str = 'sc2-explore'
STANDARD_SYMMETRIC_PROFILE: str = 'sc2-symmetric'
STANDARD_PROFILES: tuple[str, ...] = (
    STANDARD_PROFILE,
    STANDARD_EXPLORE_PROFILE,
    STANDARD_SYMMETRIC_PROFILE,
)


def _suggest(name: str, choices: Mapping[str, Any] | set[str] | list[str]) -> str:
    matches = difflib.get_close_matches(name, list(choices), n=3, cutoff=0.5)
    if not matches:
        return ''
    return f'; did you mean {", ".join(repr(m) for m in matches)}?'


def list_profiles(include_legacy: bool = False) -> list[str]:
    """Return profile names (the 5 public profiles by default, or all 22 with legacy aliases)."""
    return list(PROFILES) if include_legacy else list(PUBLIC_PROFILES)


def profile_overrides(
    name: str, pyref_options: Mapping[str, Any] | None = None
) -> tuple[BackendName, dict[str, Any]]:
    """Return ``(default_backend, overrides_dict)`` for a named profile.

    When non-empty ``pyref_options`` are provided, they are deep-merged into
    ``backend.pyref_options`` (only valid for ``pyref``-family profiles).
    """
    try:
        p = PROFILES[name]
    except KeyError:
        hint = _suggest(name, PROFILES)
        raise ValueError(
            f'Unknown profile {name!r}{hint} '
            f'Available public profiles: {list(PUBLIC_PROFILES)} '
            '(call list_profiles(include_legacy=True) for legacy aliases)'
        ) from None
    over = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.overrides.items()}
    if not pyref_options:
        return p.backend, over
    if p.backend != 'pyref':
        raise ValueError(
            f'pyref_options only applies to pyref-family profiles; '
            f'{name!r} uses backend {p.backend!r}'
        )
    return p.backend, merge_overrides(over, {'backend.pyref_options': dict(pyref_options)})


def merge_overrides(prof: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Overlay caller overrides on top of profile overrides without mutating inputs.

    When both mappings define ``backend.pyref_options``, their dictionaries are
    merged key-by-key with caller entries taking precedence.
    """
    merged = {**prof, **(overrides or {})}
    if 'backend.pyref_options' in prof and overrides and 'backend.pyref_options' in overrides:
        merged['backend.pyref_options'] = {
            **prof['backend.pyref_options'],
            **dict(overrides['backend.pyref_options']),
        }
    return merged


def pyref_extra_overrides(
    profile: str, pyref_options: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Wrap ``pyref_options`` into ``{'backend.pyref_options': ...}`` for pyref-family profiles."""
    backend, _ = profile_overrides(profile)
    if backend != 'pyref' or not pyref_options:
        return None
    return {'backend.pyref_options': dict(pyref_options)}


def parse_pyref_options(text: str | None) -> dict[str, Any] | None:
    """Parse and validate a JSON object of ``PyrefConfig`` field overrides."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'--pyref-options is not valid JSON: {exc}') from None
    if not isinstance(value, dict):
        raise ValueError(f'--pyref-options must be a JSON object, got {value!r}')
    import dataclasses

    from .backends.pyref import PyrefConfig
    from .env.config import PYREF_NAMED

    named = sorted(set(value) & set(PYREF_NAMED))
    if named:
        raise ValueError(
            f'--pyref-options cannot contain promoted fields {named}; '
            'set them via backend.pyref_<name> instead'
        )
    known = {f.name for f in dataclasses.fields(PyrefConfig)}
    unknown = sorted(set(value) - known)
    if unknown:
        hints = [
            f'{k!r} -> {m[0]!r}'
            for k in unknown
            if (m := difflib.get_close_matches(k, sorted(known), n=1, cutoff=0.5))
        ]
        suffix = f' (did you mean: {", ".join(hints)}?)' if hints else ''
        raise ValueError(f'--pyref-options contains unknown PyrefConfig fields {unknown}{suffix}')
    return value


def backend_label(profile: str, backend: str | None = None) -> str:
    """Return a normalized run label like ``'sc2'`` or ``'zgame-sc2-native'``."""
    own, _ = profile_overrides(profile)
    if own == 'sc2':
        return 'sc2'
    if backend not in (None, 'pyref', 'zgame'):
        raise ValueError(f'Unknown backend {backend!r}; expected pyref or zgame')
    return f'{backend or own}-{profile}'
