"""Environment configuration: nested frozen dataclasses, flat name mapping, and override merging.

Merge precedence: **preset < scenario defaults < preset scenario overrides < user overrides**
(see `resolve_config`).

* Presets (`unismac.presets`) supply a complete `EnvConfig`.
* Scenario defaults come from the scenario registry: `termination.episode_limit`,
  `players.agent_race`, and `players.enemy_race` (`None` in a preset means "use scenario value").
* User overrides are a dict keyed by dotted paths (`'reward.win'`), upstream `StarCraft2Env` flat
  parameter names (`'reward_win'`, see `FLAT_NAMES`), or nested dicts (`{'reward': {'win': 100}}`).
  Unknown keys and type mismatches raise `ValueError` rather than being ignored. Nested dicts
  merge field-by-field into existing groups, except the flat key `capability_config`, which
  replaces the entire `capability` group (matching upstream).
* Preset scenario overrides (`Preset.scenario_overrides`) provide per-scenario defaults, such as
  SMACv2 `10gen_*` procedural `capability` and `use_unit_ranges`.

Every upstream behavioral difference is represented as an explicit config field rather than
branching on preset names. See field comments and `FLAT_NAMES` for details.
"""

from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    'ActionConfig',
    'AttackConfig',
    'BackendConfig',
    'CapabilityConfig',
    'CatalogConfig',
    'DIFFICULTY_CODES',
    'EnemyMaskConfig',
    'EnvConfig',
    'FLAT_NAMES',
    'HealthConfig',
    'MapsConfig',
    'ObsConfig',
    'OpponentConfig',
    'PlayersConfig',
    'QuirksConfig',
    'RewardConfig',
    'StartPositionsConfig',
    'StateConfig',
    'TeamGenConfig',
    'TerminationConfig',
    'UnitsConfig',
    'VisibilityConfig',
    'apply_overrides',
    'resolve_config',
    'scenario_defaults',
]

Race = Literal['terran', 'zerg', 'protoss', 'random']
Difficulty = Literal[
    'very_easy',
    'easy',
    'medium',
    'medium_hard',
    'hard',
    'harder',
    'very_hard',
    'cheat_vision',
    'cheat_money',
    'cheat_insane',
]

# Upstream SMAC difficulty code (e.g. difficulty='7') -> contract difficulty name.
DIFFICULTY_CODES: dict[str, str] = {
    '1': 'very_easy',
    '2': 'easy',
    '3': 'medium',
    '4': 'medium_hard',
    '5': 'hard',
    '6': 'harder',
    '7': 'very_hard',
    '8': 'cheat_vision',
    '9': 'cheat_money',
    'A': 'cheat_insane',
}


@dataclass(frozen=True)
class BackendConfig:
    """Backend execution and spawning configuration."""

    step_mul: int = 8  # Game loops advanced per environment step
    # map_trigger: spawn units via map triggers (matches upstream; requires map_trigger capability);
    # explicit: spawn units from `unismac.scenarios.spawns` records.
    spawn_mode: Literal['map_trigger', 'explicit'] = 'map_trigger'
    # Unit variant used for explicit spawns: recorded uses recorded variants (SMAC allies are *_RL);
    # std forces standard units (new_maps has no *_RL units); smac uses Variant.RL only for Player 1
    # Medivacs (disabling Medivac_RL auto-heal) and Variant.STD for all other units (preserving
    # standard combat timing with bitwise parity between pyref and zgame).
    spawn_variant: Literal['recorded', 'std', 'smac'] = 'recorded'
    # Clamp spawn coordinates into the playable area (SMACv2 samples the full 32x32 map and may
    # fall outside (2, 2, 30, 30); SMAC-Hard 7q_vs_2bc spawns Battlecruisers near y=0.56; SC2
    # relocates them while pyref rejects out-of-bounds spawns). Applies to procedural and explicit.
    spawn_clamp: bool = False
    # Mirror spawning for two-sided mode (opponent.kind=external): auto mirrors Player 1 spawns
    # (position and facing) to Player 2 on mirrorable maps (see unismac.env.dual.mirror_axis);
    # off uses spawn records verbatim. Applies only to non-procedural explicit spawns.
    spawn_mirror: Literal['auto', 'off'] = 'auto'
    # Forwarded to `PyrefConfig` (without the `pyref_` prefix) when constructing pyref/zgame by
    # name: ability subsystem (none | heal), initial unit energy (None = max), and standard Medivac
    # idle/HOLD heal autocast. Ignored when passing a pre-built backend instance.
    pyref_abilities: Literal['none', 'heal'] = 'none'
    pyref_spawn_energy: float | None = None
    pyref_heal_autocast: bool = False
    # Command latency settings forwarded to PyrefConfig (commands / command_delay / move_lag) to
    # model SC2 command execution latency.
    pyref_commands: Literal['immediate', 'delayed'] = 'immediate'
    pyref_command_delay: int = 0
    pyref_move_lag: int = 0
    # Extra `PyrefConfig` fields (combat timing, collision avoidance, pathing, etc.): accepts a dict
    # in overrides, stored as sorted `((name, value), ...)`. Validated by `PyrefConfig`; fields with
    # dedicated attributes above (abilities, spawn_energy, heal_autocast, commands, command_delay,
    # move_lag) cannot appear here. Example:
    # `{'backend.pyref_options': {'attack': 'sc2', 'damage': 'sc2', 'auto_acquire': True}}`.
    pyref_options: tuple[Any, ...] = ()
    # SC2 backend only (`Sc2Backend.slot_order`): order same-owner slots from map_trigger spawns by
    # engine order (`engine`) or ascending tag (`tag`, creation order; used by SMAC-Hard scripts).
    sc2_slot_order: Literal['engine', 'tag'] = 'engine'
    # Map-specific unit stat overrides (`Scenario.unit_overrides`): asset loads entries from
    # `assets/unit_overrides.json` (e.g. new_maps/6m_vs_10m Player 1 Marine has 55 HP); off disables
    # overrides (for debugging). pyref applies them; SC2 validates them after reset.
    unit_overrides: Literal['asset', 'off'] = 'asset'


@dataclass(frozen=True)
class MapsConfig:
    """Map directory overrides; `None` uses the registry's `map_id` (`SMAC_Maps/<file>`)."""

    dir: str | None = None  # Map dir for non-SC2 backends (and SC2 when sc2_dir is None)
    sc2_dir: str | None = None  # SC2-only map dir (smac-portable uses new_maps with 2 user slots)


@dataclass(frozen=True)
class PlayersConfig:
    """Player races; `None` inherits `a_race` / `b_race` from the scenario registry."""

    agent_race: Race | None = None
    enemy_race: Race | None = None


@dataclass(frozen=True)
class OpponentConfig:
    """Opponent (Player 2) configuration."""

    # engine_ai: built-in SC2 AI (ComputerPlayer, requires engine_ai capability);
    # opening_orders: player 2 issues a single opening attack-move toward player 1's
    # centroid and then relies on engine auto-acquire;
    # native_smac: player 2 reproduces opening orders and Medivac escort healing across
    # all 23 classic SMAC maps (see unismac.env.opponents.smac_bot);
    # scripted: player 2 runs decision-tree scripts ported from SMAC-Hard
    # (unismac.env.opponents.scripted), sampled from the scenario's script pool per episode;
    # external: player 2 is controlled by caller-supplied enemy_actions in step()
    # (two-sided / self-play mode, see unismac.env.dual).
    kind: Literal['engine_ai', 'opening_orders', 'native_smac', 'scripted', 'external'] = (
        'engine_ai'
    )
    difficulty: Difficulty = 'very_hard'  # engine_ai difficulty (upstream difficulty='7')
    # Target point for opening_orders: ally_centroid = centroid of Player 1 spawn positions.
    target: Literal['ally_centroid'] = 'ally_centroid'
    # Script pool for scripted opponents (names in unismac.env.opponents.scripted.SCRIPTS);
    # None uses the scenario's upstream SCRIPT_DICT pool (duplicates act as sampling weights).
    scripts: tuple[str, ...] | None = None
    # Handling of scripted ability commands (Stim, Blink, Siege, etc. - not move/attack/stop):
    # issue forwards them (SC2); drop ignores them (pyref does not implement these abilities).
    script_abilities: Literal['issue', 'drop'] = 'issue'


@dataclass(frozen=True)
class UnitsConfig:
    """Unit view configuration: enemy visibility source, death rule, and ordering."""

    # visible: enemies outside Player 1's visibility are treated as health 0 (matching upstream);
    # all: use ground-truth `alive`. SMAC_Maps disables fog of war, so both match on SC2.
    enemy_source: Literal['visible', 'all'] = 'visible'
    # Ally agent ordering: type_xy sorts by (type_code, x, y) (upstream SMAC sorts by native
    # unit_type, pos.x, pos.y); slot preserves backend slot order.
    ally_order: Literal['type_xy', 'slot'] = 'type_xy'
    # Enemy ordering: slot preserves backend slot order (SMAC obs order); type_xy sorts by
    # (native unit_type, x, y) (SMAC-Hard opponent controller obs order).
    enemy_order: Literal['slot', 'type_xy'] = 'slot'


@dataclass(frozen=True)
class CatalogConfig:
    """Unit catalog: type encoding, max shield, cooldown normalization, shoot/sight ranges."""

    # smac: encode allies via map_type role table + min_unit_type offset (upstream SMAC rule);
    # standard: encode both sides via map_type branches on standard unit_type (SMAC-Hard new_maps),
    # sorting allies by native unit_type.
    type_id: Literal['smac', 'standard'] = 'smac'
    # table: upstream unit_max_shield table (Stalker 80, Zealot 50, Colossus 150); static: use each
    # unit's own shield_max (SMAC-Hard uses unit.shield_max).
    shield_max: Literal['table', 'static'] = 'table'
    # table: upstream unit_max_cooldown table (Medivac uses max energy 200), defaulting to 15.
    cooldown_max: Literal['table'] = 'table'
    # fixed: uniform shoot_range / sight_range across units; per_unit: SMACv2 use_unit_ranges table,
    # clamping shoot range to max(table_value, min_attack_range).
    ranges: Literal['fixed', 'per_unit'] = 'fixed'
    shoot_range: float = 6.0
    sight_range: float = 9.0
    min_attack_range: float = 2.0  # Used only when ranges='per_unit'


@dataclass(frozen=True)
class ObsConfig:
    """Observation feature toggles matching upstream parameters."""

    all_health: bool = True  # Implies own_health=True (upstream behavior)
    own_health: bool = True
    last_action: bool = False
    pathing_grid: bool = False
    terrain_height: bool = False
    timestep_number: bool = False
    own_pos: bool = False  # SMACv2 obs_own_pos: include (x / map_x, y / map_y) in Own block


@dataclass(frozen=True)
class StateConfig:
    last_action: bool = True
    timestep_number: bool = False
    obs_instead_of_state: bool = False


@dataclass(frozen=True)
class ActionConfig:
    move_amount: float = 2.0  # Target move distance in map tiles
    # mmm_rule: on MMM-like maps, Medivac targets are non-Medivac allies (heal ability 386).
    heal: Literal['mmm_rule', 'off'] = 'mmm_rule'
    # Target segment length: enemies = n_enemies (SMAC; errors if heal targets exceed n_enemies);
    # mmm_rule = n_agents on MMM-like maps when n_agents > n_enemies (SMACv2);
    # max_units_9 = max(n_enemies, n_agents, 9) (SMAC-Hard padding).
    target_pad: Literal['enemies', 'mmm_rule', 'max_units_9'] = 'enemies'
    # SMAC-Hard use_ability: append an equal-length ability segment (requires ability_query).
    ability: bool = False
    # Require targets to be within shoot range (circle) or range cone (cone) in action masks.
    # False matches SMACv2 action_mask=False (all alive targets selectable, checked on dispatch).
    mask_by_range: bool = True
    # Check range when translating actions to commands and drop out-of-range attacks (SMACv2
    # get_agent_action). No-op when mask_by_range=True.
    range_check: bool = True


@dataclass(frozen=True)
class VisibilityConfig:
    """Observation visibility model."""

    # circle: dist < sight_range; cone: SMACv2 conic_fov (within sight range and facing sector,
    # adding n_fov FovTurn actions).
    model: Literal['circle', 'cone'] = 'circle'
    n_fov: int = 12  # Used only when model='cone': number of turn actions, sector = 2*pi / n_fov
    fully_observable: bool = False  # SMACv2: alive enemies and allies are always visible
    # SMACv2 spotter: when an enemy is first spotted by an agent, each other agent also observes it
    # with probability prob_obs_enemy; unspotted enemies have zeroed features.
    spotter_gate: bool = False
    prob_obs_enemy: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    sparse: bool = False
    only_positive: bool = True
    # abs: upstream SMAC abs(delta_enemy + delta_deaths) (enemy shield regen counts positively);
    # clamp: max(0, ...) (SMACv2 / SMAC-Hard).
    only_positive_mode: Literal['abs', 'clamp'] = 'abs'
    death_value: float = 10
    win: float = 200
    defeat: float = 0
    negative_scale: float = 0.5
    scale: bool = True
    scale_rate: float = 20


@dataclass(frozen=True)
class TerminationConfig:
    episode_limit: int | None = None  # None inherits limit from scenario registry
    continuing_episode: bool = False  # Include episode_limit=True in info on timeout
    # last_unit_is_medivac: win on MMM-like maps when 1 enemy survives and is a Medivac (SMAC);
    # no_non_medivac_left: win when no non-Medivac enemy survives (SMACv2); off: disabled.
    # Ally "only Medivacs left -> loss" is always active on MMM-like maps.
    enemy_medivac_rule: Literal['last_unit_is_medivac', 'no_non_medivac_left', 'off'] = (
        'last_unit_is_medivac'
    )


@dataclass(frozen=True)
class TeamGenConfig:
    """SMACv2 `team_gen`: sample ally team by weights; mirror to enemy (pad or truncate)."""

    dist_type: Literal['weighted_teams'] = 'weighted_teams'
    unit_types: tuple[str, ...] = ()  # SMACv2 unit names, e.g. marine, marauder, medivac
    weights: tuple[float, ...] = ()
    exception_unit_types: tuple[str, ...] = ()  # Reject teams composed exclusively of these units
    observe: bool = True  # False zeroes ally type bits


@dataclass(frozen=True)
class StartPositionsConfig:
    """SMACv2 `start_positions`: `surrounded`, `reflect_position`, or `surrounded_and_reflect`."""

    dist_type: Literal['surrounded_and_reflect', 'surrounded', 'reflect_position'] = (
        'surrounded_and_reflect'
    )
    p: float = 0.5  # surrounded_and_reflect: choose surrounded when random() > p
    map_x: float = 32
    map_y: float = 32


@dataclass(frozen=True)
class AttackConfig:
    """SMACv2 `attack` (`stochastic_attack`): per-agent hit probability ~ U(lower, upper)."""

    dist_type: Literal['per_agent_uniform'] = 'per_agent_uniform'
    lower_bound: tuple[float, ...] = (1.0,)
    upper_bound: tuple[float, ...] = (1.0,)
    observe: bool = False  # Expose attack_probability in ally features (zeroed if False)


@dataclass(frozen=True)
class HealthConfig:
    """SMACv2 `health` (`stochastic_health`). Requires mid-step unit kill primitive, which is not
    currently in the backend contract; enabling raises `CapabilityError` at construction."""

    dist_type: Literal['per_agent_uniform'] = 'per_agent_uniform'
    lower_bound: tuple[float, ...] = (0.0,)
    upper_bound: tuple[float, ...] = (0.0,)
    observe: bool = False


@dataclass(frozen=True)
class EnemyMaskConfig:
    """SMACv2 `enemy_mask`: `mask[agent, enemy] ~ choice([0, 1], p=[mp, 1 - mp])`; when 1, hides
    that enemy's health, shield, and type from that agent."""

    dist_type: Literal['mask'] = 'mask'
    mask_probability: float = 0.5


@dataclass(frozen=True)
class CapabilityConfig:
    """SMACv2 `capability_config`. `None` entries are disabled; `n_units` / `n_enemies` set team
    sizes when `team_gen` is enabled, and otherwise must be `None` or match the scenario.

    Per-episode sampling order is fixed to `team_gen`, `start_positions`, `attack`, `health`,
    `enemy_mask` (matching SMACv2 example YAML key order), all drawn from one `generators` stream.
    """

    n_units: int | None = None
    n_enemies: int | None = None
    team_gen: TeamGenConfig | None = None
    start_positions: StartPositionsConfig | None = None
    attack: AttackConfig | None = None
    health: HealthConfig | None = None
    enemy_mask: EnemyMaskConfig | None = None

    @property
    def enabled(self) -> tuple[str, ...]:
        names = ('team_gen', 'start_positions', 'attack', 'health', 'enemy_mask')
        return tuple(n for n in names if getattr(self, n) is not None)


@dataclass(frozen=True)
class QuirksConfig:
    """Compatibility switches reproducing upstream SMAC/SMACv2/SMAC-Hard edge behaviors."""

    # When obs_timestep_number=True, appends a zero dim to Own and reports get_obs_size() off by 1.
    timestep_pad: bool = False
    # Upstream transposes + flips terrain_height so obs reads grid[H-1-y, x] (square maps only);
    # disabled reads grid[y, x].
    terrain_height_flip: bool = False
    # When a spotter gate is 0 during the per-enemy loop, zeroes the entire enemy feature block
    # written so far; disabled zeroes only that enemy's row.
    enemy_block_reset: bool = False
    # Upstream SMACv2 applies map_type encoding to allied *_RL units too (silently mis-encoding
    # MMM, 1c3s5z, bane). Paths that raise IndexError (e.g. 2s3z) are not reproduced.
    ally_type_override: bool = False
    # Shuffle agent evaluation order in get_obs using the obs_shuffle stream (affects spotter tags).
    obs_shuffle: bool = False
    # Under catalog.type_id='standard', compute role IDs as min_unit_type + role_index (matching
    # upstream _init_ally_unit_types misalignment on non-contiguous standard IDs, e.g. MMM
    # medivac_id=50 vs Medivac 54). Disabled matches roles directly by native unit_type.
    role_offset: bool = False


@dataclass(frozen=True)
class EnvConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    players: PlayersConfig = field(default_factory=PlayersConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    units: UnitsConfig = field(default_factory=UnitsConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    state: StateConfig = field(default_factory=StateConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    capability: CapabilityConfig = field(default_factory=CapabilityConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    quirks: QuirksConfig = field(default_factory=QuirksConfig)

    def get(self, path: str) -> Any:
        """Look up a nested value by dotted path, e.g. `cfg.get('reward.win')`."""
        obj: Any = self
        for part in path.split('.'):
            obj = getattr(obj, part)
        return obj

    def flat(self) -> dict[str, Any]:
        """Return `{dotted_path: value}` for all leaf fields."""
        out: dict[str, Any] = {}

        def walk(obj: Any, prefix: str) -> None:
            for f in dataclasses.fields(obj):
                v = getattr(obj, f.name)
                if dataclasses.is_dataclass(v):
                    walk(v, f'{prefix}{f.name}.')
                else:
                    out[f'{prefix}{f.name}'] = v

        walk(self, '')
        return out


# Upstream StarCraft2Env flat parameter names -> dotted config paths.
FLAT_NAMES: dict[str, str] = {
    'step_mul': 'backend.step_mul',
    'move_amount': 'actions.move_amount',
    'difficulty': 'opponent.difficulty',
    'continuing_episode': 'termination.continuing_episode',
    'obs_all_health': 'obs.all_health',
    'obs_own_health': 'obs.own_health',
    'obs_last_action': 'obs.last_action',
    'obs_pathing_grid': 'obs.pathing_grid',
    'obs_terrain_height': 'obs.terrain_height',
    'obs_instead_of_state': 'state.obs_instead_of_state',
    'obs_timestep_number': 'obs.timestep_number',
    'state_last_action': 'state.last_action',
    'state_timestep_number': 'state.timestep_number',
    'reward_sparse': 'reward.sparse',
    'reward_only_positive': 'reward.only_positive',
    'reward_death_value': 'reward.death_value',
    'reward_win': 'reward.win',
    'reward_defeat': 'reward.defeat',
    'reward_negative_scale': 'reward.negative_scale',
    'reward_scale': 'reward.scale',
    'reward_scale_rate': 'reward.scale_rate',
    'episode_limit': 'termination.episode_limit',
    # SMACv2
    'conic_fov': 'visibility.model',
    'num_fov_actions': 'visibility.n_fov',
    'fully_observable': 'visibility.fully_observable',
    'prob_obs_enemy': 'visibility.prob_obs_enemy',
    'use_unit_ranges': 'catalog.ranges',
    'min_attack_range': 'catalog.min_attack_range',
    'obs_own_pos': 'obs.own_pos',
    'action_mask': 'actions.mask_by_range',
    'capability_config': 'capability',
    # SMAC-Hard
    'use_ability': 'actions.ability',
}

# Flat keys that replace their entire group rather than merging field-by-field.
_REPLACE_GROUPS = {'capability_config'}

_RACE_CODES = {'T': 'terran', 'P': 'protoss', 'Z': 'zerg', 'R': 'random'}


def _normalize(path: str, value: Any) -> Any:
    if path == 'opponent.difficulty' and isinstance(value, str | int):
        return DIFFICULTY_CODES.get(str(value), value)
    if path in ('players.agent_race', 'players.enemy_race') and isinstance(value, str):
        return _RACE_CODES.get(value, value)
    if path == 'visibility.model' and isinstance(value, bool):
        return 'cone' if value else 'circle'
    if path == 'catalog.ranges' and isinstance(value, bool):
        return 'per_unit' if value else 'fixed'
    if path == 'backend.pyref_options':
        return pyref_options(value)
    return value


PYREF_NAMED = (
    'abilities',
    'spawn_energy',
    'heal_autocast',
    'commands',
    'command_delay',
    'move_lag',
)


def pyref_options(value: Any) -> tuple[tuple[str, Any], ...]:
    """Normalize `backend.pyref_options` to sorted `((name, value), ...)` (lists -> tuples)."""
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, tuple | list):
        items = list(value)
    else:
        raise ValueError(
            f'backend.pyref_options must be a dict or sequence of (name, value), got {value!r}'
        )
    out: dict[str, Any] = {}
    for item in items:
        if not (isinstance(item, tuple | list) and len(item) == 2 and isinstance(item[0], str)):
            raise ValueError(
                f'backend.pyref_options must be a dict or sequence of (name, value), got {value!r}'
            )
        key, v = item
        if key in PYREF_NAMED:
            raise ValueError(
                f'backend.pyref_options cannot contain {key!r}; use backend.pyref_{key}'
            )
        out[key] = tuple(v) if isinstance(v, list) else v
    return tuple(sorted(out.items()))


def _check_type(path: str, hint: Any, value: Any) -> None:
    """Validate `value` against `hint` (Literal, bool/int/float/str, tuple, Optional)."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        errors = []
        for arm in typing.get_args(hint):
            try:
                _check_type(path, arm, value)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f'{path}={value!r} does not match {hint}')
    if hint is type(None):
        if value is not None:
            raise ValueError(f'{path} must be None')
        return
    if origin is tuple:
        if not isinstance(value, tuple):
            raise ValueError(f'{path} must be a sequence, got {value!r}')
        arg = typing.get_args(hint)[0]
        for i, v in enumerate(value):
            _check_type(f'{path}[{i}]', arg, v)
        return
    if origin is Literal:
        if value not in typing.get_args(hint):
            raise ValueError(f'{path}={value!r} is not in allowed values {typing.get_args(hint)}')
        return
    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f'{path} must be bool, got {value!r}')
    elif hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{path} must be int, got {value!r}')
    elif hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f'{path} must be numeric, got {value!r}')
    elif hint is str and not isinstance(value, str):
        raise ValueError(f'{path} must be str, got {value!r}')


def _resolve_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls, globalns=globals())


def _group_class(hint: Any) -> type | None:
    """Return the dataclass type if `hint` is a (possibly optional) config group."""
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        for arm in typing.get_args(hint):
            if dataclasses.is_dataclass(arm) and isinstance(arm, type):
                return arm
    return None


def _set(obj: Any, parts: list[str], value: Any, path: str) -> Any:
    names = {f.name for f in dataclasses.fields(obj)}
    head = parts[0]
    if head not in names:
        candidates = sorted(names | (set(FLAT_NAMES) if isinstance(obj, EnvConfig) else set()))
        matches = difflib.get_close_matches(head, candidates, n=3, cutoff=0.5)
        hint = f'; did you mean {", ".join(repr(m) for m in matches)}?' if matches else ''
        raise ValueError(
            f'Unknown config key {path!r} ({type(obj).__name__} has no field {head!r}{hint})'
        )
    hint = _resolve_hints(type(obj))[head]
    group = _group_class(hint)
    cur = getattr(obj, head)
    if group is not None and cur is None and (len(parts) > 1 or isinstance(value, Mapping)):
        cur = group()  # Instantiate default optional group (e.g. capability.attack) before merging
    if len(parts) == 1:
        if group is not None and value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f'{path} is a config group and must be overridden with a dict')
            new = cur
            for k, v in value.items():
                new = _set(new, [str(k)], v, f'{path}.{k}')
            return dataclasses.replace(obj, **{head: new})
        value = _normalize(path, value)
        if isinstance(value, list) and typing.get_origin(hint) is tuple:
            value = tuple(value)
        _check_type(path, hint, value)
        return dataclasses.replace(obj, **{head: value})
    if group is None:
        if head == 'pyref_options':
            subkey = '.'.join(parts[1:])
            raise ValueError(
                f'Unknown config key {path!r} ({head} is a dict, not a nested config group; '
                f"pass {{'backend.pyref_options': {{{subkey!r}: ...}}}} instead)"
            )
        raise ValueError(f'Unknown config key {path!r} ({head} is not a group)')
    return dataclasses.replace(obj, **{head: _set(cur, parts[1:], value, path)})


def apply_overrides(config: EnvConfig, overrides: Mapping[str, Any] | None) -> EnvConfig:
    """Return a new config with `overrides` applied (dotted paths, flat names, or group dicts).

    The flat key `capability_config` replaces the entire `capability` group (matching upstream);
    all other group dicts merge field-by-field.
    """
    if not overrides:
        return config
    for key, value in overrides.items():
        path = FLAT_NAMES.get(key, key)
        if key in _REPLACE_GROUPS:
            config = dataclasses.replace(config, capability=CapabilityConfig())
        config = _set(config, path.split('.'), value, path)
    return config


def scenario_defaults(spec: Any) -> dict[str, Any]:
    """Derive defaults from a `unismac.scenarios.ScenarioSpec`."""
    return {
        'termination.episode_limit': int(spec.limit),
        'players.agent_race': _RACE_CODES[spec.a_race],
        'players.enemy_race': _RACE_CODES[spec.b_race],
    }


def resolve_config(
    preset: EnvConfig,
    spec: Any,
    overrides: Mapping[str, Any] | None = None,
    scenario_overrides: Mapping[str, Any] | None = None,
) -> EnvConfig:
    """Merge preset < scenario defaults < preset scenario overrides < user overrides.

    Scenario defaults fill only fields set to `None` in `preset`.
    """
    defaults = {k: v for k, v in scenario_defaults(spec).items() if preset.get(k) is None}
    config = apply_overrides(apply_overrides(preset, defaults), scenario_overrides)
    return apply_overrides(config, overrides)
