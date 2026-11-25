"""`smacv2` and `smacv2-portable` presets.

`smacv2` matches upstream SMACv2 `StarCraft2Env.__init__` defaults. For `10gen_*` scenarios,
`scenario_overrides` (precedence: scenario defaults < here < user overrides) layers the
`env_args` from `examples/configs/sc2_gen_*.yaml`: `capability_config` (5v5 `weighted_teams` and
`surrounded_and_reflect`), `obs_own_pos`, and `use_unit_ranges`. Variants like 10v10 or 10v11
replace `capability_config` wholesale, matching upstream.

SMACv2 mechanisms that differ from `smac`:

| Option | smac | smacv2 |
| --- | --- | --- |
| `reward.only_positive_mode` | `abs` | `clamp` (`max(., 0)`) |
| `termination.enemy_medivac_rule` | `last_unit_is_medivac` | `no_non_medivac_left` |
| `actions.target_pad` | `enemies` | `mmm_rule` |
| `visibility.spotter_gate` | off | on |
| `quirks.timestep_pad` | on | off (absent in SMACv2) |
| `quirks.enemy_block_reset / ally_type_override / obs_shuffle` | -- | on |

Upstream bugs in `marine_id` on marines maps, returning `None` on spawn failure, and
`agent_features` name ordering are fixed directly without toggle flags. `capability.health`
(`stochastic_health`) is not implemented and raises `CapabilityError` if enabled.

`10gen_*` scenarios use `explicit` spawning on `SMAC_Maps/32x32_flat` (sampled per episode from
capability config, with `*_RL` allies); the other 23 classic maps use `map_trigger`.

`smacv2-portable` combines SMACv2 features/actions with `smac-portable` mechanics: `explicit`
spawning, standard units (`spawn_variant='std'`), `opening_orders` opponent, `enemy_source='all'`,
and all quirks off. Medivac healing matches `smac-portable` (`actions.heal='mmm_rule'`, pyref
`abilities='heal'`, `spawn_energy=50`, `heal_autocast=True`; standard Medivacs autocast while
idle/HOLD whereas `Medivac_RL` does not, see `presets/smac.py`). Generated spawn positions are
clamped into the playable area (`backend.spawn_clamp`, whereas upstream samples across the full
32x32 map). `10gen_*` scenarios are currently unavailable on SC2 under `smacv2-portable`:
`SMAC_Maps/32x32_flat` exposes only 1 user slot (`NoSpaceForUser` on JoinGame), `opening_orders`
requires 2, and `new_maps` has no `32x32_flat`, so construction raises `CapabilityError`.

**Intentional deviations from upstream** (`smacv2` preset):

* Randomness: upstream gives each distribution an unseeded `np.random.default_rng()`, uses
  `random.shuffle` in `get_obs`, and uses global/temporary Generators for spotter and
  `stochastic_attack`, preventing reproducibility. UniSMAC draws counter-based Philox streams
  keyed by `(seed, stream, env_id, episode, step)` (`env/rng.py`): procedural generation shares
  one `generators` stream per episode in `capability_config` key order, and agent ordering uses
  `Generator.permutation` instead of `random.shuffle`. Distributions match statistically, though
  values do not match unseeded upstream runs draw-for-draw.
* `ally_type_override`: on `stalkers_and_zealots` maps (`2s3z`, `3s5z`), upstream computes
  `unit_type - min_unit_type` for `*_RL` units, exceeding `unit_type_bits` and raising
  `IndexError` in `get_obs`; UniSMAC encodes by role index so the environment runs cleanly.
* `max_reward` is tracked per env (`max_rewards[B]`, since procedural scenarios vary in initial
  composition across envs); the `max_reward` property returns env 0's value.
* `capability.health` (`stochastic_health`) is unimplemented because the contract has no
  mid-step unit-kill primitive.
* `obs_instead_of_state=True`: upstream calls `get_obs` a second time inside `get_state`,
  consuming extra agent-shuffle and spotter draws each step; UniSMAC draws once per step and
  reuses that step's obs for state, so RNG stream counts differ while formulas match.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..env.config import (
    ActionConfig,
    BackendConfig,
    EnvConfig,
    MapsConfig,
    OpponentConfig,
    QuirksConfig,
    RewardConfig,
    TerminationConfig,
    UnitsConfig,
    VisibilityConfig,
)

__all__ = [
    'GEN_CAPABILITY',
    'SMACV2',
    'SMACV2_PORTABLE',
    'smacv2_overrides',
    'smacv2_portable_overrides',
]

_START = {'dist_type': 'surrounded_and_reflect', 'p': 0.5, 'map_x': 32, 'map_y': 32}

# capability_config from examples/configs/sc2_gen_*.yaml.
GEN_CAPABILITY: dict[str, dict[str, Any]] = {
    '10gen_terran': {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {
            'dist_type': 'weighted_teams',
            'unit_types': ('marine', 'marauder', 'medivac'),
            'weights': (0.45, 0.45, 0.1),
            'exception_unit_types': ('medivac',),
            'observe': True,
        },
        'start_positions': _START,
    },
    '10gen_zerg': {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {
            'dist_type': 'weighted_teams',
            'unit_types': ('zergling', 'baneling', 'hydralisk'),
            'weights': (0.45, 0.1, 0.45),
            'exception_unit_types': ('baneling',),
            'observe': True,
        },
        'start_positions': _START,
    },
    '10gen_protoss': {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': {
            'dist_type': 'weighted_teams',
            'unit_types': ('stalker', 'zealot', 'colossus'),
            'weights': (0.45, 0.45, 0.1),
            'observe': True,
        },
        'start_positions': _START,
    },
}

SMACV2 = EnvConfig(
    backend=BackendConfig(step_mul=8, spawn_mode='map_trigger'),
    opponent=OpponentConfig(kind='engine_ai', difficulty='very_hard'),
    units=UnitsConfig(enemy_source='visible', ally_order='type_xy'),
    actions=ActionConfig(target_pad='mmm_rule', mask_by_range=True, range_check=True),
    visibility=VisibilityConfig(spotter_gate=True, prob_obs_enemy=1.0),
    reward=RewardConfig(only_positive_mode='clamp'),
    termination=TerminationConfig(enemy_medivac_rule='no_non_medivac_left'),
    quirks=QuirksConfig(
        timestep_pad=False,
        terrain_height_flip=True,
        enemy_block_reset=True,
        ally_type_override=True,
        obs_shuffle=True,
    ),
)

SMACV2_PORTABLE = EnvConfig(
    backend=BackendConfig(
        step_mul=8,
        spawn_mode='explicit',
        spawn_variant='std',
        spawn_clamp=True,
        pyref_abilities='heal',
        pyref_spawn_energy=50.0,
        pyref_heal_autocast=True,
    ),
    maps=MapsConfig(dir=None, sc2_dir='new_maps'),
    opponent=OpponentConfig(kind='opening_orders'),
    units=UnitsConfig(enemy_source='all', ally_order='type_xy'),
    actions=ActionConfig(
        heal='mmm_rule', target_pad='mmm_rule', mask_by_range=True, range_check=True
    ),
    visibility=VisibilityConfig(spotter_gate=True, prob_obs_enemy=1.0),
    reward=RewardConfig(only_positive_mode='clamp'),
    termination=TerminationConfig(enemy_medivac_rule='no_non_medivac_left'),
    quirks=QuirksConfig(timestep_pad=False, terrain_height_flip=False),
)


def _gen_overrides(spec: Any) -> dict[str, Any]:
    if spec.name not in GEN_CAPABILITY:
        return {}
    return {
        'capability_config': GEN_CAPABILITY[spec.name],
        'obs_own_pos': True,
        'use_unit_ranges': True,
        'min_attack_range': 2,
    }


def smacv2_overrides(spec: Any) -> Mapping[str, Any]:
    """Scenario overrides for `smacv2`: YAML env_args and `explicit` spawning for `10gen_*`."""
    out = _gen_overrides(spec)
    if out:
        out['backend.spawn_mode'] = 'explicit'
    return out


def smacv2_portable_overrides(spec: Any) -> Mapping[str, Any]:
    """Scenario overrides for `smacv2-portable`: same `10gen_*` env_args (`explicit` spawning)."""
    return _gen_overrides(spec)
