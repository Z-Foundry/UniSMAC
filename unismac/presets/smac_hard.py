"""`smac-hard` and `smac-hard-portable` presets for the SMAC-Hard benchmark.

`smac-hard` reproduces upstream SMAC-Hard `StarCraft2Env` defaults: a dual-agent session on
`new_maps/<scenario>` where Player 2 is controlled by a decision-tree script
(`unismac.env.scripted`, sampled each episode from the scenario's script pool). The only
intentional departure from upstream defaults is `use_ability`: upstream defaults to True, while
here it defaults to False (SMAC-Hard paper experiments disabled abilities, which strongly affects
training); enable via override `{'use_ability': True}`. That flag only controls the agent's
ability action segment; opponent script abilities are controlled separately by
`opponent.script_abilities`, which keeps V2's `'issue'` default (arXiv:2412.17707 used v1 where
both agent and scripts disabled abilities by default). Shared SMAC settings (reward, observation,
`timestep_pad`, `terrain_height_flip`) follow upstream SMAC.

| Option | smac-hard | smac-hard-portable |
| --- | --- | --- |
| `backend.spawn_mode` | `map_trigger` | `explicit` (`spawns_hard.json`) |
| `backend.sc2_slot_order` | `tag` | `tag` |
| `backend.spawn_clamp` | -- | on (`7q_vs_2bc` spawns lie outside playable area) |
| `backend.pyref_*` | -- | heal as in `smac-portable`; command delay below |
| `opponent.kind` | `scripted` | `scripted` |
| `opponent.script_abilities` | `issue` | `drop` (pyref lacks Stim/Blink etc.) |
| `actions.ability` (`use_ability`) | off (upstream on; see above) | off (no ability query) |
| `actions.target_pad` | `max_units_9` | `max_units_9` |
| `catalog.type_id` / `shield_max` | `standard` / `static` | same |
| `quirks.*` | `timestep_pad`, `terrain_height_flip`, `role_offset` | off |

Upstream correspondence and deviations:

* **Slot order**: upstream sorts `self.agents` / `self.enemies` by `(unit_type, x, y)` with
  observation-order tie-breaking. Because dual-agent observation order varies in SC2, the backend
  uses `slot_order='tag'` (ascending tag order) with `enemy_order='type_xy'` so scripts see the
  same unit ordering as upstream.
* **Stepping**: upstream launches with `realtime=True` and steps only controller 0, making
  progression wall-clock dependent and non-reproducible. UniSMAC always uses lockstep, advancing
  both players by `step_mul` game loops per env step.
* **Restarts**: upstream `_restart` kills alive units with `DebugKillUnit`, calls `step(1)`, and
  waits for map triggers to respawn. `Sc2Backend`'s `map_trigger` reset is semantically identical
  on normal restarts (no separate config option), but avoids an upstream realtime bug where
  `init_units` counts units without excluding stale tags and can return unkilled units from the
  previous episode ("ghost episodes").
* **Ability segment**: upstream `agent_ability_dict` takes the first available ability returned
  by the engine; UniSMAC takes the smallest ID (each unit has at most one listed ability, so both
  agree; see `unismac.env.abilities`).
* **`role_offset` quirk**: upstream `_init_ally_unit_types` assumes `*_RL` custom unit types and
  computes roles by offset from `min_unit_type`; on `new_maps` both sides use standard types, so
  the offset maps to wrong roles (e.g., on MMM, Marine cooldown cap becomes 25 and Medivac is not
  recognized as a healer). `role_offset=True` reproduces this; portable disables it.
* **Script fix**: two upstream `7q_vs_2bc` scripts missed a parameter and raised `TypeError` on
  invocation; fixed during porting.
* Scenarios without a script pool (`2m_vs_1z`, `2s_vs_1sc`, `so_many_baneling`, `bane_vs_bane`,
  `unit_test`, `pvt_large`) or without map files (`3s_vs_3z`, `3s_vs_4z`) raise on construction;
  `unit_test` and `pvt_large` crash SC2 4.10 on load.

`smac-hard-portable` configures pyref with `commands='delayed'`, `command_delay=1`, and
`move_lag=1` to match dual-agent command latency (orders and displacement match SC2). Combat
timing options (`attack` / `damage` / `spawn_facing` / `auto_acquire`) default to off because
they were calibrated on SMAC units and not on SMAC-Hard additions (Battlecruiser, SiegeTank,
Phoenix, Viking, Queen, etc.). Enable via overrides such as
`{'backend.pyref_options': {'attack': 'sc2', 'damage': 'sc2', 'spawn_facing': 'random',
'auto_acquire': True, 'acquire_start_loop': 2}}` (see `unismac.env.config.BackendConfig`).
"""

from __future__ import annotations

from ..env.config import (
    ActionConfig,
    BackendConfig,
    CatalogConfig,
    EnvConfig,
    OpponentConfig,
    QuirksConfig,
    RewardConfig,
    UnitsConfig,
)

__all__ = ['SMAC_HARD', 'SMAC_HARD_PORTABLE']

_UNITS = UnitsConfig(enemy_source='all', ally_order='type_xy', enemy_order='type_xy')
_CATALOG = CatalogConfig(type_id='standard', shield_max='static')
_REWARD = RewardConfig(only_positive_mode='clamp')  # upstream: max(0, delta_enemy + delta_deaths)

SMAC_HARD = EnvConfig(
    backend=BackendConfig(step_mul=8, spawn_mode='map_trigger', sc2_slot_order='tag'),
    opponent=OpponentConfig(kind='scripted'),
    units=_UNITS,
    catalog=_CATALOG,
    actions=ActionConfig(
        heal='mmm_rule', target_pad='max_units_9', ability=False, range_check=False
    ),
    reward=_REWARD,
    quirks=QuirksConfig(timestep_pad=True, terrain_height_flip=True, role_offset=True),
)

SMAC_HARD_PORTABLE = EnvConfig(
    backend=BackendConfig(
        step_mul=8,
        spawn_mode='explicit',
        spawn_variant='std',
        spawn_clamp=True,  # 7q_vs_2bc Battlecruisers spawn outside playable area
        sc2_slot_order='tag',
        pyref_abilities='heal',
        pyref_spawn_energy=50.0,
        pyref_heal_autocast=True,
        pyref_commands='delayed',
        pyref_command_delay=1,
        pyref_move_lag=1,
    ),
    opponent=OpponentConfig(kind='scripted', script_abilities='drop'),
    units=_UNITS,
    catalog=_CATALOG,
    actions=ActionConfig(
        heal='mmm_rule', target_pad='max_units_9', ability=False, range_check=False
    ),
    reward=_REWARD,
    quirks=QuirksConfig(timestep_pad=False, terrain_height_flip=False, role_offset=False),
)
