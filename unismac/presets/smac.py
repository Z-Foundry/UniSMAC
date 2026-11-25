"""`smac` and `smac-portable` presets.

`smac` matches the default arguments of upstream `StarCraft2Env.__init__`. Differences in
`smac-portable`:

| Option | smac | smac-portable |
| --- | --- | --- |
| `backend.spawn_mode` | `map_trigger` | `explicit` (`spawns.json`) |
| `backend.spawn_variant` | -- | `std` (`new_maps` and pyref lack `*_RL` units) |
| `maps.sc2_dir` | -- | `new_maps` (2 user slots) |
| `opponent.kind` | `engine_ai` | `opening_orders` |
| `units.enemy_source` | `visible` | `all` |
| `backend.pyref_*` | -- | `abilities='heal'`, `spawn_energy=50`, `heal_autocast=True` |
| `quirks.*` | on | off |

Both presets use `actions.heal='mmm_rule'` (on MMM-type maps, Medivac attack actions target
allies to heal within upstream `shoot_range=6`; actual cast range is edge-to-edge 4, so both
SC2 and pyref move closer automatically).

**Medivac autocast healing**: `smac-portable` uses standard Medivacs (`new_maps` and pyref have
no `*_RL` units), which enable Heal autocast by default in SC2: while idle or on HOLD with
energy >= 5, they automatically heal nearby injured biological allies. pyref reproduces this via
`heal_autocast=True`. SMAC's `Medivac_RL` does not autocast and heals only on explicit heal
orders, so `smac-portable` Medivacs have additional idle-healing behavior. Explicitly spawned
Medivacs start with 50 energy (verified in SC2), matched in pyref by `spawn_energy=50`.

**pyref alignment options default to off**: command delay, collision avoidance, navmesh, and
combat timing use pyref defaults unless enabled via overrides, e.g.::

    {'backend.pyref_commands': 'delayed', 'backend.pyref_command_delay': 1,
     'backend.pyref_move_lag': 1,
     'backend.pyref_options': {'attack': 'sc2', 'damage': 'sc2', 'spawn_facing': 'random',
                               'auto_acquire': True, 'auto_acquire_owners': (2,),
                               'acquire_start_loop': 2}}

Combat timing parameters are calibrated from empirical measurements on SMAC units; see
`unismac.profiles` (e.g., `aligned`, `aligned-acq`, `aligned-v4`) for curated alignment profiles.
"""

from __future__ import annotations

from ..env.config import (
    ActionConfig,
    BackendConfig,
    EnvConfig,
    MapsConfig,
    OpponentConfig,
    QuirksConfig,
    UnitsConfig,
)

__all__ = ['SMAC', 'SMAC_PORTABLE']

SMAC = EnvConfig(
    backend=BackendConfig(step_mul=8, spawn_mode='map_trigger'),
    opponent=OpponentConfig(kind='engine_ai', difficulty='very_hard'),
    units=UnitsConfig(enemy_source='visible', ally_order='type_xy'),
    quirks=QuirksConfig(timestep_pad=True, terrain_height_flip=True),
)

SMAC_PORTABLE = EnvConfig(
    backend=BackendConfig(
        step_mul=8,
        spawn_mode='explicit',
        spawn_variant='std',
        pyref_abilities='heal',
        pyref_spawn_energy=50.0,
        pyref_heal_autocast=True,
    ),
    maps=MapsConfig(dir=None, sc2_dir='new_maps'),
    opponent=OpponentConfig(kind='opening_orders'),
    units=UnitsConfig(enemy_source='all', ally_order='type_xy'),
    actions=ActionConfig(heal='mmm_rule'),
    quirks=QuirksConfig(timestep_pad=False, terrain_height_flip=False),
)
