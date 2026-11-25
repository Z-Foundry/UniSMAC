# Benchmarks, Presets & SC2 Setup

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Code Map](code-map.md) · [Environment Interface](interface.md) · [Ecosystem Adapters](adapters.md) · [Profiles & Configuration](profiles-and-config.md)

UniSMAC unifies **SMAC**, **SMACv2**, and **SMAC-Hard** under six named presets and a single 84-scenario registry. All three benchmarks can be constructed side-by-side in the same Python process without upstream PySC2 conflicts.

---

## 1. Benchmark Comparison Matrix

| Dimension | `smac` | `smacv2` | `smac-hard` |
| --- | --- | --- | --- |
| Upstream reference | `oxwhirl/smac` | `oxwhirl/smacv2` | `devindeng94/smac-hard` (V2) |
| Registered scenarios | **23** | **26** (23 SMAC + 3 `10gen_*`) | **35** (23 SMAC + 12 new maps) |
| SC2 map directory | `$SC2PATH/Maps/SMAC_Maps` | `$SC2PATH/Maps/SMAC_Maps` | `$SC2PATH/Maps/new_maps` |
| Stochasticity source | Fixed maps & teams | Procedural unit types & spawns | Fixed maps + scripted opponent pool + fog |
| Opponent | Built-in SC2 AI (or `OpeningOrders` / `NativeSMACOpponent` on portable) | Built-in SC2 AI (or `OpeningOrders` on portable) | 45 deterministic decision-tree opponent scripts |
| Key defaults | Fixed sight 9 / shoot 6 | Built-in `10gen_*` capability configs; per-unit ranges | `use_ability=False` by default (matching SMAC-Hard paper; opt-in via `use_ability=True`) |

### Six Named Presets

- **Upstream-parity presets (SC2 backend)**: `smac`, `smacv2`, `smac-hard` — reproduce each upstream benchmark's exact observation/state structure, action masks, rewards, termination, and SC2 session protocol.
- **Cross-backend portable presets (`zgame` / `pyref` / `sc2`)**: `smac-portable`, `smacv2-portable`, `smac-hard-portable` — use explicit spawning (`backend.spawn_mode='explicit'`), backend-independent opponents, and portable capabilities so identical scenarios run seamlessly across `zgame`, `pyref`, and `sc2`.

---

## 2. Portable Scenario Coverage (76 / 84 Runnable)

UniSMAC bundles portable map geometry (`assets/maps/`), unit/weapon stats (`assets/static/`), and spawn records (`scenarios/`) so **76 / 84** scenarios run immediately on `zgame` and `pyref` with **zero StarCraft II installation**—and all **76 / 76** automatically hit full Rust acceleration (`accel='rust'`):

| Portable Preset | Runnable | `accel='rust'` | Notes |
| --- | :---: | :---: | --- |
| `smac-portable` | **23 / 23** | 23 / 23 | Full coverage of all 23 classic SMAC scenarios. |
| `smacv2-portable` | **26 / 26** | 26 / 26 | All 23 classic maps + procedural `10gen_terran`, `10gen_zerg`, `10gen_protoss`. |
| `smac-hard-portable` | **27 / 35** | 27 / 27 | 8 upstream SMAC-Hard scenarios are excluded from portable execution (listed below). |
| **Total** | **76 / 84** | **76 / 76** | Bitwise identical across `zgame` and `pyref`. |

You can inspect runnable scenarios and per-scenario metadata directly from the CLI:

```bash
unismac list-scenarios --preset smac-hard-portable --runnable --details
```

### Excluded SMAC-Hard Portable Scenarios (8 / 35)

The 8 `smac-hard` scenarios that raise `CapabilityError` on `smac-hard-portable` fall into two categories:

1. **Missing or crashing `.SC2Map` assets in SC2 4.10 (4 scenarios)**: `3s_vs_3z`, `3s_vs_4z`, `unit_test`, `pvt_large`.
2. **Missing explicit spawn records in `spawns_hard.json` (4 scenarios)**: `2m_vs_1z`, `2s_vs_1sc`, `so_many_baneling`, `bane_vs_bane` (all four remain runnable under `smac-portable` and `smacv2-portable`).

### Two-Sided Self-Play Coverage (75 / 76 Portable)

With `opponent.kind='external'` and `sc2-symmetric`, **75 / 76** portable scenarios run two-sided (all hitting `accel='rust'`). The sole exception is `smac-portable/MMM2`: under SMAC v1's `target_pad='enemies'` rule, Player 2 controls 12 units (including a healing Medivac targeting allies) while Player 1 has only 10 units, so Player 2's ally-heal targets do not fit into a 10-slot enemy target pad. Both `smacv2-portable/MMM2` and `smac-hard-portable/MMM2` use wider target pads and support two-sided self-play cleanly.

### Multi-Level Terrain & Map-Gated Abilities

- **Multi-level elevation maps (10 unique maps / 11 registered scenarios)**: `2c_vs_64zg` (across SMAC, SMACv2, SMAC-Hard) plus 8 SMAC-Hard multi-level maps—3-level `3st_vs_5zl`, and 2-level `mmmt`, `3rp_vs_5zl`, `3hl_vs_24zl`, `6m_vs_10m`, `7q_vs_2bc`, `mmmt_vs_zhb`, `mmmt_vs_zspi`. Both `pyref` and `zgame` support layered high-ground vision (`visibility='sc2'`, `lost_vision='sc2'`) and 3D projectile flight (`projectile_3d=True`).
- **Map-gated Zealot Charge (`charge=True`)**: Active on the 4 runnable `smac-hard-portable` scenarios whose map data grants Charge (`corridor`, `3rp_vs_5zl`, `3st_vs_5zl`, `mmmt_vs_zspi`), supported identically in both `pyref` and `zgame`.

---

## 3. Behavior-to-Configuration Mapping

Every behavioral difference across the three upstream benchmarks maps to an explicit `EnvConfig` field:

| Behavior | Config Field | `smac` | `smacv2` | `smac-hard` |
| --- | --- | --- | --- | --- |
| SC2 session type | `backend.sc2.session` | `single` | `single` | `duel` |
| Episode restart protocol | `backend.sc2.restart` | `kill_alive_step2` | `kill_all_until_empty` | `kill_alive_step1_c0` |
| Unit spawning | `backend.spawn_mode` | `map_trigger` | `map_trigger` (`10gen_*`: `explicit`) | `map_trigger` |
| Opponent | `opponent.kind` | `engine_ai` | `engine_ai` | `scripted` |
| Enemy source & ordering | `units.enemy_source` / `order` | `visible` / `raw` | `visible` / `raw` | `all` / `type_xy` |
| Unit type encoding | `catalog.type_id` | `smac` | `smacv2` | `standard` |
| Shield max source | `catalog.shield_max` | `table` | `table` | `static` |
| Sight / shoot ranges | `catalog.ranges` | `fixed` (9 / 6) | `per_unit` | `fixed` (9 / 6) |
| Visibility & spotter | `visibility.model` / `spotter_gate` | `circle` / `False` | `circle` or `cone` / `True` | `circle` / `False` |
| Target action padding | `actions.target.pad` | `n_enemies` | `mmm_rule` | `max_units_9` |
| Range check on attack | `actions.target.range_check` | `False` | `True` | `False` |
| Agent ability actions | `actions.ability.enabled` | `False` | `False` | `False` (override `use_ability=True`) |
| Positive-only reward | `reward.only_positive_mode` | `abs` | `clamp` | `clamp` |
| Enemy Medivac termination | `termination.enemy_medivac_rule` | `last_unit_is_medivac` | `no_non_medivac_left` | `last_unit_is_medivac` |

### Upstream Quirks (`quirks.*`)

Upstream SMAC, SMACv2, and SMAC-Hard contain several well-known quirks (such as an extra constant-zero timestep feature when `obs_timestep_number=True`, clearing an enemy feature block when spotter visibility drops, random agent observation shuffling, and unit-type offset shifts). UniSMAC follows a strict rule:

- **Stable quirks that affect published training numbers** are isolated behind explicit `quirks.*` booleans so upstream presets reproduce published behavior faithfully while allowing clean ablation.
- **Unrecoverable crash paths** (missing script-pool entries, out-of-range index errors, leaked ports on `close()`, misaligned feature metadata) are fixed unconditionally.

### SMACv2 Procedural Team Generation

The three `10gen_*` scenarios (`10gen_terran`, `10gen_zerg`, `10gen_protoss`) ship with built-in capability configurations matching the official SMACv2 YAML files (all using the `32x32_flat` map geometry):

| Scenario | Unit Types (Weights) | Generation Note |
| --- | --- | --- |
| `10gen_terran` | `marine` (0.45), `marauder` (0.45), `medivac` (0.10) | `medivac` uses exception-unit handling |
| `10gen_zerg` | `zergling` (0.45), `baneling` (0.10), `hydralisk` (0.45) | `baneling` uses exception-unit handling |
| `10gen_protoss` | `stalker` (0.45), `zealot` (0.45), `colossus` (0.10) | No exception unit |

### SMAC-Hard Scripted Opponents & Abilities

- `use_ability` defaults to `False` (matching the published SMAC-Hard paper experiments where neither agents nor opponents use extra ability actions); pass `overrides={'use_ability': True}` on live SC2 when evaluating the V2 ability action segment.
- On `smac-hard-portable`, opponent script abilities not implemented by `pyref`/`zgame` (such as Stim or Blink on 4 of 35 maps) are dropped cleanly (`opponent.script_abilities='drop'`).
- Unlike upstream SMAC-Hard's unseeded global `random.choice`, UniSMAC draws opponent scripts from a deterministic per-episode `opponent` RNG stream and supports explicit per-slot overrides via `reset(opponents=[...])`.

---

## 4. Eliminating Upstream Process Conflicts

In the original upstream repositories, SMAC, SMACv2, and SMAC-Hard cannot coexist in a single Python process for two reasons:

1. **Global `pysc2` map-class collisions**: Importing both SMAC and SMACv2 registers duplicate map classes (`3m`, `8m`, etc.), causing `pysc2.maps.get()` to raise `DuplicateMapError`.
2. **Incompatible `pysc2` forks**: SMAC-Hard historically required a patched `pysc2` whose `observe()` returned `(obs, abilities)` instead of `ResponseObservation`.

UniSMAC solves both cleanly:

- Uses a single unmodified PyPI `pysc2==4.0.0`.
- Loads maps directly via `LocalMap(map_path, map_data)` without registering global map classes.
- Queries unit abilities on demand via official `controller.query()`.

---

## 5. Optional Live StarCraft II Setup (`[sc2]`)

Live StarCraft II is **not required** for `zgame` or `pyref`. To enable `backend='sc2'` for ground-truth evaluation or upstream parity checks:

### 1. Install the `[sc2]` Extra

```bash
pip install "unismac[sc2]"
```

> **Python 3.14 compatibility note**: Upstream `pysc2==4.0.0` pins `protobuf==3.20.3` and `s2clientprotocol==5.0.16.97563.0`, and depends on `pygame`, which currently publishes prebuilt wheels only through Python 3.13. For live SC2 evaluation, use **Python 3.10–3.13** out-of-the-box, or install `pygame-ce>=2.5.8` on Python 3.14. UniSMAC automatically patches PySC2's removed 2-argument `random.shuffle` call on Python 3.11+.

### 2. Install StarCraft II 4.10 & Maps

Install StarCraft II Linux headless binary (`4.10`, following the official [SC2 Linux Packages](https://github.com/Blizzard/s2client-proto#linux-packages) instructions) and place the map packs under `$SC2PATH`:

```text
$SC2PATH/
└── Maps/
    ├── SMAC_Maps/    # 23 SMAC maps + SMACv2 32x32_flat.SC2Map
    └── new_maps/     # SMAC-Hard maps
```

Export `SC2PATH` before running with `backend='sc2'`:

```bash
export SC2PATH=/path/to/StarCraftII
```

### 3. Re-Extracting Local Portable Assets (Optional)

UniSMAC already bundles the portable map and static-data JSON assets needed for all 76 portable scenarios. If you add custom `.SC2Map` files or wish to regenerate local assets from your own `$SC2PATH` installation, use the built-in extractor:

```bash
# Extract specific maps into portable MapInfo assets
python -m unismac.backends.sc2.extract maps SMAC_Maps/3m new_maps/3m

# Extract unit & weapon StaticData from SC2
python -m unismac.backends.sc2.extract static

# Extract all maps and static data
python -m unismac.backends.sc2.extract all
```
