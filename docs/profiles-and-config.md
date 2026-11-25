# Profiles & Configuration Reference

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Code Map](code-map.md) · [Environment Interface](interface.md) · [Benchmarks & Presets](benchmarks.md)

UniSMAC uses **named profiles** (`unismac.profiles`) to bundle validated simulator configurations into concise, human-readable names, while keeping the full underlying `PyrefConfig` surface accessible for fine-grained research and ablations.

---

## 1. Public Profiles (`PUBLIC_PROFILES`)

Most users only need the five public profiles listed by `unismac.profiles.list_profiles()` (or `unismac list-profiles`):

| Profile | Constant | Recommended Backend | Intended Use |
| --- | --- | --- | --- |
| **`sc2-native`** | `STANDARD_PROFILE` | `zgame` (or `pyref`) | **Default recommended profile.** Full Q12 fixed-point movement, steering, collision, stop dynamics, combat/targeting timing, assist-on-hit, race-fallback regen, health-fraction healing, and map-driven passive behavior (62 non-default `pyref_options` + 3 command-delay fields). Best for sim-to-SC2 training and cross-backend evaluation. |
| **`sc2-explore`** | `STANDARD_EXPLORE_PROFILE` | `zgame` (or `pyref`) | **Fast short-budget exploration / curriculum stage 1.** Same Q12 fixed-point movement/collision and map-driven passive behavior as `sc2-native`, paired with simplified combat/targeting without the early assist-on-hit barrier (55 + 3 non-default fields). |
| **`sc2-symmetric`** | `STANDARD_SYMMETRIC_PROFILE` | `zgame` (or `pyref`) | **Two-sided symmetric self-play.** Symmetric edition of `sc2-native` (`auto_acquire_owners=()`, `retreat_owners=(1, 2)`, `hit_slow_owner1_only=False`, `passive_from_map=False`; 60 + 3 non-default fields) for `opponent.kind='external'`. |
| **`sc2-live`** | — | `sc2` | Live StarCraft II backend with no simulator overrides. |
| **`minimal`** | — | `pyref` (or `zgame`) | Minimal baseline using only preset-provided backend defaults (`{}` overrides), ideal for building custom `pyref_options` ablations from scratch. |

### Using Profiles in Python

```python
from unismac import STANDARD_PROFILE, make_vec_env

# 1. Recommended single-side rollout / training profile (auto-selects zgame when installed)
with make_vec_env("smac-portable", "8m", profile=STANDARD_PROFILE, batch_size=64, seed=0) as env:
    obs, state = env.reset()

# 2. Overlay individual PyrefConfig fields on top of a profile (deep-merged automatically)
with make_vec_env(
    "smac-portable",
    "2c_vs_64zg",
    profile="sc2-native",
    batch_size=16,
    overrides={"backend.pyref_options": {"projectile_3d": True}},
) as env:
    obs, state = env.reset()

# 3. Two-sided symmetric self-play (automatically selects sc2-symmetric + opponent.kind='external')
with make_vec_env("smac-portable", "2s3z", dual=True, batch_size=32, seed=0) as env:
    env.reset()
```

> **Lower-level helpers**: `unismac.profiles.profile_overrides()` and `merge_overrides()` remain available for callers that construct override dictionaries explicitly. All `pyref`-family profiles (`sc2-native`, `sc2-explore`, `sc2-symmetric`, `minimal`) run identically and bitwise-equivalently on both `backend="zgame"` (recommended for speed) and `backend="pyref"`.

---

## 2. Backward-Compatible Legacy Profiles (`LEGACY_PROFILES`)

To preserve full reproducibility for historical alignment baselines and ablation tables, `unismac.profiles.PROFILES` also accepts **17 legacy profile names** (accessible via `list_profiles(include_legacy=True)` or `unismac list-profiles --all`):

| Legacy Name | Public Equivalent / Role |
| --- | --- |
| `rghf-nat-cbt` | Exact alias of **`sc2-native`** (`STANDARD_PROFILE`) |
| `rghf-nat` | Exact alias of **`sc2-explore`** (`STANDARD_EXPLORE_PROFILE`) |
| `rghf-nat-cbt-sym` | Exact alias of **`sc2-symmetric`** (`STANDARD_SYMMETRIC_PROFILE`) |
| `sc2` | Exact alias of **`sc2-live`** |
| `preset` | Exact alias of **`minimal`** |
| `aligned`, `aligned-acq`, `aligned-full`, `aligned-v3`, `aligned-v3-noretreat`, `aligned-v4`, `aligned-v5`, `aligned-v6` | Incremental historical alignment baselines (floating-point collision/avoidance ladder; `aligned-v4` is the standard historical transfer baseline) |
| `acc-ram-cov-axo`, `acc-ram-cov-axo-rghf` | Historical floating-point heuristic candidates prior to Q12 fixed-point native movement |
| `aligned-v4-sym`, `acc-ram-cov-axo-sym` | Symmetric variants of `aligned-v4` and `acc-ram-cov-axo` for two-sided comparison |

For all new projects, prefer the 5 semantic names in `PUBLIC_PROFILES`.

---

## 3. Core `PyrefConfig` Configuration Surface

`PyrefConfig` (`unismac.backends.pyref.PyrefConfig`) defines **164** fields shared between `pyref` and `zgame` (with **134 / 164** implemented in `zgame`; the remaining 30 are legacy research and diagnostic heuristic options plus `rl_turn='curve'` and `min_damage`).

### A. Promoted Top-Level Fields (`EnvConfig.backend.pyref_*`)

These six fields are configured via `backend.pyref_<name>` in `EnvConfig` overrides rather than inside `backend.pyref_options`:

| EnvConfig Key | Default | Description |
| --- | --- | --- |
| `backend.pyref_abilities` | `'none'` | `'none'` or `'heal'` (enables Medivac Heal + energy regeneration; portable presets enable `'heal'`). |
| `backend.pyref_spawn_energy` | `None` | Starting unit energy (`50.0` on portable presets; `None` = full energy). |
| `backend.pyref_heal_autocast` | `False` | Enables standard Medivac autocast healing while idle/holding (`True` on portable presets; disabled for `Variant.RL`). |
| `backend.pyref_commands` | `'immediate'` | `'immediate'` (single-client zero delay) or `'delayed'` (dual-client lockstep; set by `sc2-native`). |
| `backend.pyref_command_delay` | `0` | Delay in game loops before newly issued orders activate (`1` when `'delayed'`). |
| `backend.pyref_move_lag` | `0` | Extra 1-loop displacement lag on order switch (`1` when `'delayed'`). |

### B. Subsystem Selectors (`backend.pyref_options`)

| Option | Default | `sc2-native` | Description |
| --- | --- | --- | --- |
| `pathing` | `'straight'` | `'navmesh'` | Pathfinding: `'straight'`, `'grid'`, or `'navmesh'` (navigation mesh around obstacles and cliffs). |
| `kinematics` | `'constant_speed'` | `'accel'` | Unit kinematics: `'constant_speed'` or `'accel'` (measured acceleration/deceleration and turn limits). |
| `steering` | `'none'` | `'native'` | Local avoidance: `'none'`, `'separation'`, `'sc2_avoid'`, or `'native'` (Q12 fixed-point steering). |
| `collision` | `'none'` | `'native'` | Unit collision resolution: `'none'`, `'hard'`, or `'native'` (Q12 fixed-point contact batches). |
| `targeting` | `'basic'` | `'sc2'` | Target acquisition & retargeting: `'basic'` or `'sc2'` (periodic scan, acquire range, hit response). |
| `attack` | `'instant'` | `'sc2'` | Weapon firing: `'instant'` or `'sc2'` (facing check, damage-point windup, randomized cooldown). |
| `damage` | `'standard'` | `'sc2'` | Damage resolution: `'standard'` or `'sc2'` (projectiles, multi-hit intervals, hit-slow, regeneration). |
| `visibility` | `'full'` | `'full'` | Line-of-sight model: `'full'` (fast path) or `'sc2'` (multi-level terrain vision & fire-reveal). |
| `spawn_facing` | `'zero'` | `'random'` | Initial unit facing on reset: `'zero'` or uniform `'random'` in `[0, 2π)`. |

### C. Q12 Fixed-Point Native Movement & Collision (`sc2-native` / `sc2-explore`)

Enabled when `steering='native'` and `collision='native'`:

| Option | Default | `sc2-native` | Description |
| --- | --- | --- | --- |
| `native_share_rule` | `'legacy'` | `'priority_tree'` | Contact-push share resolution tree (`'legacy'` or `'priority_tree'`). |
| `native_batch_flags` | `False` | `True` | Tracks multi-pass contact and wall-correction flags across the 4-pass batch solver. |
| `native_nbr_order` | `'slot'` | `'quantized'` | Neighbor enumeration order (`'slot'` or `'quantized'` 0.25-cell quantized distance key). |
| `native_steer_avoid` | `False` | `True` | Native ground steering avoidance with lateral acceleration clamping. |
| `native_move_reissue` | `False` | `True` | Zero-tolerance Q12 same-point MOVE reissue skipping and progress retention. |
| `native_stop` | `False` | `True` | Native arrival deceleration and two-stage blocked-tolerance stop clearing. |
| `native_air_sep` | `False` | `True` | Air-unit separation (stationary throttled separation and moving flock aggregation). |
| `native_wall_clearance` | `False` | `True` | Exact axis-aligned blocking-edge wall clearance, bisection, and wall sliding. |

### D. Native Combat, Targeting & Loop Timing (`sc2-native`)

| Option | Default | `sc2-native` | Description |
| --- | --- | --- | --- |
| `native_backswing` | `False` | `True` | Native weapon backswing window and deferred retarget promotion on order completion. |
| `native_retarget` | `False` | `True` | Q12 target-scoring retargeting with in-range hysteresis and busy-window deferral. |
| `hit_assist` | `False` | `True` | Nearby idle / ATTACK_MOVE allies assist attacked teammates within `hit_assist_range` (`4.0`). |
| `native_assist_on_hit` | `False` | `True` | Full multi-slot threat memory and Q12 distance scoring for ally assist-on-hit (disabled in `sc2-explore`). |
| `native_scan_cadence` | `False` | `True` | Unified 8-loop scan cadence shared by target acquisition and Medivac autocast healing. |
| `native_loop_order` | `False` | `True` | Two-pass loop execution (Pass 1 unit attacks/regen, Pass 2 projectile flight/hits & cleanup) with per-loop update-order shuffling. |
| `kill_rescan_delay` | `2` | `0` | Delay loops before rescanning after target death (`0` = same-loop rescan). |
| `kill_rescan_order` | `False` | `True` | Order-dependent same-loop vs next-loop rescan when a target is killed by another unit. |
| `regen_race_fallback` | `False` | `True` | Race-level fallback shield/health regeneration for units without per-unit measured overrides. |
| `heal_pick` | `'life'` | `'fraction'` | Autocast Medivac target tie-breaking (`'life'` lowest HP vs `'fraction'` lowest HP ratio). |

### E. Terrain, Elevation, Abilities & Symmetry Options

| Option | Default | Description |
| --- | --- | --- |
| `visibility` / `vision_levels` / `reveal_loops` | `'full'` / `'cell'` / `47` | Multi-level terrain line-of-sight (`visibility='sc2'`, `vision_levels='cell'\|'interp'`, fire reveal duration in loops). Supported in both `pyref` and `zgame`. |
| `lost_vision` / `lost_vision_giveup` | `'immediate'` / `48` | Behavior when an engaged target leaves sight (`'sc2'` waits 2 frames before chasing and gives up after 48 loops). |
| `projectile_3d` | `False` | Computes 3D Euclidean projectile flight distance using map terrain elevation (`zgame` + `pyref`). |
| `charge` | `False` | Enables map-gated Zealot Charge on maps whose static map abilities grant it (`zgame` + `pyref`). |
| `passive_from_map` | `False` | Automatically derives Player 1 passive auto-acquire/retreat flags from map metadata (`True` in `sc2-native` and `sc2-explore`, `False` in `sc2-symmetric`). |
| `auto_acquire_owners` / `retreat_owners` / `hit_slow_owner1_only` | `None` / `()` / `False` | Per-player ownership asymmetry controls; `sc2-symmetric` sets `()`, `(1, 2)`, and `False` respectively for unbiased two-sided play. |

### F. Native Single-Client SMAC Helper (`native_smac_overrides`)

Use `unismac.env.native_smac_overrides(...)` when comparing `smac-portable` directly against single-client native SMAC (`SMAC_Maps`):

```python
from unismac.env import native_smac_overrides
from unismac.profiles import merge_overrides, profile_overrides

_, base = profile_overrides("sc2-native")
overrides = merge_overrides(
    base,
    native_smac_overrides("2c_vs_64zg", terrain_elevation="auto"),
)
```

Supported `terrain_elevation` values:

- `'auto'` (default): enables `projectile_3d=True` on multi-level SMAC maps (`2c_vs_64zg`) while keeping `visibility='full'`.
- `'projectile_3d'`: enables `projectile_3d=True` explicitly.
- `'full'` or `True`: enables `visibility='sc2'`, `lost_vision='sc2'`, and `projectile_3d=True`.
- `'off'` or `False`: leaves terrain options unchanged.

### G. Inspecting All 164 `PyrefConfig` Options Programmatically & via CLI

Browse options by topic (`stages`, `commands`, `targeting`, `attack`, `abilities`, `steering`, `collision`, `damage`, `visibility`), tier (`core` / `native` / `legacy_heuristic`), and `zgame` support from the CLI:

```bash
unismac show-profile sc2-native
unismac show-options --group collision --zgame-only
```

Or query the read-only option catalog directly from Python (`unismac.backends.pyref.options_meta`):

```python
from unismac.backends.pyref.options_meta import describe_pyref_options

for entry in describe_pyref_options(group="collision", tier="native", zgame_only=True):
    print(entry)
```

Fields not yet implemented in `zgame` (30 legacy heuristic knobs, `rl_turn='curve'`, and non-default `min_damage`) remain fully executable on `backend='pyref'` and fail fast with `CapabilityError` if enabled on `backend='zgame'`.
