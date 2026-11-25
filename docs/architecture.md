# UniSMAC Architecture

[Back to README](../README.md) · Related: [Code Map](code-map.md) · [Environment Interface](interface.md) · [Ecosystem Adapters](adapters.md) · [Benchmarks & Presets](benchmarks.md) · [Profiles & Configuration](profiles-and-config.md) · [Migration Guide](../MIGRATION.md)

UniSMAC separates **RL framework adapters** (PyMARL, Gymnasium, PettingZoo), **benchmark environment semantics** (observations, global state, action masks, rewards, termination/truncation, procedural team generation, and opponents), and **simulation execution** through explicit, one-way boundaries. A single purified environment kernel runs unchanged across three interchangeable backends—**ZGame** (Rust), **PyRef** (pure Python/NumPy), and **SC2** (live StarCraft II)—inside the same Python process.

---

## 1. Layered Design

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ RL Adapters     unismac.adapters (strictly one-way -> unismac.env)      │
│ (Optional)      pymarl (PyMARLEnv, 0 extra deps)                        │
│                 gymnasium (GymnasiumEnv / GymnasiumVecEnv, [gymnasium]) │
│                 pettingzoo (PettingZooParallelEnv, [pettingzoo])        │
├─────────────────────────────────────────────────────────────────────────┤
│ Core Public API unismac.make_vec_env() · VecEnv · StepBatch             │
│                 RuntimeView (env.runtime) · unismac.profiles · cli      │
├─────────────────────────────────────────────────────────────────────────┤
│ Environment     EnvConfig & presets (smac / smacv2 / smac-hard +        │
│ Layer           *-portable), UnitView, UnitCatalog, Visibility,         │
│                 ActionSpace, FeatureBlocks, Reward, Termination,        │
│                 Opponents (EngineAI / OpeningOrders / Scripted /        │
│                 NativeSMAC / External), Procedural Generators,          │
│                 3-Tier Acceleration Dispatch (python / services / rust) │
├═════════════════════════════════════════════════════════════════════════┤
│ Backend         unismac.contract (CONTRACT_VERSION = "0.2")             │
│ Contract        Backend Protocol · StaticData · MapInfo · Scenario ·    │
│                 WorldState [B, U] · Command [B, U] · Capabilities       │
├──────────────────────┬───────────────────────┬──────────────────────────┤
│ zgame Backend        │ pyref Backend         │ sc2 Backend (optional)   │
│ Rust native engine   │ Pure Python / NumPy   │ PySC2 + StarCraft II     │
│ (zgame._zgame)       │ executable reference  │ Single & Duel sessions   │
│ Rayon batched step   │ 164 PyrefConfig knobs │ LocalMap + query()       │
└──────────────────────┴───────────────────────┴──────────────────────────┘
```

### Design Principles

1. **The backend contract is the sole simulation boundary**: The environment layer only reads and writes contract types (`WorldState`, `Command`, `Scenario`, `MapInfo`, `StaticData`, `Capabilities`). Engine-specific concerns—PySC2 protobuf translation, dual-client sessions, restart protocols, or Rust FFI buffers—stay strictly inside each backend.
2. **Strict one-way adapter boundary**: Framework conventions (PyMARL `MultiAgentEnv`, Gymnasium `Env`/`VectorEnv` spaces and autoreset modes, PettingZoo `ParallelEnv` per-agent dicts) live in `unismac.adapters` on top of the purified `VecEnv` kernel. `unismac.env` never imports `unismac.adapters`, `gymnasium`, or `pettingzoo`, and `VecEnv` public methods never call each other internally.
3. **Behavior as configuration**: Every behavioral difference across SMAC, SMACv2, and SMAC-Hard is expressed as a named field in `EnvConfig` rather than ad-hoc benchmark branching.
4. **Explicit initial state**: Every episode starts from an explicit `Scenario` (unit slots, owners, canonical unit types, variants, coordinates, facings, and optional initial orders), ensuring reproducible starts across backends.
5. **Self-describing batched features**: Observation and state feature blocks declare their own feature names and compute across `[B, N]` batches at once, eliminating dimension-name drift.
6. **Explicit counter-based RNG**: All environment and backend randomness flows through named, deterministic counter-based streams keyed by `(seed, stream, env_id, episode, frame)` so Python and Rust remain bitwise identical.

---

## 2. Backend Contract (`unismac.contract`)

Rather than mirroring the wire-level `s2client` protobuf protocol—which requires per-unit Python loops and one game process per instance—UniSMAC places the contract at the **semantic simulation layer** using fixed-shape batched NumPy arrays (`[B, U]`, where `B` is `batch_size` and `U` is the maximum unit slot count).

### Core Types

| Type | Role |
| --- | --- |
| `StaticData` / `UnitTypeSpec` | Static unit attributes indexed by `(canonical_type, variant)`: health/shield/energy maxima, armor, attributes, radius, flying flag, movement speed, sight range, weapons, and empirically measured kinematics/timing parameters. |
| `MapInfo` | Portable map geometry extracted once from SC2: dimensions, playable bounds, `pathing: bool[H, W]`, `placement: bool[H, W]`, and `height: float32[H, W]`. |
| `Scenario` | Explicit episode starting specification: `map_id`, unit spawns `(slot, owner, unit_type, variant, x, y, facing)`, optional `initial_orders`, optional `mirror` axis metadata, and optional engine AI flag. |
| `WorldState` | Batched `[B, U]` simulation snapshot: `alive`, `owner`, `unit_type`, `variant`, `x`, `y`, `facing`, `radius`, `health`, `health_max`, `shield`, `shield_max`, `energy`, `weapon_cooldown`, `order_kind`, `order_target_slot`, `engaged_target_slot`, `is_flying`, `visible`, optional abilities, and `game_loop: int64[B]`. |
| `Command` | Batched `[B, U]` command buffer: `kind` (`NOOP`, `STOP`, `MOVE`, `ATTACK_UNIT`, `ATTACK_MOVE`, `ABILITY`, `HOLD`), `target_slot`, `target_x`, `target_y`, and `ability`. |
| `Capabilities` | Declarative capability flags advertised by a backend (`pathing`, `terrain_height`, `collision`, `abilities`, `ability_query`, `fog`, `engine_ai`, `projectiles`, `deterministic`, `mirror_spawn`). Requests for unsupported capabilities fail fast with `CapabilityError`. |

### Key Conventions

- **Stable unit slots**: Units are identified by slot index `0..U-1` assigned at `Scenario` creation rather than transient engine tags. Upstream ordering conventions (`raw`, `slot`, `type_xy`) are computed as views in the environment layer.
- **Canonical types + variants**: Custom SMAC map units (`*_RL`) are represented by their canonical SC2 `UnitTypeId` paired with `variant=Variant.RL`, keeping attributes explicit in `StaticData`.
- **Game-loop stepping**: One environment step advances the simulation by `config.backend.step_mul` game loops (default `8`, i.e. 0.5 s at Faster speed).

---

## 3. Interchangeable Backends

### `zgame` (Rust Native Backend)

- Powered by the independent companion package [`zfoundry-zgame`](https://github.com/Z-Foundry/ZGame) (imported as `zgame`: `zgame._zgame` + `zgame._zgame.env`), installed automatically on mainstream platforms via `pip install unismac`.
- Uses a Rayon thread pool (`zgame_threads`) to step batches of environments in parallel with deterministic per-slot ordering, so trajectories are **invariant to thread count**.
- Shares the exact `PyrefConfig` schema (supporting 134/164 configuration fields, including all options in `sc2-native`, `sc2-explore`, and `sc2-symmetric`, plus multi-level terrain vision, 3D projectiles, and map-gated Zealot charge). Unsupported non-default options raise `CapabilityError` on construction.
- Also supplies the native kernels for `accel='services'` and `accel='rust'` (see [Environment Interface: Acceleration Tiers](interface.md#6-acceleration-tiers-accel)).

### `pyref` (Pure Python / NumPy Reference Backend)

- Pure-Python/NumPy simulation engine with zero native or StarCraft II dependencies.
- Implements the full 164-field `PyrefConfig` surface (`unismac.backends.pyref.PyrefConfig`), serving as the readable executable specification for physics, pathing, collision, combat timing, vision, and healing.
- Matches `zgame` bitwise (`L∞ == 0.0` across all 24 `WorldState` fields on verified probe suites and portable scenarios).

### `sc2` (Live StarCraft II Backend, Optional `[sc2]` Extra)

- Wraps official PyPI `pysc2==4.0.0` with no patched PySC2 fork required.
- Supports both single-client sessions (Participant + built-in Computer AI, for `smac`/`smacv2`) and dual-client sessions (two Participants, for `smac-hard` and `*-portable`), plus both `map_trigger` and `explicit` spawning.
- Loads maps directly via `LocalMap` by file path and queries unit abilities via official `controller.query()`, completely eliminating upstream PySC2 duplicate-map and module-conflict errors.
- Includes `python -m unismac.backends.sc2.extract` to regenerate local `MapInfo` and `StaticData` assets from `$SC2PATH`.

---

## 4. Configuration Composition & Override Precedence

`EnvConfig` (`unismac.env.config`) is a hierarchy of frozen dataclasses covering backend parameters, map selection, unit views, unit catalogs, visibility, action segments, observation/state features, rewards, termination, opponents, procedural capabilities, and upstream compatibility quirks.

Configurations resolve strictly in six layers (later layers override earlier ones):

```text
Preset defaults  <  Scenario defaults  <  Preset scenario overrides  <  Profile overrides  <  dual ('opponent.kind': 'external')  <  User overrides
```

When both `profile=` and `overrides=` supply `backend.pyref_options`, their dictionaries are deep-merged key-by-key so user options cleanly overlay the profile without clearing unrelated fields. User overrides accept:

- **Dotted paths**: e.g., `{'opponent.kind': 'external', 'backend.step_mul': 8, 'termination.continuing_episode': True}`.
- **Upstream flat names**: familiar SMAC/SMACv2/SMAC-Hard parameter names mapped cleanly into `EnvConfig`.
- **Direct factory shortcuts & helper dicts**: `profile='sc2-native'`, `dual=True`, `unismac.profiles.profile_overrides(...)`, `merge_overrides(...)`, and `unismac.env.native_smac_overrides(...)`.

Six core `PyrefConfig` fields (`abilities`, `spawn_energy`, `heal_autocast`, `commands`, `command_delay`, `move_lag`) are promoted to top-level `backend.pyref_*` fields on `EnvConfig.backend`; all remaining `PyrefConfig` options are passed via `backend.pyref_options` (deep-merged by key when using `merge_overrides`).

---

## 5. Cross-Package Contract with `zgame`

`unismac` and `zfoundry-zgame` (imported as `zgame`) are versioned and distributed as two independent MIT-licensed packages (`unismac` is a pure-Python wheel; `zfoundry-zgame` ships prebuilt `cp310-abi3` native wheels). To guarantee safe interoperability across upgrades, `unismac` checks four explicit version constants exposed by `zgame._zgame`:

| Constant | Module | Current Value | Guarded Boundary |
| --- | --- | --- | --- |
| `CONTRACT_VERSION` | `unismac` & `zgame._zgame` | `"0.2"` | `Backend`, `WorldState`, `Command`, `Scenario`, `Capabilities` |
| `ENGINE_SCHEMA_VERSION` | `zgame._zgame` | `1` | `PyrefConfig` / `StaticData` / `MapInfo` FFI serialization schema |
| `SERVICES_VERSION` | `zgame._zgame.env` | `1` | Rust system-layer kernels and numerical services (`accel='services'`) |
| `SEMANTICS` | `zgame._zgame.env` | `3` | Rust semantic-layer environment kernels (`accel='rust'`) |

In addition, `accel='rust'` verifies `unismac/env/accel/rust_manifest.json`, which records LF-normalized SHA-256 digests of the Python semantic-layer source files, the expected `_zgame.env.SEMANTICS` integer, and the supported NumPy version range (`>=1.26, <2.7`). If a developer modifies the Python semantic layer without updating the Rust kernels, `accel='auto'` automatically and safely falls back to `'services'` with a single `RuntimeWarning`.

For the Rust crate layout and internal engine architecture of `zgame`, see [`Z-Foundry/ZGame/docs/architecture.md`](https://github.com/Z-Foundry/ZGame/blob/main/docs/architecture.md).

---

## 6. Package Layout

```text
unismac/
├── __init__.py       # Top-level exports: CONTRACT_VERSION, STANDARD_PROFILE, StepBatch, VecEnv, list_profiles, make_vec_env
├── contract/         # Contract types, Backend Protocol, Capabilities, deterministic RNG spec
├── backends/
│   ├── _common.py    # Shared passive-owner resolution & config serialization helpers (pyref + zgame)
│   ├── zgame/        # ZgameBackend wrapper, build_info(), 4D compatibility verification
│   ├── pyref/        # Pure-Python reference simulator, PyrefConfig (164 fields), options_meta.py,
│   │                 # and subsystems/ (incl. 7-file native/{q12,wall,avoid,stop,air,steer,contact})
│   └── sc2/          # Optional PySC2 backend (sessions, adapter, extract, restart, abilities)
├── assets/           # Bundled portable MapInfo, StaticData, measured params, map abilities
├── scenarios/        # 84-scenario registry across SMAC, SMACv2, and SMAC-Hard + spawn records
├── presets/          # smac, smacv2, smac-hard, and *-portable presets
├── env/              # 4 core runtime files (__init__, config, core, dual) + 3 subpackages:
│   ├── semantics/    # 14 pure environment-semantics modules (actions, features, reward, rng, ...)
│   ├── opponents/    # Opponent protocol, smac_bot.py, and scripted/ (45 SMAC-Hard scripts)
│   └── accel/        # Accel tier routing, classic.py, services.py, semantics.py, manifest.py + rust_manifest.json
├── adapters/         # One-way RL ecosystem adapters (strictly external to env/):
│   ├── __init__.py   # Zero-extra exports: PyMARLEnv, UnismacEnv, make_pymarl_env
│   ├── _common.py    # Shared optional-extra guards and adapter helpers
│   ├── pymarl.py     # PyMARL / EPyMARL MultiAgentEnv wrapper (0 extra deps)
│   ├── gymnasium.py  # Lazy-imported Gymnasium 1.0+ Env & VectorEnv ([gymnasium])
│   └── pettingzoo.py # Lazy-imported PettingZoo ParallelEnv ([pettingzoo])
├── profiles.py       # Public semantic profiles (sc2-native, sc2-explore, sc2-symmetric, ...)
└── cli.py            # CLI entry point (info, list-scenarios, list-profiles, show-profile, show-options, bench)
```

For a step-by-step trace of `env.step()` and the full 1:1 module mirror against `zgame`, see [Architecture & Code Map](code-map.md). For framework adapters, see [Ecosystem Adapters](adapters.md). For contributor setup and test-suite structure, see [Contributing](../CONTRIBUTING.md).

---

## 7. Verification Layers

UniSMAC validates correctness across three distinct layers:

1. **Upstream preset & ecosystem contract parity**: Offline comparison of `smac`, `smacv2`, and `smac-hard` presets against upstream semantics, plus official Gymnasium `check_env`, PettingZoo `parallel_api_test`, and `parallel_to_aec` `api_test` compliance suites.
2. **Cross-backend bitwise parity (`zgame` ↔ `pyref`)**: Verified across all 76 runnable portable scenarios, all three `accel` tiers (`python`, `services`, `rust`), and over 6,400 deterministic movement, combat, terrain, and charge probes (`L∞ == 0.0`).
3. **Behavioral alignment & zero-shot SC2 transfer**: Statistical comparison against live StarCraft II 4.10 trajectories, policy-pool rank correlation, and zero-shot evaluation of policies trained purely in `zgame` when deployed on live SC2.
