# UniSMAC Architecture & Code Map

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Environment Interface](interface.md) · [Benchmarks & Presets](benchmarks.md) · [Profiles & Configuration](profiles-and-config.md) · [Contributing](../CONTRIBUTING.md)

This guide is a fast navigation map for readers moving from running examples to reading or modifying the `unismac` and `zgame` codebases.

---

## 1. Learning Ladder (3 min → 15 min → 30 min)

| Stage | Goal | Start Here |
| --- | --- | --- |
| **3 minutes** | Install and run vectorized rollouts or two-sided self-play | [README](../README.md) · [`examples/quickstart_vec.py`](../examples/quickstart_vec.py) · [`examples/self_play.py`](../examples/self_play.py) |
| **15 minutes** | Integrate into a MARL trainer, switch backends, and explore profiles | [`examples/marl_loop_minimal.py`](../examples/marl_loop_minimal.py) · [`examples/sc2_eval.py`](../examples/sc2_eval.py) · [Environment Interface](interface.md) · [Profiles & Configuration](profiles-and-config.md) |
| **30 minutes** | Trace one `step()`, compare Python and Rust kernels 1:1, and run contributor checks | This Code Map · [Architecture](architecture.md) · [Contributing](../CONTRIBUTING.md) |

---

## 2. End-to-End Trace of One `env.step()` Call

Follow a single batched step from the public factory down through the Rust or Python simulation kernel:

```text
make_vec_env() / make_env()               unismac/env/__init__.py
  ├── resolve preset + scenario + profile unismac/presets/, unismac/scenarios/, unismac/profiles.py
  ├── merge EnvConfig overrides           unismac/env/config.py
  ├── select accel tier (auto/python/     unismac/env/accel/__init__.py
  │   services/rust) + manifest check     unismac/env/accel/classic.py, unismac/env/accel/manifest.py
  └── construct backend + VecEnv          unismac/env/core.py (and UnismacEnv in unismac/env/single.py)

VecEnv.step(actions, enemy_actions)       unismac/env/core.py
  1. Translate actions -> Command[B, U]   env/semantics/actions.py       (or env/accel/services.py -> _zgame.env)
  2. Merge Player 2 / opponent orders     env/dual.py, env/opponents/{__init__,smac_bot}.py, env/opponents/scripted/
                                          (or env/accel/semantics.py -> _zgame.env in accel='rust')
  3. Advance backend by `step_mul` loops  backends/zgame/backend.py      (or backends/pyref/backend.py / sc2)
     └── 10-stage per-loop pipeline:
         commands -> targeting -> attack -> abilities -> pathing ->
         steering -> kinematics -> collision -> damage -> visibility
         (+ post-move late_scan / kill_scan hooks)
  4. Extract Frame + compute rewards,     env/semantics/{units,reward,termination,settle}.py
     termination, and settlement          (or env/accel/{services,semantics}.py)
  5. Lazily encode obs / state / masks    env/semantics/{features,visibility,spotter}.py
                                          (or env/accel/semantics.py in accel='rust')
```

### `unismac/env/` Layout (5 Core Files + 3 Subpackages)

```text
unismac/env/
├── __init__.py          # Public factories & exports (make_env, make_vec_env, VecEnv, UnismacEnv, ...)
├── config.py            # EnvConfig hierarchy, FLAT_NAMES, and apply_overrides
├── core.py              # VecEnv batched environment
├── single.py            # UnismacEnv single-env PyMARL/EPyMARL wrapper
├── dual.py              # Two-sided self-play Seat and mirror-spawn geometry
├── semantics/           # 14 pure environment-semantics modules
│                        # (abilities, actions, catalog, features, generators, reward, rng,
│                        #  services, settle, spotter, termination, terrain, units, visibility)
├── opponents/           # Opponent protocol (__init__.py), smac_bot.py, and scripted/ (45 scripts)
└── accel/               # Tier dispatch (__init__.py), classic.py, services.py, semantics.py,
                         # manifest.py, and rust_manifest.json
```

- **Shared backend helpers**: `unismac/backends/_common.py` (shared passive-owner resolution, capability derivation, and scenario validation used by both `pyref` and `zgame`).

---

## 3. 1:1 Mirror Map: `unismac` (`pyref`) ↔ `zgame` (`zgame-smac`)

Both repositories share an intentional 1:1 module structure so readers can inspect the readable Python reference implementation alongside the compiled Rust kernel.

### Top-Level Simulation & Config Modules

| Concept | Python Reference (`unismac/backends/pyref/`) | Rust Kernel (`zgame/crates/zgame-smac/src/`) |
| --- | --- | --- |
| Backend / engine entry | `backend.py`, `pipeline.py` | `engine.rs`, `game.rs`, `lib.rs` |
| Configuration & validation | `config.py`, `options_meta.py` | `config/{mod,types,parse,validate,tests}.rs` |
| SoA world & unit state | `sim.py` | `world.rs`, `data.rs`, `kinds.rs`, `command.rs` |
| Navigation mesh & terrain | `navmesh.py`, `terrain.py` | `navmesh.rs`, `terrain.rs` |
| Subsystem stages (10-stage loop) | `subsystems/{commands,targeting,attack,abilities,pathing,steering,kinematics,collision,damage,visibility}.py` | `stages/{commands,targeting,attack,abilities,pathing,steering,kinematics,collision,damage,visibility}.rs` |
| Native combat timing | `subsystems/native_combat.py` | `stages/native_combat.rs` |
| Environment & scripted kernels | `unismac/env/semantics/` + `unismac/env/opponents/scripted/` | `env/` + `env/scripted/` |

### Q12 Fixed-Point Native Movement 1:1 Mirror (7 Submodules)

The Q12 fixed-point movement, steering, and collision subsystem is split identically into **7 submodules** across both languages:

| Submodule | Python (`backends/pyref/subsystems/native/`) | Rust (`crates/zgame-smac/src/stages/native/`) | Responsibility |
| --- | --- | --- | --- |
| Entry | `__init__.py` | `mod.rs` (+ `tests.rs`) | Public stage entry points & re-exports |
| **`q12`** | `q12.py` | `q12.rs` | Q12 fixed-point arithmetic, rounding, vector math, radius tables |
| **`wall`** | `wall.py` | `wall.rs` | Axis-aligned blocking-edge wall clearance, bisection, wall sliding |
| **`avoid`** | `avoid.py` | `avoid.rs` | Local ground avoidance neighbor filtering, blocked intervals, lateral clamping |
| **`stop`** | `stop.py` | `stop.rs` | Arrival deceleration, blocked-tolerance progression, early stop clearing |
| **`air`** | `air.py` | `air.rs` | Air-unit stationary separation and moving flock aggregation |
| **`steer`** | `steer.py` | `steer.rs` | Per-environment ground/air steering drivers and state sync |
| **`contact`** | `contact.py` | `contact.rs` | 4-pass contact batch solver, priority share tree, and displacement clamping |

---

## 4. Disambiguating Same-Named Modules Across Layers

Several short module names appear in both the environment layer (`unismac/env/`) and the Python simulator (`unismac/backends/pyref/`). Each pair serves a distinct architectural layer:

| Name | Environment Layer (`unismac/env/`) | Reference Backend (`unismac/backends/pyref/`) |
| --- | --- | --- |
| `config.py` | `env/config.py`: `EnvConfig` (14 nested groups, `FLAT_NAMES`, override merging) | `config.py`: `PyrefConfig` (164 simulator options + validators) |
| `rng.py` | `env/semantics/rng.py`: Counter-based Philox streams (`STREAMS`) for generators/opponents/obs | `rng.py`: Uniform helper over counter-based Philox for backend physics |
| `terrain.py` | `env/semantics/terrain.py`: Grid lookups for observation features and move action masks | `terrain.py`: Walkability & terrain height queries for pathing/vision |
| `visibility.py` | `env/semantics/visibility.py`: Observation visibility rules (`circle`, `cone`, `spotter_gate`) | `subsystems/visibility.py`: Backend fog-of-war & fire-reveal simulation |
| `abilities.py` | `env/semantics/abilities.py`: SMAC-Hard agent ability action mapping (`ABILITY_DICT`) | `subsystems/abilities.py`: Backend Medivac Heal & energy regeneration |
| `accel/` vs `native/` | `env/accel/{services,semantics}.py`: Rust FFI acceleration layer for `VecEnv` | `subsystems/native/`: Q12 fixed-point movement/collision reference |

---

## 5. `accel` Routing & `rust_manifest.json` Hash Guard

`unismac/env/accel/manifest.py` guards `accel='rust'` by verifying LF-normalized SHA-256 hashes of **47 semantic-layer source files** and **12 `VecEnv` methods** (`__init__`, `_generated_scenario`, `rng_stream`, `rng_streams`, `reset`, `step`, `_context`, `get_avail_actions`, `_obs_rng`, `get_obs`, `get_state`, `query_abilities`) against `unismac/env/accel/rust_manifest.json`.

- **List all locked entries**: `python -m unismac.env.accel.manifest --list-locked`
- **Check current status**: `python -m unismac.env.accel.manifest`
- **Re-verify parity and refresh hashes** (after intentional semantic-layer edits):

```bash
OMP_NUM_THREADS=1 python -m unismac.env.accel.manifest --write --jobs 6
```

Changes inside `unismac/backends/pyref/`, `unismac/backends/zgame/`, `unismac/profiles.py`, `unismac/cli.py`, `unismac/env/__init__.py`, `unismac/env/single.py`, `unismac/env/accel/`, or new `VecEnv` helper methods do **not** affect `rust_manifest.json`.

---

## 6. Test Suite Layout (`tests/`)

The `unismac` test suite is organized into seven domain-aligned directories (plus shared fixtures in `tests/conftest.py` and helpers in `tests/_support/`):

| Directory | Coverage |
| --- | --- |
| `tests/api/` | Public API, CLI, profiles, boundary/import isolation |
| `tests/contract/` | Backend contract invariants across backends |
| `tests/data/` | Assets, scenario registry, spawns, unit overrides |
| `tests/env/` | `VecEnv`/`UnismacEnv`, presets, dual mode, mirror, seeds, opponents, `accel` parity |
| `tests/pyref/` | Pure-Python reference simulator subsystems & physics probes |
| `tests/zgame/` | `zgame` backend & environment-layer Rust parity |
| `tests/sc2/` | Optional live StarCraft II backend tests (`-m sc2`) |

---

## 7. Frequently Asked Questions

- **Why is the PyPI package `zfoundry-zgame` while the Python import is `import zgame`?**
  The distribution name on PyPI is `zfoundry-zgame` to avoid naming collisions, while the top-level import package (`zgame`) and compiled extension module (`zgame._zgame`) remain short and ergonomic. `pip install unismac` pulls `zfoundry-zgame` automatically on mainstream platforms.
- **Why does `make_env("smac-portable", "8m")` without `profile=` or `backend=` still request SC2?**
  Preserving `backend="sc2"` as the bare default guarantees 100% backward compatibility with `make_env("smac", "3m")` and v0.1.0 callers. Pass `profile="sc2-native"` (recommended), `dual=True`, or `backend="auto"` to automatically select `zgame` when installed (falling back to `pyref`).
- **Why do finished slots remain frozen in `VecEnv` until `reset(done_ids)`?**
  Keeping terminal states frozen avoids silent off-by-one observation overwrites on the terminal step and lets callers record final infos/rewards before resetting specific slots via `env.reset(np.flatnonzero(terminated).tolist())`. See [`examples/marl_loop_minimal.py`](../examples/marl_loop_minimal.py).
- **Why does `smac-hard` default to `use_ability=False`?**
  This matches the published SMAC-Hard benchmark evaluation settings. Enable the V2 ability action segment explicitly with `overrides={"use_ability": True}` on live SC2.
- **Why did `accel="auto"` emit a `RuntimeWarning` and fall back to `'services'`?**
  A file or `VecEnv` method tracked by `rust_manifest.json` was edited locally, or the installed NumPy / `zfoundry-zgame` version is outside the verified manifest range. Run `unismac info` or `python -m unismac.env.accel.manifest` to see the exact mismatch.
