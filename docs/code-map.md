# UniSMAC Architecture & Code Map

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Environment Interface](interface.md) · [Ecosystem Adapters](adapters.md) · [Benchmarks & Presets](benchmarks.md) · [Profiles & Configuration](profiles-and-config.md) · [Migration Guide](../MIGRATION.md) · [Contributing](../CONTRIBUTING.md)

This guide is a fast navigation map for readers moving from running examples to reading or modifying the `unismac` and `zgame` codebases.

---

## 1. Learning Ladder (3 min → 15 min → 30 min)

| Stage | Goal | Start Here |
| --- | --- | --- |
| **3 minutes** | Install and run vectorized rollouts, self-play, or Gymnasium / PettingZoo adapters | [README](../README.md) · [`examples/quickstart_vec.py`](../examples/quickstart_vec.py) · [`examples/gymnasium_vec.py`](../examples/gymnasium_vec.py) · [`examples/pettingzoo_parallel.py`](../examples/pettingzoo_parallel.py) |
| **15 minutes** | Integrate into a MARL trainer, compare ecosystem adapters, switch backends, and explore profiles | [`examples/marl_loop_minimal.py`](../examples/marl_loop_minimal.py) · [`examples/sc2_eval.py`](../examples/sc2_eval.py) · [Environment Interface](interface.md) · [Ecosystem Adapters](adapters.md) · [Profiles & Configuration](profiles-and-config.md) |
| **30 minutes** | Trace one `step()`, compare Python and Rust kernels 1:1, and run contributor checks | This Code Map · [Architecture](architecture.md) · [Migration Guide](../MIGRATION.md) · [Contributing](../CONTRIBUTING.md) |

---

## 2. End-to-End Trace of One `env.step()` Call

Follow a single batched step from the public factory (or an optional RL adapter) down through the Rust or Python simulation kernel:

```text
Optional RL adapter (PyMARL / Gymnasium / PettingZoo)   unismac/adapters/{pymarl,gymnasium,pettingzoo}.py
  └── delegates strictly one-way to make_vec_env()

make_vec_env()                            unismac/env/__init__.py
  ├── resolve preset + scenario + profile unismac/presets/, unismac/scenarios/, unismac/profiles.py
  ├── merge EnvConfig overrides           unismac/env/config.py
  ├── select accel tier (auto/python/     unismac/env/accel/__init__.py
  │   services/rust) + manifest check     unismac/env/accel/classic.py, unismac/env/accel/manifest.py
  └── construct backend + VecEnv          unismac/env/core.py

VecEnv.step(actions, enemy_actions) -> StepBatch        unismac/env/core.py
  1. Translate actions -> Command[B, U]   env/semantics/actions.py       (or env/accel/services.py -> _zgame.env)
  2. Merge Player 2 / opponent orders     env/dual.py, env/opponents/{__init__,smac_bot}.py, env/opponents/scripted/
                                          (or env/accel/semantics.py -> _zgame.env in accel='rust')
  3. Advance backend by `step_mul` loops  backends/zgame/backend.py      (or backends/pyref/backend.py / sc2)
     └── 10-stage per-loop pipeline:
         commands -> targeting -> attack -> abilities -> pathing ->
         steering -> kinematics -> collision -> damage -> visibility
         (+ post-move late_scan / kill_scan hooks)
  4. Extract Frame + compute rewards,     env/semantics/{units,reward,termination,settle}.py
     terminated / truncated (via timeouts (or env/accel/{services,semantics}.py)
     delta), and StepBatch
  5. Lazily project obs() / state() /     env/semantics/{features,visibility,spotter}.py
     avail_actions() on demand            (or env/accel/semantics.py in accel='rust')
```

### `unismac/env/` & `unismac/adapters/` Layout

```text
unismac/
├── env/                 # 4 core runtime files + 3 subpackages (zero imports of adapters/):
│   ├── __init__.py      # Public kernel factory & exports (make_vec_env, VecEnv, StepBatch, RuntimeView, ...)
│   ├── config.py        # EnvConfig hierarchy, FLAT_NAMES, and apply_overrides
│   ├── core.py          # Purified VecEnv kernel, StepBatch, and RuntimeView (0 pub->pub calls)
│   ├── dual.py          # Two-sided self-play Seat and mirror-spawn geometry
│   ├── semantics/       # 14 pure environment-semantics modules
│   │                    # (abilities, actions, catalog, features, generators, reward, rng,
│   │                    #  services, settle, spotter, termination, terrain, units, visibility)
│   ├── opponents/       # Opponent protocol (__init__.py), smac_bot.py, and scripted/ (45 scripts)
│   └── accel/           # Tier dispatch (__init__.py), classic.py, services.py, semantics.py,
│                        # manifest.py, and rust_manifest.json
└── adapters/            # One-way RL framework wrappers (strictly external to env/):
    ├── __init__.py      # Zero-extra exports: PyMARLEnv, UnismacEnv, make_pymarl_env
    ├── _common.py       # Shared optional-extra import guard & single-env/action helpers
    ├── pymarl.py        # PyMARL / EPyMARL MultiAgentEnv wrapper (0 extra deps)
    ├── gymnasium.py     # Lazy-imported Gymnasium 1.0+ Env & VectorEnv ([gymnasium])
    └── pettingzoo.py    # Lazy-imported PettingZoo ParallelEnv ([pettingzoo])
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

`unismac/env/accel/manifest.py` guards `accel='rust'` by verifying LF-normalized SHA-256 hashes of the semantic-layer source files (`SOURCES`) and core `VecEnv` kernel methods/private computation primitives (`FUNCTIONS`, including `__init__`, `reset`, `step`, and private `_obs`/`_state`/`_avail`/`_context`/`_ensure_spotter` helpers) against `unismac/env/accel/rust_manifest.json`.

- **List all locked entries**: `python -m unismac.env.accel.manifest --list-locked`
- **Check current status**: `python -m unismac.env.accel.manifest`
- **Re-verify parity and refresh hashes** (after intentional semantic-layer edits):

```bash
OMP_NUM_THREADS=1 python -m unismac.env.accel.manifest --write --jobs 6
```

Changes inside `unismac/adapters/`, `unismac/backends/pyref/`, `unismac/backends/zgame/`, `unismac/profiles.py`, `unismac/cli.py`, `unismac/env/__init__.py`, `unismac/env/accel/`, or read-only `RuntimeView` helpers do **not** affect `rust_manifest.json`.

---

## 6. Test Suite Layout (`tests/`)

The `unismac` test suite is organized into **eight** domain-aligned directories (plus shared fixtures in `tests/conftest.py` and helpers in `tests/_support/`):

| Directory | Coverage |
| --- | --- |
| `tests/api/` | Public API, CLI, profiles, static & runtime boundary/import isolation |
| `tests/adapters/` | `PyMARLEnv`, missing-extra errors, Gymnasium `check_env` + 3-mode `VectorEnv`, PettingZoo `parallel_api_test` + `parallel_to_aec` `api_test` |
| `tests/contract/` | Backend contract invariants across backends |
| `tests/data/` | Assets, scenario registry, spawns, unit overrides |
| `tests/env/` | Purified `VecEnv`, `StepBatch`, `RuntimeView`, AST zero-self-call checks, presets, dual mode, mirror, seeds, opponents, `accel` parity |
| `tests/pyref/` | Pure-Python reference simulator subsystems & physics probes |
| `tests/zgame/` | `zgame` backend & environment-layer Rust parity |
| `tests/sc2/` | Optional live StarCraft II backend tests (`-m sc2`) |

---

## 7. Frequently Asked Questions

- **Why is the PyPI package `zfoundry-zgame` while the Python import is `import zgame`?**
  The distribution name on PyPI is `zfoundry-zgame` to avoid naming collisions, while the top-level import package (`zgame`) and compiled extension module (`zgame._zgame`) remain short and ergonomic. `pip install unismac` pulls `zfoundry-zgame` automatically on mainstream platforms.
- **Where did `make_env` and `UnismacEnv` move in `v0.2.0`?**
  They moved out of the core kernel into `unismac.adapters.pymarl` as `make_pymarl_env(...) -> PyMARLEnv` (with `UnismacEnv = PyMARLEnv`). Core vectorized environments use `from unismac import make_vec_env`. See [MIGRATION.md](../MIGRATION.md) for the full before/after mapping.
- **Why does `make_vec_env("smac-portable", "8m")` without `profile=` or `backend=` still request SC2?**
  Preserving `backend="sc2"` as the bare default keeps upstream-parity calls like `make_vec_env("smac", "3m")` consistent across versions. Pass `profile="sc2-native"` (recommended), `dual=True`, or `backend="auto"` to automatically select `zgame` when installed (falling back to `pyref`).
- **Why do finished slots remain frozen in `VecEnv` until `reset(done_ids)`?**
  Keeping terminal states frozen avoids silent off-by-one observation overwrites on the terminal step and lets callers record final infos/rewards before resetting specific slots via `done = batch.terminated | batch.truncated; env.reset(np.flatnonzero(done).tolist())`. If you prefer automatic `NEXT_STEP` or `SAME_STEP` autoreset, use `make_gym_vec_env(...)` from `unismac.adapters.gymnasium`.
- **Why does `smac-hard` default to `use_ability=False`?**
  This matches the published SMAC-Hard benchmark evaluation settings. Enable the V2 ability action segment explicitly with `overrides={"use_ability": True}` on live SC2.
- **Why did `accel="auto"` emit a `RuntimeWarning` and fall back to `'services'`?**
  A file or `VecEnv` method tracked by `rust_manifest.json` was edited locally, or the installed NumPy / `zfoundry-zgame` version is outside the verified manifest range. Run `unismac info` or `python -m unismac.env.accel.manifest` to see the exact mismatch.
