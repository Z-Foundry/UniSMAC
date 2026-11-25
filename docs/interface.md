# Environment Interface

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Code Map](code-map.md) · [Ecosystem Adapters](adapters.md) · [Benchmarks & Presets](benchmarks.md) · [Profiles & Configuration](profiles-and-config.md) · [Migration Guide](../MIGRATION.md)

This document specifies the purified `v0.2.0` public Python API and CLI surface of `unismac`: `make_vec_env()`, `VecEnv`, `StepBatch`, `RuntimeView`, two-sided self-play, per-episode seeding and opponent control, acceleration tiers (`accel`), and native SMAC opponent configuration. For PyMARL, Gymnasium 1.0+, and PettingZoo wrappers, see [Ecosystem Adapters](adapters.md).

---

## 1. Factory Entry Point

```python
from unismac import CONTRACT_VERSION, STANDARD_PROFILE, StepBatch, VecEnv, list_profiles, make_vec_env
```

```python
make_vec_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal["auto", "sc2", "pyref", "zgame"] | Backend | None = None,
    profile: str | None = None,
    dual: bool = False,
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal["auto", "python", "services", "rust"] = "auto",
) -> VecEnv
```

- **`make_vec_env(...)`** is the single core factory and always returns a batched `VecEnv` (even when `batch_size=1`). Single-environment RL framework interfaces are provided by monomorphic adapter factories (`make_pymarl_env`, `make_gym_env`, `make_gym_vec_env`, `make_pettingzoo_env` — see [Ecosystem Adapters](adapters.md)).
- **`profile`**: Named profile from `unismac.profiles` (e.g. `'sc2-native'`, `'sc2-explore'`, `'sc2-symmetric'`, `'sc2-live'`, `'minimal'`). Applied underneath user `overrides` with key-by-key deep merging of `backend.pyref_options`.
- **`dual`**: Shorthand for two-sided self-play (`opponent.kind='external'`). When `dual=True` and `profile` is omitted, automatically selects `profile='sc2-symmetric'`.
- **`backend`**:
  - `None` (default): stays `'sc2'` when both `profile is None` and `dual is False`; when `profile` or `dual=True` is supplied, resolves `'sc2'` for `sc2`-family profiles or `'zgame'` (if installed, else `'pyref'`) for `pyref`-family profiles.
  - `'auto'`: selects `'sc2'` for non-portable presets (`smac`, `smacv2`, `smac-hard`), or `'zgame'` (if installed, else `'pyref'`) for `*-portable` presets.
  - Explicit `'zgame'`, `'pyref'`, `'sc2'` (`batch_size=1` only), or a pre-constructed `Backend` instance with matching `batch_size`. Conflicting combinations (such as `profile='sc2-live'` with `backend='zgame'`, or `dual=True` with an asymmetric profile) raise a descriptive `ValueError`.
- **`zgame_threads`**: Rayon worker thread count for `backend='zgame'` (default `1`). Simulation results are bitwise identical regardless of thread count. Passing `zgame_threads` with any non-`zgame` backend raises `ValueError`.
- **`overrides`**: Flat upstream parameter names, dotted `EnvConfig` paths (e.g. `'termination.continuing_episode'`), or dicts returned by `profile_overrides(...)` / `native_smac_overrides(...)`. Precedence: `preset < scenario defaults < preset scenario overrides < profile overrides < dual < user overrides`.

---

## 2. Purified `VecEnv`, `StepBatch`, and `RuntimeView`

### `StepBatch`

Returned by `VecEnv.step(...)`. A frozen, slotted dataclass (`@dataclass(frozen=True, slots=True)`) that does **not** implement `__iter__` (no tuple unpacking):

| Field | Type / Shape | Description |
| --- | --- | --- |
| `reward` | `float64[B]` | Step reward for Player 1 (`0.0` for previously frozen slots). |
| `terminated` | `bool[B]` | `True` only for active slots whose battle ended on this step (`WON` / `LOST` / `DRAW`); `False` for timeout slots and previously frozen slots. |
| `truncated` | `bool[B]` | `True` only for active slots hitting `episode_limit` on this step (derived unconditionally from `timeouts` delta without requiring `termination.continuing_episode=True`); `False` for previously frozen slots. |
| `infos` | `list[dict[str, Any]]` | Length-`B` per-slot dictionaries (`battle_won`, `dead_allies`, `dead_enemies`, plus `battle_lost`/`reward_enemy` in `dual=True` mode; `{}` for frozen slots). |
| `reward_enemy` | `float64[B] \| None` | Player 2 step reward in two-sided (`dual=True`) mode; `None` in single-sided mode. |

### `VecEnv` Public Surface

`VecEnv` enforces strict orthogonality: public methods never call one another internally, and internal mutable buffers remain private.

| Member | Signature / Type | Description |
| --- | --- | --- |
| `reset` | `reset(env_ids=None, *, opponents=None, seeds=None) -> None` | Resets all environments (or the subset `env_ids`) in place and returns `None`. Supports partial reset on the very first call, per-episode scripted opponents, per-episode seeds, and per-slot idempotent spotter gating. |
| `step` | `step(actions, enemy_actions=None) -> StepBatch` | Advances active environments by `step_mul` loops (`actions: int64[B, A]`). Finished slots remain frozen until reset. |
| `obs` | `obs(side=1) -> float32[B, A, D]` | Local observations for Player 1 (`side=1`) or Player 2 (`side=2`, two-sided mode). |
| `state` | `state(side=1) -> float32[B, S]` | Global state for `side=1` or `side=2`. |
| `avail_actions` | `avail_actions(side=1) -> int64[B, A, K]` | Binary action mask (`1` = valid, `0` = invalid) for `side=1` or `side=2`. |
| `alive` | `alive(side=1) -> bool[B, A]` | Fresh boolean array indicating living agents (`health > 0`). |
| `capabilities` | `capabilities() -> float32[B, A * C]` | Per-episode SMACv2 capability vector (empty when no procedural capabilities are enabled). |
| `obs_dim` / `state_dim` | `int` | Exact feature vector widths (`obs_layout.size` and `state_layout.size`). |
| `obs_layout` / `obs_layout_for(side)` | `ObsLayout` | Structured feature metadata: `move`, `enemy`, `ally`, `own`, `timestep`, `n_enemies`, `n_allies`, `size`, `reported_size`. |
| `state_layout` / `state_layout_for(side)` | `StateLayout` | Structured global-state metadata including `.size`. |
| `n_agents` / `n_actions` / `n_enemies` | `int` | Agent count, action space size, and enemy count for Player 1. |
| `n_agents_for(side)` / `n_actions_for(side)` | `int` | Agent and action counts for `side=1` or `side=2` (may differ on asymmetric maps such as `5m_vs_6m`). |
| `batch_size` / `episode_limit` | `int` | Batch size `B` and maximum steps per episode before timeout truncation. |
| `config` / `spec` / `catalog` / `attack_prob` | Read-only properties | Resolved `EnvConfig`, `ScenarioSpec`, `UnitCatalog`, and per-slot attack probability array (if enabled). |
| `accel` | `'python' \| 'services' \| 'rust'` | Effective environment-layer acceleration tier. |
| `dual` / `mirror` | `bool` / `'x' \| 'y' \| None` | Whether two-sided mode is active, and the active mirror-spawn axis (if any). |
| `runtime` | `RuntimeView` | Read-only diagnostic and low-level runtime handle (see below). |
| `close()` / `with env:` | `None` / context manager | Releases backend resources permanently; supports `__enter__`/`__exit__` and informative `__repr__`. |

### `RuntimeView` (`env.runtime`)

Low-level simulation inspection, counters, and lifecycle controls live under `env.runtime`:

| Member | Signature / Type | Description |
| --- | --- | --- |
| `world` / `backend` | `WorldState \| None` / `Backend` | Current batched `WorldState` snapshot and underlying backend instance. |
| `healers` / `env_seeds` | `bool[...]` / `tuple[int \| None, ...]` | Healer unit mask (Medivacs) and active per-environment seeds. |
| `frame(side=1)` | `Frame` | Ordered ally/enemy `Frame` snapshot from `side`'s perspective. |
| `game_loop()` | `int64[B]` | Fresh array of current backend simulation loops (`world.game_loop`). |
| `done()` | `bool[B]` | Fresh boolean snapshot of slots currently in the terminal frozen state. |
| `episode_steps()` / `episode_count()` / `reset_count()` | `int64[B]` | Fresh snapshots of per-slot step, completed-episode, and reset counters. |
| `max_rewards()` | `float64[B]` | Fresh snapshot of per-slot reward scaling denominators. |
| `counters()` | `dict[str, int]` | Aggregate counters: `{'battles_won': int, 'battles_game': int, 'timeouts': int, 'episodes': int}`. |
| `opponent_names()` / `opponent_errors()` | Per-slot metadata | Active scripted opponent name per slot and caught script-error diagnostics. |
| `last_scenarios()` | Per-slot `Scenario` info | Most recently generated/loaded scenario records per batch slot. |
| `rng_stream(name, env_id=0, ...)` | `np.random.Generator` | Deterministic counter-based RNG stream for an environment slot. |
| `close_backend()` | `None` | Shuts down the underlying backend while preserving `VecEnv` config, episode counters, and seeds; the next full-batch `reset()` relaunches the backend. |

### Ecosystem Adapters (`unismac.adapters`)

Framework-specific interfaces (`PyMARLEnv` / `make_pymarl_env`, Gymnasium 1.0+ `GymnasiumEnv` / `GymnasiumVecEnv`, and PettingZoo `PettingZooParallelEnv`) live in `unismac.adapters` and wrap `VecEnv` cleanly without adding dependencies to the core kernel. See [Ecosystem Adapters](adapters.md).

### Additional Public Exports

- `unismac`: `CONTRACT_VERSION` (`"0.2"`), `STANDARD_PROFILE` (`"sc2-native"`), `StepBatch`, `VecEnv`, `list_profiles`, `make_vec_env`.
- `unismac.env`: `EnvConfig`, `MULTI_LEVEL_SMAC_MAPS`, `NativeSMACOpponent`, `RuntimeView`, `SMAC_TRIGGER_SPECS`, `StepBatch`, `VecEnv`, `apply_overrides`, `make_vec_env`, `native_smac_overrides`.
- `unismac.adapters`: `PyMARLEnv`, `UnismacEnv` (`= PyMARLEnv`), `make_pymarl_env` (optional extras in `unismac.adapters.gymnasium` and `unismac.adapters.pettingzoo`).
- `unismac.contract.CapabilityError`: Raised when a backend cannot satisfy a requested capability or option.
- `unismac.profiles`: `PROFILES`, `PUBLIC_PROFILES`, `LEGACY_PROFILES`, `Profile`, `STANDARD_PROFILE`, `STANDARD_EXPLORE_PROFILE`, `STANDARD_SYMMETRIC_PROFILE`, `STANDARD_PROFILES`, `profile_overrides`, `merge_overrides`, `list_profiles`, `parse_pyref_options`, `pyref_extra_overrides`, `backend_label`.
- `unismac.backends.pyref.options_meta`: `describe_pyref_options(group=None, zgame_only=False)` for browsing grouped `PyrefConfig` metadata.
- `unismac.env.opponents.scripted.script_pool(opponent_config, scenario)` (also re-exported from `unismac.env.opponents`): Returns the valid scripted-opponent pool for a SMAC-Hard scenario.

---

## 3. Two-Sided Self-Play Mode

Enable two-sided control with `dual=True` (which automatically sets `opponent.kind='external'` and `profile='sc2-symmetric'`):

```python
from unismac import make_vec_env

with make_vec_env("smac-portable", "2s3z", dual=True, batch_size=32, seed=0) as env:
    env.reset()
    obs_p1, obs_p2 = env.obs(side=1), env.obs(side=2)
    batch = env.step(
        env.avail_actions(side=1).argmax(axis=-1),
        enemy_actions=env.avail_actions(side=2).argmax(axis=-1),
    )
    r_p1, r_p2 = batch.reward, batch.reward_enemy
```

(Equivalently, pass `profile="sc2-symmetric"` with `overrides={"opponent.kind": "external"}`.)

Key semantics:

- **Stepping**: `step(actions, enemy_actions=...)` requires `enemy_actions` of shape `int64[B, n_agents_for(2)]` in two-sided mode (and rejects it in single-sided mode). Invalid Player 2 actions raise errors prefixed with `enemy_actions:`.
- **Player 2 perspective**: `obs(side=2)`, `state(side=2)`, `avail_actions(side=2)`, `alive(side=2)`, and `runtime.frame(side=2)` apply identical rules from Player 2's point of view (allies = Player 2, enemies = Player 1, independent death tracking, `last_action`, `max_rewards`, and `obs_shuffle_p2` RNG stream).
- **Rewards, infos & termination**: `batch.reward_enemy` returns Player 2's `float64[B]` reward directly alongside `batch.reward`. Active episode `batch.infos[b]` also include `reward_enemy` and `battle_lost` (Player 1 defeat) alongside `battle_won` (Player 1 victory). Simultaneous elimination counts as a draw (`battle_won=False`, `battle_lost=False`).
- **Symmetry enforcement**: Asymmetric backend options (`auto_acquire_owners`, `retreat_owners`, `hit_slow_owner1_only`, etc.) are rejected at construction time. Use `sc2-symmetric` (`STANDARD_SYMMETRIC_PROFILE`), which sets `auto_acquire_owners=()`, `retreat_owners=(1, 2)`, `hit_slow_owner1_only=False`, and `passive_from_map=False`.
- **Mirror spawning**: With `backend.spawn_mirror='auto'` (default) on explicit non-procedural maps that are geometrically mirrorable, Player 2 spawns as the exact mirror of Player 1 (position and facing), exposed via `env.mirror` (`'x'`, `'y'`, or `None`). Pass `'backend.spawn_mirror': 'off'` to use the raw spawn table instead.
  - Mirrorable maps: SMAC `3m`, `8m`, `25m`, `2s3z`, `3s5z`, `MMM`, `1c3s5z` (`'x'`) and `bane_vs_bane` (`'y'`); SMAC-Hard `3m`, `8m`, `25m`, `2s3z`, `3s5z`, `MMM`, `1c3s5z` (`'x'`).
- **Unsupported combinations** (rejected at construction): `backend='sc2'` (`CapabilityError`), `units.enemy_source='visible'`, conic FOV, `capability.attack` / `enemy_mask`, and the `actions.ability` segment.
- **Acceleration**: Two-sided mode with `sc2-symmetric` qualifies as a classic recipe and automatically runs on `accel='rust'` across all **75 / 76** runnable portable scenarios (see [Benchmarks & Presets](benchmarks.md#2-portable-scenario-coverage-76--84-runnable) for the `smac-portable/MMM2` note).
- **PettingZoo self-play**: For agent-dictionary self-play (including heterogeneous spaces on asymmetric maps such as `5m_vs_6m`), see `make_pettingzoo_env(..., dual=True)` in [Ecosystem Adapters](adapters.md#5-pettingzoo-adapter-unismacadapterspettingzoo).

---

## 4. Per-Episode Opponent Selection

On SMAC-Hard (`opponent.kind='scripted'`), callers can pin specific scripted opponents per batch slot at reset time:

```python
env.reset(env_ids=[0, 1], opponents=["attack_nearest", None])
```

- `opponents` aligns 1-to-1 with `env_ids` (or all slots when `env_ids=None`).
- Passing a script name pins that opponent for the episode; passing `None` draws from the scenario's script pool using the deterministic `opponent` RNG stream.
- Script names are validated against `script_pool(env.config.opponent, env.spec.name)` before mutating state; invalid names or non-scripted modes raise `ValueError`.
- Pinning script `X` only overrides the initial draw and leaves the opponent's internal step RNG stream identical to an episode that randomly drew `X`.

---

## 5. Per-Episode Seeds & Deterministic RNG Streams

UniSMAC supports two mutually exclusive seeding modes:

- **Batch seed (`seed=int`)**: Environment `b` derives its RNG streams from `(seed, stream, b, episode, frame)` and its backend seed from `seed + b`.
- **Per-environment seeds (`seeds=[s_0, ..., s_{B-1}]`)**: Environment `b` derives its RNG streams from `(seeds[b], stream, 0, episode, frame)` and its backend seed from `seeds[b]`. As a result, slot `b` in a `batch_size=K` vector environment is **bitwise identical** to a standalone `batch_size=1` environment initialized with `seed=seeds[b]`.
- **Dynamic re-seeding**: `env.reset(env_ids, seeds=[...])` assigns new per-environment seeds to selected slots (passing `None` keeps a slot's existing seed), resets those slots' episode counters, and yields trajectories bitwise identical to freshly constructed `batch_size=1` environments.

Named RNG streams (`unismac.env.semantics.rng.STREAMS`):

| Stream | Consumer |
| --- | --- |
| `generators` | SMACv2 procedural team and spawn-position sampling |
| `attack` | Stochastic attack-failure rolls |
| `obs_shuffle` | Player 1 observation-order shuffling when spotter gating is active |
| `opponent` | SMAC-Hard opponent pool draws and scripted opponent tie-breaking |
| `obs_shuffle_p2` | Player 2 observation-order shuffling in two-sided mode |
| `smac_trigger` | Deterministic opening-target segment sampling for `NativeSMACOpponent` on `2m_vs_1z` |

---

## 6. Acceleration Tiers (`accel`)

`make_vec_env(..., accel=...)` controls how much of the **environment layer** runs in Rust via `zgame._zgame.env`. All tiers produce **bitwise-identical** trajectories, observations, rewards, and infos, and work with any backend (`zgame`, `pyref`, or `sc2`).

| Tier | Execution Path | Behavior when Unavailable |
| --- | --- | --- |
| `'auto'` (default) | Selects `'rust'` if the config is a classic recipe and the source manifest matches; else `'services'` if `zgame._zgame.env` is installed; else `'python'`. | Never fails; warns once (`RuntimeWarning`) if the manifest mismatches and falls back to `'services'`. |
| `'python'` | Pure-Python/NumPy reference environment layer. | Always available. |
| `'services'` | Rust system-layer kernels (command translation, unit view framing, row copying, reward settlement, unit catalog tables) + Rust numerical services (`pairwise`, `grid_lookup`, `ordered_sum`); Python semantic layer. | Raises `ImportError` if `zfoundry-zgame` (`import zgame`) is not installed. |
| `'rust'` | Full Rust system + semantic layer (observations, state, action masks, rewards, termination, opening-order & all 45 SMAC-Hard scripted opponents, SMACv2 generators, spotter, stochastic attack). | Raises `ImportError` without `zfoundry-zgame` (`import zgame`), or `ValueError` explaining which recipe/manifest condition failed. |

### Classic Recipe Eligibility (`accel='rust'`)

A configuration qualifies for `'rust'` when:

1. Its base preset is `smac-portable`, `smacv2-portable`, or `smac-hard-portable` (plus scenario defaults/overrides).
2. User overrides only touch `backend.*`, `termination.continuing_episode`, or symmetric two-sided `opponent.kind='external'` with a symmetric profile. (Modifying `opponent.scripts`, enabling `use_ability=True`, or using `opponent.kind='native_smac'` falls back cleanly to `'services'` under `accel='auto'`.)
3. `unismac/env/accel/rust_manifest.json` matches the local semantic-layer source hashes, `_zgame.env.SEMANTICS`, and the NumPy version range (`>=1.26, <2.7`). Maintainers modifying the Python semantic layer can re-verify and regenerate the manifest after passing parity tests with:

```bash
python -m unismac.env.accel.manifest --list-locked         # Inspect locked files and VecEnv methods
python -m unismac.env.accel.manifest --write --jobs 6      # Re-verify parity tests and refresh hashes
```

---

## 7. Native SMAC Opponent (`NativeSMACOpponent`)

For users who want to evaluate or train against the exact map-trigger opening behavior of the original 23 SMAC maps without running live StarCraft II, `unismac.env` exports `NativeSMACOpponent`, `SMAC_TRIGGER_SPECS`, `MULTI_LEVEL_SMAC_MAPS`, and `native_smac_overrides(...)`:

```python
from unismac import make_vec_env
from unismac.env import native_smac_overrides

with make_vec_env(
    "smac-portable",
    "2c_vs_64zg",
    profile="sc2-native",
    seed=0,
    overrides=native_smac_overrides("2c_vs_64zg", terrain_elevation="auto"),
) as env:
    env.reset()
    obs, state = env.obs(), env.state()
```

`native_smac_overrides(...)` parameters:

- `scenario_name`: Scenario name (used to auto-enable multi-level terrain options on `MULTI_LEVEL_SMAC_MAPS` such as `2c_vs_64zg`).
- `single_client=True`: Switches command latency to zero-delay single-client timing (`pyref_commands='immediate'`, `pyref_command_delay=0`, `pyref_move_lag=0`).
- `smac_spawn_variant=True`: Sets `backend.spawn_variant='smac'` (Player 1 Medivac uses `Variant.RL` with autocast disabled; Player 2 and all other units use `Variant.STD`).
- `rl_no_retreat=True`: Disables Player 1 idle flee-on-hit (`retreat_owners=()`, `passive_from_map=False`) to match standard SMAC RL unit responses.
- `terrain_elevation='auto'`: Controls multi-level terrain handling (`'auto'` enables `projectile_3d=True` on `MULTI_LEVEL_SMAC_MAPS` while keeping full map visibility; `'projectile_3d'` enables 3D projectiles explicitly; `'full'` or `True` enables `visibility='sc2'`, `lost_vision='sc2'`, and `projectile_3d=True`; `'off'` or `False` omits terrain options).
- `enemy_source=None`: Optional override for `units.enemy_source` (`'all'` or `'visible'`).

---

## 8. `zgame` Build Info & Compatibility Diagnostics

```python
from unismac.backends.zgame import build_info

print(build_info())
```

Returns a dictionary describing the loaded native extension (`version`, `module`, `sha256`, `commit`, `dirty`, `built`, `source`) read directly from `zgame._zgame.build_info()`.

---

## 9. CLI Reference

Installing `unismac` provides the `unismac` command (also runnable as `python -m unismac`):

```bash
unismac --version
unismac info [--json]
unismac list-scenarios [--preset smac-portable] [--runnable] [--details]
unismac list-profiles [--all]
unismac show-profile <name> [--raw]
unismac show-options [--group <group>] [--tier <tier>] [--zgame-only]
unismac bench [--preset smac-portable] [--scenario 8m] [--backend zgame] [--batch-size 64] [--steps 200]
```

| Subcommand | Description |
| --- | --- |
| `unismac info` | Prints `unismac` version, `CONTRACT_VERSION`, `$SC2PATH` detection, `zgame` availability, four-dimensional compatibility constants, and build info (pass `--json` for machine-readable output). |
| `unismac list-scenarios` | Lists registered scenarios (filter with `--preset`; add `--runnable` for portable-runnable scenarios and `--details` for agent/enemy/limit/mirror/elevation metadata). |
| `unismac list-profiles` | Lists the 5 public semantic profiles (pass `--all` to also show the 17 legacy reproducibility aliases). |
| `unismac show-profile` | Displays a profile's concise non-default overrides relative to `PyrefConfig()` (pass `--raw` to show the full raw override dict). |
| `unismac show-options` | Browses `PyrefConfig` options filtered by topic (`--group`), tier (`--tier`), and `zgame` support (`--zgame-only`). |
| `unismac bench` | Runs a quick random-action rollout benchmark and prints effective `accel`, episodes finished, and steps/sec. |

---

## 10. Versioning Policy

- **`unismac.__version__`** (current release **`0.2.0`** — purified `VecEnv` kernel + multi-ecosystem RL adapters; see [Migration Guide](../MIGRATION.md)) tracks the environment interface under Semantic Versioning. During `0.x`, breaking API changes or default-trajectory changes bump the minor version (`0.y.0`), while bug fixes and backward-compatible ergonomic/internal improvements preserving trajectories bump the patch version (`0.y.z`).
- **`unismac.CONTRACT_VERSION`** (**`"0.2"`**) tracks the backend contract (`Backend`, `WorldState`, `Command`, `Scenario`, `Capabilities`) independently of environment-layer releases. `zfoundry-zgame` remains at **`0.1.1`** with zero Rust or trajectory changes across `unismac v0.2.0`.
