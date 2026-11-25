# Environment Interface

[Back to README](../README.md) · Related: [Architecture](architecture.md) · [Benchmarks & Presets](benchmarks.md) · [Profiles & Configuration](profiles-and-config.md)

This document specifies the public Python API and CLI surface of `unismac`, including vectorized rollouts (`VecEnv`), PyMARL/EPyMARL compatibility (`UnismacEnv`), two-sided self-play, per-episode seeding and opponent control, acceleration tiers (`accel`), and native SMAC opponent configuration.

---

## 1. Factory Entry Points

```python
from unismac import make_env, make_vec_env
```

```python
make_vec_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal["sc2", "pyref", "zgame"] | Backend = "sc2",
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal["auto", "python", "services", "rust"] = "auto",
) -> VecEnv
```

- **`make_vec_env(...)`** always returns a batched `VecEnv` (even when `batch_size=1`).
- **`make_env(...)`** accepts the exact same arguments and returns a PyMARL/EPyMARL-compatible `UnismacEnv` when `batch_size == 1`, or a `VecEnv` when `batch_size > 1`.
- **`backend`**: `'zgame'` (fast Rust native backend), `'pyref'` (pure-Python reference backend), `'sc2'` (live StarCraft II via optional `[sc2]` extra; `batch_size=1` only), or a pre-constructed `Backend` instance with matching `batch_size`.
- **`zgame_threads`**: Rayon worker thread count for `backend='zgame'` (default `1`). Simulation results are bitwise identical regardless of thread count. Passing `zgame_threads` with any non-`zgame` backend raises `ValueError`.
- **`overrides`**: Flat upstream parameter names, dotted `EnvConfig` paths (e.g. `'opponent.kind'`), or dicts returned by `profile_overrides(...)` / `native_smac_overrides(...)`.

---

## 2. Public `VecEnv` API

| Member | Signature / Type | Description |
| --- | --- | --- |
| `reset` | `reset(env_ids=None, *, opponents=None, seeds=None) -> tuple[ndarray, ndarray]` | Resets all environments (or the subset `env_ids`) and returns full-batch `(obs, state)`. Supports per-episode scripted opponents and per-episode seeds. |
| `step` | `step(actions, enemy_actions=None) -> tuple[ndarray, ndarray, list[dict]]` | Advances active environments by `step_mul` loops. `actions` has shape `int64[B, A]`. Returns `(reward float64[B], terminated bool[B], infos list[dict])`. Finished slots remain frozen until reset. |
| `get_obs` | `get_obs(side=1) -> float32[B, A, D]` | Returns local observations for Player 1 (`side=1`) or Player 2 (`side=2`, two-sided mode only). |
| `get_state` | `get_state(side=1) -> float32[B, S]` | Returns global state for `side=1` or `side=2`. |
| `get_avail_actions` | `get_avail_actions(side=1) -> int64[B, A, K]` | Binary action mask (`1` = valid, `0` = invalid) for `side=1` or `side=2`. |
| `alive` | `alive(side=1) -> bool[B, A]` | Fresh boolean array indicating living agents (`frame_for(side).allies.health > 0`). |
| `game_loop` | `game_loop() -> int64[B]` | Fresh array of current backend simulation loops (`world.game_loop`). |
| `obs_layout` / `obs_layout_for(side)` | `ObsLayout` | Structured feature metadata: `move`, `enemy`, `ally`, `own`, `timestep` (tuples of feature names), `n_enemies`, `n_allies`, `size`. |
| `state_layout` / `state_layout_for(side)` | `StateLayout` | Structured global-state metadata including `.size`. |
| `n_agents` / `n_actions` / `n_enemies` | `int` | Agent count, action space size, and enemy count for Player 1. |
| `n_agents_for(side)` / `n_actions_for(side)` | `int` | Agent and action counts for `side=1` or `side=2` (may differ on asymmetric maps such as `5m_vs_6m`). |
| `episode_limit` | `int` | Maximum steps per episode before timeout. |
| `accel` | `'python' \| 'services' \| 'rust'` | Effective environment-layer acceleration tier. |
| `dual` / `mirror` | `bool` / `'x' \| 'y' \| None` | Whether two-sided mode is active, and the active mirror-spawn axis (if any). |
| `opponent_names()` | `list[str \| None]` | Active scripted opponent name per batch slot (`None` for non-scripted modes). |
| `opponent_errors()` | `int64[B]` | Cumulative step count of caught opponent-script exceptions per slot. |
| `env_seeds` | `tuple[int \| None, ...]` | Active per-environment seeds (for environments using per-episode seeding). |
| `rng_stream(name, env_id, frame=0)` | `np.random.Generator` | Deterministic counter-based RNG stream for an environment slot. |
| `frame` / `frame_for(side)` / `world` | Read-only views | Current `Frame` (ordered ally/enemy arrays) and raw `WorldState` snapshot. |
| `get_env_info()` / `get_stats()` | `dict[str, Any]` | Standard environment metadata and cumulative win/game/timeout counters. |
| `close_backend()` | `None` | Shuts down the underlying backend while preserving `VecEnv` config, episode counters, and seeds; the next full-batch `reset()` relaunches the backend. |
| `close()` | `None` | Releases backend resources permanently. |

### `UnismacEnv` (PyMARL / EPyMARL Single-Env Wrapper)

Returned by `make_env(..., batch_size=1)`. Mirrors the upstream `StarCraft2Env` (`MultiAgentEnv`) API:

- `reset() -> tuple[list[np.ndarray], np.ndarray]` (per-agent `float32[D]` list and `float32[S]` state)
- `step(actions, enemy_actions=None) -> tuple[float, bool, dict[str, Any]]`
- `get_obs(side=1)`, `get_obs_agent(agent_id)`, `get_obs_size()`
- `get_state(side=1)`, `get_state_size()`
- `get_avail_actions(side=1)`, `get_avail_agent_actions(agent_id)`, `get_total_actions()`
- `get_env_info()`, `get_stats()`, `get_capabilities()`, `get_cap_size()`, `seed()`, `close()`
- `env.vec` exposes the underlying `batch_size=1` `VecEnv` (for inspecting `env.vec.accel`, `env.vec.mirror`, etc.).

### Additional Public Exports

- `unismac.CONTRACT_VERSION`: Backend contract version string (`"0.2"`).
- `unismac.contract.CapabilityError`: Raised when a backend cannot satisfy a requested capability or option.
- `unismac.profiles`: `PROFILES`, `PUBLIC_PROFILES`, `LEGACY_PROFILES`, `Profile`, `STANDARD_PROFILE`, `STANDARD_EXPLORE_PROFILE`, `STANDARD_SYMMETRIC_PROFILE`, `STANDARD_PROFILES`, `profile_overrides`, `merge_overrides`, `list_profiles`, `parse_pyref_options`, `pyref_extra_overrides`, `backend_label`.
- `unismac.env`: `EnvConfig`, `build_config`, `resolve_config`, `apply_overrides`, `NativeSMACOpponent`, `SMAC_TRIGGER_SPECS`, `MULTI_LEVEL_SMAC_MAPS`, `native_smac_overrides`.
- `unismac.env.scripted.script_pool(opponent_config, scenario)`: Returns the valid scripted-opponent pool for a SMAC-Hard scenario.

---

## 3. Two-Sided Self-Play Mode

Enable two-sided control by combining the symmetric profile with `opponent.kind='external'`:

```python
from unismac import make_vec_env
from unismac.profiles import merge_overrides, profile_overrides

_, sym = profile_overrides("sc2-symmetric")
overrides = merge_overrides(sym, {"opponent.kind": "external"})
env = make_vec_env("smac-portable", "2s3z", backend="zgame", batch_size=32, seed=0, overrides=overrides)
```

Key semantics:

- **Stepping**: `step(actions, enemy_actions=...)` requires `enemy_actions` of shape `int64[B, n_agents_for(2)]` in two-sided mode (and rejects it in single-sided mode). Invalid Player 2 actions raise errors prefixed with `enemy_actions:`.
- **Player 2 perspective**: `get_obs(side=2)`, `get_state(side=2)`, `get_avail_actions(side=2)`, `alive(side=2)`, and `frame_for(side=2)` apply identical rules from Player 2's point of view (allies = Player 2, enemies = Player 1, independent death tracking, `last_action`, `max_rewards`, and `obs_shuffle_p2` RNG stream).
- **Infos & termination**: Active episode `infos[b]` include `reward_enemy` (Player 2's scalar reward) and `battle_lost` (Player 1 defeat), alongside `battle_won` (Player 1 victory). Simultaneous elimination counts as a draw (`battle_won=False`, `battle_lost=False`).
- **Symmetry enforcement**: Asymmetric backend options (`auto_acquire_owners`, `retreat_owners`, `hit_slow_owner1_only`, etc.) are rejected at construction time. Use `sc2-symmetric` (`STANDARD_SYMMETRIC_PROFILE`), which sets `auto_acquire_owners=()`, `retreat_owners=(1, 2)`, `hit_slow_owner1_only=False`, and `passive_from_map=False`.
- **Mirror spawning**: With `backend.spawn_mirror='auto'` (default) on explicit non-procedural maps that are geometrically mirrorable, Player 2 spawns as the exact mirror of Player 1 (position and facing), exposed via `env.mirror` (`'x'`, `'y'`, or `None`). Pass `'backend.spawn_mirror': 'off'` to use the raw spawn table instead.
  - Mirrorable maps: SMAC `3m`, `8m`, `25m`, `2s3z`, `3s5z`, `MMM`, `1c3s5z` (`'x'`) and `bane_vs_bane` (`'y'`); SMAC-Hard `3m`, `8m`, `25m`, `2s3z`, `3s5z`, `MMM`, `1c3s5z` (`'x'`).
- **Unsupported combinations** (rejected at construction): `backend='sc2'` (`CapabilityError`), `units.enemy_source='visible'`, conic FOV, `capability.attack` / `enemy_mask`, and the `actions.ability` segment.
- **Acceleration**: Two-sided mode with `sc2-symmetric` qualifies as a classic recipe and automatically runs on `accel='rust'` across all **75 / 76** runnable portable scenarios (see [Benchmarks & Presets](benchmarks.md#2-portable-scenario-coverage-76--84-runnable) for the `smac-portable/MMM2` note).

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

Named RNG streams (`unismac.env.rng.STREAMS`):

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
3. `unismac/env/rust_manifest.json` matches the local semantic-layer source hashes, `_zgame.env.SEMANTICS`, and the NumPy version range (`>=1.26, <2.7`). Maintainers modifying the Python semantic layer can re-verify and regenerate the manifest after passing parity tests with:

```bash
python -m unismac.env.manifest --write
```

---

## 7. Native SMAC Opponent (`NativeSMACOpponent`)

For users who want to evaluate or train against the exact map-trigger opening behavior of the original 23 SMAC maps without running live StarCraft II, `unismac.env` exports `NativeSMACOpponent`, `SMAC_TRIGGER_SPECS`, `MULTI_LEVEL_SMAC_MAPS`, and `native_smac_overrides(...)`:

```python
from unismac import make_env
from unismac.env import native_smac_overrides
from unismac.profiles import merge_overrides, profile_overrides

_, base = profile_overrides("sc2-native")
overrides = merge_overrides(base, native_smac_overrides("2c_vs_64zg", terrain_elevation="auto"))
env = make_env("smac-portable", "2c_vs_64zg", backend="zgame", seed=0, overrides=overrides)
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
unismac info
unismac list-scenarios [--preset smac-portable]
unismac list-profiles [--all]
unismac bench [--preset smac-portable] [--scenario 8m] [--backend zgame] [--batch-size 64] [--steps 200]
```

| Subcommand | Description |
| --- | --- |
| `unismac info` | Prints `unismac` version, `CONTRACT_VERSION`, `zgame` availability, four-dimensional compatibility constants, and build info. |
| `unismac list-scenarios` | Lists registered scenarios (optionally filtered by `--preset`). |
| `unismac list-profiles` | Lists the 5 public semantic profiles (pass `--all` to also show the 17 legacy reproducibility aliases). |
| `unismac bench` | Runs a quick random-action rollout benchmark and prints effective `accel`, episodes finished, and steps/sec. |

---

## 10. Versioning Policy

- **`unismac.__version__`** (initial public release **`0.1.0`**) tracks the environment interface under Semantic Versioning. During `0.x`, breaking API changes or default-trajectory changes bump the minor version (`0.y.0`), while bug fixes preserving public APIs and trajectories bump the patch version (`0.y.z`).
- **`unismac.CONTRACT_VERSION`** (**`"0.2"`**) tracks the backend contract (`Backend`, `WorldState`, `Command`, `Scenario`, `Capabilities`) independently of environment-layer releases.
