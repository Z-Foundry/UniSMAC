# Ecosystem Adapters (`unismac.adapters`)

[Back to README](../README.md) · Related: [Environment Interface](interface.md) · [Architecture](architecture.md) · [Code Map](code-map.md) · [Migration Guide](../MIGRATION.md)

Starting in **`v0.2.0`**, UniSMAC decouples its purified vectorized simulation kernel (`VecEnv`) from RL framework conventions through three thin, one-way adapters under `unismac.adapters`:

- **`unismac.adapters.pymarl`** — Zero extra dependencies. Drop-in PyMARL / EPyMARL `MultiAgentEnv` compatibility (`PyMARLEnv`, alias `UnismacEnv`).
- **`unismac.adapters.gymnasium`** — Optional extra `[gymnasium]`. Standard Gymnasium 1.0+ `Env` and `VectorEnv` for CleanRL, TorchRL, Stable-Baselines3, and custom PyTorch MAPPO/IPPO loops.
- **`unismac.adapters.pettingzoo`** — Optional extra `[pettingzoo]`. Standard PettingZoo `ParallelEnv` (with both single-sided cooperative play and first-class `dual=True` two-sided self-play, plus standard `parallel_to_aec` conversion) for RLlib, AgileRL, BenchMARL, and multi-agent research.

All adapters wrap the exact same `VecEnv` kernel and preserve **bitwise-identical** simulation semantics and full `accel='rust'` throughput.

---

## 1. Choosing the Right Interface

| Interface | Install Extra | Target Frameworks / Use Cases | Batching | Two-Sided Self-Play |
| --- | --- | --- | --- | --- |
| **Native `VecEnv`** (`unismac.make_vec_env`) | None (core) | High-throughput custom MAPPO/IPPO, research ablations, zero-copy rollouts, evaluation harnesses | Native `B >= 1` (`accel='rust'`) | Yes (`dual=True`, `side=1/2`, `batch.reward_enemy`) |
| **PyMARL `PyMARLEnv`** (`unismac.adapters.pymarl`) | None (core) | PyMARL, EPyMARL, existing `StarCraft2Env` (`MultiAgentEnv`) codebases | `batch_size=1` | Yes (`dual=True`, `enemy_actions=...`) |
| **Gymnasium `GymnasiumVecEnv` / `GymnasiumEnv`** (`unismac.adapters.gymnasium`) | `unismac[gymnasium]` | CleanRL, TorchRL (`GymWrapper`), Stable-Baselines3, custom PyTorch trainers | Native `VectorEnv` (`B >= 1`) or single `Env` | Single-sided only (use PettingZoo or `VecEnv` for self-play) |
| **PettingZoo `PettingZooParallelEnv`** (`unismac.adapters.pettingzoo`) | `unismac[pettingzoo]` | Ray RLlib, BenchMARL, AgileRL, AEC pipelines (`parallel_to_aec`), heterogeneous self-play | `batch_size=1` (parallel across agents) | **First-class** (`dual=True`: `ally_*` vs `enemy_*`, zero-padding on asymmetric maps) |

---

## 2. Optional Dependencies & Import Isolation

```bash
pip install "unismac[gymnasium]"   # Installs gymnasium>=1.0
pip install "unismac[pettingzoo]"  # Installs pettingzoo>=1.24.3 and gymnasium>=1.0
pip install "unismac[adapters]"    # Installs both Gymnasium and PettingZoo adapters
```

- **Strict one-way dependency**: `unismac.env` never imports `unismac.adapters`, `gymnasium`, or `pettingzoo`.
- **Lazy third-party imports**: `import unismac`, `import unismac.env`, and `import unismac.adapters` never load `gymnasium` or `pettingzoo`. Importing `unismac.adapters.gymnasium` or `unismac.adapters.pettingzoo` without the corresponding extra raises an actionable `ImportError` showing the exact `pip install` command.
- **Python compatibility**: `gymnasium>=1.0` depends only on lightweight pure-Python packages plus `numpy` (`cloudpickle`, `farama-notifications`, `typing-extensions`) and installs cleanly across Python 3.10–3.14.

---

## 3. PyMARL Adapter (`unismac.adapters.pymarl`)

Zero extra dependencies. Replaces the `v0.1.x` `unismac.env.single.UnismacEnv` and `make_env(..., batch_size=1)`.

### Entry Points

```python
from unismac.adapters.pymarl import PyMARLEnv, UnismacEnv, make_pymarl_env
# Also re-exported directly from `unismac.adapters`:
# from unismac.adapters import PyMARLEnv, UnismacEnv, make_pymarl_env
```

- `make_pymarl_env(preset, scenario, *, backend=None, profile=None, dual=False, seed=None, overrides=None, zgame_threads=None, accel='auto') -> PyMARLEnv` — Monomorphic factory that always returns a `batch_size=1` `PyMARLEnv`.
- `PyMARLEnv(vec: VecEnv)` — Wraps an existing `batch_size=1` `VecEnv` (`UnismacEnv` is an exact alias of `PyMARLEnv`).

### Interface Summary

Mirrors upstream SMAC `StarCraft2Env` (`MultiAgentEnv`):

- `reset() -> tuple[list[np.ndarray], np.ndarray]` (per-agent `float32[D]` list and global `float32[S]` state)
- `step(actions, enemy_actions=None) -> tuple[float, bool, dict[str, Any]]` (combines `batch.terminated | batch.truncated` into a single `terminated` bool; stepping an already-done environment returns `(0.0, True, {})`)
- `get_obs(side=1)`, `get_obs_agent(agent_id)`, `get_obs_size()` (uses `obs_layout.reported_size` to preserve upstream SMAC `timestep_pad` reporting parity)
- `get_state(side=1)`, `get_state_size()`, `get_avail_actions(side=1)`, `get_avail_agent_actions(agent_id)`, `get_total_actions()`
- `get_env_info()`, `get_stats()`, `get_capabilities()`, `get_cap_size()`, `max_reward`, `seed()`, `close()`, `render()`, `save_replay()`
- Context manager (`with make_pymarl_env(...) as env:`), read-only pass-throughs (`accel`, `dual`, `mirror`, `obs_layout`, `state_layout`, `n_agents_for`, `n_actions_for`, `alive`), and `env.vec` access to the underlying `VecEnv`.

See [`examples/sc2_eval.py`](../examples/sc2_eval.py) for a complete runnable example switching between `zgame`, `pyref`, and live `sc2`.

---

## 4. Gymnasium 1.0+ Adapter (`unismac.adapters.gymnasium`)

Requires `pip install "unismac[gymnasium]"` (`gymnasium>=1.0`). Designed for single-sided cooperative MARL (single team policy controlling `A` agents); passing `dual=True` raises `ValueError` and directs two-sided self-play users to `PettingZooParallelEnv` or `VecEnv`.

### Entry Points

```python
from unismac.adapters.gymnasium import (
    GymnasiumEnv,
    GymnasiumVecEnv,
    make_gym_env,
    make_gym_vec_env,
)
```

| Class / Factory | Signature Options |
| --- | --- |
| `GymnasiumEnv(vec, ...)` / `make_gym_env(...)` | `obs_mode: Literal['dict', 'obs_only'] = 'dict'`, `include_state: bool = True`, `invalid_action_mode: Literal['raise', 'fallback'] = 'raise'`, `render_mode: str | None = None` |
| `GymnasiumVecEnv(vec, ...)` / `make_gym_vec_env(...)` | All of the above plus `autoreset_mode: AutoresetMode \| Literal['next_step', 'same_step', 'disabled'] = 'next_step'` |

### Spaces & Exact Numeric Contracts

- **`observation_space`**:
  - Default `obs_mode='dict'` (CTDE-ready):
    - `obs`: `Box(-inf, inf, shape=(A, D), dtype=float32)` where `D = vec.obs_dim` (`obs_layout.size`, avoiding PyMARL's `timestep_pad` off-by-1 quirk)
    - `state` (when `include_state=True`): `Box(-inf, inf, shape=(S,), dtype=float32)` where `S = vec.state_dim`
    - `action_mask`: `Box(0, 1, shape=(A, K), dtype=int8)` (8× smaller than `int64` and directly accepted by Gymnasium / TorchRL / SB3-contrib masks)
  - Optional `obs_mode='obs_only'`: `Box(-inf, inf, shape=(A, D), dtype=float32)`, with `state` and `action_mask` placed in `info`.
  - Why `[-inf, inf]` bounds? In SMACv2 (`fully_observable=True`) and high-cooldown/override units, legitimate observation features (`d / sight`, `cooldown / cooldown_max`, normalized coordinates) exceed `[-1, 1]`. Unbounded `float32` `Box` prevents false `obs in observation_space` failures.
- **`action_space`**:
  - Single env: `MultiDiscrete([K] * A)` (`shape=(A,)`, `dtype=int64`).
  - Vector env: `single_action_space = MultiDiscrete([K] * A)` and batched `action_space = MultiDiscrete(full((B, A), K))` (`shape=(B, A)`, `dtype=int64`).
- **`terminated` vs `truncated`**:
  - Derived directly from `StepBatch.terminated` (battle end: victory, defeat, or draw) and `StepBatch.truncated` (reached `episode_limit`). They are strictly mutually exclusive and do not require setting `termination.continuing_episode=True`.

### Vectorized `AutoresetMode` Bridge

`GymnasiumVecEnv` bridges Gymnasium 1.0+'s three `AutoresetMode` policies directly onto `VecEnv`'s terminal-slot freezing + partial `reset(done_ids)` with zero wasted simulation work:

| `autoreset_mode` | Behavior on Episode End | Info / Final Snapshot | Best For |
| --- | --- | --- | --- |
| `'next_step'` **(default)** | Step `t` returns terminal obs/reward/info; step `t+1` steps only active slots, resets finished slots, and reads obs once (zero `final_obs` copy). | `infos['battle_won']` + `infos['_battle_won']` (dict-of-arrays) on step `t`; cleared on step `t+1`. | Gymnasium 1.0+ / CleanRL / TorchRL / maximum `accel='rust'` throughput. |
| `'same_step'` | Step `t` immediately resets finished slots and returns their initial next-episode obs. | `infos['final_obs']` + `infos['_final_obs']` and `infos['final_info']` + `infos['_final_info']` (Gymnasium 1.0+ naming). | Legacy step-inline reset loops. |
| `'disabled'` | Finished slots remain frozen until explicitly reset. | Call `env.reset(options={'reset_mask': done_mask})` for partial resets. | Custom evaluation harnesses and manual rollout scheduling. |

### `invalid_action_mode` & Official `check_env` Compliance

- **`invalid_action_mode='raise'` (default)**: Fails fast with `ValueError` whenever a masked-out action (`action_mask == 0`) is passed—protecting RL trainers from silent action-masking bugs.
- **`invalid_action_mode='fallback'`**: Remaps any masked-out action to an always-valid safe action (`1 = STOP` for living agents, `0 = NOOP` for dead agents) and records the count in `info['invalid_actions_remapped']`. Because `gymnasium.utils.env_checker.check_env` calls `env.action_space.sample()` without applying `action_mask`, pass `invalid_action_mode='fallback'` when running `check_env`:

```python
from gymnasium.utils.env_checker import check_env
from unismac.adapters.gymnasium import make_gym_env

with make_gym_env("smac-portable", "8m", profile="sc2-native", invalid_action_mode="fallback") as env:
    check_env(env, skip_render_check=False)
```

- `metadata = {'render_modes': []}` (`render_mode=None`), `render()` returns `None`, and `reset(seed=...)` delegates to `super().reset(seed=seed)` plus `VecEnv.reset(seeds=...)` for strict `check_reset_seed_determinism` compliance.

See [`examples/gymnasium_vec.py`](../examples/gymnasium_vec.py) for a full runnable example covering all three `AutoresetMode` settings.

---

## 5. PettingZoo Adapter (`unismac.adapters.pettingzoo`)

Requires `pip install "unismac[pettingzoo]"` (`pettingzoo>=1.24.3`, `gymnasium>=1.0`). Subclasses `pettingzoo.ParallelEnv` directly and passes official `pettingzoo.test.parallel_api_test` in both single-sided and two-sided (`dual=True`) modes.

### Entry Points

```python
from unismac.adapters.pettingzoo import PettingZooParallelEnv, make_pettingzoo_env
```

- `make_pettingzoo_env(preset, scenario, *, backend=None, profile=None, dual=False, seed=None, overrides=None, zgame_threads=None, accel='auto', dead_agent_mode='keep', obs_with_mask=True, invalid_action_mode='raise', render_mode=None) -> PettingZooParallelEnv`
- `PettingZooParallelEnv(vec, *, dead_agent_mode='keep', obs_with_mask=True, invalid_action_mode='raise', render_mode=None)` (requires `vec.batch_size == 1`).

### Agent IDs, Observations & CTDE Global State

- **Consistent agent naming**:
  - Single-sided (`dual=False`): `possible_agents = ['ally_0', ..., f'ally_{N-1}']`.
  - Two-sided (`dual=True`): `possible_agents = ['ally_0', ..., f'ally_{N-1}', 'enemy_0', ..., f'enemy_{M-1}']`.
  - On asymmetric maps (e.g., `5m_vs_6m`), `ally_*` and `enemy_*` expose their exact native per-side `observation_space(agent)` and `action_space(agent)` with **zero padding** (cached by identity so repeated calls return the same space object).
- **Per-agent observation & `int8` action mask**:
  - Default `obs_with_mask=True`: `Dict({'observation': Box(-inf, inf, (D,), float32), 'action_mask': Box(0, 1, (K,), int8)})`. The `int8` dtype works directly with `env.action_space(agent).sample(mask=obs[agent]['action_mask'])`.
  - Optional `obs_with_mask=False`: raw `Box(-inf, inf, (D,), float32)` with `'action_mask'` placed in `infos[agent]`.
- **CTDE global state**:
  - `env.state()` / `env.state_space`: Player 1 global state (`float32[S1]`).
  - `env.state_for(side)` / `env.state_space_for(side)`: Per-side global state (`side=1` or `side=2` in `dual=True` self-play).
- **Two-sided rewards & mirrored infos**:
  - `ally_*` agents receive `batch.reward[0]` and Player 1's info (`battle_won`, `dead_allies`, `dead_enemies`).
  - `enemy_*` agents receive `batch.reward_enemy[0]` and mirrored Player 2 info (`battle_won = raw['battle_lost']`, with `dead_allies` and `dead_enemies` swapped).

### Dead-Agent Semantics (`dead_agent_mode`)

| Mode | Behavior when Unit `i` Dies Mid-Episode | Credit Assignment | Recommended For |
| --- | --- | --- | --- |
| `'keep'` **(default)** | Agent stays in `env.agents` until battle end or timeout with `action_mask = [1, 0, ..., 0]` (`0 = NOOP` only); all agents terminate/truncate together and `env.agents` becomes `[]`. | **Exact SMAC team credit assignment** (early-killed allies still receive subsequent team & terminal win/loss rewards). | Standard SMAC / SMACv2 / SMAC-Hard training, QMIX, MAPPO, RLlib, BenchMARL. |
| `'remove'` | On the death step, `terminations[agent] = True` and `agent` is removed from `env.agents` before `step()` returns; omitted dead slots auto-fill `NOOP` on later steps. | Truncates per-agent returns at individual death. | Per-agent survival / dynamic-population MARL research. |

### Turn-Based AEC Conversion (`parallel_to_aec`)

Rather than duplicating turn-based state management, use PettingZoo's official `parallel_to_aec` conversion (verified in CI with `pettingzoo.test.api_test`):

```python
from pettingzoo.utils.conversions import parallel_to_aec
from unismac.adapters.pettingzoo import make_pettingzoo_env

par_env = make_pettingzoo_env("smac-portable", "8m", profile="sc2-native", seed=0)
aec_env = parallel_to_aec(par_env)
aec_env.reset(seed=0)
for agent in aec_env.agent_iter():
    obs, reward, terminated, truncated, info = aec_env.last()
    action = None if (terminated or truncated) else aec_env.action_space(agent).sample(mask=obs["action_mask"])
    aec_env.step(action)
aec_env.close()
```

See [`examples/pettingzoo_parallel.py`](../examples/pettingzoo_parallel.py) for a full runnable example demonstrating both single-sided cooperative play and `dual=True` two-sided self-play.
