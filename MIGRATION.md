# Migration Guide: `v0.1.x` -> `v0.2.0`

[Back to README](README.md) · Related: [Environment Interface](docs/interface.md) · [Ecosystem Adapters](docs/adapters.md) · [Architecture](docs/architecture.md)

`unismac v0.2.0` separates the **purified vectorized environment kernel** (`VecEnv` + `StepBatch` + `RuntimeView`) from **ecosystem interface adapters** (`unismac.adapters.{pymarl,gymnasium,pettingzoo}`).

- **Zero Rust / simulation changes**: `zfoundry-zgame` stays at `v0.1.1` (`CONTRACT_VERSION = "0.2"`, `ENGINE_SCHEMA_VERSION = 1`, `SERVICES_VERSION = 1`, `SEMANTICS = 3`). All trajectories, observations, rewards, and `accel='rust'` outputs remain **bitwise identical**.
- **Clean break on legacy PyMARL wrappers**: `UnismacEnv` and `make_env` are removed from top-level `unismac` and `unismac.env` (no deprecation aliases) and now live in `unismac.adapters.pymarl` as `PyMARLEnv` (`UnismacEnv` alias) and `make_pymarl_env`.

---

## 1. Quick Symbol Reference

| `v0.1.x` | `v0.2.0` | Notes |
| --- | --- | --- |
| `from unismac import make_env, UnismacEnv` | `from unismac.adapters.pymarl import make_pymarl_env, PyMARLEnv` | `UnismacEnv` is also re-exported as an alias of `PyMARLEnv` in `unismac.adapters` and `unismac.adapters.pymarl`. For batched envs (`batch_size > 1`), use `from unismac import make_vec_env`. |
| `unismac.env.single` | `unismac.adapters.pymarl` | Module removed from `unismac.env`. |
| `obs, state = vec.reset(...)` | `vec.reset(...)`; `obs = vec.obs()`; `state = vec.state()` | `VecEnv.reset()` returns `None` to avoid redundant full-batch encoding on partial resets. |
| `rewards, terminated, infos = vec.step(...)` | `batch = vec.step(...)` | Returns frozen `StepBatch(reward, terminated, truncated, infos, reward_enemy)`. Does not support tuple unpacking. |
| `info.get('episode_limit')` (requiring `continuing_episode=True`) | `batch.terminated` vs `batch.truncated` | Battle termination and timeout truncation are natively separated in `StepBatch` without needing `termination.continuing_episode=True`. |
| `[info.get('reward_enemy', 0.0) for info in infos]` | `batch.reward_enemy` | Direct `float64[B]` array in two-sided (`dual=True`) mode (`None` in single-sided mode). |
| `vec.get_obs(side=1)` | `vec.obs(side=1)` | Pure noun projection. |
| `vec.get_state(side=1)` | `vec.state(side=1)` | Pure noun projection. |
| `vec.get_avail_actions(side=1)` | `vec.avail_actions(side=1)` | Pure noun projection. |
| `vec.get_capabilities()` | `vec.capabilities()` | Pure noun projection. |
| `vec.get_obs_size()` / `vec.get_state_size()` / `vec.get_total_actions()` | `vec.obs_dim` / `vec.state_dim` / `vec.n_actions` | Note: `vec.obs_dim == vec.obs_layout.size` (exact vector width). If you need upstream SMAC's PyMARL `get_obs_size()` (`reported_size`), use `PyMARLEnv` or `vec.obs_layout.reported_size`. |
| `vec.get_env_info()` / `vec.get_cap_size()` / `vec.get_stats()` / `vec.max_reward` | `PyMARLEnv` methods, or `vec.runtime.counters()` / `vec.runtime.max_rewards()` | PyMARL-specific helpers moved to `unismac.adapters.pymarl.PyMARLEnv`. |
| `vec.world`, `vec.backend`, `vec.healers`, `vec.env_seeds` | `vec.runtime.world`, `vec.runtime.backend`, `vec.runtime.healers`, `vec.runtime.env_seeds` | Low-level diagnostics grouped under read-only `vec.runtime` (`RuntimeView`). |
| `vec.frame` / `vec.frame_for(side)` / `vec.game_loop()` | `vec.runtime.frame(side=1)` / `vec.runtime.game_loop()` | Grouped under `vec.runtime`. |
| `vec.done`, `vec.episode_steps`, `vec.episode_count`, `vec.reset_count` | `vec.runtime.done()`, `vec.runtime.episode_steps()`, `vec.runtime.episode_count()`, `vec.runtime.reset_count()` | Internal mutable buffers are private; read safe snapshots via `vec.runtime`. |
| `vec.opponent_names()`, `vec.opponent_errors()`, `vec.last_scenarios`, `vec.rng_stream(...)`, `vec.close_backend()` | `vec.runtime.opponent_names()`, `vec.runtime.opponent_errors()`, `vec.runtime.last_scenarios()`, `vec.runtime.rng_stream(...)`, `vec.runtime.close_backend()` | Grouped under `vec.runtime`. |

---

## 2. Side-by-Side Code Examples

### A. Vectorized Rollout (`VecEnv`)

**Before (`v0.1.x`):**

```python
import numpy as np
from unismac import make_vec_env

with make_vec_env("smac-portable", "8m", profile="sc2-native", batch_size=64, seed=0) as env:
    obs, state = env.reset()
    avail = env.get_avail_actions()
    rewards, terminated, infos = env.step(avail.argmax(axis=-1))
    if terminated.any():
        env.reset(np.flatnonzero(terminated).tolist())
```

**After (`v0.2.0`):**

```python
import numpy as np
from unismac import make_vec_env

with make_vec_env("smac-portable", "8m", profile="sc2-native", batch_size=64, seed=0) as env:
    env.reset()
    obs, state = env.obs(), env.state()
    avail = env.avail_actions()
    batch = env.step(avail.argmax(axis=-1))
    done = batch.terminated | batch.truncated
    if done.any():
        env.reset(np.flatnonzero(done).tolist())
```

### B. Two-Sided Self-Play (`dual=True`)

**Before (`v0.1.x`):**

```python
r_p1, terminated, infos = env.step(acts_p1, enemy_actions=acts_p2)
r_p2 = np.array([info.get("reward_enemy", 0.0) for info in infos])
```

**After (`v0.2.0`):**

```python
batch = env.step(acts_p1, enemy_actions=acts_p2)
r_p1, r_p2 = batch.reward, batch.reward_enemy
done = batch.terminated | batch.truncated
```

### C. Single-Environment PyMARL / EPyMARL (`PyMARLEnv`)

**Before (`v0.1.x`):**

```python
from unismac import make_env

with make_env("smac-portable", "8m", backend="sc2", seed=0) as env:
    obs, state = env.reset()
    reward, terminated, info = env.step(actions)
```

**After (`v0.2.0`):**

```python
from unismac.adapters.pymarl import make_pymarl_env

with make_pymarl_env("smac-portable", "8m", backend="sc2", seed=0) as env:
    obs, state = env.reset()
    reward, terminated, info = env.step(actions)
```

> `make_pymarl_env(...)` always constructs a `batch_size=1` `PyMARLEnv` with no union return type (`UnismacEnv | VecEnv`), so callers no longer need `assert isinstance(env, UnismacEnv)`.

### D. New Ecosystem Adapters in `v0.2.0`

If you previously wrapped `VecEnv` by hand to integrate with CleanRL, TorchRL, SB3, RLlib, or BenchMARL, consider switching to the official adapters in [Ecosystem Adapters](docs/adapters.md):

```python
# Gymnasium 1.0+ vectorized environment (`pip install "unismac[gymnasium]"`)
from unismac.adapters.gymnasium import make_gym_vec_env

# PettingZoo ParallelEnv with single-sided & dual=True self-play (`pip install "unismac[pettingzoo]"`)
from unismac.adapters.pettingzoo import make_pettingzoo_env
```
