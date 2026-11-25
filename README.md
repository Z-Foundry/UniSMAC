# UniSMAC

English | [Chinese (zh-CN)](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![ZGame](https://img.shields.io/badge/Backend-ZGame%20Rust-orange.svg)](https://github.com/Z-Foundry/ZGame)

**UniSMAC** is a unified, backend-swappable StarCraft Multi-Agent Challenge environment that brings **SMAC**, **SMACv2**, and **SMAC-Hard** under one purified vectorized kernel, first-class RL ecosystem adapters, and three interchangeable simulation backends:

- **`zgame`** (Rust, via [Z-Foundry/ZGame](https://github.com/Z-Foundry/ZGame), PyPI package `zfoundry-zgame`, imported as `zgame`): High-performance, deterministic simulation with **27×–91×** single-core speedups over Python and **zero StarCraft II dependency**.
- **`pyref`** (Pure Python / NumPy): Readable, hackable executable reference specification with bitwise parity against `zgame`.
- **`sc2`** (StarCraft II via PySC2, optional `[sc2]` extra): Real StarCraft II client backend for ground-truth evaluation and upstream parity verification.

## Highlights

- **3 Benchmarks, 1 Process, Zero Conflicts**: Consolidates **84** registered benchmark scenarios across SMAC (23), SMACv2 (26), and SMAC-Hard (35) into 6 named presets (`smac`, `smacv2`, `smac-hard`, and 3 backend-portable `*-portable` presets). Eliminates upstream PySC2 map-registry collisions and patched-PySC2 conflicts so all benchmarks coexist in one process.
- **76/84 Portable Scenarios Out-of-the-Box**: **76 / 84** scenarios run immediately on both `pyref` and `zgame` (`smac-portable` 23/23, `smacv2-portable` 26/26, `smac-hard-portable` 27/35; the remaining 8 SMAC-Hard maps either lack map/spawn assets or crash in SC2 4.10), with **76 / 76** automatically hitting full Rust environment acceleration (`accel='rust'`).
- **Native Two-Sided Self-Play**: Built-in `dual=True` (`opponent.kind='external'`) mode with mirror spawning and the unbiased `sc2-symmetric` profile runs on **75 / 76** portable scenarios (all hitting `accel='rust'`), achieving mirror win-rate 95% CIs covering `0.50` across `2s3z`, `3s5z`, and `8m`.
- **High-Fidelity SC2 Alignment & Zero-Shot Transfer**: Policies trained purely in `zgame` transfer zero-shot to real StarCraft II 4.10 with dramatic improvements over heuristic baselines.
- **Purified Kernel + Multi-Ecosystem Adapters (`v0.2.0`)**: `make_vec_env()` exposes a minimal, orthogonal `VecEnv` kernel (`reset() -> None`, `step() -> StepBatch` with native `terminated`/`truncated`/`reward_enemy`, `obs()`/`state()`/`avail_actions()`, and `env.runtime`), while `unismac.adapters` provides drop-in adapters for **PyMARL/EPyMARL** (zero extra deps), **Gymnasium 1.0+ `Env` / `VectorEnv`** (`[gymnasium]`), and **PettingZoo `ParallelEnv`** (`[pettingzoo]`, single-sided and `dual=True` self-play). Upgrading from `v0.1.x`? See [MIGRATION.md](MIGRATION.md).

## Verified Benchmarks

### Throughput & Speedup (`OMP_NUM_THREADS=1`, `accel='rust'`)

| Benchmark | Baseline | `zgame` + `accel='rust'` | Speedup |
| --- | ---: | ---: | ---: |
| 57-unit L3 suite (`sc2-native`, 128 ep/unit, 1 core) | 7,067.7 s (`pyref`) | **115.1 s** | **61.4×** |
| Single-core representative maps (`8m` / `2c_vs_64zg` + 3D) | 86.1 / 3.9 step/s (`pyref`) | **6,219 / 358 step/s** | **72.2× / 91.4×** |
| `8m` MAPPO (`B=128`, 1T / 4T) | 602 step/s (10 SC2, ~4.1 cores) | **14.4k / 24.1k step/s** | **24×–40×** |
| `2s3z` MAPPO (`B=128`, 1T / 4T) | 719 step/s (10 SC2, ~4.0 cores) | **22.6k / 34.8k step/s** | **31×–48×** |
| `sc2-native` MAPPO end-to-end (`8m` / `2s3z`, 1T) | 7,064 s / 2,486 s (`pyref`) | **286 s / 96 s** | **24.7× / 25.8×** |

### Behavioral Fidelity & Sim-to-SC2 Zero-Shot Transfer

| Evaluation | Baseline (`aligned-v4`) | `sc2-native` (`STANDARD_PROFILE`) | `sc2-explore` (`STANDARD_EXPLORE_PROFILE`) |
| --- | --- | --- | --- |
| 54-cell env-level alignment (exceeds-threshold cells) | 28 / 54 | **1 / 54** screening (**2 / 54** at 960 ep; SMACv2 0/18, SMAC-Hard 0/18) | 4 / 54 |
| Policy-pool ranking vs real SC2 (166 MAPPO ckpts) | `2s3z` ρ=0.650 / `8m` ρ=0.606 | **`2s3z` ρ=0.955** (`\|gap\|`=0.083, slope=0.983) / **`8m` ρ=0.870** | `2s3z` ρ=0.930 / `8m` ρ=0.875 |
| `zgame` ↔ `pyref` bitwise parity | — | **6,179 / 6,179** probes + **246 / 246** terrain/charge probes `L∞=0.0`; **57 / 57** L3 `1.000` | Bitwise identical (`L∞=0.0`) |
| `2s3z` zero-shot SC2 4.10 win rate (3× MAPPO) | 37.4% (gap `+47.5 pp`) | **64.6%** (10 seeds, **+27.2 pp** `[+9.8, +44.7]`, gap `+11.3 pp`, 0 stuck) | **86.3%** (5 seeds, gap **`+0.3 pp`**) |
| Two-sided mirror symmetry (`sc2-symmetric`, 95% CI) | Biased (`2s3z` 0.609, `3s5z` 0.671) | **Unbiased** (`2s3z` **0.490**, `3s5z` **0.535**, `8m` **0.512** — all CIs cover 0.50) | — |

## Installation

On mainstream platforms (Linux `x86_64`/`aarch64`, macOS `arm64`/`x86_64`, Windows `AMD64`, Python 3.10–3.14), `pip install unismac` automatically installs the prebuilt `zfoundry-zgame` Rust acceleration package (imported as `zgame`):

```bash
pip install unismac
```

### Optional Ecosystem Adapters

```bash
pip install "unismac[gymnasium]"   # Gymnasium 1.0+ Env & VectorEnv adapter
pip install "unismac[pettingzoo]"  # PettingZoo ParallelEnv adapter (single & dual self-play)
pip install "unismac[adapters]"    # Both Gymnasium and PettingZoo adapters
```

The PyMARL / EPyMARL adapter (`unismac.adapters.pymarl`) requires **zero extra dependencies**.

### Pure-Python / Unsupported Platform Fallback

On platforms without a matching `cp310-abi3` `zfoundry-zgame` wheel (e.g., glibc < 2.28, Alpine ARM64, RISC-V, or free-threaded Python 3.13t/3.14t), install UniSMAC in pure-Python mode (`pyref` backend, `accel='python'`):

```bash
pip install --no-deps unismac && pip install "numpy>=1.26"
```

If a Rust ≥ 1.85 toolchain is available, you can build `zfoundry-zgame` from source with `pip install "unismac[zgame]"`.

### Optional Real StarCraft II Backend (`[sc2]`)

```bash
pip install "unismac[sc2]"
export SC2PATH=/path/to/StarCraftII
```

> **Python 3.14 note for `[sc2]`**: Upstream `pysc2==4.0.0` depends on `pygame`, which only publishes prebuilt wheels up to Python 3.13 (and pins `protobuf==3.20.3` and `s2clientprotocol==5.0.16.97563.0`). For the optional `[sc2]` extra, **Python 3.10–3.13** works out-of-the-box; on Python 3.14, install `pygame-ce>=2.5.8` (or use a Python 3.12/3.13 environment for SC2 evaluation). Note that `unismac` requires no `[sc2]` extra for `zgame` or `pyref`, and automatically patches PySC2's removed 2-argument `random.shuffle` call on Python 3.11+.

## 30-Second Quickstart

### 1. Fast Zero-SC2 Vectorized Rollout (`VecEnv`)

```python
import numpy as np
from unismac import make_vec_env

with make_vec_env("smac-portable", "8m", profile="sc2-native", batch_size=64, seed=0) as env:
    env.reset()
    obs, state = env.obs(), env.state()  # obs: [64, 8, D], state: [64, S]; env.accel == "rust"
    rng = np.random.default_rng(0)
    for _ in range(100):
        avail = env.avail_actions()
        actions = ((1.0 + rng.random(avail.shape)) * avail).argmax(axis=-1)
        batch = env.step(actions)  # StepBatch(reward, terminated, truncated, infos)
        done = batch.terminated | batch.truncated
        if done.any():
            env.reset(np.flatnonzero(done).tolist())
```

### 2. Symmetric Two-Sided Self-Play

```python
from unismac import make_vec_env

with make_vec_env("smac-portable", "2s3z", dual=True, batch_size=32, seed=0) as env:
    env.reset()
    obs_p1, obs_p2 = env.obs(side=1), env.obs(side=2)
    avail_p1, avail_p2 = env.avail_actions(side=1), env.avail_actions(side=2)
    batch = env.step(avail_p1.argmax(-1), enemy_actions=avail_p2.argmax(-1))
    r_p1, r_p2 = batch.reward, batch.reward_enemy
```

### 3. Ecosystem Adapters (Gymnasium, PettingZoo, PyMARL) & Live SC2 Switching

```python
# Gymnasium 1.0+ VectorEnv (`pip install "unismac[gymnasium]"`)
from unismac.adapters.gymnasium import make_gym_vec_env
gym_env = make_gym_vec_env("smac-portable", "8m", profile="sc2-native", batch_size=32, seed=0)
obs, info = gym_env.reset(seed=0)  # obs = {"obs": ..., "state": ..., "action_mask": int8}

# PettingZoo ParallelEnv, including two-sided self-play (`pip install "unismac[pettingzoo]"`)
from unismac.adapters.pettingzoo import make_pettingzoo_env
pz_env = make_pettingzoo_env("smac-portable", "2s3z", dual=True, seed=0)
obs, infos = pz_env.reset(seed=0)  # agents: ally_0..ally_4, enemy_0..enemy_4

# PyMARL / EPyMARL MultiAgentEnv & one-line live SC2 evaluation (zero extra deps for zgame/pyref)
from unismac.adapters.pymarl import make_pymarl_env
sc2_env = make_pymarl_env("smac-portable", "8m", backend="sc2", seed=0)
pyref_env = make_pymarl_env("smac-portable", "8m", backend="pyref", profile="sc2-native", seed=0)
```

Runnable scripts are in [`examples/`](examples/quickstart_vec.py):
- [`examples/quickstart_vec.py`](examples/quickstart_vec.py) — vectorized single-side rollout (`VecEnv` + `StepBatch`)
- [`examples/self_play.py`](examples/self_play.py) — two-sided symmetric self-play (`dual=True`, `batch.reward_enemy`)
- [`examples/marl_loop_minimal.py`](examples/marl_loop_minimal.py) — zero-dependency MARL rollout, `terminated`/`truncated` bootstrapping, and per-env `seeds` eval
- [`examples/sc2_eval.py`](examples/sc2_eval.py) — PyMARL adapter & backend switching (`zgame` / `pyref` / `sc2`)
- [`examples/gymnasium_vec.py`](examples/gymnasium_vec.py) — Gymnasium 1.0+ `VectorEnv` across `next_step`, `same_step`, and `disabled` autoreset modes
- [`examples/pettingzoo_parallel.py`](examples/pettingzoo_parallel.py) — PettingZoo `ParallelEnv` for single-sided cooperative play and `dual=True` self-play

## CLI

```bash
unismac info [--json]                                                 # Versions, contract, SC2PATH & zgame status
unismac list-scenarios --preset smac-hard-portable --runnable --details  # Inspect scenarios & portable coverage
unismac list-profiles [--all]                                         # List public profiles (add --all for legacy aliases)
unismac show-profile sc2-native                                       # Show concise non-default profile overrides
unismac show-options --group collision --zgame-only                   # Browse grouped PyrefConfig options
unismac bench --preset smac-portable --scenario 8m \
    --backend zgame --batch-size 64 --steps 200                       # Quick throughput benchmark
```

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Layered design, purified kernel + adapters boundary, SC2 / PyRef / ZGame backends, and cross-package versioning |
| [Architecture & Code Map](docs/code-map.md) | 30-minute reading map: `step()` trace, 1:1 `pyref` ↔ `zgame` mirror, `accel` routing, and FAQ |
| [Environment Interface](docs/interface.md) | Purified `VecEnv`, `StepBatch`, `RuntimeView`, `profile=`/`dual=`, per-episode seeds/opponents, `accel` tiers, and CLI |
| [Ecosystem Adapters](docs/adapters.md) | PyMARL, Gymnasium 1.0+ (`Env`/`VectorEnv`), and PettingZoo (`ParallelEnv` + `parallel_to_aec`) selection matrix and contracts |
| [Benchmarks & Presets](docs/benchmarks.md) | SMAC / SMACv2 / SMAC-Hard differences, 84 scenarios (76 portable runnable), SC2 setup & asset extraction |
| [Profiles & Configuration](docs/profiles-and-config.md) | Public profiles (`sc2-native`, `sc2-explore`, `sc2-symmetric`), legacy aliases, and grouped `PyrefConfig` reference |
| [Migration Guide (`v0.1.x` → `v0.2.0`)](MIGRATION.md) | Symbol mapping and before/after examples for upgrading to the purified `v0.2.0` kernel and `unismac.adapters` |
| [Contributing](CONTRIBUTING.md) | Developer setup, [`Makefile`](Makefile) targets, 8-layer test suite, and `rust_manifest.json` workflow |

## Citation

If you use UniSMAC in your research, please cite it using [CITATION.cff](CITATION.cff).

## License & Disclaimer

Licensed under the [MIT License](LICENSE). Third-party attributions (including SMAC, SMACv2, and SMAC-Hard) are listed in [NOTICE](NOTICE).

StarCraft® II is a registered trademark of Blizzard Entertainment, Inc. UniSMAC is an independent open-source research project and is not affiliated with, endorsed by, or sponsored by Blizzard Entertainment, Inc.
