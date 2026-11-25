# UniSMAC

English | [Chinese (zh-CN)](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![ZGame](https://img.shields.io/badge/Backend-ZGame%20Rust-orange.svg)](https://github.com/Z-Foundry/ZGame)

**UniSMAC** is a unified, backend-swappable StarCraft Multi-Agent Challenge environment that brings **SMAC**, **SMACv2**, and **SMAC-Hard** under one clean API, one vectorized contract, and three interchangeable simulation backends:

- **`zgame`** (Rust, via [Z-Foundry/ZGame](https://github.com/Z-Foundry/ZGame), PyPI package `zfoundry-zgame`, imported as `zgame`): High-performance, deterministic simulation with **27×–91×** single-core speedups over Python and **zero StarCraft II dependency**.
- **`pyref`** (Pure Python / NumPy): Readable, hackable executable reference specification with bitwise parity against `zgame`.
- **`sc2`** (StarCraft II via PySC2, optional `[sc2]` extra): Real StarCraft II client backend for ground-truth evaluation and upstream parity verification.

## Highlights

- **3 Benchmarks, 1 Process, Zero Conflicts**: Consolidates **84** registered benchmark scenarios across SMAC (23), SMACv2 (26), and SMAC-Hard (35) into 6 named presets (`smac`, `smacv2`, `smac-hard`, and 3 backend-portable `*-portable` presets). Eliminates upstream PySC2 map-registry collisions and patched-PySC2 conflicts so all benchmarks coexist in one process.
- **76/84 Portable Scenarios Out-of-the-Box**: **76 / 84** scenarios run immediately on both `pyref` and `zgame` (`smac-portable` 23/23, `smacv2-portable` 26/26, `smac-hard-portable` 27/35; the remaining 8 SMAC-Hard maps either lack map/spawn assets or crash in SC2 4.10), with **76 / 76** automatically hitting full Rust environment acceleration (`accel='rust'`).
- **Native Two-Sided Self-Play**: Built-in `opponent.kind='external'` mode with mirror spawning and the unbiased `sc2-symmetric` profile runs on **75 / 76** portable scenarios (all hitting `accel='rust'`), achieving mirror win-rate 95% CIs covering `0.50` across `2s3z`, `3s5z`, and `8m`.
- **High-Fidelity SC2 Alignment & Zero-Shot Transfer**: Policies trained purely in `zgame` transfer zero-shot to real StarCraft II 4.10 with dramatic improvements over heuristic baselines.
- **Drop-In RL Ecosystem Interfaces**: `make_vec_env()` returns a batched `VecEnv` with per-episode `seeds` (batch-position invariant) and per-episode opponent selection; `make_env(..., batch_size=1)` returns a PyMARL/EPyMARL-compatible `UnismacEnv`.

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

### 1. Fast Zero-SC2 Vectorized Rollout (`zgame`)

```python
import numpy as np
from unismac import make_vec_env
from unismac.profiles import profile_overrides

_, overrides = profile_overrides("sc2-native")
env = make_vec_env(
    "smac-portable", "8m", backend="zgame", batch_size=64, seed=0, overrides=overrides
)
obs, state = env.reset()  # obs: [64, 8, D], state: [64, S]; env.accel == "rust"

rng = np.random.default_rng(0)
for _ in range(100):
    avail = env.get_avail_actions()
    actions = np.array([[int(rng.choice(np.flatnonzero(r))) for r in row] for row in avail])
    rewards, terminated, infos = env.step(actions)
    if terminated.any():
        env.reset(np.flatnonzero(terminated).tolist())
env.close()
```

### 2. Symmetric Two-Sided Self-Play

```python
import numpy as np
from unismac import make_vec_env
from unismac.profiles import merge_overrides, profile_overrides

_, sym = profile_overrides("sc2-symmetric")
overrides = merge_overrides(sym, {"opponent.kind": "external"})
env = make_vec_env(
    "smac-portable", "2s3z", backend="zgame", batch_size=32, seed=0, overrides=overrides
)
env.reset()

obs_p1, obs_p2 = env.get_obs(side=1), env.get_obs(side=2)
avail_p1, avail_p2 = env.get_avail_actions(side=1), env.get_avail_actions(side=2)
r_p1, terminated, infos = env.step(avail_p1.argmax(-1), enemy_actions=avail_p2.argmax(-1))
r_p2 = np.array([info.get("reward_enemy", 0.0) for info in infos])
env.close()
```

### 3. One-Line Switch to Real SC2 Evaluation or Pure-Python `pyref`

```python
from unismac import make_env

# Real StarCraft II evaluation (requires `pip install "unismac[sc2]"` and SC2PATH)
sc2_env = make_env("smac-portable", "8m", backend="sc2", seed=0)  # MultiAgentEnv-compatible UnismacEnv

# Or readable pure-Python reference backend (no SC2 or Rust needed)
pyref_env = make_env("smac-portable", "8m", backend="pyref", seed=0)
```

Runnable scripts are in [`examples/`](examples/quickstart_vec.py):
- [`examples/quickstart_vec.py`](examples/quickstart_vec.py) — vectorized single-side rollout
- [`examples/self_play.py`](examples/self_play.py) — two-sided symmetric self-play
- [`examples/sc2_eval.py`](examples/sc2_eval.py) — backend switching (`zgame` / `pyref` / `sc2`)

## CLI

```bash
unismac info                                            # Package, contract, and zgame acceleration status
unismac list-scenarios --preset smac-portable           # List scenarios for a preset
unismac list-profiles                                   # List public profiles (add --all for legacy aliases)
unismac bench --preset smac-portable --scenario 8m \
    --backend zgame --batch-size 64 --steps 200         # Quick throughput benchmark
```

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Layered design, backend contract, SC2 / PyRef / ZGame backends, and cross-package versioning |
| [Environment Interface](docs/interface.md) | Public `VecEnv`/`UnismacEnv` API, two-sided self-play, per-episode seeds/opponents, `accel` tiers, and `NativeSMACOpponent` |
| [Benchmarks & Presets](docs/benchmarks.md) | SMAC / SMACv2 / SMAC-Hard differences, 84 scenarios (76 portable runnable), SC2 setup & asset extraction |
| [Profiles & Configuration](docs/profiles-and-config.md) | Public profiles (`sc2-native`, `sc2-explore`, `sc2-symmetric`), legacy aliases, and `PyrefConfig` reference |

## Citation

If you use UniSMAC in your research, please cite it using [CITATION.cff](CITATION.cff).

## License & Disclaimer

Licensed under the [MIT License](LICENSE). Third-party attributions (including SMAC, SMACv2, and SMAC-Hard) are listed in [NOTICE](NOTICE).

StarCraft® II is a registered trademark of Blizzard Entertainment, Inc. UniSMAC is an independent open-source research project and is not affiliated with, endorsed by, or sponsored by Blizzard Entertainment, Inc.
