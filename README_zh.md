# UniSMAC

[English](README.md) | 中文

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![ZGame](https://img.shields.io/badge/Backend-ZGame%20Rust-orange.svg)](https://github.com/Z-Foundry/ZGame)

**UniSMAC** 是可替换后端的统一星际争霸多智能体基准环境（StarCraft Multi-Agent Challenge），将 **SMAC**、**SMACv2** 与 **SMAC-Hard** 统一于同一套简洁接口、同一份向量化契约与三类可互换仿真后端之下：

- **`zgame`**（Rust，独立仓 [Z-Foundry/ZGame](https://github.com/Z-Foundry/ZGame)，PyPI 包名 `zfoundry-zgame`，导入名 `zgame`）：高性能确定性仿真，单核速度达到 Python 参考后端的 **27×～91×**，**无需安装 StarCraft II 客户端**；
- **`pyref`**（纯 Python / NumPy）：可读、易改的可执行参考规范，与 `zgame` 保持逐位一致；
- **`sc2`**（真实 StarCraft II 客户端，可选 `[sc2]` 依赖组）：用于实机行为真值评估与上游逐值对照。

## 核心亮点

- **三基准同进程共存、零冲突**：整合 SMAC（23）、SMACv2（26）与 SMAC-Hard（35）共 **84 个注册基准场景**，提供 6 套命名预设（`smac`、`smacv2`、`smac-hard` 及三套跨后端可移植预设 `*-portable`），彻底消除上游 PySC2 地图注册表重名冲突与补丁版 PySC2 独占限制。
- **76/84 可移植场景开箱即跑**：**76 / 84** 个可移植场景在 `pyref` 与 `zgame` 上开箱即运行（`smac-portable` 23/23、`smacv2-portable` 26/26、`smac-hard-portable` 27/35；其余 8 张 SMAC-Hard 场景因缺地图/出生点资产或在 SC2 4.10 中崩溃而隔离），且 **76 / 76** 全部自动命中全 Rust 环境层加速（`accel='rust'`）。
- **原生双边对称自博弈（Self-Play）**：内置 `opponent.kind='external'` 双边模式、可镜像地图自动镜像出生与无偏对称配置 `sc2-symmetric`，支持 **75 / 76** 个可移植场景（全部命中 `accel='rust'`），在 `2s3z`、`3s5z`、`8m` 上镜像胜率 95% 置信区间全部覆盖 `0.50`。
- **高保真 SC2 行为对齐与零样本实机迁移**：纯 `zgame` 上训练的策略可直接零样本迁移至真实 StarCraft II 4.10 评估，实机胜率大幅领先旧启发式基线。
- **无缝对接现有 MARL 生态**：`make_vec_env()` 返回批量 `VecEnv`（支持按局种子 `seeds` 实现批位置无关复现、支持按局指定脚本对手）；`make_env(..., batch_size=1)` 返回兼容 PyMARL / EPyMARL 的 `UnismacEnv`。

## 实测指标

### 吞吐与加速比（`OMP_NUM_THREADS=1`，`accel='rust'`）

| 基准项 | 参照基线 | `zgame` + `accel='rust'` | 加速比 |
| --- | ---: | ---: | ---: |
| 57 单元 L3 全网格（`sc2-native`，每单元 128 局，单核） | 7,067.7 s（`pyref`） | **115.1 s** | **61.4×** |
| 代表场景单核步速（`8m` / `2c_vs_64zg` + 3D 地形） | 86.1 / 3.9 step/s（`pyref`） | **6,219 / 358 step/s** | **72.2× / 91.4×** |
| `8m` MAPPO（`B=128`，1T / 4T） | 602 step/s（10 SC2，约 4.1 核） | **14.4k / 24.1k step/s** | **24×～40×** |
| `2s3z` MAPPO（`B=128`，1T / 4T） | 719 step/s（10 SC2，约 4.0 核） | **22.6k / 34.8k step/s** | **31×～48×** |
| `sc2-native` MAPPO 端到端训练（`8m` / `2s3z`，单线程） | 7,064 s / 2,486 s（`pyref`） | **286 s / 96 s** | **24.7× / 25.8×** |

### SC2 行为对齐与零样本 Sim-to-SC2 迁移

| 评测维度 | 基线（`aligned-v4`） | `sc2-native`（`STANDARD_PROFILE`） | `sc2-explore`（`STANDARD_EXPLORE_PROFILE`） |
| --- | --- | --- | --- |
| 54 单元环境级对齐（超阈值行数） | 28 / 54 | **1 / 54** 初筛（960 局复核 **2 / 54**；SMACv2 0/18，SMAC-Hard 0/18） | 4 / 54 |
| 策略池排序相关性（166 个 MAPPO checkpoint 对实机 SC2） | `2s3z` ρ=0.650 / `8m` ρ=0.606 | **`2s3z` ρ=0.955**（`\|gap\|`=0.083，斜率 0.983）/ **`8m` ρ=0.870** | `2s3z` ρ=0.930 / `8m` ρ=0.875 |
| `zgame` ↔ `pyref` 逐位等价 | — | **6,179 / 6,179** 探针 + **246 / 246** 地形/冲锋探针 `L∞=0.0`；**57 / 57** L3 `1.000` | 逐位一致（`L∞=0.0`） |
| `2s3z` 零样本实机 SC2 4.10 胜率（3× 预算 MAPPO） | 37.4%（差距 `+47.5 pp`） | **64.6%**（10 seed，**+27.2 pp** `[+9.8, +44.7]`，差距 `+11.3 pp`，0 卡住） | **86.3%**（5 seed，差距 **`+0.3 pp`**） |
| 双边镜像公平性（`sc2-symmetric`，95% CI） | 有左偏（`2s3z` 0.609、`3s5z` 0.671） | **无偏**（`2s3z` **0.490**、`3s5z` **0.535**、`8m` **0.512**，CI 全含 0.50） | — |

## 安装

在主流平台（Linux `x86_64`/`aarch64`、macOS `arm64`/`x86_64`、Windows `AMD64`，Python 3.10–3.14）上，`pip install unismac` 会自动拉取预编译的 `zfoundry-zgame` Rust 加速包（Python 导入名 `zgame`）：

```bash
pip install unismac
```

### 纯 Python / 无预编译包平台兜底

在缺少匹配 `cp310-abi3` wheel 的平台（如 glibc < 2.28、Alpine ARM64、RISC-V 或自由线程 Python 3.13t/3.14t）上，可按纯 Python 模式安装（自动使用 `pyref` 后端与 `accel='python'`）：

```bash
pip install --no-deps unismac && pip install "numpy>=1.26"
```

如果本机具备 Rust ≥ 1.85 工具链，可执行 `pip install "unismac[zgame]"` 从源码构建加速扩展。

### 可选真实 StarCraft II 后端（`[sc2]`）

```bash
pip install "unismac[sc2]"
export SC2PATH=/path/to/StarCraftII
```

> **`[sc2]` 在 Python 3.14 下的注意事项**：上游 `pysc2==4.0.0` 依赖 `pygame`（其 PyPI 官方包仅提供至 Python 3.13 的 wheel，并锁定 `protobuf==3.20.3` 与 `s2clientprotocol==5.0.16.97563.0`）。如需启用 `[sc2]` 实机评估，推荐在 **Python 3.10–3.13** 环境中开箱安装；若在 Python 3.14 下使用，需配合 `pygame-ce>=2.5.8` 替代（或单独用 Python 3.12/3.13 跑实机评估）。仅使用 `zgame` 或 `pyref` 后端时无需安装 `[sc2]`。此外 `unismac` 在 Python 3.11+ 上会自动修补 PySC2 调用两参数 `random.shuffle` 的兼容问题。

## 30 秒快速开始

### 1. 零 SC2 依赖、单边向量化高速采样（`zgame`）

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

### 2. 双边对称自博弈（Self-Play）

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

### 3. 一行切换实机 SC2 评估或纯 Python `pyref`

```python
from unismac import make_env

# 真实 SC2 实机评估（需先 `pip install "unismac[sc2]"` 并设置 SC2PATH）
sc2_env = make_env("smac-portable", "8m", backend="sc2", seed=0)  # MultiAgentEnv 兼容的 UnismacEnv

# 或切换纯 Python 参考后端调试（无需 SC2 与 Rust）
pyref_env = make_env("smac-portable", "8m", backend="pyref", seed=0)
```

完整可运行示例位于 [`examples/`](examples/quickstart_vec.py)：
- [`examples/quickstart_vec.py`](examples/quickstart_vec.py) — 单边向量化采样示例
- [`examples/self_play.py`](examples/self_play.py) — 双边对称自博弈示例
- [`examples/sc2_eval.py`](examples/sc2_eval.py) — 多后端一键切换示例（`zgame` / `pyref` / `sc2`）

## 命令行工具

```bash
unismac info                                            # 查看版本、契约与 zgame 加速状态
unismac list-scenarios --preset smac-portable           # 列出指定预设的全部场景
unismac list-profiles                                   # 列出公开 Profile（加 --all 显示兼容别名）
unismac bench --preset smac-portable --scenario 8m \
    --backend zgame --batch-size 64 --steps 200         # 快速吞吐基准测试
```

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [架构设计](docs/architecture.md) | 总体分层、后端契约、SC2 / PyRef / ZGame 三后端架构与跨包版本契约 |
| [环境接口规范](docs/interface.md) | `VecEnv`/`UnismacEnv` 公开 API、双边自博弈、按局种子与对手、`accel` 档位及 `NativeSMACOpponent` |
| [基准与预设](docs/benchmarks.md) | SMAC / SMACv2 / SMAC-Hard 差异矩阵、84 场景（76 可移植可运行）与 SC2 本地资产提取 |
| [Profile 与配置指南](docs/profiles-and-config.md) | 公开语义化 Profile（`sc2-native`、`sc2-explore`、`sc2-symmetric`）、兼容别名与 `PyrefConfig` 配置说明 |

## 引用

如果在研究中使用了 UniSMAC，请参阅 [CITATION.cff](CITATION.cff)。

## 许可证与免责声明

本项目采用 [MIT 许可证](LICENSE)，第三方归属声明（含 SMAC、SMACv2、SMAC-Hard）见 [NOTICE](NOTICE)。

StarCraft® II 是 Blizzard Entertainment, Inc. 的注册商标。UniSMAC 是独立的开源研究项目，与 Blizzard Entertainment, Inc. 无关，亦未获得其认可或赞助。
