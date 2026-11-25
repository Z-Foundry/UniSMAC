# Contributing to UniSMAC

Thank you for your interest in contributing to **UniSMAC**. This guide covers local setup,
test tiers, the `rust_manifest.json` parity gate, and side-by-side development with
[ZGame](https://github.com/Z-Foundry/ZGame).

For a guided tour of the codebase, see [docs/code-map.md](docs/code-map.md) and
[docs/architecture.md](docs/architecture.md).

## 1. Development Environment Setup

UniSMAC uses PEP 735 `[dependency-groups]` in `pyproject.toml` so development tools stay out of
published package metadata.

### Using `uv` (recommended)

```bash
uv sync --group dev
```

To include optional Gymnasium and PettingZoo adapter test dependencies:

```bash
uv sync --group dev --group adapters
```

For pure-Python development without installing the compiled `zfoundry-zgame` wheel:

```bash
uv venv
uv pip install --no-deps -e .
uv pip install "numpy>=1.26" "pytest>=8.0" "pytest-xdist>=3.6" "ruff>=0.9" "pyrefly>=0.30"
```

### Using `pip` (`pip >= 25.1`)

```bash
python -m pip install -e . --group dev
```

On older `pip` versions that do not support `--group`, install the dev tools explicitly:

```bash
python -m pip install -e . "pytest>=8.0" "pytest-xdist>=3.6" "ruff>=0.9" "pyrefly>=0.30"
```

## 2. Everyday Commands (`Makefile`)

A lightweight `Makefile` wraps the standard local checks (all test commands pin
`OMP_NUM_THREADS=1` to avoid OpenMP/BLAS oversubscription):

```bash
make help            # list available targets
make lint            # ruff check + ruff format --check (unismac, tests, examples)
make fmt             # auto-format with ruff
make typecheck       # pyrefly check
make test-fast       # fast unit suite (skips sc2, slow, and heavy AVX-512 parity)
make test            # full default suite (matches CI: not sc2 and not slow)
make test-zgame      # zgame backend & accel tier tests
make test-adapters   # PyMARL, Gymnasium, and PettingZoo adapter contract tests
make test-slow       # full E1 pyref grid (-m slow)
make manifest-check  # verify env/accel/rust_manifest.json matches current sources
make manifest-locked # print all files and VecEnv methods locked by rust_manifest.json
make manifest-write  # re-verify all accel='rust' bitwise parity tests and update manifest
make check           # lint + typecheck + manifest-check + test-fast
```

Adjust worker concurrency via `JOBS=4` (for example `make test JOBS=6`) and the interpreter via
`PYTHON=.venv/bin/python`.

## 3. Test Suite Layout and Markers

`tests/` is organized into eight layers plus shared helpers:

- `tests/api/` -- public API surface (`test_public_api.py`), architecture invariants, and dependency boundaries (`test_boundary.py`)
- `tests/adapters/` -- PyMARL (`PyMARLEnv`), Gymnasium (`check_env`, `AutoresetMode`), PettingZoo (`parallel_api_test`, `parallel_to_aec`), and missing-extra diagnostics
- `tests/contract/` -- backend contract types and conformance checks
- `tests/data/` -- static assets, presets, scenarios, unit overrides, and SMAC bot helpers
- `tests/env/` -- purified `VecEnv` / `StepBatch` / `RuntimeView`, RNG, generators, opponents, dual/mirror mode, and 3-tier `accel`
- `tests/pyref/` -- pure-Python reference simulation engine unit tests
- `tests/zgame/` -- `zgame` Rust backend step-by-step parity with `pyref`
- `tests/sc2/` -- StarCraft II backend tests (opt-in via `-m sc2`)
- `tests/conftest.py` & `tests/_support/` -- session-cached asset loaders and shared parity helpers

Pytest markers:

- `sc2` -- requires a local StarCraft II installation (`$SC2PATH`)
- `slow` -- full cross-tier E1 grid on `pyref` (`pytest -m slow`)
- `parity` -- heavy cross-backend and AVX-512-disabled subprocess bitwise parity tests
- `adapters` -- tests requiring optional `gymnasium` or `pettingzoo` extras

## 4. Working with `rust_manifest.json`

`unismac/env/accel/rust_manifest.json` locks 47 semantic-layer source files and the core
`VecEnv` methods whose behavior is mirrored bitwise in `zgame._zgame.env` (`accel='rust'`).

Before editing files under `unismac/env/` or `unismac/presets/`, check whether a file is locked:

```bash
python -m unismac.env.accel.manifest --list-locked
```

- Editing **unlocked** files (for example `adapters/`, `backends/`, `cli.py`, `profiles.py`,
  `scenarios/`, `env/accel/manifest.py`) requires **no** manifest refresh.
- Editing **locked** files requires verifying parity with `zfoundry-zgame` installed and
  regenerating the manifest:

```bash
OMP_NUM_THREADS=1 python -m unismac.env.accel.manifest --write --jobs 4
```

`--write` always executes the full `accel='rust'` bitwise parity suite before writing
`rust_manifest.json`; there is intentionally no flag to bypass parity verification.

## 5. Side-by-Side Local Development with `ZGame`

When changing both `ZGame` (Rust kernel / `zgame._zgame.env`) and `UniSMAC`, check out both
repositories as siblings:

```text
workspace/
├── ZGame/
├── UniSMAC/
└── dist/       # shared local wheel directory
```

### Build a local `zfoundry-zgame` wheel and test in `UniSMAC`

```bash
# In ../ZGame
export CARGO_TARGET_DIR=/tmp/zgame-target
export RUSTFLAGS="--remap-path-prefix=$HOME/.cargo=cargo --remap-path-prefix=$PWD=."
maturin build --release --out ../dist

# Verify Rust tests and Python stubs in ../ZGame
cargo test --workspace --exclude zgame-py --locked
python -m mypy.stubtest --ignore-disjoint-bases zgame._zgame

# In ../UniSMAC
uv pip install --reinstall --find-links ../dist --no-index --no-deps zfoundry-zgame
python -m unismac.env.accel.manifest
OMP_NUM_THREADS=1 pytest -n 4 tests/zgame tests/env/test_env_accel.py
```

Alternatively, if both checkouts share a single virtual environment, run
`maturin develop --release` inside `../ZGame` to rebuild the extension in place.

### Cross-package contract version coordination

`UniSMAC` and `ZGame` enforce four explicit compatibility constants:

1. `CONTRACT_VERSION` (`'0.2'`) -- world/command/event contract
2. `ENGINE_SCHEMA_VERSION` (`1`) -- `PyrefConfig` option schema passed to `zgame.Engine`
3. `zgame.env.SERVICES_VERSION` (`1`) -- `accel='services'` system-layer ABI
4. `zgame.env.SEMANTICS` (`3`) -- `accel='rust'` semantic-layer parity version

When bumping any of these constants, release the matching `zfoundry-zgame` version first, then
update `UniSMAC` and refresh `rust_manifest.json`.

## 6. Pull Request Checklist

- All code, docstrings, comments, tests, and English docs use ASCII/English only (`README_zh.md`
  is the sole localized Chinese file).
- `make lint` and `make typecheck` pass with zero warnings or errors.
- `make manifest-check` exits `0` (or `make manifest-write` was run if locked files changed).
- Tests pass in both pure-Python mode (without `zfoundry-zgame`) and accelerated mode (with
  `zfoundry-zgame` installed).
- Preserve backwards compatibility for `zfoundry-zgame` (`0.2` / `1` / `1` / `3` contract),
  zero public-to-public calls inside `VecEnv`, strict unidirectional `adapters -> env` imports,
  and bitwise determinism across `pyref`, `zgame`, and all three `accel` tiers.
