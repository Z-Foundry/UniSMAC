PYTHON ?= python3
JOBS ?= 4

.PHONY: help lint fmt typecheck test-fast test test-zgame test-slow manifest-check manifest-locked manifest-write check

help:
	@printf '%s\n' \
	  'UniSMAC developer targets:' \
	  '  make lint             Run ruff check and ruff format --check' \
	  '  make fmt              Format unismac/, tests/, and examples/ with ruff' \
	  '  make typecheck        Run pyrefly type checker' \
	  '  make test-fast        Run unit tests (skips sc2, slow, parity)' \
	  '  make test             Run default test suite (not sc2 and not slow)' \
	  '  make test-zgame       Run zgame backend and accel tier parity tests' \
	  '  make test-slow        Run full E1 pyref grid (-m slow)' \
	  '  make manifest-check   Check env/accel/rust_manifest.json consistency' \
	  '  make manifest-locked  List files and VecEnv methods locked by manifest' \
	  '  make manifest-write   Re-verify accel=rust parity and update manifest' \
	  '  make check            Run lint, typecheck, manifest-check, and test-fast'

lint:
	$(PYTHON) -m ruff check unismac tests examples
	$(PYTHON) -m ruff format --check unismac tests examples

fmt:
	$(PYTHON) -m ruff format unismac tests examples

typecheck:
	$(PYTHON) -m pyrefly check

test-fast:
	OMP_NUM_THREADS=1 $(PYTHON) -m pytest -n $(JOBS) -m "not sc2 and not slow and not parity"

test:
	OMP_NUM_THREADS=1 $(PYTHON) -m pytest -n $(JOBS) -m "not sc2 and not slow"

test-zgame:
	OMP_NUM_THREADS=1 $(PYTHON) -m pytest -n $(JOBS) tests/zgame tests/env/test_env_accel.py tests/env/test_env_scripted_rust.py -m "not sc2 and not slow"

test-slow:
	OMP_NUM_THREADS=1 $(PYTHON) -m pytest -n $(JOBS) -m "slow"

manifest-check:
	$(PYTHON) -m unismac.env.accel.manifest

manifest-locked:
	$(PYTHON) -m unismac.env.accel.manifest --list-locked

manifest-write:
	OMP_NUM_THREADS=1 $(PYTHON) -m unismac.env.accel.manifest --write --jobs $(JOBS)

check: lint typecheck manifest-check test-fast
