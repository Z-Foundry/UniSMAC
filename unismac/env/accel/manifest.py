"""Source-hash and compatibility manifest for ``accel='rust'`` (``env/accel/rust_manifest.json``).

The Rust semantic layer in ``zgame._zgame.env`` mirrors the Python reference
implementation bitwise. This manifest records the verified state: LF-normalized
SHA-256 digests of semantic-layer source files (`SOURCES`) and `VecEnv` methods
(`FUNCTIONS`), the supported NumPy version range (``numpy_min`` inclusive to
``numpy_max_exclusive`` exclusive), and the native semantic layer version
(``_zgame.env.SEMANTICS``).

At runtime (`accel.resolve`), any mismatch causes ``accel='auto'`` to fall back
to ``'services'`` with a single ``RuntimeWarning``, while explicit
``accel='rust'`` raises `ValueError`. After editing semantic-layer files,
re-verify parity and regenerate the manifest with:

    python -m unismac.env.accel.manifest --write [--jobs N]

Running ``python -m unismac.env.accel.manifest`` without ``--write`` prints any
mismatches and exits non-zero when the manifest is stale. Pass ``--list-locked``
to inspect the exact list of manifest-locked files and ``VecEnv`` methods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    'FUNCTIONS',
    'MANIFEST',
    'NUMPY_MAX_EXCLUSIVE',
    'NUMPY_MIN',
    'SOURCES',
    'current',
    'diff',
    'load',
    'mismatches',
    'write',
]

_PKG = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('rust_manifest.json')

NUMPY_MIN = '1.26'
NUMPY_MAX_EXCLUSIVE = '2.7'

SOURCES = (
    'env/config.py',
    'env/dual.py',
    'env/opponents/__init__.py',
    'env/semantics/actions.py',
    'env/semantics/catalog.py',
    'env/semantics/features.py',
    'env/semantics/generators.py',
    'env/semantics/reward.py',
    'env/semantics/rng.py',
    'env/semantics/services.py',
    'env/semantics/settle.py',
    'env/semantics/spotter.py',
    'env/semantics/termination.py',
    'env/semantics/terrain.py',
    'env/semantics/units.py',
    'env/semantics/visibility.py',
    'presets/registry.py',
    'presets/smac.py',
    'presets/smac_hard.py',
    'presets/smacv2.py',
    *sorted(
        p.relative_to(_PKG).as_posix()
        for p in (_PKG / 'env' / 'opponents' / 'scripted').rglob('*.py')
    ),
)
FUNCTIONS = tuple(
    f'env/core.py:VecEnv.{m}'
    for m in (
        '__init__',
        '_generated_scenario',
        'rng_stream',
        'rng_streams',
        'reset',
        'step',
        '_context',
        'get_avail_actions',
        '_obs_rng',
        'get_obs',
        'get_state',
        'query_abilities',
    )
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def _parse_version(version: str) -> tuple[int, ...]:
    match = re.match(r'^(\d+(?:\.\d+)*)', version.strip())
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split('.'))


def _version_in_range(version: str, min_inclusive: str, max_exclusive: str) -> bool:
    got = _parse_version(version)
    lo = _parse_version(min_inclusive)
    hi = _parse_version(max_exclusive)
    if not got or not lo or not hi:
        return False
    width = max(len(got), len(lo), len(hi))
    g = got + (0,) * (width - len(got))
    low = lo + (0,) * (width - len(lo))
    high = hi + (0,) * (width - len(hi))
    return low <= g < high


def _function_source(entry: str) -> str:
    path, qual = entry.split(':')
    mod = importlib.import_module('unismac.' + path[: -len('.py')].replace('/', '.'))
    obj: Any = mod
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return inspect.getsource(obj)


def _semantics_version() -> int | None:
    try:
        from . import services as rust_services
    except ImportError:
        return None
    return getattr(rust_services._env, 'SEMANTICS', None)


def current() -> dict[str, Any]:
    """Compute the manifest dictionary for the current sources and runtime."""
    return {
        'numpy_min': NUMPY_MIN,
        'numpy_max_exclusive': NUMPY_MAX_EXCLUSIVE,
        'zgame_env_semantics': _semantics_version(),
        'files': {p: _sha((_PKG / p).read_bytes()) for p in SOURCES},
        'functions': {f: _sha(_function_source(f).encode('utf-8')) for f in FUNCTIONS},
    }


def load() -> dict[str, Any] | None:
    """Load the checked-in manifest, or return None if missing."""
    if not MANIFEST.exists():
        return None
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


_HOW = 'run `python -m unismac.env.accel.manifest --write` after verifying parity tests'


def diff(want: dict[str, Any] | None, got: dict[str, Any]) -> list[str]:
    """Return human-readable mismatch descriptions between `want` (manifest) and `got`."""
    if want is None:
        return [f'Missing manifest {MANIFEST.name} ({_HOW})']
    out = []
    ver = str(got.get('numpy', np.__version__))
    lo = want.get('numpy_min')
    hi = want.get('numpy_max_exclusive')
    if not (isinstance(lo, str) and isinstance(hi, str) and _version_in_range(ver, lo, hi)):
        out.append(
            f'numpy version {ver} is outside manifest range [{lo}, {hi}) '
            f'(Generator distributions are verified for [{NUMPY_MIN}, {NUMPY_MAX_EXCLUSIVE}))'
        )
    if want.get('zgame_env_semantics') != got['zgame_env_semantics']:
        out.append(
            f'zgame env semantics version {got["zgame_env_semantics"]} does not match manifest '
            f'{want.get("zgame_env_semantics")} (install a compatible `zfoundry-zgame` build)'
        )
    for kind in ('files', 'functions'):
        w, g = want.get(kind, {}), got[kind]
        changed = sorted(k for k in w.keys() | g.keys() if w.get(k) != g.get(k))
        if changed:
            out.append(
                f'Semantic-layer sources differ from manifest: {", ".join(changed)} ({_HOW})'
            )
    return out


def _print_cli_diff(want: dict[str, Any] | None, got: dict[str, Any]) -> None:
    """Print categorized, multi-line mismatch diagnostics for interactive CLI usage."""
    if want is None:
        print(f'[missing] {MANIFEST.name} not found ({_HOW})')
        return
    ver = str(got.get('numpy', np.__version__))
    lo, hi = want.get('numpy_min'), want.get('numpy_max_exclusive')
    if not (isinstance(lo, str) and isinstance(hi, str) and _version_in_range(ver, lo, hi)):
        print(
            f'[numpy] version {ver} is outside manifest range [{lo}, {hi}) '
            f'(verified for [{NUMPY_MIN}, {NUMPY_MAX_EXCLUSIVE}))'
        )
    if want.get('zgame_env_semantics') != got['zgame_env_semantics']:
        print(
            f'[semantics] zgame.env.SEMANTICS={got["zgame_env_semantics"]!r} != '
            f'manifest {want.get("zgame_env_semantics")!r} '
            '(install a compatible `zfoundry-zgame` build)'
        )
    for kind, label in (('files', 'locked source files'), ('functions', 'locked VecEnv methods')):
        w, g = want.get(kind, {}), got[kind]
        changed = sorted(k for k in w.keys() | g.keys() if w.get(k) != g.get(k))
        if not changed:
            continue
        if len(changed) <= 5:
            print(f'[sources] {label} differ ({len(changed)}): {", ".join(changed)}')
        else:
            print(f'[sources] {label} differ ({len(changed)}):')
            for item in changed:
                print(f'  - {item}')
    print(f'Next step: {_HOW}')


@cache
def mismatches() -> tuple[str, ...]:
    """Return cached mismatches between the current runtime/sources and the manifest."""
    return tuple(diff(load(), current()))


def write() -> None:
    MANIFEST.write_text(
        json.dumps(current(), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def _run_comparison(jobs: int | None = None) -> int:
    tests = [
        _PKG.parent / 'tests' / 'env' / f
        for f in ('test_env_accel.py', 'test_env_scripted_rust.py')
    ]
    for t in tests:
        if not t.exists():
            print(f'Parity test file not found: {t}', file=sys.stderr)
            return 2
    env = dict(os.environ, OMP_NUM_THREADS='1')
    cmd = [sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider']
    if jobs is not None:
        if jobs < 1:
            print(f'--jobs must be >= 1, got {jobs}', file=sys.stderr)
            return 2
        if jobs > 1:
            if importlib.util.find_spec('xdist') is not None:
                cmd.extend(['-n', str(jobs)])
            else:
                print(
                    'Note: pytest-xdist is not installed; running parity tests serially.',
                    file=sys.stderr,
                )
    cmd.extend([*map(str, tests), '-m', 'not sc2', '-k', 'rust and not manifest'])
    return subprocess.run(cmd, env=env).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m unismac.env.accel.manifest',
        description='Verify or regenerate unismac/env/accel/rust_manifest.json.',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Run accel=rust bitwise parity tests and update rust_manifest.json on success.',
    )
    parser.add_argument(
        '--list-locked',
        action='store_true',
        help='List all source files and VecEnv methods tracked by rust_manifest.json.',
    )
    parser.add_argument(
        '-j',
        '--jobs',
        type=int,
        default=None,
        help='Parallel pytest workers (via pytest-xdist) when running --write parity tests.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if ns.list_locked:
        print(f'Locked source files ({len(SOURCES)}):')
        for rel in SOURCES:
            print(f'  unismac/{rel}')
        print(f'Locked VecEnv methods ({len(FUNCTIONS)}):')
        for fn in FUNCTIONS:
            print(f'  unismac/{fn}')
        return 0
    if ns.jobs is not None and not ns.write:
        print('--jobs is only used together with --write', file=sys.stderr)
        return 2
    want = load()
    got = current()
    problems = diff(want, got)
    if not ns.write:
        if problems:
            _print_cli_diff(want, got)
            return 1
        return 0
    if not problems:
        print(f'{MANIFEST.name} is already up to date')
        return 0
    print('Running accel=rust bitwise parity tests before updating manifest...')
    code = _run_comparison(jobs=ns.jobs)
    if code != 0:
        print('Parity tests failed; manifest not updated', file=sys.stderr)
        return code
    write()
    print(f'Wrote {MANIFEST}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
