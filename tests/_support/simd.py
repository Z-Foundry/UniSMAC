"""Subprocess runner for AVX-512-disabled bitwise parity verification."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def run_without_avx512(
    script_file: str,
    body: str,
    *,
    verify_disabled: bool = False,
    timeout: int = 1800,
) -> None:
    """Re-execute ``script_file`` in a subprocess with NumPy AVX-512 dispatch disabled.

    Appends the repository root to ``PYTHONPATH`` (after existing ``sys.path`` entries so
    installed wheel tests still import ``unismac`` from site-packages while resolving
    ``tests._support`` from the checkout).
    """
    entries: list[str] = []
    for path in (*(p for p in sys.path if p), str(_REPO_ROOT)):
        if path not in entries:
            entries.append(path)
    env = dict(
        os.environ,
        NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL',
        OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(entries),
    )
    preamble = 'import runpy\n'
    if verify_disabled:
        preamble += (
            'import importlib\n'
            'umath = importlib.import_module("numpy._core._multiarray_umath")\n'
            'assert not umath.__cpu_features__.get("AVX512_SKX")\n'
        )
    code = preamble + f'm = runpy.run_path({script_file!r})\n' + body
    result = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr[-4000:]
