"""Tests verifying helpful ImportError messages when optional adapter extras are absent."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ('module', 'blocked', 'extra'),
    [
        ('unismac.adapters.gymnasium', 'gymnasium', 'gymnasium'),
        ('unismac.adapters.pettingzoo', 'pettingzoo', 'pettingzoo'),
    ],
)
def test_missing_optional_extra_error_message(module: str, blocked: str, extra: str) -> None:
    """Importing an optional adapter without its extra raises an actionable `ImportError`."""
    code = (
        'import importlib, sys\n'
        f'sys.modules[{blocked!r}] = None\n'
        'try:\n'
        f'    importlib.import_module({module!r})\n'
        'except ImportError as exc:\n'
        '    msg = str(exc)\n'
        f'    assert {extra!r} in msg and "pip install" in msg, msg\n'
        'else:\n'
        '    raise AssertionError("expected ImportError")\n'
    )
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]
