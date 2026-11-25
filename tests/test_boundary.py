"""Package dependency boundary tests for `unismac`.

Ensures `unismac` remains self-contained and never imports training frameworks or
plotting libraries (`torch`, `matplotlib`, etc.) statically or at runtime.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / 'unismac'
FORBIDDEN = frozenset({'torch', 'matplotlib', 'tensorflow', 'jax'})


def _imports(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield `(lineno, absolute_module_name)` for static and string-literal dynamic imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            arg = node.args[0]
            if name in ('import_module', '__import__') and isinstance(arg, ast.Constant):
                if isinstance(arg.value, str):
                    yield node.lineno, arg.value


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return [
        f'{path.relative_to(PKG.parent).as_posix()}:{line} {name}'
        for line, name in _imports(tree)
        if name.split('.')[0] in FORBIDDEN
    ]


def test_scanner_sees_all_import_forms() -> None:
    src = (
        'import jax.numpy\n'
        'from tensorflow import keras\n'
        'from . import env\n'
        'def f():\n'
        '    import importlib\n'
        "    importlib.import_module('torch')\n"
        "    __import__('matplotlib.pyplot')\n"
    )
    names = sorted(name for _, name in _imports(ast.parse(src)))
    want = ['importlib', 'jax.numpy', 'matplotlib.pyplot', 'tensorflow', 'torch']
    assert names == want


def test_unismac_does_not_import_training_or_plotting() -> None:
    files = sorted(PKG.rglob('*.py'))
    assert len(files) > 50 and (PKG / 'env' / 'core.py') in files
    bad = [v for path in files for v in _violations(path)]
    assert not bad, '\n'.join(bad)


def test_runtime_imports_stay_inside() -> None:
    """Importing public `unismac` entry points must not load heavy ML/plotting packages."""
    code = (
        'import sys\n'
        'before = set(sys.modules)\n'
        'import unismac, unismac.env, unismac.profiles\n'
        'new = set(sys.modules) - before\n'
        f'bad = sorted(m for m in new if m.split(".")[0] in {sorted(FORBIDDEN)})\n'
        'assert not bad, bad\n'
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
