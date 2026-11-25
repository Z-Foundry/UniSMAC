"""Package dependency boundary and unidirectional layering tests for `unismac`.

Ensures:
1. `unismac` (including `unismac.adapters`) never imports deep-learning or plotting libraries
   (`torch`, `matplotlib`, `tensorflow`, `jax`) statically or at runtime.
2. Core `unismac` modules outside `unismac/adapters/` never import `unismac.adapters`,
   `gymnasium`, or `pettingzoo` (strict unidirectional `adapters -> env` layering).
3. Importing `unismac`, `unismac.env`, `unismac.profiles`, `unismac.adapters`, and
   `unismac.adapters.pymarl` never eagerly loads `gymnasium` or `pettingzoo`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import unismac

PKG = Path(unismac.__file__).resolve().parent
FORBIDDEN = frozenset({'torch', 'matplotlib', 'tensorflow', 'jax'})
ADAPTER_DEPS = frozenset({'gymnasium', 'pettingzoo'})


def _imports(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield `(lineno, module_name)` for static and string-literal dynamic imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module
            elif node.level > 0:
                if node.module:
                    yield node.lineno, f'{"." * node.level}{node.module}'
                for alias in node.names:
                    yield node.lineno, f'{"." * node.level}{alias.name}'
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            arg = node.args[0]
            if name in ('import_module', '__import__') and isinstance(arg, ast.Constant):
                if isinstance(arg.value, str):
                    yield node.lineno, arg.value


def _violations(path: Path, blocked: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return [
        f'{path.relative_to(PKG.parent).as_posix()}:{line} {name}'
        for line, name in _imports(tree)
        if not name.startswith('.') and name.split('.')[0] in blocked
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
    names = sorted(name for _, name in _imports(ast.parse(src)) if not name.startswith('.'))
    want = ['importlib', 'jax.numpy', 'matplotlib.pyplot', 'tensorflow', 'torch']
    assert names == want


def test_unismac_does_not_import_training_or_plotting() -> None:
    files = sorted(PKG.rglob('*.py'))
    assert len(files) > 50 and (PKG / 'env' / 'core.py') in files
    bad = [v for path in files for v in _violations(path, FORBIDDEN)]
    assert not bad, '\n'.join(bad)


def test_core_does_not_import_adapters_or_optional_rl_libs() -> None:
    """Core modules outside `unismac/adapters/` stay strictly decoupled from adapters and extras."""
    adapters_dir = PKG / 'adapters'
    bad: list[str] = []
    for path in sorted(PKG.rglob('*.py')):
        if adapters_dir in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        rel = path.relative_to(PKG.parent).as_posix()
        for line, name in _imports(tree):
            top = name.lstrip('.').split('.')[0]
            if (
                top in ADAPTER_DEPS
                or name.startswith('unismac.adapters')
                or (name.startswith('.') and 'adapters' in name.lstrip('.').split('.'))
            ):
                bad.append(f'{rel}:{line} {name}')
    assert not bad, '\n'.join(bad)


def test_zero_dep_adapter_entrypoints_do_not_import_extras() -> None:
    """`unismac/adapters/__init__.py`, `_common.py`, and `pymarl.py` never import extras."""
    for name in ('__init__.py', '_common.py', 'pymarl.py'):
        path = PKG / 'adapters' / name
        assert path.is_file(), path
        bad = _violations(path, ADAPTER_DEPS)
        assert not bad, '\n'.join(bad)


def test_runtime_imports_stay_inside() -> None:
    """Importing core and PyMARL entry points must not load ML, plotting, or optional RL libs."""
    blocked = sorted(FORBIDDEN | ADAPTER_DEPS)
    code = (
        'import sys\n'
        'before = set(sys.modules)\n'
        'import unismac, unismac.env, unismac.profiles, unismac.adapters, unismac.adapters.pymarl\n'
        'new = set(sys.modules) - before\n'
        f'bad = sorted(m for m in new if m.split(".")[0] in {blocked!r})\n'
        'assert not bad, bad\n'
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
