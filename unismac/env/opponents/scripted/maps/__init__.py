"""Per-scenario scripts (one module per upstream `s_<map>/` directory), registered on import."""

from __future__ import annotations

import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{_m.name}')
