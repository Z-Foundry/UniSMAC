"""SMAC-Hard scripted opponents, ported from upstream `smac_hard/env/scripts` (standalone).

* `api`: unit view `ScriptUnit`, `RawAction`, and ported script helper utilities;
* `base`: base class `Script`, registry `SCRIPTS`, and upstream `SCRIPT_DICT` pools `POOLS`;
* `common`: `base` / `atkw` / `atkn` / `noop`; `maps`: per-scenario scripts;
* `opponent`: `ScriptedOpponent` environment opponent wrapper.
"""

from __future__ import annotations

from . import common, maps  # noqa: F401  Register built-in scripts
from .base import POOLS, SCRIPTS, Script, register
from .opponent import ScriptedOpponent, StreamChoice, script_pool

__all__ = [
    'POOLS',
    'SCRIPTS',
    'Script',
    'ScriptedOpponent',
    'StreamChoice',
    'register',
    'script_pool',
]
