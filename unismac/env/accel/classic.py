"""Classic recipe detection for `accel='rust'` eligibility.

A classic recipe is one of `CLASSIC_PRESETS` (`smac-portable`, `smacv2-portable`,
`smac-hard-portable`) plus its scenario overrides; user overrides may freely
customize the `backend` group (e.g. profile options) and `ALLOWED_KEYS`
(`termination.continuing_episode`).

For `smac-hard-portable`, the built-in script registry and script pools must be
unmodified (`registry_changes`). In two-sided mode (`opponent.kind='external'`),
the `opponent` config group is ignored and backend symmetry is checked by
`unismac.env.dual.check_backend`.

Options in `unsupported` (such as `opponent.kind='native_smac'` or conic FOV)
automatically fall back to `accel='services'` under `accel='auto'`.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any

from ..config import EnvConfig, resolve_config
from ..opponents.scripted import POOLS, SCRIPTS

__all__ = [
    'ALLOWED_GROUPS',
    'ALLOWED_KEYS',
    'BUILTIN_POOLS',
    'BUILTIN_SCRIPTS',
    'CLASSIC_PRESETS',
    'recipe_diff',
    'registry_changes',
    'rust_blockers',
    'unsupported',
]

CLASSIC_PRESETS = ('smac-portable', 'smacv2-portable', 'smac-hard-portable')
# Config groups users are allowed to customize
ALLOWED_GROUPS = ('backend',)
# Individual keys outside ALLOWED_GROUPS that users may customize (bitwise identical in Rust)
ALLOWED_KEYS = ('termination.continuing_episode',)
# Built-in script registry and pools captured at import time (before user code runs)
BUILTIN_SCRIPTS = MappingProxyType(dict(SCRIPTS))
BUILTIN_POOLS = MappingProxyType(dict(POOLS))


def _flat(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        path = f'{prefix}{f.name}'
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            _flat(v, path + '.', out)
        else:
            out[path] = v


def flat(config: EnvConfig, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Flatten config into `{dotted_path: value}`, excluding `ALLOWED_GROUPS`, `ALLOWED_KEYS`,
    and groups listed in `skip`."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        if f.name in ALLOWED_GROUPS or f.name in skip:
            continue
        v = getattr(config, f.name)
        _flat(v, f'{f.name}.', out)
    for k in ALLOWED_KEYS:
        out.pop(k, None)
    return out


def recipe_diff(config: EnvConfig, spec: Any) -> tuple[str | None, list[str]]:
    """Return `(closest_classic_preset, differing_keys)`. Preset is None if no classic preset
    matches the scenario's benchmark."""
    from ... import presets

    best: tuple[str | None, list[str]] = (None, [])
    skip = ('opponent',) if config.opponent.kind == 'external' else ()
    for name in CLASSIC_PRESETS:
        p = presets.get(name)
        if p.benchmark != getattr(spec, 'benchmark', None):
            continue
        ref = flat(resolve_config(p.config, spec, None, p.scenario_overrides(spec)), skip)
        got = flat(config, skip)
        diff = sorted(k for k in ref.keys() | got.keys() if ref.get(k, ...) != got.get(k, ...))
        if not diff:
            return name, []
        if best[0] is None or len(diff) < len(best[1]):
            best = (name, diff)
    return best


def unsupported(config: EnvConfig) -> list[str]:
    """Return options not supported by the Rust environment layer."""
    out = []
    if config.visibility.model == 'cone':
        out.append(
            'visibility.model=cone '
            '(conic FOV uses float64 np.arctan, matching libm only without AVX-512)'
        )
    if config.actions.ability:
        out.append('actions.ability (SMAC-Hard ability segment)')
    if config.opponent.kind not in ('opening_orders', 'scripted', 'external'):
        out.append(
            f'opponent.kind={config.opponent.kind} '
            '(only opening_orders, scripted, external supported)'
        )
    if config.capability.health is not None:
        out.append('capability.health')
    return out


def registry_changes(config: EnvConfig, spec: Any) -> list[str]:
    """Return modifications to the scripted opponent registry/pools relative to built-ins.

    Returns an empty list when `opponent.kind != 'scripted'`. Uses `opponent.scripts` if set,
    otherwise `POOLS[scenario]`, verifying every name maps to its built-in script class.
    """
    if config.opponent.kind != 'scripted':
        return []
    out = []
    name = str(getattr(spec, 'name', ''))
    if config.opponent.scripts is None and POOLS.get(name) != BUILTIN_POOLS.get(name):
        out.append(f'script pool POOLS[{name!r}] differs from built-in')
    pool = config.opponent.scripts
    if pool is None:
        pool = POOLS.get(name) or ()
    changed = sorted({n for n in pool if SCRIPTS.get(n) is not BUILTIN_SCRIPTS.get(n)})
    if changed:
        out.append(f'script registry entries {changed} are not built-in script classes')
    return out


def rust_blockers(config: EnvConfig, spec: Any) -> list[str]:
    """Return reasons `config`/`spec` is ineligible for `accel='rust'`; empty if eligible."""
    out = unsupported(config) + registry_changes(config, spec)
    name, diff = recipe_diff(config, spec)
    where = f'{getattr(spec, "benchmark", "?")}/{getattr(spec, "name", "?")}'
    if name is None:
        out.append(f'scenario {where} does not belong to classic presets {CLASSIC_PRESETS}')
    elif diff:
        shown = ', '.join(diff[:6]) + ('...' if len(diff) > 6 else '')
        out.append(
            f'config differs from classic recipe {name} ({where}): {shown} '
            '(only backend.*, termination.continuing_episode, and two-sided '
            'opponent.kind=external may change)'
        )
    return out
