"""Preset registry."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..env.config import EnvConfig
from .smac import SMAC, SMAC_PORTABLE
from .smac_hard import SMAC_HARD, SMAC_HARD_PORTABLE
from .smacv2 import SMACV2, SMACV2_PORTABLE, smacv2_overrides, smacv2_portable_overrides

__all__ = ['PRESETS', 'Preset', 'get', 'names']


def _no_overrides(spec: Any) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class Preset:
    name: str
    benchmark: str  # benchmark key in scenarios registry
    config: EnvConfig
    description: str
    # Per-scenario overrides (e.g., SMACv2 10gen_* capability_config);
    # precedence sits between scenario defaults and user overrides.
    scenario_overrides: Callable[[Any], Mapping[str, Any]] = _no_overrides


PRESETS: dict[str, Preset] = {
    'smac': Preset('smac', 'smac', SMAC, 'Reproduces upstream SMAC (SC2 only)'),
    'smac-portable': Preset(
        'smac-portable', 'smac', SMAC_PORTABLE, 'Cross-backend SMAC preset (zgame / pyref / SC2)'
    ),
    'smacv2': Preset(
        'smacv2', 'smacv2', SMACV2, 'Reproduces upstream SMACv2 (SC2 only)', smacv2_overrides
    ),
    'smacv2-portable': Preset(
        'smacv2-portable',
        'smacv2',
        SMACV2_PORTABLE,
        'Cross-backend SMACv2 preset (zgame / pyref / SC2)',
        smacv2_portable_overrides,
    ),
    'smac-hard': Preset(
        'smac-hard',
        'smac-hard',
        SMAC_HARD,
        'Reproduces upstream SMAC-Hard (SC2 only, scripted opponents)',
    ),
    'smac-hard-portable': Preset(
        'smac-hard-portable',
        'smac-hard',
        SMAC_HARD_PORTABLE,
        'Cross-backend SMAC-Hard preset (zgame / pyref / SC2)',
    ),
}


def get(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        matches = difflib.get_close_matches(name, list(PRESETS), n=3, cutoff=0.5)
        hint = f'; did you mean {", ".join(repr(m) for m in matches)}?' if matches else ''
        raise KeyError(f'Unknown preset {name!r}{hint} Available: {", ".join(PRESETS)}') from None


def names() -> list[str]:
    return list(PRESETS)
