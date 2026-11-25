"""Scenario registry for SMAC, SMACv2, and SMAC-Hard.

Scenario metadata is stored statically in `data.py` so importing `unismac`
does not require StarCraft II or upstream benchmark repositories at runtime.

The primary key is `(benchmark, name)` because multiple benchmarks define
scenarios with identical names (e.g. `3m`) that differ in parameters or map IDs.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from functools import cache
from typing import Literal

__all__ = ['BENCHMARKS', 'Benchmark', 'ScenarioSpec', 'all_specs', 'get', 'names']

Benchmark = Literal['smac', 'smacv2', 'smac-hard']
BENCHMARKS: tuple[Benchmark, ...] = ('smac', 'smacv2', 'smac-hard')


@dataclass(frozen=True)
class ScenarioSpec:
    """Registered scenario parameters matching upstream `map_param_registry`.

    * `n_agents` / `n_enemies` / `limit` / `unit_type_bits` / `map_type`: copied verbatim.
    * `a_race` / `b_race`: upstream single-letter race codes (`'T'`, `'Z'`, `'P'`).
    * `map_id`: map path relative to `$SC2PATH/Maps` without `.SC2Map`. SMAC / SMACv2 use
      `SMAC_Maps/<filename>` (SMACv2 takes registry `map_name`); SMAC-Hard uses
      `new_maps/<scenario>`.
    * `procedural`: unit types and spawns are generated procedurally by SMACv2 capability config
      (`10gen_*`, `map_type` ending in `_gen`); `n_agents` / `n_enemies` are registry upper
      bounds, with actual counts determined by capability config.
    """

    benchmark: Benchmark
    name: str
    n_agents: int
    n_enemies: int
    limit: int
    a_race: str
    b_race: str
    unit_type_bits: int
    map_type: str
    map_id: str
    procedural: bool


@cache
def _index() -> dict[Benchmark, dict[str, ScenarioSpec]]:
    """Index grouped by benchmark preserving upstream registration order (built on first use)."""
    from .data import SCENARIOS

    index: dict[Benchmark, dict[str, ScenarioSpec]] = {b: {} for b in BENCHMARKS}
    for spec in SCENARIOS:
        group = index[spec.benchmark]
        if spec.name in group:
            raise RuntimeError(f'Duplicate scenario registration: {spec.benchmark}/{spec.name}')
        group[spec.name] = spec
    return index


def _group(benchmark: str) -> dict[str, ScenarioSpec]:
    index = _index()
    if benchmark not in index:
        matches = difflib.get_close_matches(benchmark, list(BENCHMARKS), n=2, cutoff=0.5)
        hint = f'; did you mean {", ".join(repr(m) for m in matches)}?' if matches else ''
        raise KeyError(f'Unknown benchmark {benchmark!r}{hint} Available: {", ".join(BENCHMARKS)}')
    return index[benchmark]  # type: ignore[index]


def get(benchmark: str, name: str) -> ScenarioSpec:
    """Return scenario for `(benchmark, name)`, or raise `KeyError` if unregistered."""
    group = _group(benchmark)
    try:
        return group[name]
    except KeyError:
        matches = difflib.get_close_matches(name, list(group), n=3, cutoff=0.5)
        hint = f'; did you mean {", ".join(repr(m) for m in matches)}?' if matches else ''
        raise KeyError(
            f'No scenario {name!r} registered for benchmark {benchmark!r}{hint}'
        ) from None


def names(benchmark: str) -> list[str]:
    """Return all scenario names for `benchmark` in upstream registration order."""
    return list(_group(benchmark))


def all_specs() -> tuple[ScenarioSpec, ...]:
    """Return all scenarios ordered by `BENCHMARKS`, then upstream registration order."""
    return tuple(spec for b in BENCHMARKS for spec in _index()[b].values())
