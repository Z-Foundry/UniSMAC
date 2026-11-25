"""Scenario registry (static benchmark metadata and map IDs, no SC2 required) and spawn tables."""

from .registry import BENCHMARKS, Benchmark, ScenarioSpec, all_specs, get, names
from .spawns import SPAWNS_FILE, SpawnRecord, load_spawns, spawn_record

__all__ = [
    'BENCHMARKS',
    'SPAWNS_FILE',
    'Benchmark',
    'ScenarioSpec',
    'SpawnRecord',
    'all_specs',
    'get',
    'load_spawns',
    'names',
    'spawn_record',
]
