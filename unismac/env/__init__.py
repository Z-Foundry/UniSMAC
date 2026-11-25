"""Environment layer: `make_env`, batched `VecEnv`, and SMAC-compatible `UnismacEnv`.

Components: `units`, `catalog`, `visibility`, `actions`, `features`, `reward`,
`termination`, `opponents`, `terrain`, and `config` (`EnvConfig`). Built-in
presets are defined in `unismac.presets`.

Example::

    env = make_env('smac-portable', '8m', backend='zgame', batch_size=8)
    env = make_env('smac-portable', '3m', backend='pyref', batch_size=4)

`accel` (default `'auto'`) selects the environment-layer acceleration tier:
`'python'` (pure Python), `'services'` (Rust system kernels + numeric services),
or `'rust'` (full Rust environment layer for classic recipes). All tiers produce
bitwise-identical results; see `unismac.env.accel`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from ..contract import Backend, CapabilityError
from . import accel as accel_mod
from .config import BackendConfig, EnvConfig, apply_overrides, resolve_config
from .core import UnismacEnv, VecEnv, resolve_map_ids
from .smac_bot import (
    MULTI_LEVEL_SMAC_MAPS,
    SMAC_TRIGGER_SPECS,
    NativeSMACOpponent,
    native_smac_overrides,
)

if TYPE_CHECKING:
    from ..backends.pyref import PyrefConfig

__all__ = [
    'MULTI_LEVEL_SMAC_MAPS',
    'SMAC_TRIGGER_SPECS',
    'EnvConfig',
    'NativeSMACOpponent',
    'UnismacEnv',
    'VecEnv',
    'apply_overrides',
    'build_config',
    'make_env',
    'make_vec_env',
    'native_smac_overrides',
    'resolve_config',
]


def build_config(
    preset: str, scenario: str, overrides: Mapping[str, Any] | None = None
) -> tuple[EnvConfig, Any]:
    """Return merged config and scenario spec.

    Precedence: preset < scenario defaults < preset scenario overrides < user overrides.
    """
    from .. import presets, scenarios

    p = presets.get(preset)
    spec = scenarios.get(p.benchmark, scenario)
    config = resolve_config(p.config, spec, overrides, p.scenario_overrides(spec))
    if spec.procedural and config.capability.team_gen is None:
        raise ValueError(
            f'{preset}/{scenario} is a procedural scenario and requires capability_config.team_gen'
        )
    return config, spec


def _backend_name(backend: str | Backend) -> str:
    return backend if isinstance(backend, str) else backend.capabilities.name


def _pyref_config(bc: BackendConfig, dual: bool = False) -> PyrefConfig:
    """Build `PyrefConfig` from `backend.pyref_*` and `backend.pyref_options` (pyref and zgame).

    In two-sided mode (`dual`), `passive_from_map` is expanded to symmetric passive for both
    sides via `dual.symmetric_options`.
    """
    from ..backends.pyref import PyrefConfig
    from .dual import symmetric_options

    extra = symmetric_options(bc.pyref_options) if dual else dict(bc.pyref_options)
    unknown = set(extra) - {f.name for f in dataclasses.fields(PyrefConfig)}
    if unknown:
        raise ValueError(f'backend.pyref_options has unknown PyrefConfig fields: {sorted(unknown)}')
    return PyrefConfig(
        abilities=bc.pyref_abilities,
        spawn_energy=bc.pyref_spawn_energy,
        heal_autocast=bc.pyref_heal_autocast,
        commands=bc.pyref_commands,
        command_delay=bc.pyref_command_delay,
        move_lag=bc.pyref_move_lag,
        **extra,
    )


def make_vec_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['sc2', 'pyref', 'zgame'] | Backend = 'sc2',
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
) -> VecEnv:
    """Always return a `VecEnv`. `backend` may also be a pre-built instance with matching
    `batch_size`.

    `seed` is a batch-wide seed (env `b` derives its RNG stream from `(seed, b)` and backend seed
    `seed + b`); `seeds` provides per-env seeds of length `batch_size` (mutually exclusive with
    `seed`), so env `b` is bitwise-identical to a `batch_size=1` env with `seed=seeds[b]` (see
    `unismac.env.core`). `zgame_threads` sets the zgame kernel thread pool size (per-env
    parallelism, result invariant to thread count), valid only with `backend='zgame'` (defaults
    to `ZgameBackend`'s default of 1).

    `accel` selects the environment-layer tier: `'python'`, `'services'`, `'rust'` (for classic
    recipes: `smac-portable`, `smacv2-portable`, `smac-hard-portable` plus scenario overrides,
    with user overrides limited to `backend.*`; raises ValueError detailing blockers otherwise),
    or `'auto'` (`'rust'` when eligible, else `'services'` if zgame is installed, else `'python'`).
    All tiers are bitwise-identical and backend-agnostic (`sc2`, `pyref`, `zgame`); see
    `VecEnv.accel` for the resolved tier.
    """
    from ..assets import load_map_info
    from ..scenarios import spawn_record

    accel_mod.resolve('services' if accel == 'rust' else accel)  # Validate tier name and extension

    if zgame_threads is not None and backend != 'zgame':
        raise ValueError(
            f"zgame_threads is only valid for backend='zgame', got {_backend_name(backend)!r}"
        )
    if seed is not None and seeds is not None:
        raise ValueError(
            'Specify at most one of seed or seeds (seed is batch-wide; seeds is per-env)'
        )
    if seeds is not None and len(seeds) != batch_size:
        raise ValueError(f'seeds has {len(seeds)} items, expected batch_size={batch_size}')
    config, spec = build_config(preset, scenario, overrides)
    tier = accel_mod.resolve(accel, config, spec)
    name = _backend_name(backend)
    map_id = resolve_map_ids(config, spec, name)
    try:
        info = load_map_info(map_id)
    except FileNotFoundError:
        raise CapabilityError(
            f'Missing map asset {map_id} ({preset}/{scenario} is unavailable on backend {name})'
        ) from None
    explicit = config.backend.spawn_mode == 'explicit' and not spec.procedural
    dual = config.opponent.kind == 'external'
    record = None
    if explicit:
        try:
            record = spawn_record(
                spec.name, 'smac-hard' if spec.benchmark == 'smac-hard' else 'smac'
            )
        except KeyError as exc:
            raise CapabilityError(f'{preset}/{scenario} has no spawn record: {exc}') from None
    if isinstance(backend, str):
        if backend == 'sc2':
            if batch_size != 1:
                raise CapabilityError('SC2 backend only supports batch_size=1')
            from ..backends.sc2 import Sc2Backend

            inst: Backend = Sc2Backend(slot_order=config.backend.sc2_slot_order)
        elif backend == 'pyref':
            from ..assets import load_static_data
            from ..backends.pyref import PyrefBackend

            pyref = _pyref_config(config.backend, dual)
            inst = PyrefBackend(load_static_data(), {map_id: info}, batch_size, pyref)
        elif backend == 'zgame':
            # Rust backend shares PyrefConfig fields; unsupported options raise CapabilityError.
            from ..assets import load_static_data
            from ..backends.zgame import ZgameBackend

            pyref = _pyref_config(config.backend, dual)
            threads = 1 if zgame_threads is None else int(zgame_threads)  # 1 = ZgameBackend default
            inst = ZgameBackend(
                load_static_data(), {map_id: info}, batch_size, pyref, threads=threads
            )
        else:
            raise ValueError(f'Unknown backend {backend!r}; expected sc2 / pyref / zgame')
    else:
        inst = backend
        if inst.batch_size != batch_size:
            raise ValueError(f'Backend batch_size={inst.batch_size} does not match {batch_size}')
    return VecEnv(config, spec, inst, info, seed=seed, seeds=seeds, spawn_record=record, accel=tier)


def make_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['sc2', 'pyref', 'zgame'] | Backend = 'sc2',
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
) -> UnismacEnv | VecEnv:
    """Create an environment: `UnismacEnv` (SMAC API) when `batch_size=1`, else `VecEnv`.

    `overrides` keys may be dotted paths, upstream flat parameter names, or group names (see
    `unismac.env.config`). See `make_vec_env` for `seeds`, `zgame_threads`, and `accel`.
    """
    vec = make_vec_env(
        preset,
        scenario,
        backend=backend,
        batch_size=batch_size,
        seed=seed,
        seeds=seeds,
        overrides=overrides,
        zgame_threads=zgame_threads,
        accel=accel,
    )
    return UnismacEnv(vec) if batch_size == 1 else vec
