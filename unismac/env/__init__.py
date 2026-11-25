"""Environment layer: `make_env`, batched `VecEnv`, and SMAC-compatible `UnismacEnv`.

Components: `units`, `catalog`, `visibility`, `actions`, `features`, `reward`,
`termination`, `opponents`, `terrain`, and `config` (`EnvConfig`). Built-in
presets are defined in `unismac.presets`.

Example::

    with make_vec_env('smac-portable', '8m', profile='sc2-native', batch_size=8) as env:
        obs, state = env.reset()

    with make_vec_env('smac-portable', '2s3z', dual=True, batch_size=4) as env:
        obs, state = env.reset()

`accel` (default `'auto'`) selects the environment-layer acceleration tier:
`'python'` (pure Python), `'services'` (Rust system kernels + numeric services),
or `'rust'` (full Rust environment layer for classic recipes). All tiers produce
bitwise-identical results; see `unismac.env.accel`.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from ..contract import Backend, CapabilityError
from . import accel as accel_mod
from .config import BackendConfig, EnvConfig, apply_overrides, resolve_config
from .core import VecEnv, resolve_map_ids
from .opponents import (
    MULTI_LEVEL_SMAC_MAPS,
    SMAC_TRIGGER_SPECS,
    NativeSMACOpponent,
    native_smac_overrides,
)
from .single import UnismacEnv

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

_ASYMMETRIC_PROFILES = frozenset({'sc2-native', 'sc2-explore', 'rghf-nat-cbt', 'rghf-nat'})
_BACKEND_NAMES = ('auto', 'sc2', 'pyref', 'zgame')


def _suggest(value: str, candidates: Sequence[str]) -> str:
    matches = difflib.get_close_matches(value, list(candidates), n=3, cutoff=0.5)
    if not matches:
        return ''
    return f'; did you mean {", ".join(repr(m) for m in matches)}?'


def _zgame_backend_available() -> bool:
    from ..backends.zgame.backend import _module

    try:
        _module()
    except ImportError:
        return False
    return True


def _resolve_auto_backend(
    preset: str, prof_backend: str | None
) -> Literal['sc2', 'pyref', 'zgame']:
    if prof_backend == 'sc2':
        return 'sc2'
    if prof_backend == 'pyref':
        return 'zgame' if _zgame_backend_available() else 'pyref'
    from .. import presets

    p = presets.get(preset)
    if p.config.backend.spawn_mode == 'explicit':
        return 'zgame' if _zgame_backend_available() else 'pyref'
    return 'sc2'


def _extract_opponent_kind(overrides: Mapping[str, Any]) -> Any:
    if 'opponent.kind' in overrides:
        return overrides['opponent.kind']
    opp = overrides.get('opponent')
    if isinstance(opp, Mapping) and 'kind' in opp:
        return opp['kind']
    return None


def _overlay_overrides(base: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Overlay user overrides on top of profile/dual overrides, deep-merging pyref_options."""
    from ..profiles import merge_overrides

    if not overrides:
        return base
    merged = merge_overrides(base, overrides)
    if (
        'backend.pyref_options' in base
        and isinstance(overrides.get('backend'), Mapping)
        and 'pyref_options' in overrides['backend']
    ):
        backend_dict = dict(overrides['backend'])
        backend_dict['pyref_options'] = {
            **base['backend.pyref_options'],
            **dict(backend_dict['pyref_options']),
        }
        merged.pop('backend.pyref_options', None)
        merged['backend'] = backend_dict
    return merged


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
    known = sorted(f.name for f in dataclasses.fields(PyrefConfig))
    unknown = sorted(set(extra) - set(known))
    if unknown:
        hints = [
            f'{k!r} -> {m[0]!r}'
            for k in unknown
            if (m := difflib.get_close_matches(k, known, n=1, cutoff=0.5))
        ]
        suffix = f' (did you mean: {", ".join(hints)}?)' if hints else ''
        raise ValueError(f'backend.pyref_options has unknown PyrefConfig fields: {unknown}{suffix}')
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
    backend: Literal['auto', 'sc2', 'pyref', 'zgame'] | Backend | None = None,
    profile: str | None = None,
    dual: bool = False,
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
) -> VecEnv:
    """Always return a `VecEnv`. `backend` may also be a pre-built instance with matching
    `batch_size`.

    Backend and profile resolution:
    * When ``backend is None``, ``profile is None``, and ``dual is False``, ``backend`` defaults
      to ``'sc2'`` (preserving 0.1.0 compatibility).
    * When ``profile`` is given (e.g. ``profile='sc2-native'``) and ``backend`` is ``None`` or
      ``'auto'``, ``'pyref'``-family profiles select ``'zgame'`` if ``zfoundry-zgame`` is
      installed, else ``'pyref'``; ``'sc2-live'`` selects ``'sc2'``.
    * When ``dual=True``, sets ``opponent.kind='external'`` and defaults ``profile`` to
      ``'sc2-symmetric'`` for simulator backends.
    * When ``backend='auto'`` without a profile, portable presets select ``'zgame'`` (if
      available) else ``'pyref'``, and live SC2 presets select ``'sc2'``.

    Override precedence: ``preset < scenario defaults < preset scenario overrides < profile
    overrides < dual ('opponent.kind': 'external') < user overrides`` (``backend.pyref_options``
    entries are deep-merged key-by-key).

    `seed` is a batch-wide seed (env `b` derives its RNG stream from `(seed, b)` and backend seed
    `seed + b`); `seeds` provides per-env seeds of length `batch_size` (mutually exclusive with
    `seed`), so env `b` is bitwise-identical to a `batch_size=1` env with `seed=seeds[b]` (see
    `unismac.env.core`). `zgame_threads` sets the zgame kernel thread pool size (per-env
    parallelism, result invariant to thread count), valid only with `backend='zgame'` (defaults
    to `ZgameBackend`'s default of 1).

    `accel` selects the environment-layer tier: `'python'`, `'services'`, `'rust'` (for classic
    recipes: `smac-portable`, `smacv2-portable`, `smac-hard-portable` plus scenario overrides,
    with user overrides limited to `backend.*`, `termination.continuing_episode`, and two-sided
    `opponent.kind='external'`), or `'auto'` (`'rust'` when eligible, else `'services'` if zgame
    is installed, else `'python'`). All tiers are bitwise-identical and backend-agnostic.
    """
    from ..assets import load_map_info
    from ..profiles import STANDARD_SYMMETRIC_PROFILE, merge_overrides, profile_overrides
    from ..scenarios import spawn_record

    accel_mod.resolve('services' if accel == 'rust' else accel)  # Validate tier name and extension

    if isinstance(backend, str) and backend not in _BACKEND_NAMES:
        raise ValueError(
            f'Unknown backend {backend!r}; expected auto / sc2 / pyref / zgame'
            f'{_suggest(backend, _BACKEND_NAMES)}'
        )

    if dual and overrides:
        opp_kind = _extract_opponent_kind(overrides)
        if opp_kind is not None and opp_kind != 'external':
            raise ValueError(
                f'dual=True conflicts with overrides setting opponent.kind={opp_kind!r}; '
                'remove the conflicting override or pass dual=False'
            )

    effective_profile = profile
    if dual and effective_profile is None and backend in (None, 'auto', 'pyref', 'zgame'):
        effective_profile = STANDARD_SYMMETRIC_PROFILE
    if dual and effective_profile in _ASYMMETRIC_PROFILES:
        raise ValueError(
            f'dual=True cannot be combined with asymmetric profile {effective_profile!r}; '
            f'use profile={STANDARD_SYMMETRIC_PROFILE!r} (or omit profile when dual=True)'
        )

    prof_backend: str | None = None
    prof_over: dict[str, Any] = {}
    if effective_profile is not None:
        prof_backend, prof_over = profile_overrides(effective_profile)
        if prof_backend == 'sc2' and backend in ('pyref', 'zgame'):
            raise ValueError(
                f'Profile {effective_profile!r} targets live SC2 (backend={prof_backend!r}), '
                f"not backend={backend!r}; use profile='sc2-native' or backend='sc2'"
            )
        if prof_backend == 'pyref' and backend == 'sc2':
            raise ValueError(
                f'Profile {effective_profile!r} targets simulator backends (zgame/pyref), '
                "not backend='sc2'; use profile='sc2-live' or backend='zgame'"
            )

    effective_backend: Literal['sc2', 'pyref', 'zgame'] | Backend
    if backend is None and profile is None and not dual:
        effective_backend = 'sc2'
    elif backend is None or backend == 'auto':
        effective_backend = _resolve_auto_backend(preset, prof_backend)
    else:
        effective_backend = backend

    if zgame_threads is not None and effective_backend != 'zgame':
        raise ValueError(
            f"zgame_threads is only valid for backend='zgame', "
            f'got {_backend_name(effective_backend)!r}'
        )
    if seed is not None and seeds is not None:
        raise ValueError(
            'Specify at most one of seed or seeds (seed is batch-wide; seeds is per-env)'
        )
    if seeds is not None and len(seeds) != batch_size:
        raise ValueError(f'seeds has {len(seeds)} items, expected batch_size={batch_size}')

    combined_overrides: Mapping[str, Any] | None
    if prof_over or dual:
        merged = dict(prof_over)
        if dual:
            merged = merge_overrides(merged, {'opponent.kind': 'external'})
        combined_overrides = _overlay_overrides(merged, overrides)
    else:
        combined_overrides = overrides

    config, spec = build_config(preset, scenario, combined_overrides)
    tier = accel_mod.resolve(accel, config, spec)
    name = _backend_name(effective_backend)
    map_id = resolve_map_ids(config, spec, name)
    try:
        info = load_map_info(map_id)
    except FileNotFoundError:
        raise CapabilityError(
            f'Missing map asset {map_id} ({preset}/{scenario} is unavailable on backend {name})'
        ) from None
    explicit = config.backend.spawn_mode == 'explicit' and not spec.procedural
    is_dual = config.opponent.kind == 'external'
    record = None
    if explicit:
        try:
            record = spawn_record(
                spec.name, 'smac-hard' if spec.benchmark == 'smac-hard' else 'smac'
            )
        except KeyError as exc:
            raise CapabilityError(f'{preset}/{scenario} has no spawn record: {exc}') from None
    if isinstance(effective_backend, str):
        if effective_backend == 'sc2':
            if batch_size != 1:
                raise CapabilityError('SC2 backend only supports batch_size=1')
            from ..backends.sc2 import Sc2Backend

            inst: Backend = Sc2Backend(slot_order=config.backend.sc2_slot_order)
        elif effective_backend == 'pyref':
            from ..assets import load_static_data
            from ..backends.pyref import PyrefBackend

            pyref = _pyref_config(config.backend, is_dual)
            inst = PyrefBackend(load_static_data(), {map_id: info}, batch_size, pyref)
        elif effective_backend == 'zgame':
            # Rust backend shares PyrefConfig fields; unsupported options raise CapabilityError.
            from ..assets import load_static_data
            from ..backends.zgame import ZgameBackend

            pyref = _pyref_config(config.backend, is_dual)
            threads = 1 if zgame_threads is None else int(zgame_threads)  # 1 = ZgameBackend default
            inst = ZgameBackend(
                load_static_data(), {map_id: info}, batch_size, pyref, threads=threads
            )
        else:
            raise ValueError(
                f'Unknown backend {effective_backend!r}; expected auto / sc2 / pyref / zgame'
            )
    else:
        inst = effective_backend
        if inst.batch_size != batch_size:
            raise ValueError(f'Backend batch_size={inst.batch_size} does not match {batch_size}')
    return VecEnv(config, spec, inst, info, seed=seed, seeds=seeds, spawn_record=record, accel=tier)


def make_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['auto', 'sc2', 'pyref', 'zgame'] | Backend | None = None,
    profile: str | None = None,
    dual: bool = False,
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
) -> UnismacEnv | VecEnv:
    """Create an environment: `UnismacEnv` (SMAC API) when `batch_size=1`, else `VecEnv`.

    `overrides` keys may be dotted paths, upstream flat parameter names, or group names (see
    `unismac.env.config`). See `make_vec_env` for `profile`, `dual`, `backend='auto'`, `seeds`,
    `zgame_threads`, and `accel`.
    """
    vec = make_vec_env(
        preset,
        scenario,
        backend=backend,
        profile=profile,
        dual=dual,
        batch_size=batch_size,
        seed=seed,
        seeds=seeds,
        overrides=overrides,
        zgame_threads=zgame_threads,
        accel=accel,
    )
    return UnismacEnv(vec) if batch_size == 1 else vec
