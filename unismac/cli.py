"""Command-line interface for UniSMAC (`unismac` / `python -m unismac`).

Subcommands:
* ``unismac info [--json]`` — package version, contract version, SC2PATH, and `zgame` status.
* ``unismac list-scenarios [--preset PRESET] [--runnable] [--details]`` — list scenarios.
* ``unismac list-profiles [--all]`` — list public profiles (or include legacy aliases).
* ``unismac show-profile NAME [--raw]`` — inspect non-default options for a named profile.
* ``unismac show-options [--group GROUP] [--tier TIER] [--zgame-only]`` — inspect `PyrefConfig`
  option metadata.
* ``unismac bench [--preset ...] [--scenario ...] [--backend ...] [--batch-size ...] [--steps ...]``
  — run a quick random-action rollout throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _detect_sc2_path() -> str | None:
    """Check whether a local StarCraft II installation exists without modifying `os.environ`."""
    candidates: list[Path] = []
    env_path = os.environ.get('SC2PATH')
    if env_path:
        candidates.append(Path(env_path))
    else:
        candidates.append(Path('~/StarCraftII'))
        if os.name == 'nt':
            candidates.append(Path('C:/Program Files (x86)/StarCraft II'))
        elif hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
            candidates.append(Path('/Applications/StarCraft II'))
    for cand in candidates:
        path = cand.expanduser().resolve()
        if (path / 'Versions').is_dir():
            return str(path)
    return None


def _cmd_info(args: argparse.Namespace) -> int:
    from . import CONTRACT_VERSION, __version__
    from .backends.zgame.backend import ENGINE_SCHEMA_VERSION, _module, build_info
    from .env import accel
    from .env.accel.manifest import NUMPY_MAX_EXCLUSIVE, NUMPY_MIN, mismatches

    sc2_path = _detect_sc2_path()
    payload: dict[str, Any] = {
        'unismac': __version__,
        'contract_version': CONTRACT_VERSION,
        'engine_schema_version': ENGINE_SCHEMA_VERSION,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'numpy_manifest_range': [NUMPY_MIN, NUMPY_MAX_EXCLUSIVE],
        'sc2_path': sc2_path,
    }

    try:
        z = _module()
    except ImportError as exc:
        payload['zgame_available'] = False
        payload['zgame_error'] = str(exc)
        payload['accel_default'] = 'python'
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(f'unismac: {__version__}')
        print(f'contract_version: {CONTRACT_VERSION}')
        print(f'engine_schema_version: {ENGINE_SCHEMA_VERSION}')
        print(f'python: {platform.python_version()}')
        print(f'numpy: {np.__version__} (manifest range: >={NUMPY_MIN}, <{NUMPY_MAX_EXCLUSIVE})')
        print(f'sc2_path: {sc2_path or "not found"}')
        print(f'zgame: unavailable ({exc})')
        print('accel_default: python')
        return 0

    info = build_info()
    z_env = getattr(z, 'env', None)
    svc_ver = getattr(z_env, 'SERVICES_VERSION', None) if z_env is not None else None
    sem_ver = getattr(z_env, 'SEMANTICS', None) if z_env is not None else None
    native_ok = accel.native_available()
    probs = list(mismatches()) if native_ok else []

    payload.update(
        {
            'zgame_available': True,
            'zgame': info['version'],
            'zgame_module': info['module'],
            'zgame_contract_version': getattr(z, 'CONTRACT_VERSION', None),
            'zgame_engine_schema_version': getattr(z, 'ENGINE_SCHEMA_VERSION', None),
            'zgame_services_version': svc_ver,
            'zgame_env_semantics': sem_ver,
            'zgame_commit': info.get('commit'),
            'zgame_dirty': info.get('dirty'),
            'zgame_built': info.get('built'),
            'native_env_available': native_ok,
            'rust_manifest_ok': (not probs) if native_ok else None,
            'manifest_mismatches': probs,
        }
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f'unismac: {__version__}')
    print(f'contract_version: {CONTRACT_VERSION}')
    print(f'engine_schema_version: {ENGINE_SCHEMA_VERSION}')
    print(f'python: {platform.python_version()}')
    print(f'numpy: {np.__version__} (manifest range: >={NUMPY_MIN}, <{NUMPY_MAX_EXCLUSIVE})')
    print(f'sc2_path: {sc2_path or "not found"}')
    print(f'zgame: {info["version"]} ({info["module"]})')
    print(f'zgame_contract_version: {payload["zgame_contract_version"]}')
    print(f'zgame_engine_schema_version: {payload["zgame_engine_schema_version"]}')
    print(f'zgame_services_version: {svc_ver}')
    print(f'zgame_env_semantics: {sem_ver}')
    if info.get('commit'):
        dirty = ' (dirty)' if info.get('dirty') else ''
        print(f'zgame_commit: {info["commit"]}{dirty}')
    if info.get('built'):
        print(f'zgame_built: {info["built"]}')
    print(f'native_env_available: {native_ok}')
    if native_ok:
        print(f'rust_manifest_ok: {not probs}')
        for p in probs:
            print(f'  manifest_mismatch: {p}')
    return 0


def _portable_preset_for(benchmark: str) -> str:
    return {
        'smac': 'smac-portable',
        'smacv2': 'smacv2-portable',
        'smac-hard': 'smac-hard-portable',
    }[benchmark]


def _scenario_status(preset_obj: Any, spec: Any) -> tuple[bool, str, bool]:
    """Return `(portable_runnable, mirror_axis_label, is_multi_level)` for `spec`."""
    from dataclasses import replace

    from .assets import load_map_info, load_unit_overrides
    from .contract import Spawn, Variant
    from .env.core import resolve_map_ids
    from .env.dual import mirror_axis
    from .env.opponents import MULTI_LEVEL_SMAC_MAPS
    from .env.semantics.catalog import MEDIVAC
    from .scenarios import spawn_record

    multi_level = spec.benchmark == 'smac' and spec.name in MULTI_LEVEL_SMAC_MAPS
    map_id = resolve_map_ids(preset_obj.config, spec, 'pyref')
    try:
        info = load_map_info(map_id)
    except FileNotFoundError:
        return False, '-', multi_level
    if spec.procedural:
        return True, '-', multi_level
    try:
        rec = spawn_record(spec.name, 'smac-hard' if spec.benchmark == 'smac-hard' else 'smac')
    except KeyError:
        return False, '-', multi_level
    spawns = tuple(rec.spawns)
    variant = preset_obj.config.backend.spawn_variant
    if variant == 'std':
        spawns = tuple(replace(s, variant=int(Variant.STD)) for s in spawns)
    elif variant == 'smac':
        spawns = tuple(
            replace(
                s,
                variant=int(Variant.RL)
                if (s.owner == 1 and s.unit_type == MEDIVAC)
                else int(Variant.STD),
            )
            for s in spawns
        )
    spawns = tuple(Spawn(**vars(s)) for s in spawns)
    unit_ov = (
        load_unit_overrides(map_id) if preset_obj.config.backend.unit_overrides == 'asset' else ()
    )
    axis = mirror_axis(spawns, info, unit_ov)
    return True, (axis if axis is not None else '-'), multi_level


def _print_scenario_table(rows: list[tuple[str, str, int, bool, str, bool]]) -> None:
    headers = ('scenario', 'units', 'limit', 'portable', 'mirror', 'multi_level')
    str_rows = [
        (name, units, str(limit), 'yes' if ok else 'no', mirror, 'yes' if ml else 'no')
        for name, units, limit, ok, mirror, ml in rows
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in str_rows)) if str_rows else len(h)
        for i, h in enumerate(headers)
    ]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*('-' * w for w in widths)))
    for r in str_rows:
        print(fmt.format(*r))


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    from . import presets, scenarios

    if args.preset is not None:
        try:
            preset = presets.get(args.preset)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        check_preset = (
            preset
            if preset.config.backend.spawn_mode == 'explicit'
            else presets.get(_portable_preset_for(preset.benchmark))
        )
        rows = []
        for name in scenarios.names(preset.benchmark):
            spec = scenarios.get(preset.benchmark, name)
            ok, mirror, ml = _scenario_status(check_preset, spec)
            if args.runnable and not ok:
                continue
            rows.append((name, f'{spec.n_agents}v{spec.n_enemies}', spec.limit, ok, mirror, ml))
        if args.details:
            _print_scenario_table(rows)
        else:
            for row in rows:
                print(row[0])
        return 0

    for benchmark in scenarios.BENCHMARKS:
        check_preset = presets.get(_portable_preset_for(benchmark))
        rows = []
        for name in scenarios.names(benchmark):
            spec = scenarios.get(benchmark, name)
            ok, mirror, ml = _scenario_status(check_preset, spec)
            if args.runnable and not ok:
                continue
            rows.append((name, f'{spec.n_agents}v{spec.n_enemies}', spec.limit, ok, mirror, ml))
        print(f'[{benchmark}] ({len(rows)} scenarios)')
        if args.details:
            _print_scenario_table(rows)
        else:
            for row in rows:
                print(f'  {row[0]}')
    return 0


def _cmd_list_profiles(args: argparse.Namespace) -> int:
    from .profiles import PROFILES, list_profiles

    names = list_profiles(include_legacy=bool(args.all))
    width = max(len(n) for n in names)
    for name in names:
        p = PROFILES[name]
        print(f'{name:<{width}}  [{p.backend:<5}]  {p.doc}')
    return 0


def _cmd_show_profile(args: argparse.Namespace) -> int:
    from .backends.pyref import PyrefConfig
    from .profiles import PROFILES, profile_overrides

    try:
        backend, over = profile_overrides(args.name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    prof = PROFILES[args.name]
    if args.raw:
        payload = {
            'name': prof.name,
            'backend': backend,
            'doc': prof.doc,
            'overrides': over,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    print(f'profile: {prof.name}')
    print(f'default_backend: {backend}')
    print(f'description: {prof.doc}')
    top_keys = sorted(k for k in over if k != 'backend.pyref_options')
    if top_keys:
        print('env_overrides:')
        for k in top_keys:
            print(f'  {k} = {over[k]!r}')
    raw_opts = over.get('backend.pyref_options', {})
    defaults = PyrefConfig()
    non_default = {
        k: v
        for k, v in sorted(raw_opts.items())
        if not hasattr(defaults, k) or getattr(defaults, k) != v
    }
    print(f'non_default_pyref_options ({len(non_default)}):')
    if not non_default:
        print('  (none)')
    else:
        width = max(len(k) for k in non_default)
        for k, v in non_default.items():
            print(f'  {k:<{width}} = {v!r}')
    return 0


def _cmd_show_options(args: argparse.Namespace) -> int:
    from .backends.pyref.options_meta import describe_pyref_options

    try:
        items = describe_pyref_options(
            group=args.group, tier=args.tier, zgame_only=bool(args.zgame_only)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not items:
        print('(no matching options)')
        return 0

    grouped: dict[str, list[Any]] = {}
    for item in items:
        grouped.setdefault(item.group, []).append(item)
    for grp, members in grouped.items():
        print(f'[{grp}] ({len(members)} options)')
        w_name = max(len(m.name) for m in members)
        w_tier = max(len(m.tier) for m in members)
        for m in members:
            zg = 'zgame' if m.zgame_supported else 'pyref'
            print(
                f'  {m.name:<{w_name}}  {m.tier:<{w_tier}}  {zg:<5}  '
                f'default={m.default!r}  {m.summary}'
            )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .contract import CapabilityError
    from .env import make_vec_env
    from .profiles import STANDARD_PROFILE

    if args.batch_size < 1:
        print('--batch-size must be >= 1', file=sys.stderr)
        return 2
    if args.steps < 1:
        print('--steps must be >= 1', file=sys.stderr)
        return 2

    profile = args.profile if args.backend in ('pyref', 'zgame') else None
    kwargs: dict[str, Any] = {}
    if args.backend == 'zgame' and args.threads is not None:
        kwargs['zgame_threads'] = args.threads

    try:
        env = make_vec_env(
            args.preset,
            args.scenario,
            backend=args.backend,
            profile=profile,
            batch_size=args.batch_size,
            seed=args.seed,
            **kwargs,
        )
    except (ImportError, ValueError, KeyError, CapabilityError) as exc:
        print(f'Error creating environment: {exc}', file=sys.stderr)
        return 1

    with env:
        rng = np.random.default_rng(args.seed)
        env.reset()
        episodes = 0
        t0 = time.perf_counter()
        for _ in range(args.steps):
            avail = env.get_avail_actions()
            actions = (rng.random(avail.shape) * avail).argmax(-1)
            _, terminated, _ = env.step(actions)
            if terminated.any():
                done_ids = np.flatnonzero(terminated).tolist()
                episodes += len(done_ids)
                env.reset(done_ids)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        total_env_steps = args.steps * args.batch_size
        print(
            f'preset={args.preset} scenario={args.scenario} backend={args.backend} '
            f'profile={args.profile or STANDARD_PROFILE} accel={env.accel} '
            f'batch_size={args.batch_size} steps={args.steps}'
        )
        print(
            f'elapsed={elapsed:.3f}s env_steps={total_env_steps} episodes={episodes} '
            f'throughput={total_env_steps / elapsed:.1f} env-steps/s'
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    from .profiles import STANDARD_PROFILE

    parser = argparse.ArgumentParser(
        prog='unismac',
        description=f'UniSMAC {__version__}: unified StarCraft Multi-Agent Challenge environment.',
    )
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command')

    p_info = sub.add_parser(
        'info', help='Show package versions, contract info, SC2PATH, and zgame status.'
    )
    p_info.add_argument('--json', action='store_true', help='Output diagnostic info as JSON.')
    p_info.set_defaults(func=_cmd_info)

    p_scen = sub.add_parser('list-scenarios', help='List registered benchmark scenarios.')
    p_scen.add_argument(
        '--preset',
        default=None,
        help=(
            'Filter scenarios by preset name '
            '(e.g. smac-portable, smacv2-portable, smac-hard-portable).'
        ),
    )
    p_scen.add_argument(
        '--runnable',
        action='store_true',
        help='Only list scenarios runnable on portable simulator backends (zgame/pyref).',
    )
    p_scen.add_argument(
        '--details',
        action='store_true',
        help='Show detailed columns (units, episode limit, portable, mirror, multi-level).',
    )
    p_scen.set_defaults(func=_cmd_list_scenarios)

    p_prof = sub.add_parser('list-profiles', help='List named simulator configuration profiles.')
    p_prof.add_argument(
        '--all',
        action='store_true',
        help='Include backward-compatible legacy profile aliases.',
    )
    p_prof.set_defaults(func=_cmd_list_profiles)

    p_show_prof = sub.add_parser(
        'show-profile', help='Show effective non-default options for a named profile.'
    )
    p_show_prof.add_argument('name', help='Profile name (e.g. sc2-native, sc2-symmetric).')
    p_show_prof.add_argument(
        '--raw', action='store_true', help='Output raw profile overrides dictionary as JSON.'
    )
    p_show_prof.set_defaults(func=_cmd_show_profile)

    p_opts = sub.add_parser(
        'show-options', help='Inspect PyrefConfig options grouped by subsystem.'
    )
    p_opts.add_argument('--group', default=None, help='Filter by option group name.')
    p_opts.add_argument(
        '--tier', default=None, help='Filter by tier (core, native, legacy_heuristic).'
    )
    p_opts.add_argument(
        '--zgame-only',
        action='store_true',
        help='Show only options supported by the zgame Rust backend.',
    )
    p_opts.set_defaults(func=_cmd_show_options)

    p_bench = sub.add_parser('bench', help='Run a quick random-action rollout benchmark.')
    p_bench.add_argument('--preset', default='smac-portable', help='Preset name.')
    p_bench.add_argument('--scenario', default='8m', help='Scenario name.')
    p_bench.add_argument(
        '--backend',
        default='zgame',
        choices=('zgame', 'pyref', 'sc2'),
        help='Simulation backend.',
    )
    p_bench.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Simulator profile for pyref/zgame backends (default: {STANDARD_PROFILE}).',
    )
    p_bench.add_argument('--batch-size', type=int, default=64, help='Vectorized batch size.')
    p_bench.add_argument('--steps', type=int, default=200, help='Number of rollout steps.')
    p_bench.add_argument('--threads', type=int, default=None, help='Rayon threads for zgame.')
    p_bench.add_argument('--seed', type=int, default=0, help='Random seed.')
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(args)
    if ns.command is None:
        parser.print_help()
        return 0
    return int(ns.func(ns))
