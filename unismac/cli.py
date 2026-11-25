"""Command-line interface for UniSMAC (`unismac` / `python -m unismac`).

Subcommands:
* ``unismac info`` — package version, contract version, and `zgame` acceleration status.
* ``unismac list-scenarios [--preset PRESET]`` — list registered scenarios.
* ``unismac list-profiles [--all]`` — list public profiles (or include legacy aliases).
* ``unismac bench [--preset ...] [--scenario ...] [--backend ...] [--batch-size ...] [--steps ...]``
  — run a quick random-action rollout throughput benchmark.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Sequence

import numpy as np


def _cmd_info(_args: argparse.Namespace) -> int:
    from . import CONTRACT_VERSION, __version__
    from .backends.zgame.backend import ENGINE_SCHEMA_VERSION, _module, build_info
    from .env import accel
    from .env.manifest import NUMPY_MAX_EXCLUSIVE, NUMPY_MIN, mismatches

    print(f'unismac: {__version__}')
    print(f'contract_version: {CONTRACT_VERSION}')
    print(f'engine_schema_version: {ENGINE_SCHEMA_VERSION}')
    print(f'python: {platform.python_version()}')
    print(f'numpy: {np.__version__} (manifest range: >={NUMPY_MIN}, <{NUMPY_MAX_EXCLUSIVE})')

    try:
        z = _module()
    except ImportError as exc:
        print(f'zgame: unavailable ({exc})')
        print('accel_default: python')
        return 0

    info = build_info()
    z_env = getattr(z, 'env', None)
    svc_ver = getattr(z_env, 'SERVICES_VERSION', None) if z_env is not None else None
    sem_ver = getattr(z_env, 'SEMANTICS', None) if z_env is not None else None
    print(f'zgame: {info["version"]} ({info["module"]})')
    print(f'zgame_contract_version: {getattr(z, "CONTRACT_VERSION", None)}')
    print(f'zgame_engine_schema_version: {getattr(z, "ENGINE_SCHEMA_VERSION", None)}')
    print(f'zgame_services_version: {svc_ver}')
    print(f'zgame_env_semantics: {sem_ver}')
    if info.get('commit'):
        dirty = ' (dirty)' if info.get('dirty') else ''
        print(f'zgame_commit: {info["commit"]}{dirty}')
    if info.get('built'):
        print(f'zgame_built: {info["built"]}')
    native_ok = accel.native_available()
    print(f'native_env_available: {native_ok}')
    if native_ok:
        probs = mismatches()
        print(f'rust_manifest_ok: {not probs}')
        for p in probs:
            print(f'  manifest_mismatch: {p}')
    return 0


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    from . import presets, scenarios

    if args.preset is not None:
        try:
            preset = presets.get(args.preset)
        except KeyError:
            print(
                f'Unknown preset {args.preset!r}. Available presets: {", ".join(presets.names())}',
                file=sys.stderr,
            )
            return 2
        for name in scenarios.names(preset.benchmark):
            print(name)
        return 0

    for benchmark in scenarios.BENCHMARKS:
        names = scenarios.names(benchmark)
        print(f'[{benchmark}] ({len(names)} scenarios)')
        for name in names:
            print(f'  {name}')
    return 0


def _cmd_list_profiles(args: argparse.Namespace) -> int:
    from .profiles import PROFILES, list_profiles

    names = list_profiles(include_legacy=bool(args.all))
    width = max(len(n) for n in names)
    for name in names:
        p = PROFILES[name]
        print(f'{name:<{width}}  [{p.backend:<5}]  {p.doc}')
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .env import make_vec_env
    from .profiles import STANDARD_PROFILE, profile_overrides

    if args.batch_size < 1:
        print('--batch-size must be >= 1', file=sys.stderr)
        return 2
    if args.steps < 1:
        print('--steps must be >= 1', file=sys.stderr)
        return 2

    overrides = None
    if args.backend in ('pyref', 'zgame') and args.profile:
        prof_backend, prof_over = profile_overrides(args.profile)
        if prof_backend != 'pyref':
            print(
                f'Profile {args.profile!r} targets backend {prof_backend!r}, not {args.backend!r}',
                file=sys.stderr,
            )
            return 2
        overrides = prof_over

    kwargs = {}
    if args.backend == 'zgame' and args.threads is not None:
        kwargs['zgame_threads'] = args.threads

    try:
        env = make_vec_env(
            args.preset,
            args.scenario,
            backend=args.backend,
            batch_size=args.batch_size,
            seed=args.seed,
            overrides=overrides,
            **kwargs,
        )
    except (ImportError, ValueError, KeyError) as exc:
        print(f'Error creating environment: {exc}', file=sys.stderr)
        return 1

    try:
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
    finally:
        env.close()
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
        'info', help='Show package versions, contract info, and zgame acceleration status.'
    )
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
    p_scen.set_defaults(func=_cmd_list_scenarios)

    p_prof = sub.add_parser('list-profiles', help='List named simulator configuration profiles.')
    p_prof.add_argument(
        '--all',
        action='store_true',
        help='Include backward-compatible legacy profile aliases.',
    )
    p_prof.set_defaults(func=_cmd_list_profiles)

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
