"""30-second vectorized rollout quickstart with UniSMAC (zero SC2 required).

Demonstrates the purified `v0.2.0` `VecEnv` kernel:
`env.reset() -> None`, noun projections (`obs()`, `state()`, `avail_actions()`),
and `StepBatch` (`reward`, `terminated`, `truncated`, `infos`).

Run:
    python examples/quickstart_vec.py
    python examples/quickstart_vec.py --scenario 2s3z --batch-size 64 --steps 200
    python examples/quickstart_vec.py --backend pyref --batch-size 8 --steps 50
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from unismac import STANDARD_PROFILE, make_vec_env


def sample_valid_actions(avail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized uniform sampling over valid actions (`avail: int64[B, A, K]`)."""
    scores = (1.0 + rng.random(avail.shape)) * avail
    return scores.argmax(axis=-1).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Vectorized rollout quickstart for UniSMAC.')
    parser.add_argument(
        '--preset',
        default='smac-portable',
        help='Preset name (default: smac-portable).',
    )
    parser.add_argument('--scenario', default='8m', help='Scenario name (default: 8m).')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=('auto', 'zgame', 'pyref'),
        help='Simulation backend (default: auto; prefers zgame when installed).',
    )
    parser.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Simulation profile (default: {STANDARD_PROFILE}).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Parallel environments (default: 64).',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Batched steps to run (default: 200).',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='ZGame worker threads (default: 1; used when backend is zgame).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed (default: 0).')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {'zgame_threads': args.threads} if args.backend == 'zgame' else {}

    try:
        ctx = make_vec_env(
            args.preset,
            args.scenario,
            backend=args.backend,
            profile=args.profile,
            batch_size=args.batch_size,
            seed=args.seed,
            **kwargs,
        )
    except ImportError as exc:
        print(
            f"Backend '{args.backend}' is unavailable ({exc}). "
            "Install 'zfoundry-zgame' (`pip install zfoundry-zgame`) "
            "or re-run with '--backend pyref'.",
            file=sys.stderr,
        )
        return 1

    with ctx as env:
        env.reset()
        obs = env.obs()
        state = env.state()
        rng = np.random.default_rng(args.seed)
        episodes = 0
        wins = 0
        timeouts = 0
        total_reward = 0.0

        t0 = time.perf_counter()
        for _ in range(args.steps):
            avail = env.avail_actions()
            actions = sample_valid_actions(avail, rng)
            batch = env.step(actions)
            total_reward += float(batch.reward.sum())
            done = batch.terminated | batch.truncated
            if done.any():
                done_ids = np.flatnonzero(done).tolist()
                episodes += len(done_ids)
                wins += sum(1 for i in done_ids if batch.infos[i].get('battle_won'))
                timeouts += int(batch.truncated[done_ids].sum())
                env.reset(done_ids)
        elapsed = time.perf_counter() - t0

        env_steps = args.steps * args.batch_size
        sps = env_steps / max(elapsed, 1e-9)
        print(f'Environment     : {env!r}')
        print(f'Profile         : {args.profile}')
        print(f'Obs / State     : {tuple(obs.shape)} / {tuple(state.shape)}')
        print(f'Env steps       : {env_steps:,} in {elapsed:.3f}s ({sps:,.0f} step/s)')
        print(
            f'Episodes / Wins : {episodes} / {wins} '
            f'(timeouts: {timeouts}, total reward: {total_reward:.2f})'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
