"""Symmetric two-sided self-play example with UniSMAC.

Controls both Player 1 (side=1) and Player 2 (side=2) with external policies
using `dual=True` (automatically selecting the unbiased `sc2-symmetric` profile
and mirror spawning on mirrorable maps).

Run:
    python examples/self_play.py
    python examples/self_play.py --scenario 3s5z --batch-size 32 --steps 200
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from unismac import make_vec_env


def sample_valid_actions(avail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized uniform sampling over valid actions (`avail: int64[B, A, K]`)."""
    scores = (1.0 + rng.random(avail.shape)) * avail
    return scores.argmax(axis=-1).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Two-sided symmetric self-play example for UniSMAC.'
    )
    parser.add_argument(
        '--preset',
        default='smac-portable',
        help='Preset name (default: smac-portable).',
    )
    parser.add_argument('--scenario', default='2s3z', help='Scenario name (default: 2s3z).')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=('auto', 'zgame', 'pyref'),
        help='Simulation backend (default: auto; prefers zgame when installed).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Parallel environments (default: 32).',
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
            dual=True,
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
        obs_p1 = env.get_obs(side=1)
        obs_p2 = env.get_obs(side=2)
        rng_p1 = np.random.default_rng(args.seed)
        rng_p2 = np.random.default_rng(args.seed + 10_000)

        episodes = 0
        p1_wins = 0
        p2_wins = 0
        draws = 0
        p1_reward = 0.0
        p2_reward = 0.0

        t0 = time.perf_counter()
        for _ in range(args.steps):
            acts_p1 = sample_valid_actions(env.get_avail_actions(side=1), rng_p1)
            acts_p2 = sample_valid_actions(env.get_avail_actions(side=2), rng_p2)
            rewards_p1, terminated, infos = env.step(acts_p1, enemy_actions=acts_p2)
            p1_reward += float(rewards_p1.sum())
            p2_reward += sum(float(info.get('reward_enemy', 0.0)) for info in infos)
            if terminated.any():
                done_ids = np.flatnonzero(terminated).tolist()
                episodes += len(done_ids)
                for i in done_ids:
                    won = bool(infos[i].get('battle_won'))
                    lost = bool(infos[i].get('battle_lost'))
                    if won and not lost:
                        p1_wins += 1
                    elif lost and not won:
                        p2_wins += 1
                    else:
                        draws += 1
                env.reset(done_ids)
        elapsed = time.perf_counter() - t0

        env_steps = args.steps * args.batch_size
        sps = env_steps / max(elapsed, 1e-9)
        p1_na, p1_k = env.n_agents_for(1), env.n_actions_for(1)
        p2_na, p2_k = env.n_agents_for(2), env.n_actions_for(2)
        print(f'Environment     : {env!r}')
        print(f'Dual / Mirror   : {env.dual} / {env.mirror!r}')
        print(f'Side 1 Obs      : {tuple(obs_p1.shape)} (n_agents={p1_na}, n_actions={p1_k})')
        print(f'Side 2 Obs      : {tuple(obs_p2.shape)} (n_agents={p2_na}, n_actions={p2_k})')
        print(f'Env steps       : {env_steps:,} in {elapsed:.3f}s ({sps:,.0f} step/s)')
        print(
            f'Outcomes        : P1 wins={p1_wins}, P2 wins={p2_wins}, '
            f'draws={draws} (total={episodes})'
        )
        print(f'Cumulative R    : P1={p1_reward:.2f}, P2={p2_reward:.2f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
