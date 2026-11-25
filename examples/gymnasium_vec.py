"""Gymnasium 1.0+ vectorized rollout example with UniSMAC.

Requires the optional `[gymnasium]` extra (`pip install "unismac[gymnasium]"`).
Demonstrates `make_gym_vec_env(...)` with CTDE Dict observations
(`obs`, `state`, `action_mask`), `MultiDiscrete` actions, native
`terminated` / `truncated` separation, and all three Gymnasium 1.0+
`AutoresetMode` behaviors (`next_step`, `same_step`, `disabled`).

Run:
    python examples/gymnasium_vec.py
    python examples/gymnasium_vec.py --scenario 2s3z --autoreset-mode same_step
    python examples/gymnasium_vec.py --autoreset-mode disabled --batch-size 16 --steps 100
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from unismac import STANDARD_PROFILE


def sample_masked_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample valid actions uniformly from an `int8[B, A, K]` action mask."""
    scores = (1.0 + rng.random(mask.shape)) * mask
    return scores.argmax(axis=-1).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Gymnasium 1.0+ vectorized rollout example for UniSMAC.'
    )
    parser.add_argument('--preset', default='smac-portable', help='Preset name.')
    parser.add_argument('--scenario', default='8m', help='Scenario name (default: 8m).')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=('auto', 'zgame', 'pyref'),
        help='Simulation backend (default: auto).',
    )
    parser.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Simulation profile (default: {STANDARD_PROFILE}).',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Parallel environments.')
    parser.add_argument('--steps', type=int, default=150, help='Batched steps to run.')
    parser.add_argument(
        '--autoreset-mode',
        default='next_step',
        choices=('next_step', 'same_step', 'disabled'),
        help='Gymnasium 1.0+ VectorEnv autoreset mode (default: next_step).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from unismac.adapters.gymnasium import make_gym_vec_env
    except ImportError as exc:
        print(
            f'Gymnasium adapter is unavailable ({exc}). '
            'Install it with: pip install "unismac[gymnasium]"',
            file=sys.stderr,
        )
        return 1

    try:
        env = make_gym_vec_env(
            args.preset,
            args.scenario,
            backend=args.backend,
            profile=args.profile,
            batch_size=args.batch_size,
            seed=args.seed,
            autoreset_mode=args.autoreset_mode,
        )
    except ImportError as exc:
        print(
            f"Backend '{args.backend}' is unavailable ({exc}). "
            "Install 'zfoundry-zgame' or re-run with '--backend pyref'.",
            file=sys.stderr,
        )
        return 1

    try:
        obs, _ = env.reset(seed=args.seed)
        rng = np.random.default_rng(args.seed)
        episodes = 0
        wins = 0
        timeouts = 0
        total_reward = 0.0

        t0 = time.perf_counter()
        for _ in range(args.steps):
            actions = sample_masked_actions(obs['action_mask'], rng)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += float(rewards.sum())
            done = terminations | truncations
            if done.any():
                episodes += int(done.sum())
                timeouts += int(truncations.sum())
                if args.autoreset_mode == 'same_step' and '_final_info' in infos:
                    final_mask = infos['_final_info']
                    final_infos = infos['final_info']
                    for i in np.flatnonzero(final_mask).tolist():
                        entry = final_infos[i]
                        if isinstance(entry, dict) and entry.get('battle_won'):
                            wins += 1
                elif 'battle_won' in infos and '_battle_won' in infos:
                    wins += int(np.asarray(infos['battle_won'])[done & infos['_battle_won']].sum())
                if args.autoreset_mode == 'disabled':
                    obs, _ = env.reset(options={'reset_mask': done})
        elapsed = time.perf_counter() - t0

        env_steps = args.steps * args.batch_size
        sps = env_steps / max(elapsed, 1e-9)
        print(f'Autoreset mode  : {args.autoreset_mode}')
        print(f'Obs space (1x)  : {env.single_observation_space}')
        print(f'Act space (1x)  : {env.single_action_space}')
        print(
            f'Batch shapes    : obs={tuple(obs["obs"].shape)}, '
            f'state={tuple(obs["state"].shape)}, mask={tuple(obs["action_mask"].shape)}'
        )
        print(f'Env steps       : {env_steps:,} in {elapsed:.3f}s ({sps:,.0f} step/s)')
        print(
            f'Episodes / Wins : {episodes} / {wins} '
            f'(timeouts: {timeouts}, total reward: {total_reward:.2f})'
        )
    finally:
        env.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
