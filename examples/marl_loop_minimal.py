"""Minimal zero-dependency MARL rollout and evaluation loop for UniSMAC.

Demonstrates idiomatic integration directly against the purified `VecEnv`
kernel without requiring PyTorch, Gymnasium, or PettingZoo:

1. Context-managed creation via `with make_vec_env(..., profile=...) as env:`;
2. Vectorized masked action sampling (safe against zero-probability ties);
3. Native `StepBatch.terminated` vs `StepBatch.truncated` separation for value
   bootstrapping (no `continuing_episode` override required);
4. Partial reset (`env.reset(done_ids)`) so finished slots do not stay frozen;
5. Deterministic, batch-position-invariant evaluation via per-environment `seeds`.

Run:
    python examples/marl_loop_minimal.py
    python examples/marl_loop_minimal.py --scenario 2s3z --batch-size 16 --steps 100
    python examples/marl_loop_minimal.py --backend pyref --batch-size 4 --steps 40
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from unismac import STANDARD_PROFILE, make_vec_env


def sample_masked_actions(avail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized uniform sampling over valid actions (`avail: int64[B, A, K]`).

    Uses `(1.0 + U(0, 1)) * avail` so every valid action receives a strictly
    positive score while invalid actions stay 0.0 (avoiding boundary ties).
    """
    scores = (1.0 + rng.random(avail.shape)) * avail
    return scores.argmax(axis=-1).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Minimal vectorized MARL rollout and deterministic eval loop.'
    )
    parser.add_argument('--preset', default='smac-portable', help='Preset name.')
    parser.add_argument('--scenario', default='8m', help='Scenario name.')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=('auto', 'zgame', 'pyref'),
        help='Simulation backend (default: auto).',
    )
    parser.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Simulator profile (default: {STANDARD_PROFILE}).',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--steps', type=int, default=80, help='Rollout steps to collect.')
    parser.add_argument('--eval-envs', type=int, default=8, help='Parallel eval environments.')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed.')
    return parser.parse_args()


def collect_rollout(args: argparse.Namespace) -> None:
    """Collect a vectorized training rollout with truncation-aware bootstrap masks."""
    with make_vec_env(
        args.preset,
        args.scenario,
        backend=args.backend,
        profile=args.profile,
        batch_size=args.batch_size,
        seed=args.seed,
    ) as env:
        print(f'Train env       : {env!r}')
        env.reset()
        obs = env.obs()
        state = env.state()
        rng = np.random.default_rng(args.seed)

        # Pre-allocate rollout storage (CleanRL / PPO style).
        b, a, d = obs.shape
        s_dim = state.shape[-1]
        rewards_buf = np.zeros((args.steps, b), dtype=np.float32)
        bootstrap_buf = np.ones((args.steps, b), dtype=np.float32)

        completed = 0
        wins = 0
        timeouts = 0

        for t in range(args.steps):
            avail = env.avail_actions()
            actions = sample_masked_actions(avail, rng)
            batch = env.step(actions)

            # True battle terminations zero out bootstrap targets; timeout truncations keep them.
            rewards_buf[t] = batch.reward
            bootstrap_buf[t] = (~batch.terminated).astype(np.float32)

            done = batch.terminated | batch.truncated
            if done.any():
                done_ids = np.flatnonzero(done).tolist()
                completed += len(done_ids)
                wins += sum(1 for i in done_ids if batch.infos[i].get('battle_won'))
                timeouts += int(batch.truncated[done_ids].sum())
                # Reset finished slots immediately so they do not stay frozen on step t + 1.
                env.reset(done_ids)

            obs = env.obs()
            state = env.state()

        print(
            f'Collected       : T={args.steps}, B={b}, A={a}, obs_dim={d}, state_dim={s_dim}, '
            f'mean_r={float(rewards_buf.mean()):.4f}'
        )
        print(
            f'Train episodes  : completed={completed}, wins={wins}, timeouts={timeouts}, '
            f'bootstrap_frac={float(bootstrap_buf.mean()):.3f}'
        )


def evaluate_deterministic(args: argparse.Namespace) -> None:
    """Evaluate one episode per seed using batch-position-invariant per-env `seeds`."""
    eval_seeds = [1000 + i for i in range(args.eval_envs)]
    with make_vec_env(
        args.preset,
        args.scenario,
        backend=args.backend,
        profile=args.profile,
        batch_size=args.eval_envs,
        seeds=eval_seeds,
    ) as env:
        print(f'Eval env        : {env!r}')
        env.reset()
        rng = np.random.default_rng(args.seed + 1)
        finished = np.zeros(args.eval_envs, dtype=bool)
        ep_returns = np.zeros(args.eval_envs, dtype=np.float64)
        ep_won = np.zeros(args.eval_envs, dtype=bool)

        while not finished.all():
            actions = sample_masked_actions(env.avail_actions(), rng)
            batch = env.step(actions)
            active = ~finished
            ep_returns[active] += batch.reward[active]
            newly_done = batch.terminated | batch.truncated
            for i in np.flatnonzero(newly_done).tolist():
                ep_won[i] = bool(batch.infos[i].get('battle_won'))
            finished |= newly_done

        print(
            f'Eval ({args.eval_envs} seeds)  : win_rate={float(ep_won.mean()):.3f}, '
            f'mean_return={float(ep_returns.mean()):.2f}'
        )


def main() -> int:
    args = parse_args()
    try:
        collect_rollout(args)
        evaluate_deterministic(args)
    except ImportError as exc:
        print(
            f"Backend '{args.backend}' is unavailable ({exc}). "
            "Install 'zfoundry-zgame' (`pip install zfoundry-zgame`) "
            "or re-run with '--backend pyref'.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
