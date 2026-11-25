"""One-line backend switching example (`zgame` / `pyref` / live `sc2`).

Uses `make_env(..., batch_size=1)` to obtain the PyMARL/EPyMARL-compatible
`UnismacEnv` interface and runs a simple heuristic or random policy on any
backend. Pass `--native-smac` to evaluate against the native 23-map SMAC
trigger opponent on `zgame` or `pyref`.

Run:
    python examples/sc2_eval.py --backend zgame --scenario 8m --episodes 5
    python examples/sc2_eval.py --backend pyref --scenario 8m --episodes 2
    python examples/sc2_eval.py --backend zgame --scenario 2c_vs_64zg --native-smac
    SC2PATH=/path/to/SC2 python examples/sc2_eval.py --backend sc2 --scenario 8m --episodes 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from unismac import UnismacEnv, make_env
from unismac.env import native_smac_overrides
from unismac.profiles import STANDARD_PROFILE, merge_overrides, profile_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Single-env backend switching example for UniSMAC.'
    )
    parser.add_argument(
        '--preset',
        default='smac-portable',
        help='Preset name (default: smac-portable).',
    )
    parser.add_argument('--scenario', default='8m', help='Scenario name (default: 8m).')
    parser.add_argument(
        '--backend',
        default='zgame',
        choices=('zgame', 'pyref', 'sc2'),
        help='Backend to run (default: zgame).',
    )
    parser.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Profile for zgame/pyref backends (default: {STANDARD_PROFILE}).',
    )
    parser.add_argument(
        '--native-smac',
        action='store_true',
        help='Apply native_smac_overrides(...) for single-client SMAC trigger opponent alignment.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of episodes to run (default: 3).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Base seed (default: 0).')
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, object] | None:
    if args.backend == 'sc2':
        return None
    _, overrides = profile_overrides(args.profile)
    if args.native_smac:
        overrides = merge_overrides(
            overrides,
            native_smac_overrides(args.scenario, terrain_elevation='auto'),
        )
    return overrides


def main() -> int:
    args = parse_args()
    overrides = build_overrides(args)

    try:
        env = make_env(
            args.preset,
            args.scenario,
            backend=args.backend,
            batch_size=1,
            seed=args.seed,
            overrides=overrides,
        )
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        if args.backend == 'sc2':
            print(
                f'Failed to initialize live SC2 backend ({exc}). '
                'Ensure `pip install "unismac[sc2]"` is installed and SC2PATH points to '
                'StarCraft II 4.10.',
                file=sys.stderr,
            )
        else:
            print(f"Failed to initialize backend '{args.backend}': {exc}", file=sys.stderr)
        return 1

    assert isinstance(env, UnismacEnv)
    try:
        info = env.get_env_info()
        print(f'Preset/Scenario : {args.preset}/{args.scenario}')
        print(f'Backend / Accel : {args.backend} / {env.vec.accel}')
        print(
            'Env info        : '
            f'n_agents={info["n_agents"]}, n_actions={info["n_actions"]}, '
            f'obs_shape={info["obs_shape"]}, state_shape={info["state_shape"]}, '
            f'episode_limit={info["episode_limit"]}'
        )

        rng = np.random.default_rng(args.seed)
        for ep in range(args.episodes):
            env.reset()
            terminated = False
            ep_reward = 0.0
            steps = 0
            last_info: dict[str, object] = {}

            while not terminated:
                avail = env.get_avail_actions()
                actions = []
                for agent_mask in avail:
                    valid = np.flatnonzero(agent_mask)
                    # Greedy baseline: prefer an attack/heal target if valid, else random valid.
                    attack_valid = valid[valid >= 6]
                    pool = attack_valid if len(attack_valid) > 0 else valid
                    actions.append(int(pool[rng.integers(len(pool))]))
                reward, terminated, last_info = env.step(actions)
                ep_reward += reward
                steps += 1

            print(
                f'Episode {ep + 1:02d}      : steps={steps:3d}, reward={ep_reward:6.2f}, '
                f'won={bool(last_info.get("battle_won"))}'
            )

        print(f'Summary stats   : {env.get_stats()}')
    finally:
        env.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
