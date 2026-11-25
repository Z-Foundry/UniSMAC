"""PettingZoo ParallelEnv example for single-sided play and two-sided self-play.

Requires the optional `[pettingzoo]` extra (`pip install "unismac[pettingzoo]"`).
Demonstrates `make_pettingzoo_env(...)` in both modes:

1. Single-sided cooperative play (`dual=False`): `ally_0..ally_{N-1}` agents
   with `int8` `action_mask` and CTDE global `env.state()`.
2. Two-sided symmetric self-play (`dual=True`): `ally_0..ally_{N-1}` vs
   `enemy_0..enemy_{M-1}` (supporting heterogeneous spaces on asymmetric maps
   with zero padding) and per-side `env.state_for(1)` / `env.state_for(2)`.

Run:
    python examples/pettingzoo_parallel.py
    python examples/pettingzoo_parallel.py --mode dual --scenario 5m_vs_6m
    python examples/pettingzoo_parallel.py --dead-agent-mode remove --episodes 2
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from unismac import STANDARD_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PettingZoo ParallelEnv example (single-sided and two-sided self-play).'
    )
    parser.add_argument('--preset', default='smac-portable', help='Preset name.')
    parser.add_argument('--scenario', default='2s3z', help='Scenario name (default: 2s3z).')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=('auto', 'zgame', 'pyref'),
        help='Simulation backend (default: auto).',
    )
    parser.add_argument(
        '--profile',
        default=STANDARD_PROFILE,
        help=f'Single-sided profile (default: {STANDARD_PROFILE}; dual mode uses sc2-symmetric).',
    )
    parser.add_argument(
        '--mode',
        default='both',
        choices=('single', 'dual', 'both'),
        help='Which demo mode to run (default: both).',
    )
    parser.add_argument(
        '--dead-agent-mode',
        default='keep',
        choices=('keep', 'remove'),
        help='Dead-agent lifecycle semantics (default: keep).',
    )
    parser.add_argument('--episodes', type=int, default=2, help='Episodes per mode (default: 2).')
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed.')
    return parser.parse_args()


def _first_by_prefix(mapping: dict[str, Any], prefix: str) -> Any:
    for key, value in mapping.items():
        if key.startswith(prefix):
            return value
    return None


def run_demo(make_pettingzoo_env: Any, args: argparse.Namespace, *, dual: bool) -> None:
    label = 'Two-sided self-play (dual=True)' if dual else 'Single-sided cooperative (dual=False)'
    profile = None if dual else args.profile
    env = make_pettingzoo_env(
        args.preset,
        args.scenario,
        backend=args.backend,
        profile=profile,
        dual=dual,
        dead_agent_mode=args.dead_agent_mode,
        seed=args.seed,
    )
    try:
        print(f'=== {label} ===')
        print(
            f'Possible agents : {len(env.possible_agents)} '
            f'({env.possible_agents[0]} .. {env.possible_agents[-1]})'
        )
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            if ep == 0:
                if dual:
                    s1 = env.state_for(1)
                    s2 = env.state_for(2)
                    print(f'Global states   : side1={tuple(s1.shape)}, side2={tuple(s2.shape)}')
                else:
                    print(f'Global state    : {tuple(env.state().shape)}')

            steps = 0
            ally_return = 0.0
            enemy_return = 0.0
            last_infos: dict[str, dict[str, Any]] = {}

            while env.agents:
                actions = {
                    agent: int(env.action_space(agent).sample(mask=obs[agent]['action_mask']))
                    for agent in env.agents
                }
                obs, rewards, terminations, truncations, last_infos = env.step(actions)
                ally_r = _first_by_prefix(rewards, 'ally_')
                if ally_r is not None:
                    ally_return += float(ally_r)
                if dual:
                    enemy_r = _first_by_prefix(rewards, 'enemy_')
                    if enemy_r is not None:
                        enemy_return += float(enemy_r)
                steps += 1
                if not env.agents:
                    any_trunc = any(truncations.values())
                    any_term = any(terminations.values())
                    outcome = 'timeout' if (any_trunc and not any_term) else 'ended'
                    ally_info = _first_by_prefix(last_infos, 'ally_') or {}
                    ally_won = bool(ally_info.get('battle_won', False))
                    if dual:
                        enemy_info = _first_by_prefix(last_infos, 'enemy_') or {}
                        enemy_won = bool(enemy_info.get('battle_won', False))
                        print(
                            f'  Episode {ep + 1:02d}    : steps={steps:3d} ({outcome}), '
                            f'ally_R={ally_return:6.2f} (won={ally_won}), '
                            f'enemy_R={enemy_return:6.2f} (won={enemy_won})'
                        )
                    else:
                        print(
                            f'  Episode {ep + 1:02d}    : steps={steps:3d} ({outcome}), '
                            f'return={ally_return:6.2f}, won={ally_won}'
                        )
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    try:
        from unismac.adapters.pettingzoo import make_pettingzoo_env
    except ImportError as exc:
        print(
            f'PettingZoo adapter is unavailable ({exc}). '
            'Install it with: pip install "unismac[pettingzoo]"',
            file=sys.stderr,
        )
        return 1

    try:
        if args.mode in ('single', 'both'):
            run_demo(make_pettingzoo_env, args, dual=False)
        if args.mode in ('dual', 'both'):
            run_demo(make_pettingzoo_env, args, dual=True)
    except ImportError as exc:
        print(
            f"Backend '{args.backend}' is unavailable ({exc}). "
            "Install 'zfoundry-zgame' or re-run with '--backend pyref'.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
