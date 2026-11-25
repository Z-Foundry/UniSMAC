"""Default trajectory invariance tests for `VecEnv` across `pyref` and `zgame`.

Verifies that default rollout digests (observations, global states, action masks,
rewards, termination flags, infos, reset scenarios, scripted opponent names, and
final stats) match pinned reference SHA-256 hashes on Linux x86_64 with NumPy 2.5.4.
"""

from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import pytest

from unismac.env import make_vec_env
from unismac.profiles import profile_overrides

CASES = [
    ('smac-portable', '3m', None),
    ('smac-portable', '2s3z', None),
    ('smac-portable', 'MMM', None),
    ('smac-portable', '2s3z', 'acc-ram-cov-axo'),
    ('smacv2-portable', '10gen_protoss', None),
    ('smacv2-portable', '10gen_terran', None),
    ('smac-hard-portable', '3m', None),
    ('smac-hard-portable', '2s3z', None),
    ('smac-hard-portable', 'MMM', None),
]

REF = {
    'smac-portable/3m': 'bd0f3c330e95be5978aa4fa0458f045d3d4013603d711984737c7790feea6a7d',
    'smac-portable/2s3z': '65ecc22e4c70354576cd93936057421b603929a72e0162a3b08b5f081bc05e71',
    'smac-portable/MMM': '9256b9accfca4fb8a369553c9c3ce384c3706efe5404455f7b12555cf5978e45',
    'smac-portable/2s3z/acc-ram-cov-axo': (
        '35e0428b02e5c49d0c326a08a61ca19050a4ab2f7fb9e2a9ff6de30ff7f8d7f1'
    ),
    'smacv2-portable/10gen_protoss': (
        '9eba8dfecaea53ae613f103a19f56d57f41e4cbeafed69bfb5a307e730dc988a'
    ),
    'smacv2-portable/10gen_terran': (
        '5448ae8f8315658e843566828840910160974ef62a3f02c8526e9dfad39c2c8a'
    ),
    'smac-hard-portable/3m': '86094c89de9599efac4761b3bed1b455ceb3c976c3e8ea28816d1edbf4cd862c',
    'smac-hard-portable/2s3z': '4b9475787f758ef065b7be753ce5491b39173dc48271347448abac62052256c3',
    'smac-hard-portable/MMM': '7845718850bd3dfdea2e85e688dfc4425f7bd39014cb8847c711237a6ec93df6',
}


def _digest(
    preset: str,
    scenario: str,
    backend: str,
    profile: str | None = None,
    batch: int = 3,
    seed: int = 7,
    n_steps: int = 90,
) -> str:
    over = None if profile is None else profile_overrides(profile)[1]
    env = make_vec_env(
        preset,
        scenario,
        backend=backend,  # type: ignore[arg-type]
        batch_size=batch,
        seed=seed,
        overrides=over,
    )
    h = hashlib.sha256()
    act_rng = np.random.default_rng(1234)

    def put(*arrays: np.ndarray) -> None:
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())

    def put_reset(reset_ids: list[int]) -> None:
        put(env.obs(), env.state(), env.avail_actions())
        names = getattr(env._opponent, 'names', None)
        if names is not None:
            h.update(json.dumps(sorted(names.items())).encode())
        scenarios = env.runtime.last_scenarios()
        h.update(repr([(b, scenarios[b]) for b in sorted(reset_ids)]).encode())

    env.reset()
    put_reset(list(range(batch)))
    for t in range(n_steps):
        avail = env.avail_actions()
        acts = np.array([[act_rng.choice(np.flatnonzero(r)) for r in rows] for rows in avail])
        step = env.step(acts)
        term = step.terminated | step.truncated
        put(step.reward, term, env.obs(), env.state(), env.avail_actions())
        h.update(json.dumps(step.infos, sort_keys=True).encode())
        ids = set(np.flatnonzero(term).tolist())
        if t % 11 == 10:
            ids.add(t % batch)
        if ids:
            order = sorted(ids)
            env.reset(order)
            put_reset(order)
    c = env.runtime.counters()
    won, games, timeouts = c['battles_won'], c['battles_game'], c['timeouts']
    stats = {
        'battles_won': won,
        'battles_game': games,
        'battles_draw': timeouts,
        'win_rate': won / games if games else 0.0,
        'timeouts': timeouts,
        'restarts': 0,
    }
    h.update(json.dumps(stats, sort_keys=True).encode())
    env.close()
    return h.hexdigest()


def _ids() -> list[str]:
    return ['/'.join(p for p in c if p) for c in CASES]


@pytest.mark.skipif(
    platform.system() != 'Linux' or platform.machine() != 'x86_64' or np.__version__ != '2.5.4',
    reason='Reference trajectory digests are pinned for Linux x86_64 with NumPy 2.5.4',
)
@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
@pytest.mark.parametrize(('preset', 'scenario', 'profile'), CASES, ids=_ids())
def test_default_trajectory_unchanged(
    preset: str, scenario: str, profile: str | None, backend: str
) -> None:
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    key = '/'.join(p for p in (preset, scenario, profile) if p)
    assert _digest(preset, scenario, backend, profile) == REF[key]
