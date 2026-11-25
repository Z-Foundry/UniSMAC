"""SMACv2 scenario generation (`unismac.env.semantics.generators`): distributions & validation."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from unismac.contract import CapabilityError
from unismac.env import build_config, make_vec_env
from unismac.env.semantics import rng as rng_mod
from unismac.env.semantics.generators import ScenarioGenerator


def _gen(cap: dict, n=5, m=5):
    base = {'n_units': n, 'n_enemies': m}
    cfg, _ = build_config('smacv2', '10gen_terran', {'capability_config': {**base, **cap}})
    return ScenarioGenerator(cfg.capability, n, m), cfg


TERRAN = {
    'unit_types': ['marine', 'marauder', 'medivac'],
    'weights': [0.45, 0.45, 0.1],
    'exception_unit_types': ['medivac'],
}


def _draws(gen, n=400, seed=0):
    return [gen.draw(rng_mod.stream(seed, 'generators', 0, k)) for k in range(n)]


def test_team_never_all_exceptions():
    gen, _ = _gen({'team_gen': {**TERRAN, 'weights': [0.05, 0.05, 0.9]}}, n=2, m=2)
    for d in _draws(gen):
        assert d.ally_team is not None and not all(u == 'medivac' for u in d.ally_team)


@pytest.mark.parametrize(('n', 'm'), [(5, 5), (5, 8), (8, 5)])
def test_enemy_team_shares_ally_prefix(n, m):
    gen, _ = _gen({'team_gen': TERRAN}, n=n, m=m)
    for d in _draws(gen, 100):
        assert len(d.ally_team) == n and len(d.enemy_team) == m
        k = min(n, m)
        assert d.enemy_team[:k] == d.ally_team[:k]
        if m > n:
            assert not all(u == 'medivac' for u in d.enemy_team[n:])


def test_reflect_positions():
    for n, m in ((5, 5), (4, 6), (6, 4)):
        gen, _ = _gen(
            {'team_gen': TERRAN, 'start_positions': {'dist_type': 'reflect_position'}}, n, m
        )
        for d in _draws(gen, 50):
            a, e = d.ally_pos, d.enemy_pos
            assert a.shape == (n, 2) and e.shape == (m, 2)
            assert (a[:, 0] >= 0).all() and (a[:, 0] <= 15).all()
            k = min(n, m)
            np.testing.assert_allclose(e[:k, 0], 32 - a[:k, 0])
            np.testing.assert_allclose(e[:k, 1], a[:k, 1])
            assert (e[k:, 0] >= 16).all()


def test_surrounded_positions():
    gen, _ = _gen({'team_gen': TERRAN, 'start_positions': {'dist_type': 'surrounded'}}, 5, 7)
    for d in _draws(gen, 100):
        np.testing.assert_array_equal(d.ally_pos, np.full((5, 2), 16.0))
        # Enemies lie on the 4 diagonals: |x - 16| == |y - 16| with offset >= 2
        dx, dy = np.abs(d.enemy_pos[:, 0] - 16), np.abs(d.enemy_pos[:, 1] - 16)
        np.testing.assert_allclose(dx, dy)
        assert (dx >= 2 - 1e-9).all() and (dx <= 16 + 1e-9).all()


def test_attack_and_mask():
    gen, _ = _gen(
        {
            'team_gen': TERRAN,
            'attack': {'lower_bound': [0.25], 'upper_bound': [0.75]},
            'enemy_mask': {'mask_probability': 0.2},
        },
        5,
        6,
    )
    draws = _draws(gen, 300)
    probs = np.array([d.attack_prob for d in draws])
    masks = np.array([d.enemy_mask for d in draws])
    assert probs.shape == (300, 5) and ((probs >= 0.25) & (probs < 0.75)).all()
    assert masks.shape == (300, 5, 6) and set(np.unique(masks)) <= {0, 1}
    assert abs(1 - masks.mean() - 0.2) < 0.03  # P(0) == mask_probability


def test_draw_does_not_mutate_config():
    gen, cfg = _gen({'team_gen': TERRAN, 'start_positions': {}})
    before = copy.deepcopy(cfg.capability)
    _draws(gen, 20)
    assert cfg.capability == before


@pytest.mark.parametrize(
    'team_gen',
    [
        {'unit_types': ['marine', 'tank'], 'weights': [0.5, 0.5]},
        {'unit_types': ['marine'], 'weights': [0.5, 0.5]},
        {'unit_types': ['medivac'], 'weights': [1.0], 'exception_unit_types': ['medivac']},
    ],
)
def test_bad_team_gen(team_gen):
    with pytest.raises(ValueError):
        _gen({'team_gen': team_gen})


def test_stochastic_health_unsupported():
    """stochastic_health requires mid-step unit killing, unsupported by contract: reject early."""
    cap = {
        'n_units': 5,
        'n_enemies': 5,
        'team_gen': TERRAN,
        'health': {'lower_bound': [0.0], 'upper_bound': [0.5]},
    }
    with pytest.raises(CapabilityError, match=r'capability\.health'):
        make_vec_env(
            'smacv2-portable', '10gen_terran', backend='pyref', overrides={'capability_config': cap}
        )
