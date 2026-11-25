"""Mirror-spawn tests for two-sided self-play mode across `pyref` and `zgame`."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from unismac.assets import load_map_info, load_unit_overrides
from unismac.contract import CapabilityError, ContractError, Scenario, Spawn, mirror_facing
from unismac.contract.types import Variant
from unismac.env import make_vec_env
from unismac.env.dual import mirror_axis, mirror_spawns
from unismac.profiles import profile_overrides
from unismac.scenarios.spawns import load_spawns

EXTERNAL = {'opponent.kind': 'external'}
SYM = ('aligned-v4-sym', 'acc-ram-cov-axo-sym', 'rghf-nat-cbt-sym')

# Mirrorable scenarios (after normalizing spawn records to spawn_variant='std'):
# pinned static check results; re-verify here if detection logic or assets change.
MIRRORABLE = {
    'smac': {
        '3m': 'x',
        '8m': 'x',
        '25m': 'x',
        '2s3z': 'x',
        '3s5z': 'x',
        'MMM': 'x',
        '1c3s5z': 'x',
        'bane_vs_bane': 'y',
    },
    'smac-hard': {
        '3m': 'x',
        '8m': 'x',
        '25m': 'x',
        '2s3z': 'x',
        '3s5z': 'x',
        'MMM': 'x',
        '1c3s5z': 'x',
    },
}


def dual_over(profile: str = 'aligned-v4-sym', **extra: Any) -> dict[str, Any]:
    return {**profile_overrides(profile)[1], **EXTERNAL, **extra}


def std(spawns: Any) -> tuple[Spawn, ...]:
    return tuple(Spawn(**{**vars(s), 'variant': int(Variant.STD)}) for s in spawns)


# ---------------------------------------------------------------- Detection


@pytest.mark.parametrize('benchmark', sorted(MIRRORABLE))
def test_mirrorable_maps_pinned(benchmark: str) -> None:
    got = {}
    for name, rec in load_spawns(benchmark).items():
        axis = mirror_axis(
            std(rec.spawns), load_map_info(rec.map_id), load_unit_overrides(rec.map_id)
        )
        if axis is not None:
            got[name] = axis
    assert got == MIRRORABLE[benchmark]


def test_mirror_axis_rules() -> None:
    rec = load_spawns('smac')['3m']
    info = load_map_info(rec.map_id)
    spawns = std(rec.spawns)
    assert mirror_axis(spawns, info) == 'x'
    # Reject mirror on mismatched rosters, one-sided overrides, off-center COM, or
    # asymmetric terrain
    p2 = [i for i, s in enumerate(spawns) if s.owner == 2]
    other = list(spawns)
    other[p2[0]] = dataclasses.replace(other[p2[0]], unit_type=51)
    assert mirror_axis(other, info) is None
    from unismac.contract import UnitOverride

    assert mirror_axis(spawns, info, (UnitOverride(1, 48, health_max=60.0),)) is None
    both = (UnitOverride(1, 48, health_max=60.0), UnitOverride(2, 48, health_max=60.0))
    assert mirror_axis(spawns, info, both) == 'x'
    shifted = [dataclasses.replace(s, x=s.x + 1.2) if s.owner == 2 else s for s in spawns]
    assert mirror_axis(shifted, info) is None
    lifted = [dataclasses.replace(s, y=s.y + 0.8) if s.owner == 2 else s for s in spawns]
    assert mirror_axis(lifted, info) is None
    path = info.pathing.copy()
    path[10, 5] = not path[10, 5]
    assert mirror_axis(spawns, dataclasses.replace(info, pathing=path)) is None
    # Height on unpathable cells is ignored
    height = info.terrain_height.copy()
    height[~info.pathing] += 1.0
    if (~info.pathing[2:30, 2:30]).any():
        assert mirror_axis(spawns, dataclasses.replace(info, terrain_height=height)) == 'x'


def test_mirror_spawns() -> None:
    a = (Spawn(1, 48, 9.5, 15.0), Spawn(2, 48, 22.0, 16.0), Spawn(1, 51, 8.25, 17.5))
    got = mirror_spawns(a, 'x', (2, 2, 30, 30))
    assert got == (
        Spawn(1, 48, 9.5, 15.0),
        Spawn(1, 51, 8.25, 17.5),
        Spawn(2, 48, 22.5, 15.0),
        Spawn(2, 51, 23.75, 17.5),
    )
    got = mirror_spawns(a, 'y', (2, 2, 30, 30))
    assert [(s.owner, s.x, s.y) for s in got[2:]] == [(2, 9.5, 17.0), (2, 8.25, 14.5)]


# ---------------------------------------------------------------- Contract


def test_mirror_facing() -> None:
    pi32 = float(np.float32(math.pi))
    assert mirror_facing(0.0, 'x') == np.float32(-math.pi)  # pi wraps to -pi
    assert mirror_facing(0.5, 'x') == np.float32(math.pi - 0.5)
    assert mirror_facing(-0.5, 'x') == np.float32(math.pi + 0.5 - 2 * math.pi)
    assert mirror_facing(0.5, 'y') == np.float32(-0.5)
    assert mirror_facing(-pi32, 'y') == np.float32(pi32 - 2 * math.pi)
    assert mirror_facing(0.0, 'y') == 0.0
    for t in np.random.default_rng(0).uniform(-math.pi, math.pi, 200).astype(np.float32):
        for axis in ('x', 'y'):
            m = float(mirror_facing(t, axis))
            assert -math.pi <= m < math.pi
            want = math.pi - t if axis == 'x' else -t
            assert abs(math.remainder(m - want, 2 * math.pi)) < 1e-6


def test_scenario_mirror_contract() -> None:
    a = Scenario('SMAC_Maps/3m', spawns=(Spawn(1, 48, 9, 16), Spawn(2, 48, 23, 16)))
    assert 'mirror' not in repr(a)  # keep default repr unchanged (hashed by invariance tests)
    b = dataclasses.replace(a, mirror='x')
    assert repr(b).endswith("unit_overrides=(), mirror='x')")
    b.validate()
    for bad in (
        dataclasses.replace(a, spawns=a.spawns[::-1], mirror='x'),
        dataclasses.replace(a, spawns=(*a.spawns, Spawn(1, 48, 9, 17)), mirror='x'),
        dataclasses.replace(a, spawns=(Spawn(1, 48, 9, 16), Spawn(2, 51, 23, 16)), mirror='y'),
    ):
        with pytest.raises(ContractError, match='mirror'):
            bad.validate()
    with pytest.raises(ContractError, match="'x'.*'y'"):
        dataclasses.replace(a, mirror='z').validate()  # type: ignore[arg-type]
    with pytest.raises(ContractError, match='explicit'):
        Scenario('SMAC_Maps/3m', spawn_mode='map_trigger', n_slots=6, mirror='x').validate()


def test_sc2_rejects_mirror() -> None:
    from unismac.backends.sc2.backend import CAPABILITIES

    assert not CAPABILITIES.mirror_spawn
    with pytest.raises(CapabilityError, match='mirror_spawn'):
        CAPABILITIES.require(mirror_spawn=True)


# ---------------------------------------------------------------- Env & backends


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_mirror_spawn_in_env(backend: str) -> None:
    """Mirrorable maps reflect Player 2 spawn positions and facings from Player 1."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    env = make_vec_env(
        'smac-portable',
        '3s5z',
        backend=backend,  # type: ignore[arg-type]
        batch_size=3,
        seeds=[1, 2, 3],
        overrides=dual_over('acc-ram-cov-axo-sym'),
    )
    assert env.mirror == 'x' and env._scenario is not None and env._scenario.mirror == 'x'
    env.reset()
    w = env.runtime.world
    assert w is not None
    n = env.n_agents
    assert (w.owner[:, :n] == 1).all() and (w.owner[:, n:] == 2).all()
    np.testing.assert_allclose(w.x[:, n:], 32.0 - w.x[:, :n], atol=4e-6)
    assert w.y[:, n:].tobytes() == w.y[:, :n].tobytes()
    assert len(set(w.facing[:, :n].ravel().tolist())) > 1  # randomized facing
    want = np.vectorize(lambda t: mirror_facing(t, 'x'), otypes=[np.float32])(w.facing[:, :n])
    assert w.facing[:, n:].tobytes() == want.tobytes()
    assert all(s is not None and s.mirror == 'x' for s in env.runtime.last_scenarios())


def test_mirror_facing_same_on_both_backends() -> None:
    from unismac.backends.zgame.backend import _module

    try:
        _module()
    except ImportError as exc:
        pytest.skip(f'zgame extension unavailable: {exc}')
    worlds = []
    for backend in ('pyref', 'zgame'):
        env = make_vec_env(
            'smac-hard-portable',
            'MMM',
            backend=backend,
            batch_size=2,
            seeds=[5, 6],
            overrides=dual_over('acc-ram-cov-axo-sym'),
        )
        env.reset()
        assert env.runtime.world is not None
        worlds.append(env.runtime.world)
    for name in ('x', 'y', 'facing'):
        assert getattr(worlds[0], name).tobytes() == getattr(worlds[1], name).tobytes(), name


def test_mirror_only_where_it_applies() -> None:
    cases = [
        ('smac-portable', '3m', dual_over(**{'backend.spawn_mirror': 'off'}), None),
        ('smac-portable', '3m', profile_overrides('aligned-v4')[1], None),  # not two-sided
        ('smac-portable', '5m_vs_6m', dual_over(), None),  # asymmetric teams
        ('smac-portable', 'bane_vs_bane', dual_over(), 'y'),
        ('smac-hard-portable', '2s3z', dual_over(), 'x'),
        ('smacv2-portable', '10gen_terran', dual_over(), None),  # procedural
    ]
    for preset, scenario, over, want in cases:
        env = make_vec_env(preset, scenario, backend='pyref', batch_size=1, overrides=over)
        assert env.mirror == want, (preset, scenario)
        env.reset()
        sc = env.runtime.last_scenarios()[0]
        assert sc is not None and sc.mirror == want, (preset, scenario)


# ---------------------------------------------------------------- Mirrored initial observations

_SWAP = {'x': ('move_east', 'move_west'), 'y': ('move_north', 'move_south')}


def _blocks(env: Any, side: int, b: int) -> dict[int, dict[tuple[Any, ...], float]]:
    """Unpack side's obs for env b into (block, target_slot, feat) keyed by agent slot."""
    lay = env.obs_layout_for(side)
    obs = env.obs(side)[b]
    f = env.runtime.frame(side)
    allies, enemies = f.allies.slot[b].tolist(), f.enemies.slot[b].tolist()
    out = {}
    for i, s in enumerate(allies):
        it = iter(obs[i].tolist())
        rec: dict[tuple[Any, ...], float] = {}
        for n in lay.move:
            rec[('move', n)] = next(it)
        for t in enemies:
            for n in lay.enemy:
                rec[('enemy', t, n)] = next(it)
        for t in [a for a in allies if a != s]:
            for n in lay.ally:
                rec[('ally', t, n)] = next(it)
        for n in (*lay.own, *lay.timestep):
            rec[('own', n)] = next(it)
        assert next(it, None) is None
        out[s] = rec
    return out


def _mirrored(key: tuple[Any, ...], v: float, axis: str, n: int, scale: float) -> tuple[Any, float]:
    """Map a Player 1 obs entry to Player 2 (swap slot j <-> n + j, mirror dir/rel pos)."""
    name = key[-1]
    if key[0] in ('enemy', 'ally'):
        t = key[1]
        key = (key[0], t - n if t >= n else t + n, name)
    if name in _SWAP[axis]:
        a, c = _SWAP[axis]
        key = (*key[:-1], c if name == a else a)
    if name == f'rel_{axis}':
        v = -v
    if name == f'pos_{axis}':
        v = scale - v
    return key, v


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
@pytest.mark.parametrize('profile', SYM)
@pytest.mark.parametrize(
    ('preset', 'scenario'),
    [
        ('smac-portable', '3m'),
        ('smac-portable', 'MMM'),
        ('smac-portable', '1c3s5z'),
        ('smac-portable', 'bane_vs_bane'),
        ('smac-hard-portable', '2s3z'),
        ('smac-hard-portable', '25m'),
        ('smacv2-portable', '8m'),
    ],
)
def test_start_observation_mirrors(preset: str, scenario: str, profile: str, backend: str) -> None:
    """Initial observations, state, and available actions of Player 2 mirror Player 1."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    env = make_vec_env(
        preset,
        scenario,
        backend=backend,  # type: ignore[arg-type]
        batch_size=2,
        seeds=[3, 4],
        overrides=dual_over(profile),
    )
    axis = env.mirror
    assert axis is not None
    env.reset()
    n = env.n_agents
    lo, hi = env._playable[0 if axis == 'x' else 1], env._playable[2 if axis == 'x' else 3]
    map_size = env._terrain.map_x if axis == 'x' else env._terrain.map_y
    for b in range(2):
        one, two = _blocks(env, 1, b), _blocks(env, 2, b)
        assert sorted(two) == [s + n for s in sorted(one)]
        for s, rec in one.items():
            other = two[s + n]
            assert len(other) == len(rec)
            for key, v in rec.items():
                k2, want = _mirrored(key, v, axis, n, (lo + hi) / map_size)
                assert abs(other[k2] - want) <= 1e-5, (b, s, key, v, other[k2])
        # Available actions: swap move directions and map targets by mirrored slots
        av1, av2 = env.avail_actions(1)[b], env.avail_actions(2)[b]
        f1, f2 = env.runtime.frame(1), env.runtime.frame(2)
        a1, a2 = f1.allies.slot[b].tolist(), f2.allies.slot[b].tolist()
        e1, e2 = f1.enemies.slot[b].tolist(), f2.enemies.slot[b].tolist()
        swap = [0, 1, 2, 3, 5, 4] if axis == 'x' else [0, 1, 3, 2, 4, 5]
        medivac = env._require()[2].is_medivac[b]
        for i, s in enumerate(a1):
            j = a2.index(s + n)
            assert av2[j, :6].tolist() == av1[i, swap].tolist(), (b, i)
            targets = a1 if medivac[i] else e1
            others = a2 if medivac[i] else e2
            for t, slot in enumerate(targets):
                u = others.index(slot + n if slot < n else slot - n)
                assert av2[j, 6 + u] == av1[i, 6 + t], (b, i, t)
        # State: map ally/enemy blocks by mirrored slots and negate rel_<axis>
        lay1 = env.state_layout_for(1)
        st1, st2 = env.state(1)[b], env.state(2)[b]
        k_al, k_en = len(lay1.ally), len(lay1.enemy)
        for i, s in enumerate(a1):
            j = a2.index(s + n)
            x1 = st1[i * k_al : (i + 1) * k_al].copy()
            x2 = st2[j * k_al : (j + 1) * k_al]
            x1[lay1.ally.index(f'rel_{axis}')] *= -1
            np.testing.assert_allclose(x2, x1, atol=1e-5, rtol=0)
        base = n * k_al
        for e, slot in enumerate(e1):
            u = e2.index(slot - n)
            x1 = st1[base + e * k_en : base + (e + 1) * k_en].copy()
            x2 = st2[base + u * k_en : base + (u + 1) * k_en]
            x1[lay1.enemy.index(f'rel_{axis}')] *= -1
            np.testing.assert_allclose(x2, x1, atol=1e-5, rtol=0)
