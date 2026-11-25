"""Two-sided self-play mode (`opponent.kind='external'`): API, Player 2 perspective components,
termination and settlement, config/backend option checks, per-env seeds, and partial reset.

Tier bitwise parity (E0, E1) is in test_env_accel.py; mirror spawns and symmetry are in
test_env_mirror.py.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from unismac.adapters.pymarl import make_pymarl_env
from unismac.contract import CapabilityError
from unismac.env import build_config, make_vec_env
from unismac.env.config import RewardConfig
from unismac.env.dual import check_options, seat_catalog, symmetric_options
from unismac.env.semantics import termination as term
from unismac.env.semantics.catalog import AllyTables
from unismac.env.semantics.settle import settle_enemy
from unismac.env.semantics.units import Frame, Side
from unismac.profiles import PROFILES, profile_overrides

EXTERNAL = {'opponent.kind': 'external'}
MARINE, MARAUDER, MEDIVAC, ZEALOT, STALKER = 48, 51, 54, 73, 74


def dual_over(profile: str = 'aligned-v4-sym', **extra: Any) -> dict[str, Any]:
    return {**profile_overrides(profile)[1], **EXTERNAL, **extra}


def dual_env(preset: str, scenario: str, batch: int = 2, **kw: Any) -> Any:
    kw.setdefault('backend', 'pyref')
    kw.setdefault('accel', 'python')
    over = kw.pop('overrides', None) or dual_over()
    return make_vec_env(preset, scenario, batch_size=batch, overrides=over, **kw)


def random_actions(rng: np.random.Generator, avail: np.ndarray) -> np.ndarray:
    return (rng.random(avail.shape) * avail).argmax(-1)


# ---------------------------------------------------------------- API


def test_sym_profiles() -> None:
    """Symmetric profiles treat both sides like Player 1 while keeping all other options equal."""
    v4 = profile_overrides('aligned-v4')[1]['backend.pyref_options']
    v4s = profile_overrides('aligned-v4-sym')[1]['backend.pyref_options']
    assert v4s == {**v4, 'auto_acquire_owners': (), 'retreat_owners': (1, 2)}
    axo = profile_overrides('acc-ram-cov-axo')[1]['backend.pyref_options']
    axos = profile_overrides('acc-ram-cov-axo-sym')[1]['backend.pyref_options']
    sym = {'auto_acquire_owners': (), 'retreat_owners': (1, 2), 'hit_slow_owner1_only': False}
    assert axos == {**axo, **sym}
    cbt = profile_overrides('rghf-nat-cbt')[1]['backend.pyref_options']
    cbts = profile_overrides('rghf-nat-cbt-sym')[1]['backend.pyref_options']
    assert cbts == {**cbt, **sym, 'passive_from_map': False}
    for name in ('aligned-v4-sym', 'acc-ram-cov-axo-sym', 'rghf-nat-cbt-sym'):
        check_options(PROFILES[name].overrides['backend.pyref_options'], 'SMAC_Maps/3m')


def test_non_dual_rejects_side2() -> None:
    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=2, seed=0)
    env.reset()
    obs, state = env.obs(), env.state()
    assert not env.dual and env._p2 is None
    assert np.array_equal(env.obs(1), obs) and np.array_equal(env.state(1), state)
    assert env.n_agents_for(1) == 3 and env.n_actions_for(1) == env.n_actions
    for call in (env.obs, env.state, env.avail_actions, env.alive, env.runtime.frame):
        with pytest.raises(ValueError, match='(?i)two-sided'):
            call(2)
    with pytest.raises(ValueError, match='(?i)side.*1.*2'):
        env.obs(3)
    acts = np.zeros((2, 3), np.int64) + 1
    with pytest.raises(ValueError, match='(?i)enemy_actions.*two-sided'):
        env.step(acts, acts)


def test_dual_interface() -> None:
    env = dual_env('smac-portable', '5m_vs_6m', seed=3)
    env.reset()
    assert env.dual and env.n_agents_for(1) == 5 and env.n_agents_for(2) == 6
    assert (env.n_actions_for(1), env.n_actions_for(2)) == (6 + 6, 6 + 5)
    assert env.obs(2).shape == (2, 6, env.obs_layout_for(2).size)
    assert env.state(2).shape == (2, env.state_layout_for(2).size)
    assert env.avail_actions(2).shape == (2, 6, 11)
    # Player 2 perspective: allies are Player 2's units
    world, f2 = env.runtime.world, env.runtime.frame(2)
    assert world is not None
    rows = np.arange(2)[:, None]
    assert (world.owner[rows, f2.allies.slot] == 2).all()
    assert (world.owner[rows, f2.enemies.slot] == 1).all()
    assert env.alive(2).all() and env.alive().all()
    assert env.runtime.opponent_names() == [None, None]
    assert env.runtime.opponent_errors().tolist() == [0, 0]
    with pytest.raises(ValueError, match='cannot select per-env scripts'):
        env.reset(opponents=['x', None])
    env.reset(opponents=[None, None])
    stop = np.ones((2, 5), np.int64)
    with pytest.raises(ValueError, match='enemy_actions'):
        env.step(stop)
    with pytest.raises(ValueError, match='(?i)enemy_actions.*shape'):
        env.step(stop, np.ones((2, 5), np.int64))
    bad = np.zeros((2, 6), np.int64)  # live units cannot take no-op
    with pytest.raises(ValueError, match='^enemy_actions:'):
        env.step(stop, bad)
    step = env.step(stop, np.ones((2, 6), np.int64))
    assert step.reward_enemy is not None and step.reward_enemy.shape == (2,)
    for info in step.infos:
        assert set(info) == {
            'battle_won',
            'dead_allies',
            'dead_enemies',
            'battle_lost',
            'reward_enemy',
        }
    # PyMARLEnv: same two-sided arguments
    one = make_pymarl_env('smac-portable', '3m', backend='pyref', overrides=dual_over(), seed=1)
    one.reset()
    assert len(one.get_obs(2)) == 3 and len(one.get_avail_actions(2)) == 3
    _, _, info = one.step([1, 1, 1], [1, 1, 1])
    assert 'reward_enemy' in info


def test_last_action_is_per_side() -> None:
    """Player 2's last_action goes into its own state (same layout: after ally/enemy blocks)."""
    env = dual_env('smac-portable', '3m', batch=1, seed=0)
    env.reset()
    a1, a2 = [1, 2, 3], [4, 5, 1]
    env.step(np.array([a1]), np.array([a2]))
    for side, acts in ((1, a1), (2, a2)):
        lay = env.state_layout_for(side)
        assert lay.last_action and not lay.timestep
        k = env.n_actions_for(side)
        tail = env.state(side)[0, lay.size - lay.n_agents * k :]
        assert tail.reshape(lay.n_agents, k).argmax(-1).tolist() == acts
        assert tail.sum() == lay.n_agents


# ---------------------------------------------------------------- Config & backend options


def test_dual_config_checks() -> None:
    with pytest.raises(ValueError, match='enemy_source'):
        make_vec_env('smac', '3m', backend='pyref', overrides=EXTERNAL)
    with pytest.raises(ValueError, match=r'visibility\.model'):
        dual_env('smacv2-portable', '10gen_terran', overrides=dual_over(conic_fov=True))
    v4 = {**profile_overrides('aligned-v4')[1], **EXTERNAL}
    with pytest.raises(ValueError, match='symmetric player options'):
        dual_env('smac-portable', '3m', overrides=v4)
    axo = {**profile_overrides('acc-ram-cov-axo')[1], **EXTERNAL}
    with pytest.raises(ValueError, match='hit_slow_owner1_only'):
        dual_env('smac-portable', '3m', overrides=axo)
    with pytest.raises(CapabilityError, match='(?i)SC2'):
        make_vec_env('smac-portable', '3m', backend='sc2', overrides=EXTERNAL)
    # On MMM2 with 'enemies' target pad, Player 2 Medivacs cannot target all allies (12 > 10)
    with pytest.raises(ValueError, match='(?i)player 2'):
        dual_env('smac-portable', 'MMM2')


def test_check_options() -> None:
    check_options({}, 'SMAC_Maps/3m')
    check_options({'auto_acquire_owners': None, 'retreat_owners': ()}, 'SMAC_Maps/3m')
    check_options({'auto_acquire_owners': (1, 2), 'retreat_owners': (2, 1)}, 'SMAC_Maps/3m')
    for bad in ({'auto_acquire_owners': (2,)}, {'retreat_owners': (1,)}):
        with pytest.raises(ValueError, match='asymmetric fields'):
            check_options(bad, 'SMAC_Maps/3m')
    # Charge: new_maps/8m has it for Player 1 only; mmmt has it for both sides
    with pytest.raises(ValueError, match='charge'):
        check_options({'charge': True}, 'new_maps/8m')
    check_options({'charge': True}, 'new_maps/mmmt')
    with pytest.raises(ValueError, match='passive_from_map'):
        check_options({'passive_from_map': True}, 'new_maps/8m')


def test_passive_from_map_means_both_passive() -> None:
    from unismac.env import _pyref_config

    over = dual_over(**{'backend.pyref_options': {'passive_from_map': True}})
    config, _ = build_config('smac-hard-portable', '8m', over)
    assert symmetric_options({'passive_from_map': True}) == {
        'passive_from_map': False,
        'auto_acquire_owners': (),
        'retreat_owners': (1, 2),
    }
    pc = _pyref_config(config.backend, dual=True)
    assert (pc.passive_from_map, pc.auto_acquire_owners, pc.retreat_owners) == (False, (), (1, 2))
    assert _pyref_config(config.backend).passive_from_map  # unchanged in one-sided mode
    env = dual_env('smac-hard-portable', '8m', batch=1, overrides=over)
    env.reset()
    env.close()
    over_sc2 = dual_over(
        **{'backend.pyref_options': {'targeting': 'sc2', 'passive_from_map': True}}
    )
    try:
        from unismac.backends.zgame.backend import _module

        _module()
    except ImportError:
        return
    env_z = dual_env('smac-hard-portable', '8m', batch=1, backend='zgame', overrides=over_sc2)
    env_z.reset()
    env_z.close()
    # Passing a bare ZgameBackend with passive_from_map=True still fails dual.check_backend
    from unismac.assets import load_map_info, load_static_data
    from unismac.backends.pyref import PyrefConfig
    from unismac.backends.zgame import ZgameBackend

    info = load_map_info('new_maps/8m')
    bare = ZgameBackend(
        load_static_data(),
        {'new_maps/8m': info},
        1,
        PyrefConfig(targeting='sc2', passive_from_map=True),
    )
    try:
        with pytest.raises(ValueError, match='passive_from_map'):
            make_vec_env(
                'smac-hard-portable',
                '8m',
                backend=bare,
                batch_size=1,
                overrides=dual_over(),
            )
    finally:
        bare.close()


def test_seat_catalog_for_asymmetric_maps() -> None:
    """Under smac type encoding, Player 2 types missing from roles get their own role table."""
    env = dual_env('smac-portable', '3s_vs_5z', batch=1, seed=0)
    env.reset()
    assert env._p2 is not None
    assert env._p2.catalog.roles == (ZEALOT,) and env.catalog.roles == (STALKER,)
    tab = env._p2._require()[1]
    assert tab.shield_max[0].tolist() == [50.0] * 5 and tab.cooldown_max[0].tolist() == [22.0] * 5
    # Symmetric map keeps the same catalog
    env = dual_env('smac-portable', 'MMM', batch=1, seed=0)
    env.reset()
    assert env._p2 is not None and env._p2.catalog == env.catalog
    # Type absent from cooldown table (Spine Crawler) falls back to default 15
    cat = seat_catalog(env.catalog, np.array([98]))
    assert cat.roles == (98,)
    assert cat.ally_tables(np.array([[98]])).cooldown_max.tolist() == [[15.0]]


# ---------------------------------------------------------------- Termination & settlement


def _frame(health: list[list[float]], alive: list[list[bool]] | None = None) -> Frame:
    h = np.array(health, np.float64)
    a = h > 0 if alive is None else np.array(alive)
    z = np.zeros_like(h)
    side = Side(
        slot=np.zeros(h.shape, np.int64),
        alive=a,
        unit_type=np.zeros(h.shape, np.int32),
        variant=np.zeros(h.shape, np.int8),
        x=z,
        y=z,
        health=h,
        health_max=z,
        shield=z,
        shield_max=z,
        energy=z,
        cooldown=z,
    )
    return Frame(allies=side, enemies=side)


def _tables(medivac: list[list[bool]]) -> AllyTables:
    m = np.array(medivac)
    z = np.zeros(m.shape)
    return AllyTables(
        type_id=np.zeros(m.shape, np.int64),
        cooldown_max=z,
        shield_max=z,
        is_medivac=m,
        shoot_range=z,
        sight_range=z,
    )


def test_game_end_dual() -> None:
    # Envs 0..3: both alive, Player 1 wiped, Player 2 wiped, mutual wipe
    p1 = _frame([[1, 1], [0, 0], [1, 0], [0, 0]])
    p2 = _frame([[1, 0], [1, 1], [0, 0], [0, 0]])
    no = _tables([[False, False]] * 4)
    got = term.game_end_dual(p1, no, p2, no, medivac_rules=False)
    assert got.tolist() == [term.NONE, term.LOST, term.WON, term.DRAW]
    # Medivac maps: only Medivacs left loses; simultaneous Medivac-only is a draw
    med = _tables([[False, True]] * 4)
    p1 = _frame([[1, 1], [0, 1], [1, 1], [0, 1]])
    p2 = _frame([[1, 1], [1, 1], [0, 1], [0, 1]])
    got = term.game_end_dual(p1, med, p2, med, medivac_rules=True)
    assert got.tolist() == [term.NONE, term.LOST, term.WON, term.DRAW]
    assert term.game_end_dual(p1, med, p2, med, medivac_rules=False).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('scale', [False, True])
def test_settle_enemy(sparse: bool, scale: bool) -> None:
    rc = RewardConfig(sparse=sparse, scale=scale, win=200, defeat=-7, scale_rate=20)
    active = np.array([True, True, True, True, False])
    end = np.array([term.NONE, term.WON, term.LOST, term.DRAW, term.LOST], np.int8)
    reward = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
    max_r = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    infos: list[dict[str, Any]] = [{} for _ in range(5)]
    got = settle_enemy(active, end, reward, max_r, rc, infos)
    want = (
        np.array([1.5, -1.0, 1.0, 4.0, 0.0])
        if sparse
        else np.array([1.5, 2.0 - 7.0, 3.0 + 200.0, 4.0, 0.0])
    )
    if scale:
        want = want / (max_r / 20)
    want[4] = 0.0
    assert got.tolist() == want.tolist()
    assert reward.tolist() == [1.5, 2.0, 3.0, 4.0, 5.0]  # input not mutated
    assert [i.get('battle_lost') for i in infos] == [False, False, True, False, None]
    assert [i.get('reward_enemy') for i in infos[:4]] == got[:4].tolist()
    assert infos[4] == {}


def test_reward_enemy_by_side() -> None:
    """Player 2's reward uses its own perspective (damaging/killing Player 1), with max_reward
    computed from Player 1's units; under symmetric play the winner receives the win bonus."""
    env = dual_env('smac-portable', '3m', batch=1, seed=5)
    env.reset()
    assert env._p2 is not None
    assert env._p2.max_rewards[0] == env.runtime.max_rewards()[0]  # both sides have 3 Marines
    rng = np.random.default_rng(1)
    tot1 = tot2 = 0.0
    done = np.zeros(1, bool)
    infos: list[dict[str, Any]] = [{}]
    for _ in range(200):
        a1 = random_actions(rng, env.avail_actions())
        # Player 2: attack first available target, else random
        av2 = env.avail_actions(2)
        a2 = random_actions(rng, av2)
        hit = av2[..., 6:].argmax(-1)
        a2 = np.where(av2[..., 6:].any(-1), 6 + hit, a2)
        step = env.step(a1, a2)
        assert step.reward_enemy is not None
        tot1 += float(step.reward[0])
        tot2 += float(step.reward_enemy[0])
        assert step.infos[0]['reward_enemy'] == float(step.reward_enemy[0])
        done = step.terminated | step.truncated
        infos = step.infos
        if done[0]:
            break
    assert done[0]
    if infos[0]['battle_lost']:
        assert not infos[0]['battle_won'] and tot2 > tot1
    assert tot2 > 0


# ---------------------------------------------------------------- Per-env seeds & partial reset


@pytest.mark.parametrize(
    ('preset', 'scenario'),
    [('smacv2-portable', '10gen_terran'), ('smac-hard-portable', '2s3z')],
)
def test_per_env_seeds_and_partial_reset(preset: str, scenario: str) -> None:
    """Row b with per-env seeds matches a standalone batch=1 env with seeds[b] bitwise
    (obs, state, avail, reward for both sides); partial resets leave other envs untouched."""
    seeds = [11, 12]
    big = dual_env(preset, scenario, batch=2, seeds=seeds)
    small = [dual_env(preset, scenario, batch=1, seeds=[s]) for s in seeds]
    rng = np.random.default_rng(0)
    big.reset()
    for e in small:
        e.reset()

    def check(where: str) -> None:
        for b, e in enumerate(small):
            for side in (1, 2):
                for get in ('obs', 'state', 'avail_actions'):
                    x = getattr(big, get)(side)[b]
                    y = getattr(e, get)(side)[0]
                    assert x.tobytes() == y.tobytes(), f'{where} env {b} side={side} {get}'

    check('reset')
    for t in range(80):
        a1 = random_actions(rng, big.avail_actions())
        a2 = random_actions(rng, big.avail_actions(2))
        step = big.step(a1, a2)
        d = step.terminated | step.truncated
        for b, e in enumerate(small):
            sb = e.step(a1[b : b + 1], a2[b : b + 1])
            assert (step.reward[b], step.terminated[b], step.truncated[b]) == (
                sb.reward[0],
                sb.terminated[0],
                sb.truncated[0],
            ) and step.infos[b] == sb.infos[0], f'step {t} env {b}'
            assert step.reward_enemy is not None and sb.reward_enemy is not None
            assert step.reward_enemy[b] == sb.reward_enemy[0]
        check(f'step {t}')
        ids = [b for b in range(2) if d[b]]
        if t == 30:
            ids = sorted({*ids, 1})
        if ids:
            big.reset(ids)
            for b in ids:
                small[b].reset()
            check(f'step {t} reset {ids}')
    for b, e in enumerate(small):
        assert e._p2 is not None and big._p2 is not None
        assert big._p2.max_rewards[b] == e._p2.max_rewards[0]


def test_reseed_resets_side2_state() -> None:
    """Re-seeding an env via reset(seeds=...) resets Player 2's max_reward and per-env
    catalog to match a freshly constructed batch=1 env."""
    env = dual_env('smacv2-portable', '10gen_protoss', batch=2, seeds=[1, 2])
    env.reset()
    env.reset([1], seeds=[7])
    one = dual_env('smacv2-portable', '10gen_protoss', batch=1, seeds=[7])
    one.reset()
    assert env.obs(2)[1].tobytes() == one.obs(2)[0].tobytes()
    assert env._p2 is not None and one._p2 is not None
    assert env._p2.max_rewards[1] == one._p2.max_rewards[0]


def test_dataclass_contract_unchanged() -> None:
    """side=1 and one-sided components are unchanged: UnitView defaults to owners=(1, 2)."""
    env = make_vec_env('smac-portable', '3m', backend='pyref', batch_size=1, seed=0)
    assert env._view.owners == (1, 2)
    assert not any(f.name == 'dual' for f in dataclasses.fields(env.config.opponent))


# ---------------------------------------------------------------- MMM termination checks


def _aggressive(env: Any, side: int, rng: np.random.Generator) -> np.ndarray:
    """Attack/heal a random valid target when available, else march toward the opponent."""
    av = env.avail_actions(side)
    tgt = av[..., 6:]
    pick = 6 + (rng.random(tgt.shape) * tgt).argmax(-1)
    move = 4 if side == 1 else 5
    rand = (rng.random(av.shape) * av).argmax(-1)
    walk = np.where(av[..., move] > 0, move, rand)
    return np.where(tgt.any(-1), pick, walk)


@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_mmm_termination_step_by_step(backend: str) -> None:
    """Verify MMM Medivac termination rules step-by-step against world state."""
    if backend == 'zgame':
        from unismac.backends.zgame.backend import _module

        try:
            _module()
        except ImportError as exc:
            pytest.skip(f'zgame extension unavailable: {exc}')
    env = dual_env(
        'smac-portable',
        'MMM',
        batch=8,
        backend=backend,
        seeds=list(range(40, 48)),
        overrides=dual_over('acc-ram-cov-axo-sym'),
    )
    env.reset()
    assert env.catalog.heal_possible
    rng = np.random.default_rng(3)
    seen: dict[str, int] = {'won': 0, 'lost': 0, 'draw': 0, 'timeout': 0, 'medivac_left': 0}
    for _ in range(600):
        active = ~env.runtime.done()
        step = env.step(_aggressive(env, 1, rng), _aggressive(env, 2, rng))
        done = step.terminated | step.truncated
        w = env.runtime.world
        assert w is not None
        ep_steps = env.runtime.episode_steps()
        for b in np.flatnonzero(active):
            fighter = w.alive[b] & (w.unit_type[b] != MEDIVAC)
            out1 = not (fighter & (w.owner[b] == 1)).any()
            out2 = not (fighter & (w.owner[b] == 2)).any()
            timeout = ep_steps[b] >= env.episode_limit
            info = step.infos[b]
            assert bool(done[b]) == (out1 or out2 or timeout), b
            assert bool(step.terminated[b]) == (out1 or out2), b
            assert bool(step.truncated[b]) == (timeout and not (out1 or out2)), b
            assert info['battle_won'] == (out2 and not out1), b
            assert info['battle_lost'] == (out1 and not out2), b
            if done[b]:
                kind = 'draw' if out1 and out2 else 'won' if out2 else 'lost' if out1 else 'timeout'
                seen[kind] += 1
                loser = 2 if kind == 'won' else 1 if kind == 'lost' else 0
                if loser and (w.alive[b] & (w.owner[b] == loser)).any():
                    seen['medivac_left'] += 1  # losing side has only Medivacs left
        ids = np.flatnonzero(done).tolist()
        if ids:
            env.reset(ids)
    assert seen['won'] > 0 and seen['lost'] > 0, seen
    assert seen['medivac_left'] > 0, seen
