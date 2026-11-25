"""Unit tests and cross-backend parity tests for `NativeSMACOpponent`.

Covers:
1. `SMAC_TRIGGER_SPECS` coverage across all 23 classic SMAC maps (20 formation-offset
   point maps, `2c_vs_64zg` 32/32 round-robin `ATTACK_UNIT`, `2s_vs_1sc` no opening orders,
   and `2m_vs_1z` deterministic segment `[13, 19] x {12}`);
2. `MMM` / `MMM2` enemy Medivac escort, healing (386), waypoint recovery, and
   `backend.spawn_variant='smac'` Player 1 `Medivac_RL` autocast asymmetry;
3. Acceleration tier selection (`smac-portable` remains `'rust'` when `zgame` is installed;
   `native_smac` resolves to `'services'` or `'python'` cleanly);
4. Full-episode bitwise parity (`L_inf == 0.0`) between `pyref` and `zgame` when `zgame`
   is installed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import fields
from typing import Any, Literal

import numpy as np
import pytest

from unismac.contract import Command, CommandKind, Variant, WorldState
from unismac.env import (
    MULTI_LEVEL_SMAC_MAPS,
    SMAC_TRIGGER_SPECS,
    NativeSMACOpponent,
    VecEnv,
    make_vec_env,
    native_smac_overrides,
)
from unismac.env import (
    accel as accel_mod,
)
from unismac.profiles import STANDARD_PROFILE, merge_overrides, profile_overrides
from unismac.scenarios import names as scenario_names

Q12 = 4096
MEDIVAC_TYPE = 54
HEAL_ABILITY = 386


def _require_zgame() -> None:
    from unismac.backends.zgame.backend import _module

    try:
        _module()
    except ImportError as exc:
        pytest.skip(f'zgame extension unavailable: {exc}')


def _std_overrides(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    _, prof = profile_overrides(STANDARD_PROFILE)
    return merge_overrides(prof, extra)


def _world_bitwise_equal(a: WorldState, b: WorldState) -> None:
    for f in fields(WorldState):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=f'WorldState.{f.name} mismatch')
        else:
            assert va == vb, f'WorldState.{f.name}: {va!r} != {vb!r}'


def _nearest_or_random_actions(env: VecEnv, rng: np.random.Generator) -> np.ndarray:
    avail = env.avail_actions()
    out = np.zeros(avail.shape[:2], dtype=np.int64)
    for b in range(avail.shape[0]):
        for a in range(avail.shape[1]):
            row = avail[b, a]
            atk = np.flatnonzero(row[6:])
            if len(atk):
                out[b, a] = 6 + int(atk[0])
            else:
                choices = np.flatnonzero(row)
                out[b, a] = int(rng.choice(choices))
    return out


def test_smac_trigger_specs_cover_all_23_maps() -> None:
    smac_maps = set(scenario_names('smac'))
    assert len(smac_maps) == 23
    assert set(SMAC_TRIGGER_SPECS) == smac_maps
    cls_counts = Counter(spec.map_class for spec in SMAC_TRIGGER_SPECS.values())
    assert cls_counts == {
        'A': 14,
        'A+': 2,
        'B': 1,
        'C': 1,
        'D': 2,
        'E': 1,
        'F': 1,
        'G': 1,
    }
    assert MULTI_LEVEL_SMAC_MAPS == frozenset({'2c_vs_64zg'})


def test_accel_and_native_smac_overrides() -> None:
    # Classic smac-portable stays eligible for accel='rust' when zgame is available
    classic = make_vec_env(
        'smac-portable', '3m', backend='pyref', seed=0, overrides=_std_overrides()
    )
    try:
        if accel_mod.native_available():
            assert classic.accel in ('rust', 'services')
        else:
            assert classic.accel == 'python'
    finally:
        classic.close()

    # native_smac uses 'services' when zgame env services are available, else 'python'
    backends: list[Literal['pyref', 'zgame']] = ['pyref']
    if accel_mod.native_available():
        backends.append('zgame')
    for be in backends:
        ov = _std_overrides(native_smac_overrides('3m'))
        env = make_vec_env('smac-portable', '3m', backend=be, seed=0, overrides=ov)
        try:
            expected_accel = 'services' if accel_mod.native_available() else 'python'
            assert env.accel == expected_accel
            assert type(env._opponent) is NativeSMACOpponent
        finally:
            env.close()

    # native_smac_overrides parameter contract
    o_flat = native_smac_overrides('3m')
    assert o_flat['opponent.kind'] == 'native_smac'
    assert o_flat['backend.spawn_variant'] == 'smac'
    assert o_flat['backend.pyref_commands'] == 'immediate'
    assert o_flat['backend.pyref_options'] == {'retreat_owners': (), 'passive_from_map': False}

    o_2c_auto = native_smac_overrides('2c_vs_64zg')
    assert o_2c_auto['backend.pyref_options'] == {
        'retreat_owners': (),
        'passive_from_map': False,
        'projectile_3d': True,
    }

    o_2c_full = native_smac_overrides(
        '2c_vs_64zg', terrain_elevation='full', enemy_source='visible'
    )
    assert o_2c_full['backend.pyref_options'] == {
        'retreat_owners': (),
        'passive_from_map': False,
        'visibility': 'sc2',
        'lost_vision': 'sc2',
        'projectile_3d': True,
    }
    assert o_2c_full['units.enemy_source'] == 'visible'

    with pytest.raises(ValueError, match='terrain_elevation'):
        native_smac_overrides('3m', terrain_elevation='invalid')  # type: ignore[arg-type]


def test_all_23_maps_opening_orders() -> None:
    """Verify opening orders produced by `NativeSMACOpponent` across all 23 SMAC maps."""
    for map_name, spec in SMAC_TRIGGER_SPECS.items():
        ov = _std_overrides(native_smac_overrides(map_name))
        env = make_vec_env('smac-portable', map_name, backend='pyref', seed=0, overrides=ov)
        try:
            env.reset()
            opp = env._opponent
            assert isinstance(opp, NativeSMACOpponent)
            w = env.runtime.world
            assert w is not None
            enemies = np.flatnonzero(w.alive[0] & (w.owner[0] == 2)).tolist()
            allies = np.flatnonzero(w.alive[0] & (w.owner[0] == 1)).tolist()
            assert len(enemies) == env.n_enemies, map_name
            assert len(allies) == env.n_agents, map_name

            orders = opp.opening_orders(0)
            if spec.kind == 'none':
                assert map_name == '2s_vs_1sc'
                assert orders == ()
                continue

            if spec.kind == 'round_robin_unit_group':
                assert map_name == '2c_vs_64zg'
                assert len(orders) == 64
                assert all(o[1] == CommandKind.ATTACK_UNIT for o in orders)
                tgt_counts = Counter(o[2] for o in orders)
                assert sorted(tgt_counts.values()) == [32, 32]
                assert set(tgt_counts) == set(allies)
                continue

            if spec.kind == 'random_segment':
                assert map_name == '2m_vs_1z'
                assert len(orders) == len(enemies) == 1
                assert orders[0][1] == CommandKind.ATTACK_MOVE
                tx, ty = orders[0][3] or (-1.0, -1.0)
                assert 13.0 <= tx <= 19.0 and abs(ty - 12.0) < 1e-4
                continue

            # 20 fixed-target maps: Q12 formation offset preservation
            assert spec.kind == 'point' and spec.target_xy is not None
            assert len(orders) == len(enemies), map_name
            exp_tx, exp_ty = spec.target_xy

            x0, y0, x1, y1 = env._playable
            min_x, min_y = round(x0 * Q12), round(y0 * Q12)
            max_x, max_y = round(x1 * Q12), round(y1 * Q12)
            tx_q12, ty_q12 = round(exp_tx * Q12), round(exp_ty * Q12)
            xs_q12 = [round(float(w.x[0, s]) * Q12) for s in enemies]
            ys_q12 = [round(float(w.y[0, s]) * Q12) for s in enemies]
            cx_q12 = sum(xs_q12) // len(enemies)
            cy_q12 = sum(ys_q12) // len(enemies)

            by_slot = {o[0]: o for o in orders}
            max_err = 0.0
            for s, ux_q12, uy_q12 in zip(enemies, xs_q12, ys_q12, strict=True):
                _, kind, _, xy = by_slot[s]
                assert xy is not None
                pred_x = max(min_x, min(max_x, tx_q12 - cx_q12 + ux_q12)) / Q12
                pred_y = max(min_y, min(max_y, ty_q12 - cy_q12 + uy_q12)) / Q12
                err = math.hypot(xy[0] - pred_x, xy[1] - pred_y)
                max_err = max(max_err, err)
                if int(w.unit_type[0, s]) == MEDIVAC_TYPE:
                    assert kind == CommandKind.MOVE, map_name
                else:
                    assert kind == CommandKind.ATTACK_MOVE, map_name

            assert max_err < 1e-3, f'{map_name}: max_err={max_err}'
        finally:
            env.close()


def test_2m_vs_1z_deterministic_and_batch_independent() -> None:
    """`2m_vs_1z` segment target sampling is deterministic and batch-position invariant."""
    ov = _std_overrides(native_smac_overrides('2m_vs_1z'))
    batch = make_vec_env(
        'smac-portable', '2m_vs_1z', backend='pyref', batch_size=3, seeds=(10, 20, 30), overrides=ov
    )
    singles = [
        make_vec_env('smac-portable', '2m_vs_1z', backend='pyref', seed=sd, overrides=ov)
        for sd in (10, 20, 30)
    ]
    try:
        batch.reset()
        b_opp = batch._opponent
        assert isinstance(b_opp, NativeSMACOpponent)
        for i, s_env in enumerate(singles):
            s_env.reset()
            s_opp = s_env._opponent
            assert isinstance(s_opp, NativeSMACOpponent)
            assert b_opp.opening_targets[i] == s_opp.opening_targets[0]
            assert b_opp.opening_orders(i) == s_opp.opening_orders(0)

        # Partial reset of slot 1 leaves slots 0 and 2 untouched
        t0_before, t2_before = b_opp.opening_targets[0], b_opp.opening_targets[2]
        t1_before = b_opp.opening_targets[1]
        batch.reset([1])
        assert b_opp.opening_targets[0] == t0_before
        assert b_opp.opening_targets[2] == t2_before
        assert b_opp.opening_targets[1] != t1_before

        # 16 episodes stay within [13, 19] x {12} and vary across episodes
        xs: list[float] = []
        for _ in range(16):
            singles[0].reset()
            tgt = singles[0]._opponent.opening_targets[0]  # type: ignore[attr-defined]
            assert tgt is not None
            assert 13.0 <= tgt[0] <= 19.0 and abs(tgt[1] - 12.0) < 1e-6
            xs.append(tgt[0])
        assert len(set(xs)) >= 12
    finally:
        batch.close()
        for s_env in singles:
            s_env.close()


@pytest.mark.parametrize('scenario', ['MMM', 'MMM2'])
@pytest.mark.parametrize('backend', ['pyref', 'zgame'])
def test_medivac_escort_and_rl_asymmetry(scenario: str, backend: Literal['pyref', 'zgame']) -> None:
    """Verify `spawn_variant='smac'` disables P1 Medivac autocast while P2 escorts and heals."""
    if backend == 'zgame':
        _require_zgame()
    ov = _std_overrides(native_smac_overrides(scenario))
    env = make_vec_env('smac-portable', scenario, backend=backend, seed=0, overrides=ov)
    try:
        env.reset()
        opp = env._opponent
        assert isinstance(opp, NativeSMACOpponent)
        w = env.runtime.world
        assert w is not None
        p1_meds = np.flatnonzero((w.owner[0] == 1) & (w.unit_type[0] == MEDIVAC_TYPE)).tolist()
        p2_meds = np.flatnonzero((w.owner[0] == 2) & (w.unit_type[0] == MEDIVAC_TYPE)).tolist()
        assert len(p1_meds) == 1 and len(p2_meds) == 1
        m1, m2 = p1_meds[0], p2_meds[0]

        # spawn_variant='smac': only Player 1 Medivac uses Variant.RL; all others use Variant.STD
        assert int(w.variant[0, m1]) == int(Variant.RL)
        assert int(w.variant[0, m2]) == int(Variant.STD)
        for s in np.flatnonzero(w.alive[0]).tolist():
            if s != m1:
                assert int(w.variant[0, s]) == int(Variant.STD)

        # Advance 2 still steps: Player 2 Medivac advances westward with its squad
        m2_x0 = float(w.x[0, m2])

        def _still() -> np.ndarray:
            av = env.avail_actions()
            return np.where(av[..., 1] == 1, 1, 0).astype(np.int64)

        env.step(_still())
        env.step(_still())
        assert env.runtime.world is not None
        assert float(env.runtime.world.x[0, m2]) < m2_x0 - 0.5

        # Player 1 Medivac_RL never autocasts healing while Player 1 holds still
        p1_ever_healed_on_still = False
        for _ in range(18):
            step = env.step(_still())
            w_cur = env.runtime.world
            assert w_cur is not None
            if (
                int(w_cur.order_kind[0, m1]) == CommandKind.ABILITY
                and int(w_cur.order_ability[0, m1]) == HEAL_ABILITY
            ):
                p1_ever_healed_on_still = True
            if step.terminated[0] or step.truncated[0]:
                break
        assert not p1_ever_healed_on_still

        # Under active combat, Player 2 Medivac engages Heal (386)
        env.reset()
        rng = np.random.default_rng(7)
        p2_healed = False
        for _ in range(25):
            step = env.step(_nearest_or_random_actions(env, rng))
            w_cur = env.runtime.world
            assert w_cur is not None
            if (
                int(w_cur.order_kind[0, m2]) == CommandKind.ABILITY
                and int(w_cur.order_ability[0, m2]) == HEAL_ABILITY
            ):
                p2_healed = True
                break
            if step.terminated[0] or step.truncated[0]:
                break
        assert p2_healed

        # Verify escort waypoint resumption when idle and no injured allies are nearby
        env.reset()
        w_cur = env.runtime.world
        assert w_cur is not None
        wp = opp._medivac_wp[0][m2]
        assert not opp._medivac_arrived[0][m2]
        cmd = Command.noop(env.batch_size, w_cur.n_slots)
        opp._pending[0] = False
        opp._step_medivac_escort(w_cur, cmd, 0)
        assert int(cmd.kind[0, m2]) == CommandKind.MOVE
        assert (
            math.hypot(float(cmd.target_x[0, m2]) - wp[0], float(cmd.target_y[0, m2]) - wp[1])
            < 1e-5
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ('scenario', 'terrain_elevation'),
    [
        ('3m', 'auto'),
        ('MMM', 'auto'),
        ('2c_vs_64zg', 'auto'),
        ('2c_vs_64zg', 'full'),
        ('2m_vs_1z', 'auto'),
        ('2s_vs_1sc', 'auto'),
    ],
)
def test_pyref_zgame_bitwise_parity_native_smac(
    scenario: str, terrain_elevation: Literal['auto', 'full']
) -> None:
    """`NativeSMACOpponent` matches bitwise (`L_inf == 0.0`) between `pyref` and `zgame`."""
    _require_zgame()
    ov = _std_overrides(native_smac_overrides(scenario, terrain_elevation=terrain_elevation))
    e_py = make_vec_env('smac-portable', scenario, backend='pyref', seed=5, overrides=ov)
    e_zg = make_vec_env('smac-portable', scenario, backend='zgame', seed=5, overrides=ov)
    try:
        e_py.reset()
        e_zg.reset()
        assert e_py.runtime.world is not None and e_zg.runtime.world is not None
        _world_bitwise_equal(e_py.runtime.world, e_zg.runtime.world)
        np.testing.assert_array_equal(e_py.obs(), e_zg.obs())
        np.testing.assert_array_equal(e_py.state(), e_zg.state())

        rng = np.random.default_rng(13)
        max_steps = 80 if scenario == '2c_vs_64zg' else 120
        for _ in range(max_steps):
            av_py = e_py.avail_actions()
            av_zg = e_zg.avail_actions()
            np.testing.assert_array_equal(av_py, av_zg)
            acts = _nearest_or_random_actions(e_py, rng)
            s_py = e_py.step(acts)
            s_zg = e_zg.step(acts)
            assert e_py.runtime.world is not None and e_zg.runtime.world is not None
            _world_bitwise_equal(e_py.runtime.world, e_zg.runtime.world)
            np.testing.assert_array_equal(s_py.reward, s_zg.reward)
            np.testing.assert_array_equal(s_py.terminated, s_zg.terminated)
            np.testing.assert_array_equal(s_py.truncated, s_zg.truncated)
            np.testing.assert_array_equal(e_py.obs(), e_zg.obs())
            np.testing.assert_array_equal(e_py.state(), e_zg.state())
            if s_py.terminated[0] or s_py.truncated[0]:
                break
    finally:
        e_py.close()
        e_zg.close()
