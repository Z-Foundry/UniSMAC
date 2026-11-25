"""SC2 backend implementing ``unismac.contract.Backend`` with a StarCraft II client."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...contract import (
    NO_SLOT,
    AgentPlayer,
    Capabilities,
    CapabilityError,
    Command,
    CommandKind,
    ComputerPlayer,
    ContractError,
    Player,
    Scenario,
    StaticData,
    Variant,
    WorldState,
)
from .session import Session

# Native ability IDs for issuing commands (matches SMAC: move=16, attack=23, stop=4, heal=386).
ABILITY_STOP = 4
ABILITY_MOVE = 16
ABILITY_HOLD = 18
ABILITY_ATTACK = 23
ABILITY_HEAL = 386

# Map raw order ability_id -> CommandKind. Observations may report specific IDs or generic IDs
# (GeneralMove 3794, GeneralAttack 3674, Stop 3665, GeneralHoldPosition 3793). Specific IDs cover
# all pysc2 RAW_FUNCTIONS whose general_id points to these four (SC2 4.10), including Battlecruiser
# abilities (3771/3776/3783/3778 since 4.7.1). Attack orders split into ATTACK_UNIT / ATTACK_MOVE
# by target presence; Patrol (17, 3777, 3795) and unlisted IDs map to ABILITY.
_ATTACK = -1
_ORDER_KIND: dict[int, int] = {
    ABILITY_MOVE: CommandKind.MOVE,
    3776: CommandKind.MOVE,  # Move_Battlecruiser
    3794: CommandKind.MOVE,
    ABILITY_ATTACK: _ATTACK,
    19: _ATTACK,  # Scan_Move (attack-move)
    1682: _ATTACK,  # Attack_Redirect
    2048: _ATTACK,  # Attack_AttackBuilding
    3771: _ATTACK,  # Attack_Battlecruiser
    3674: _ATTACK,
    ABILITY_STOP: CommandKind.STOP,
    1691: CommandKind.STOP,  # Stop_Redirect
    2057: CommandKind.STOP,  # Stop_Building
    3783: CommandKind.STOP,  # Stop_Battlecruiser
    3665: CommandKind.STOP,
    ABILITY_HOLD: CommandKind.HOLD,
    3778: CommandKind.HOLD,  # HoldPosition_Battlecruiser
    3793: CommandKind.HOLD,
}

# Typo in SMAC custom unit name: ``Colosus_RL`` corresponds to standard ``Colossus``.
_RL_NAME_FIX = {'Colosus': 'Colossus'}

_ALLIANCE_SELF = 1
_DISPLAY_VISIBLE = 1
_SPAWN_MAX_LOOPS = 16  # explicit: max loops per phase waiting for kill/spawn (~2-3 loops)
_PLACEHOLDER_TYPE = 106  # Overlord: placeholder unit during explicit reset
_TRIGGER_MAX_LOOPS = 512  # map_trigger: max loops waiting for map triggers to spawn units
_TRIGGER_STABLE_LOOPS = 16  # if count < n_slots, done after this many stable loops

CAPABILITIES = Capabilities(
    name='sc2',
    max_batch=1,
    engine_ai=True,
    map_trigger=True,
    pathing='engine',
    collision='engine',
    terrain_height=True,
    abilities=True,
    fog=True,
    deterministic=False,
    ability_query=True,
    unit_overrides=True,  # cannot set caps; verified after reset (_check_unit_overrides)
)

# Basic commands excluded from query_abilities: generic IDs (Attack 3674, Move 3794,
# Patrol 3795, HoldPosition 3793, Stop 3665, Smart 1) and IDs remapping to them.
_BASIC_GENERAL_ABILITIES = frozenset({1, 3665, 3674, 3793, 3794, 3795})


@dataclass
class _Obs:
    """Single observation: per-controller fogged observations plus merged ground truth."""

    loop: int
    fogged: list[Any]  # ResponseObservation per AgentPlayer, ordered like Session.agent_ids
    truth: dict[int, Any]  # tag -> raw Unit, only units owned by players 1..P
    ordered: list[Any]  # units in truth, in engine return order


class Sc2Backend:
    """SC2 backend with ``batch_size=1``.

    **Session**: ``reset`` launches or relaunches SC2 on first call or whenever
    ``(map_id, players)`` changes (see ``session.Session``). ``seeds`` applies as ``random_seed``
    only at game creation; subsequent resets in the same session do not reseed. SC2 is not
    bit-deterministic (``deterministic=False``).

    **Spawning**:

    * ``explicit``: two-phase reset. Phase 1 sends ``DebugKillUnit`` for all old player 1..P units
      together with one placeholder unit per player in a single debug request, then steps loop by
      loop until old units disappear. Phase 2 sends ``DebugCreateUnit`` for ``Scenario.spawns``
      and kills the placeholders in one request, steps until placeholders vanish and all new units
      appear, then matches tags to slots by ``(owner, native_type, nearest_pos)``. Placeholders
      prevent an all-units-dead instant that would trigger respawn logic on maps such as
      ``new_maps/*``. Killing old units and creating new ones in the same request is avoided
      because old units persist for 1-2 loops and displace newly created ground units by
      ~0.31-0.56 tiles. Placeholders are Overlords (flying, unarmed, no autocast, no ground
      collision) placed at the integer grid point inside ``playable_area`` (or full map if empty)
      farthest from all spawns; they are killed at the start of phase 2 and leave no vision.
      When no old units exist (empty map on a fresh session), only phase 2 runs with no
      placeholders. Neutral units (owner 16) are untouched. Melee maps with building-loss
      victory triggers are unsuitable for ``explicit``.
    * ``map_trigger``: units are spawned by map triggers and assigned slots by
      ``(owner, engine_order)`` (``slot_order='engine'``, default) or ``(owner, ascending tag)``
      (``slot_order='tag'``, matching SMAC-Hard scripted opponent tag grouping). Subsequent resets
      kill player units and wait for triggers to respawn them, stepping both controllers in
      lockstep until all old tags disappear and new units reach ``n_slots`` or stabilize. Unlike
      upstream SMAC-Hard (whose restart check does not exclude old tags and can return stale
      units after timeout), stale tags are never reused here.

    **Observation and ground truth**:

    * Each ``AgentPlayer``'s units come from its own controller observation (``alliance=Self`` with
      full fields); dual-agent sessions merge both controllers for full ground truth.
    * In Agent + Computer sessions, Computer units come from an extra
      ``observe(disable_fog=True)`` on player 1's controller. SC2 omits ``orders``,
      ``weapon_cooldown``, and ``engaged_target_tag`` for opponent units even without fog, so
      uncontrolled units take sentinel values: ``weapon_cooldown=NaN``, ``order_kind=NOOP``,
      ``order_ability=0``, ``order_target_slot=NO_SLOT``, ``order_x/y=NaN``,
      ``engaged_target_slot=NO_SLOT``.
    * Death = unit disappears from its owner's ground-truth observation (units loaded into
      transports also disappear and count as dead, matching upstream SMAC); dead units stay dead.
    * ``visible[b, u, p]``: for ``AgentPlayer``, determined from its fogged observation
      (``display_type=Visible``). SC2 does not expose ``ComputerPlayer`` vision: when map fog is
      disabled (observed on ``SMAC_Maps``, where the visibility grid is all 0 yet all units are
      Visible), all alive units are marked visible; otherwise approximated by being within any
      unit's ``sight_range`` circle (ignoring high ground and detection).
    * ``order_kind`` maps from ``orders[0].ability_id``, with unlisted IDs recorded as
      ``ABILITY``; ``order_ability`` always stores the native ID.
    * ``weapon_cooldown`` is taken directly from raw ``Unit.weapon_cooldown`` in game loops
      (decrements by 1 per loop; e.g., Marine fires to ~13.2-14.6, i.e., ``0.861 s * 16`` plus
      random jitter).
    * ``unit_type`` normalization: map-private ``*_RL`` types map to the standard type plus
      ``Variant.RL``.

    **Commands**: only alive slots owned by an ``AgentPlayer`` may issue non-NOOP commands, and
    target slots must be alive. Commands are issued per unit (one ``ActionRawUnitCommand`` each,
    no grouping).

    **Ability query** (``AbilityQuery``): ``query_abilities`` sends
    ``RequestQueryAvailableAbilities`` via each unit owner's controller (using the protocol
    default ``ignore_resource_requirements=True``, matching patched pysc2 in SMAC-Hard) and
    returns sorted native IDs excluding basic commands. Computer units and dead slots return
    empty lists. Observed at t=0: Marine 380, Marauder 253, Medivac 386/394/2116, Stalker 1442
    (when Blink is researched), Queen 1664, etc.

    **Backend-specific attribute** ``last_action_results``: list of
    ``(slot, CommandKind, ActionResult)`` for commands issued in the last ``step``, where
    ``ActionResult`` is a ``s2clientprotocol.error_pb2`` enum value (1 = Success). Engine-rejected
    commands do not raise; e.g., ``ATTACK_UNIT`` on a currently invisible enemy yields
    2 = NotSupported and the unit keeps its previous order.
    """

    def __init__(
        self,
        *,
        static_data: StaticData | None = None,
        slot_order: Literal['engine', 'tag'] = 'engine',
    ) -> None:
        if slot_order not in ('engine', 'tag'):
            raise ValueError(f"slot_order must be 'engine' or 'tag', got {slot_order!r}")
        self._static = static_data
        self.slot_order = slot_order
        self._basic_abilities: frozenset[int] = _BASIC_GENERAL_ABILITIES
        self._session: Session | None = None
        self._key: tuple[str, tuple[Player, ...]] | None = None
        self._scenario: Scenario | None = None
        self._state: WorldState | None = None
        self._tags: list[int] = []
        self._tag_slot: dict[int, int] = {}
        self._fog_disabled = False
        self._native_to_canon: dict[int, tuple[int, int]] = {}
        self._canon_to_native: dict[tuple[int, int], int] = {}
        self._sight: dict[int, float] = {}
        self.last_action_results: list[tuple[int, CommandKind, int]] = []

    # ------------------------------------------------------------ Backend interface

    @property
    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def static_data(self) -> StaticData:
        if self._static is None:
            from ...assets import load_static_data

            self._static = load_static_data()
        return self._static

    @property
    def session(self) -> Session | None:
        """Current SC2 session (None if not started), for extraction tools and debugging."""
        return self._session

    def reset(
        self,
        scenarios: Sequence[Scenario],
        *,
        env_ids: Sequence[int] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> WorldState:
        ids = [0] if env_ids is None else [int(i) for i in env_ids]
        if any(i != 0 for i in ids) or len(set(ids)) != len(ids):
            raise ContractError(f'SC2 backend has batch_size=1; env_ids must be [0], got {ids}')
        if len(scenarios) != len(ids):
            raise ContractError(f'scenarios length {len(scenarios)} does not match env_ids {ids}')
        if seeds is not None and len(seeds) != len(ids):
            raise ContractError(f'seeds length {len(seeds)} does not match env_ids {ids}')
        if not ids:
            return self._require_state().copy()

        scenario = scenarios[0]
        scenario.validate()
        if scenario.mirror is not None:
            raise CapabilityError('Backend sc2 does not support mirror spawns (Scenario.mirror)')
        seed = seeds[0] if seeds is not None else None
        key = (scenario.map_id, tuple(scenario.players))
        fresh = self._session is None or self._session.closed or self._key != key
        if fresh:
            self._launch(scenario, seed)
        self._scenario = scenario
        if scenario.spawn_mode == 'explicit':
            ob = self._reset_explicit(scenario)
        else:
            ob = self._reset_map_trigger(scenario, fresh=fresh)
        self._fog_disabled = not any(
            np.any(_grid(o.observation.raw_data.map_state.visibility) == 2) for o in ob.fogged
        )
        self._update_state(ob)
        self._check_unit_overrides(scenario)
        return self._require_state().copy()

    def step(self, command: Command, n_loops: int) -> WorldState:
        state = self._require_state()
        session = self._require_session()
        n_slots = state.n_slots
        command.validate(1, n_slots)
        if int(n_loops) != n_loops or n_loops < 1:
            raise ContractError(f'n_loops must be a positive integer, got {n_loops!r}')

        from s2clientprotocol import sc2api_pb2 as sc_pb

        controller_of = {pid: k for k, pid in enumerate(session.agent_ids)}
        requests = [sc_pb.RequestAction() for _ in session.controllers]
        issued: list[list[tuple[int, int]]] = [[] for _ in session.controllers]
        for slot in range(n_slots):
            kind = int(command.kind[0, slot])
            if kind == CommandKind.NOOP:
                continue
            if not state.alive[0, slot]:
                raise ContractError(f'Slot {slot} is dead or empty and cannot receive commands')
            owner = int(state.owner[0, slot])
            if owner not in controller_of:
                raise ContractError(
                    f'Slot {slot} belongs to non-AgentPlayer {owner} and cannot be commanded'
                )
            k = controller_of[owner]
            raw = requests[k].actions.add().action_raw.unit_command
            raw.unit_tags.append(self._tags[slot])
            self._fill_command(raw, command, slot, kind)
            issued[k].append((slot, kind))

        results = session.actions(requests)
        # Session.actions skips empty requests and returns results for non-empty controllers.
        self.last_action_results = []
        sent = [cmds for cmds in issued if cmds]
        for cmds, response in zip(sent, results, strict=True):
            for (slot, kind), result in zip(cmds, response.result, strict=True):
                self.last_action_results.append((slot, CommandKind(kind), int(result)))
        session.step(int(n_loops))
        self._update_state(self._observe())
        return state.copy()

    def query_abilities(self, env_ids: Sequence[int] | None = None) -> list[list[list[int]]]:
        from s2clientprotocol import query_pb2 as q_pb

        ids = [0] if env_ids is None else [int(i) for i in env_ids]
        if any(i != 0 for i in ids):
            raise ContractError(f'SC2 backend has batch_size=1; env_ids must be [0], got {ids}')
        state = self._require_state()
        session = self._require_session()
        controller_of = {pid: k for k, pid in enumerate(session.agent_ids)}
        result: list[list[int]] = [[] for _ in range(state.n_slots)]
        slots_of: list[list[int]] = [[] for _ in session.controllers]
        for slot in range(state.n_slots):
            owner = int(state.owner[0, slot])
            if state.alive[0, slot] and owner in controller_of:
                slots_of[controller_of[owner]].append(slot)
        for k, slots in enumerate(slots_of):
            if not slots:
                continue
            query = q_pb.RequestQuery()
            for slot in slots:
                query.abilities.add().unit_tag = self._tags[slot]
            response = session.controllers[k].query(query)
            by_tag = {a.unit_tag: a.abilities for a in response.abilities}
            for slot in slots:
                abilities = by_tag.get(self._tags[slot], ())
                result[slot] = sorted(
                    {int(a.ability_id) for a in abilities} - self._basic_abilities
                )
        return [[list(a) for a in result] for _ in ids]

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._key = None
        self._state = None

    # ------------------------------------------------------------ Session & type maps

    def _launch(self, scenario: Scenario, seed: int | None) -> None:
        self.close()
        self._session = Session(scenario.map_id, scenario.players, seed=seed)
        self._key = (scenario.map_id, tuple(scenario.players))
        self._build_type_maps(self._session.data_raw)

    def _build_type_maps(self, data_raw: Any) -> None:
        by_name: dict[str, int] = {}
        for u in data_raw.units:
            if u.name:
                by_name.setdefault(u.name, u.unit_id)
        self._native_to_canon = {}
        for u in data_raw.units:
            if u.name.endswith('_RL'):
                base = u.name[: -len('_RL')]
                std = by_name.get(_RL_NAME_FIX.get(base, base))
                if std is not None and std != u.unit_id:
                    self._native_to_canon[u.unit_id] = (std, int(Variant.RL))
        self._canon_to_native = {v: k for k, v in self._native_to_canon.items()}
        self._sight = {u.unit_id: float(u.sight_range) for u in data_raw.units}
        self._basic_abilities = _BASIC_GENERAL_ABILITIES | frozenset(
            int(a.ability_id)
            for a in data_raw.abilities
            if int(a.remaps_to_ability_id) in _BASIC_GENERAL_ABILITIES
        )

    def canonical_type(self, native: int) -> tuple[int, int]:
        """Map native unit_type -> (standard UnitTypeId, variant)."""
        return self._native_to_canon.get(native, (native, int(Variant.STD)))

    def native_type(self, unit_type: int, variant: int = Variant.STD) -> int:
        """Map (standard UnitTypeId, variant) -> current map's native unit_type."""
        if variant == Variant.STD:
            return int(unit_type)
        try:
            return self._canon_to_native[(int(unit_type), int(variant))]
        except KeyError:
            raise ContractError(
                f'Current map has no native type for unit ({unit_type}, variant={variant})'
            ) from None

    def _require_session(self) -> Session:
        if self._session is None or self._session.closed:
            raise RuntimeError('SC2 session is not started; call reset() first')
        return self._session

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise RuntimeError('Must call reset() first')
        return self._state

    # ------------------------------------------------------------ Observation

    def _observe(self) -> _Obs:
        session = self._require_session()
        fogged = session.observe()
        loops = {int(o.observation.game_loop) for o in fogged}
        if len(loops) != 1:
            raise RuntimeError(f'Controller game_loop mismatch: {sorted(loops)}')
        for o in fogged:
            if len(o.player_result) > 0:
                raise RuntimeError(f'SC2 game ended: {list(o.player_result)}')
        n_players = len(session.players)
        truth: dict[int, Any] = {}
        ordered: list[Any] = []
        if session.has_computer:
            # Unfogged observation provides all units in engine order; player units match
            # the fogged observation, while Computer units are only available here.
            full = session.controllers[0].observe(disable_fog=True)
            for u in full.observation.raw_data.units:
                if 1 <= u.owner <= n_players:
                    truth[u.tag] = u
                    ordered.append(u)
        else:
            for o in fogged:
                for u in o.observation.raw_data.units:
                    if u.alliance == _ALLIANCE_SELF and 1 <= u.owner <= n_players:
                        truth[u.tag] = u
                        ordered.append(u)
        return _Obs(loop=loops.pop(), fogged=fogged, truth=truth, ordered=ordered)

    def _update_state(self, ob: _Obs) -> None:
        state = self._require_state()
        session = self._require_session()
        state.game_loop[0] = ob.loop
        controllable = set(session.agent_ids)
        for slot, tag in enumerate(self._tags):
            if tag == 0 or not state.alive[0, slot]:
                continue
            u = ob.truth.get(tag)
            if u is None:  # disappeared from owner observation = dead
                state.alive[0, slot] = False
                state.health[0, slot] = 0.0
                continue
            self._write_unit(state, slot, u, u.owner in controllable)
        self._write_visibility(state, ob)

    def _write_unit(self, state: WorldState, slot: int, u: Any, controllable: bool) -> None:
        i = (0, slot)
        std, variant = self.canonical_type(u.unit_type)
        state.owner[i] = u.owner
        state.unit_type[i] = std
        state.variant[i] = variant
        state.x[i] = u.pos.x
        state.y[i] = u.pos.y
        state.facing[i] = u.facing
        state.radius[i] = u.radius
        state.health[i] = u.health
        state.health_max[i] = u.health_max
        state.shield[i] = u.shield
        state.shield_max[i] = u.shield_max
        state.energy[i] = u.energy
        state.energy_max[i] = u.energy_max
        state.is_flying[i] = u.is_flying
        if not controllable:
            state.weapon_cooldown[i] = np.nan
            state.order_kind[i] = CommandKind.NOOP
            state.order_ability[i] = 0
            state.order_target_slot[i] = NO_SLOT
            state.order_x[i] = np.nan
            state.order_y[i] = np.nan
            state.engaged_target_slot[i] = NO_SLOT
            return
        state.weapon_cooldown[i] = u.weapon_cooldown
        state.engaged_target_slot[i] = (
            self._tag_slot.get(u.engaged_target_tag, NO_SLOT) if u.engaged_target_tag else NO_SLOT
        )
        if len(u.orders) == 0:
            state.order_kind[i] = CommandKind.NOOP
            state.order_ability[i] = 0
            state.order_target_slot[i] = NO_SLOT
            state.order_x[i] = np.nan
            state.order_y[i] = np.nan
            return
        order = u.orders[0]
        kind = _ORDER_KIND.get(order.ability_id, CommandKind.ABILITY)
        if kind == _ATTACK:
            kind = CommandKind.ATTACK_UNIT if order.target_unit_tag else CommandKind.ATTACK_MOVE
        state.order_kind[i] = kind
        state.order_ability[i] = order.ability_id
        state.order_target_slot[i] = (
            self._tag_slot.get(order.target_unit_tag, NO_SLOT) if order.target_unit_tag else NO_SLOT
        )
        if order.HasField('target_world_space_pos'):
            state.order_x[i] = order.target_world_space_pos.x
            state.order_y[i] = order.target_world_space_pos.y
        else:
            state.order_x[i] = np.nan
            state.order_y[i] = np.nan

    def _write_visibility(self, state: WorldState, ob: _Obs) -> None:
        session = self._require_session()
        state.visible[...] = False
        alive = state.alive[0]
        for k, pid in enumerate(session.agent_ids):
            seen = {
                u.tag
                for u in ob.fogged[k].observation.raw_data.units
                if u.display_type == _DISPLAY_VISIBLE
            }
            for slot, tag in enumerate(self._tags):
                state.visible[0, slot, pid - 1] = bool(alive[slot]) and tag in seen
        for idx, player in enumerate(session.players):
            if not isinstance(player, ComputerPlayer):
                continue
            pid = idx + 1
            if self._fog_disabled:
                state.visible[0, :, pid - 1] = alive
                continue
            watchers = [
                (u.pos.x, u.pos.y, self._sight.get(u.unit_type, 0.0))
                for u in ob.truth.values()
                if u.owner == pid
            ]
            for slot in range(len(self._tags)):
                if not alive[slot]:
                    continue
                if state.owner[0, slot] == pid:
                    state.visible[0, slot, pid - 1] = True
                    continue
                x, y = float(state.x[0, slot]), float(state.y[0, slot])
                state.visible[0, slot, pid - 1] = any(
                    math.hypot(x - wx, y - wy) <= r for wx, wy, r in watchers
                )

    # ------------------------------------------------------------ Reset

    def _check_unit_overrides(self, scenario: Scenario) -> None:
        """Verify spawned unit health_max / shield_max against ``Scenario.unit_overrides``.

        SC2 cannot set unit stat caps programmatically, so any mismatch raises
        ``ContractError`` (override asset does not match the map or SC2 version).
        """
        if not scenario.unit_overrides:
            return
        state = self._require_state()
        for o in scenario.unit_overrides:
            match = (
                state.alive[0]
                & (state.owner[0] == o.owner)
                & (state.unit_type[0] == o.unit_type)
                & (state.variant[0] == o.variant)
            )
            for name, want in (('health_max', o.health_max), ('shield_max', o.shield_max)):
                if want is None:
                    continue
                got = getattr(state, name)[0, match]
                bad = sorted({float(v) for v in got if abs(float(v) - want) > 1e-3})
                if bad:
                    raise ContractError(
                        f'{scenario.map_id}: player {o.owner} unit ({o.unit_type}, {o.variant}) '
                        f'{name} is {bad}, expected {want} from unit_overrides; '
                        'override asset does not match map or SC2 version'
                    )

    def _new_state(self, scenario: Scenario, tags: list[int]) -> None:
        n_slots = scenario.slot_count()
        self._tags = tags + [0] * (n_slots - len(tags))
        self._tag_slot = {t: s for s, t in enumerate(self._tags) if t}
        self._state = WorldState.empty(1, n_slots, len(scenario.players))
        for slot, tag in enumerate(self._tags):
            self._state.alive[0, slot] = tag != 0

    def _player_tags(self, ob: _Obs) -> list[int]:
        return [u.tag for u in ob.ordered]

    def _placeholder_point(self, scenario: Scenario) -> tuple[float, float]:
        """Placeholder position: grid point inset 1 tile in playable area farthest from spawns."""
        raw = self._require_session().game_info.start_raw
        p0, p1 = raw.playable_area.p0, raw.playable_area.p1
        x0, y0, x1, y1 = float(p0.x), float(p0.y), float(p1.x), float(p1.y)
        if x1 <= x0 or y1 <= y0:  # map without playable_area: fall back to full map
            x0, y0, x1, y1 = 0.0, 0.0, float(raw.map_size.x), float(raw.map_size.y)
        xs = np.arange(math.ceil(x0 + 1), math.floor(x1 - 1) + 1, dtype=np.float64)
        ys = np.arange(math.ceil(y0 + 1), math.floor(y1 - 1) + 1, dtype=np.float64)
        if not len(xs) or not len(ys):
            return (x0 + x1) / 2, (y0 + y1) / 2
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        if not scenario.spawns:
            return float(gx.flat[0]), float(gy.flat[0])
        sp = np.array([(s.x, s.y) for s in scenario.spawns], dtype=np.float64)
        d = np.hypot(gx[..., None] - sp[:, 0], gy[..., None] - sp[:, 1]).min(axis=-1)
        k = int(np.argmax(d))
        return float(gx.flat[k]), float(gy.flat[k])

    def _reset_explicit(self, scenario: Scenario) -> _Obs:
        from s2clientprotocol import common_pb2 as sc_common
        from s2clientprotocol import debug_pb2 as d_pb

        session = self._require_session()
        natives = [self.native_type(s.unit_type, s.variant) for s in scenario.spawns]

        def create(native: int, owner: int, x: float, y: float) -> Any:
            return d_pb.DebugCommand(
                create_unit=d_pb.DebugCreateUnit(
                    unit_type=native, owner=owner, pos=sc_common.Point2D(x=x, y=y), quantity=1
                )
            )

        def kill(tags: set[int]) -> Any:
            return d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=sorted(tags)))

        # Phase 1: kill old units, spawn one placeholder per player, wait for old units to die.
        ob = self._observe()
        old = set(self._player_tags(ob))
        holders: set[int] = set()
        if old:
            owners = sorted({int(u.owner) for u in ob.ordered} | {s.owner for s in scenario.spawns})
            px, py = self._placeholder_point(scenario)
            session.debug([kill(old), *(create(_PLACEHOLDER_TYPE, o, px, py) for o in owners)])
            for _ in range(_SPAWN_MAX_LOOPS):
                session.step(1)
                ob = self._observe()
                holders = {u.tag for u in ob.ordered if u.tag not in old}
                if not old.intersection(ob.truth) and len(holders) >= len(owners):
                    break
            else:
                raise RuntimeError(
                    f'Old units did not vanish or placeholders did not spawn within '
                    f'{_SPAWN_MAX_LOOPS} loops'
                )

        # Phase 2: create new units and kill placeholders, then wait for placeholders to vanish
        # and all new units to appear.
        commands = [
            create(n, s.owner, s.x, s.y) for s, n in zip(scenario.spawns, natives, strict=True)
        ]
        if holders:
            commands.append(kill(holders))
        session.debug(commands)
        gone = old | holders
        expected = Counter(zip((s.owner for s in scenario.spawns), natives, strict=True))
        for _ in range(_SPAWN_MAX_LOOPS):
            session.step(1)
            ob = self._observe()
            fresh = [u for u in ob.ordered if u.tag not in gone]
            got = Counter((u.owner, u.unit_type) for u in fresh)
            if not gone.intersection(ob.truth) and all(got[k] >= v for k, v in expected.items()):
                break
        else:
            raise RuntimeError(
                f'Reset cleanup/spawn did not complete within {_SPAWN_MAX_LOOPS} loops'
            )

        # Group by (owner, native_type) and greedily match tags to slots by ascending distance.
        pairs = []
        for slot, (spawn, native) in enumerate(zip(scenario.spawns, natives, strict=True)):
            for u in fresh:
                if u.owner == spawn.owner and u.unit_type == native:
                    pairs.append((math.hypot(u.pos.x - spawn.x, u.pos.y - spawn.y), slot, u.tag))
        pairs.sort()
        tags = [0] * len(scenario.spawns)
        used: set[int] = set()
        for _, slot, tag in pairs:
            if tags[slot] == 0 and tag not in used:
                tags[slot] = tag
                used.add(tag)
        self._new_state(scenario, tags)
        return ob

    def _reset_map_trigger(self, scenario: Scenario, *, fresh: bool) -> _Obs:
        from s2clientprotocol import debug_pb2 as d_pb

        session = self._require_session()
        n_slots = scenario.slot_count()
        old: set[int] = set()
        if not fresh:
            old = set(self._player_tags(self._observe()))
            if old:
                session.debug([d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=sorted(old)))])
                session.step(1)
        last_n, stable = -1, 0
        for _ in range(_TRIGGER_MAX_LOOPS):
            ob = self._observe()
            units = [u for u in ob.ordered if u.tag not in old]
            if len(units) > n_slots:
                raise ContractError(f'Map spawned {len(units)} units, exceeding n_slots={n_slots}')
            if not old.intersection(ob.truth):
                if len(units) == n_slots:
                    break
                stable = stable + 1 if len(units) == last_n and units else 0
                if stable >= _TRIGGER_STABLE_LOOPS:
                    break
            last_n = len(units)
            session.step(1)
        else:
            raise RuntimeError(
                f'Map triggers did not spawn units within {_TRIGGER_MAX_LOOPS} loops'
            )
        if self.slot_order == 'tag':
            units.sort(key=lambda u: (u.owner, u.tag))
        else:
            units.sort(key=lambda u: u.owner)  # stable sort: (owner, engine return order)
        self._new_state(scenario, [u.tag for u in units])
        return ob

    # ------------------------------------------------------------ Commands

    def _target_tag(self, slot: int, target: int) -> int:
        state = self._require_state()
        if not 0 <= target < state.n_slots:
            raise ContractError(f'Slot {slot}: target_slot={target} out of range')
        if not state.alive[0, target]:
            raise ContractError(f'Slot {slot}: target slot {target} is dead or empty')
        return self._tags[target]

    @staticmethod
    def _point(command: Command, slot: int) -> tuple[float, float]:
        x, y = float(command.target_x[0, slot]), float(command.target_y[0, slot])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ContractError(
                f'Slot {slot}: command requires finite target point, got ({x}, {y})'
            )
        return x, y

    def _fill_command(self, raw: Any, command: Command, slot: int, kind: int) -> None:
        raw.queue_command = False
        if kind == CommandKind.STOP:
            raw.ability_id = ABILITY_STOP
        elif kind == CommandKind.HOLD:
            raw.ability_id = ABILITY_HOLD
        elif kind in (CommandKind.MOVE, CommandKind.ATTACK_MOVE):
            raw.ability_id = ABILITY_MOVE if kind == CommandKind.MOVE else ABILITY_ATTACK
            raw.target_world_space_pos.x, raw.target_world_space_pos.y = self._point(command, slot)
        elif kind == CommandKind.ATTACK_UNIT:
            raw.ability_id = ABILITY_ATTACK
            raw.target_unit_tag = self._target_tag(slot, int(command.target_slot[0, slot]))
        elif kind == CommandKind.ABILITY:
            ability = int(command.ability[0, slot])
            if ability <= 0:
                raise ContractError(f'Slot {slot}: ABILITY command missing ability id')
            raw.ability_id = ability
            target = int(command.target_slot[0, slot])
            if target != NO_SLOT:
                raw.target_unit_tag = self._target_tag(slot, target)
            elif math.isfinite(float(command.target_x[0, slot])):
                raw.target_world_space_pos.x, raw.target_world_space_pos.y = self._point(
                    command, slot
                )
        else:  # pragma: no cover - already rejected by Command.validate
            raise ContractError(f'Unknown command kind {kind}')


def _grid(image: Any) -> np.ndarray:
    """Return raw bytes of ``ImageData`` (8 bpp), or bit-unpacked 1D array for 1 bpp."""
    data = np.frombuffer(image.data, dtype=np.uint8)
    if image.bits_per_pixel == 1:
        return np.unpackbits(data)
    return data


__all__ = ['CAPABILITIES', 'AgentPlayer', 'Sc2Backend']
