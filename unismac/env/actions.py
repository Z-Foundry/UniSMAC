"""Action segments: `NoopStop` (0-1), `Move4` (2-5), `FovTurn(n)` (SMACv2 conic FOV, 6..5+n),
and `Target` (one per target thereafter).

Upstream SMAC / SMACv2 semantics:

* Alive units cannot pick no-op and can always stop; movement uses `can_move` (pathable at half
  the move distance).
* FovTurn: always available while alive; sets the agent's facing to canonical direction `i`
  `(cos 2*pi*i/n, sin 2*pi*i/n)` without issuing a backend command.
* Target segment: alive enemies within shoot range (`dist <= shoot_range`, or inside the range
  cone in cone mode). On heal maps (MMM, plus SMACv2 `terran_gen`), Medivacs instead target
  alive non-Medivac allies (indexed by agent ID, heal ability 386). Range checks are skipped in
  the mask when `mask_by_range=False` (SMACv2 `action_mask=False`).
* Target segment length: `target_pad='enemies'` uses `n_enemies`; `'mmm_rule'` uses `n_agents`
  on heal maps when `n_agents > n_enemies` (SMACv2); `'max_units_9'` uses
  `max(n_enemies, n_agents, 9)` (SMAC-Hard).
* Ability segment (`ActionConfig.ability`, SMAC-Hard `use_ability`): an equal-length segment
  after targets; see `unismac.env.abilities`.
* Command translation (`to_command`, in agent index order): `stochastic_attack` draws
  `uniform()` and skips commands above the agent's hit probability; `range_check` skips
  out-of-range attacks.
* Dead units can only pick no-op.

`to_command` translates `[B, A]` actions into a contract `Command`, raising `ValueError` for
unavailable actions (upstream asserts). It belongs to the system layer: this is the Python
reference, while the Rust version lives in `zgame-smac` (`_zgame.env.to_command`) for
`accel='services'`. Masking (`avail`) uses `services.pairwise` and `services.grid_lookup`.
"""

from __future__ import annotations

import numpy as np

from ..contract import NO_SLOT, Command, CommandKind
from . import services
from .abilities import ABILITY_DICT, ability_avail, point_target
from .catalog import AllyTables
from .config import ActionConfig
from .rng import Streams
from .terrain import TerrainGrids
from .units import Frame
from .visibility import cone_visible

__all__ = ['ABILITY_HEAL', 'N_ACTIONS_MOVE', 'N_ACTIONS_NO_ATTACK', 'ActionSpace', 'FovState']

N_ACTIONS_NO_ATTACK = 6  # Excludes FovTurn segment
N_ACTIONS_MOVE = 4
ABILITY_HEAL = 386  # Upstream actions['heal']

# Direction offsets (North, South, East, West) for move actions.
_MOVE = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])


class FovState:
    """Per-agent facing vectors `float64[B, A, 2]`, initialized to `(1, 0)` on reset."""

    def __init__(self, batch_size: int, n_agents: int, n_fov: int) -> None:
        self.n_fov = n_fov
        i = np.arange(max(n_fov, 1))
        self.canonical = np.stack(
            [np.cos(2 * np.pi * i / max(n_fov, 1)), np.sin(2 * np.pi * i / max(n_fov, 1))], -1
        )
        self.angle = 2 * np.pi / n_fov if n_fov else 0.0
        self.dirs = np.zeros((batch_size, n_agents, 2))
        self.dirs[..., 0] = 1.0

    def reset(self, env_ids) -> None:
        ids = list(env_ids)
        self.dirs[ids] = 0.0
        self.dirs[ids, :, 0] = 1.0


class ActionSpace:
    def __init__(
        self,
        n_agents: int,
        n_enemies: int,
        config: ActionConfig,
        *,
        heal_enabled: bool,
        heal_map: bool,
        n_fov: int = 0,
        ops: services.Services = services.PYTHON,
    ) -> None:
        self.ops = ops
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.config = config
        self.heal_enabled = heal_enabled
        self.n_fov = n_fov
        self.n_no_attack = N_ACTIONS_NO_ATTACK + n_fov
        pad_agents = config.target_pad == 'mmm_rule' and heal_map and n_agents > n_enemies
        self.n_targets = n_agents if pad_agents else n_enemies
        if config.target_pad == 'max_units_9':
            self.n_targets = max(n_enemies, n_agents, 9)
        self.n_ability = self.n_targets if config.ability else 0
        self.n_actions = self.n_no_attack + self.n_targets + self.n_ability
        if heal_enabled and n_agents > self.n_targets:
            raise ValueError(
                'Heal targets index agents in target segment; '
                f'requires n_agents <= {self.n_targets}'
            )

    def names(self) -> list[str]:
        out = ['noop', 'stop', 'move_north', 'move_south', 'move_east', 'move_west']
        out += [f'fov_{i}' for i in range(self.n_fov)]
        out += [f'target_{i}' for i in range(self.n_targets)]
        return out + [f'ability_{i}' for i in range(self.n_ability)]

    def heals(self, tables: AllyTables) -> np.ndarray:
        """Return `bool[B, A]` indicating whether each agent's target segment heals allies."""
        return tables.is_medivac if self.heal_enabled else np.zeros_like(tables.is_medivac)

    def _in_range(
        self, frame: Frame, tables: AllyTables, fov: FovState | None, tx, ty
    ) -> np.ndarray:
        """Return `bool[B, A, N]`: targets `(tx, ty)` `[B, N]` within each agent's shoot range
        (or range cone in cone mode)."""
        al = frame.allies
        rng_ = tables.shoot_range[:, :, None]
        if fov is not None and self.n_fov:
            ax, ay = al.x[:, :, None], al.y[:, :, None]
            fx, fy = fov.dirs[..., 0][:, :, None], fov.dirs[..., 1][:, :, None]
            return cone_visible(ax, ay, tx[:, None, :], ty[:, None, :], fx, fy, rng_, fov.angle)
        _, _, dist = self.ops.pairwise(al.x, al.y, tx, ty)
        return dist <= rng_

    def target_ok(
        self, frame: Frame, tables: AllyTables, fov: FovState | None, by_range: bool
    ) -> np.ndarray:
        """Return `bool[B, A, n_targets]`: target eligibility (alive, and within range/cone when
        `by_range=True`)."""
        al, en = frame.allies, frame.enemies
        b_size = al.x.shape[0]
        out = np.zeros((b_size, self.n_agents, self.n_targets), bool)
        attack = np.broadcast_to(en.health[:, None, :] > 0, (b_size, self.n_agents, self.n_enemies))
        if by_range:
            attack = attack & self._in_range(frame, tables, fov, en.x, en.y)
        out[..., : self.n_enemies] = attack
        heals = self.heals(tables)
        if heals.any():
            heal = (al.health[:, None, :] > 0) & ~tables.is_medivac[:, None, :]
            heal = np.broadcast_to(heal, (b_size, self.n_agents, self.n_agents))
            if by_range:
                heal = heal & self._in_range(frame, tables, fov, al.x, al.y)
            pad = np.zeros_like(out)
            pad[..., : self.n_agents] = heal
            out = np.where(heals[..., None], pad, out)
        return out

    def avail(
        self,
        frame: Frame,
        tables: AllyTables,
        terrain: TerrainGrids,
        fov: FovState | None = None,
        *,
        mask_by_range: bool | None = None,
        ability_choice: np.ndarray | None = None,
        map_size: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Return `int64[B, A, n_actions]` available action mask.

        `mask_by_range=None` uses the configured value (upstream `get_avail_actions`); `True`
        matches upstream `get_true_avail_agent_actions`. Ability segments require `ability_choice`
        (`int[B, A]` from `choose_ability`) and `map_size`.
        """
        by_range = self.config.mask_by_range if mask_by_range is None else mask_by_range
        al = frame.allies
        b_size = al.x.shape[0]
        alive = al.health > 0
        out = np.zeros((b_size, self.n_agents, self.n_actions), np.int64)
        out[..., 1] = 1
        out[..., 2:6] = terrain.can_move(al.x, al.y, self.config.move_amount)
        out[..., N_ACTIONS_NO_ATTACK : self.n_no_attack] = 1
        t_end = self.n_no_attack + self.n_targets
        out[..., self.n_no_attack : t_end] = self.target_ok(frame, tables, fov, by_range)
        if self.n_ability:
            if ability_choice is None:
                raise ValueError('Ability segment requires ability_choice')
            out[..., t_end:] = ability_avail(
                ability_choice,
                al.x,
                al.y,
                al.health,
                al.health_max,
                alive,
                self.n_ability,
                map_size,
            )
        dead = np.zeros(self.n_actions, np.int64)
        dead[0] = 1
        return np.where(alive[..., None], out, dead)

    def to_command(
        self,
        actions: np.ndarray,
        active: np.ndarray,
        avail: np.ndarray,
        frame: Frame,
        tables: AllyTables,
        n_slots: int,
        *,
        fov: FovState | None = None,
        attack_prob: np.ndarray | None = None,
        attack_rng: Streams | None = None,
        ability_choice: np.ndarray | None = None,
        out: Command | None = None,
    ) -> Command:
        """Translate `[B, A]` actions into a `Command` for `active` envs.

        Validates availability first and raises `ValueError` on invalid actions. When `out` is
        provided, writes into and returns it (touching only issued slots; used in two-sided mode
        to merge Player 2 commands into Player 1's).

        FovTurn actions update `fov.dirs` in place after range evaluation (matching upstream
        per-agent semantics, since each agent only updates and queries its own facing). Supplying
        both `attack_prob` (`float64[B, A]`) and `attack_rng` (per-env `Streams`) enables
        `stochastic_attack`.

        Rust counterpart: zgame-smac `env::command::to_command` (`unismac.env.native.to_command`).
        """
        b_size = actions.shape[0]
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError(f'invalid action outside [0, {self.n_actions}): {actions.tolist()}')
        ok = np.take_along_axis(avail, actions[..., None], axis=-1)[..., 0] == 1
        bad = np.argwhere(active[:, None] & ~ok)
        if len(bad):
            b, a = (int(v) for v in bad[0])
            raise ValueError(f'env {b} agent {a}: invalid action {int(actions[b, a])}')
        cmd = Command.noop(b_size, n_slots) if out is None else out
        al = frame.allies
        heals = self.heals(tables)
        reach = None
        if self.config.range_check and not self.config.mask_by_range:
            reach = self.target_ok(frame, tables, fov, True)
        rngs: dict[int, np.random.Generator] = {}
        for b, a in np.argwhere(active[:, None] & (actions > 0)):
            b, a = int(b), int(a)
            act = int(actions[b, a])
            slot = int(al.slot[b, a])
            if act == 1:
                cmd.set(b, slot, CommandKind.STOP)
            elif act < N_ACTIONS_NO_ATTACK:
                dx, dy = _MOVE[act - 2] * self.config.move_amount
                cmd.set(b, slot, CommandKind.MOVE, target_xy=(al.x[b, a] + dx, al.y[b, a] + dy))
            elif act < self.n_no_attack:
                assert fov is not None
                fov.dirs[b, a] = fov.canonical[act - N_ACTIONS_NO_ATTACK]
            elif act >= self.n_no_attack + self.n_targets:
                assert ability_choice is not None
                self._ability_command(cmd, frame, b, a, act, int(ability_choice[b, a]))
            else:
                t = act - self.n_no_attack
                if attack_prob is not None and attack_rng is not None:
                    if b not in rngs:
                        rngs[b] = attack_rng(b)
                    if rngs[b].uniform() > attack_prob[b, a]:
                        continue  # Upstream: attack roll missed; issue no command
                if reach is not None and not reach[b, a, t]:
                    continue  # Upstream get_can_shoot is False; issue no command
                if heals[b, a]:
                    target = int(al.slot[b, t])
                    cmd.set(b, slot, CommandKind.ABILITY, target_slot=target, ability=ABILITY_HEAL)
                else:
                    target = int(frame.enemies.slot[b, t])
                    cmd.set(b, slot, CommandKind.ATTACK_UNIT, target_slot=target)
                assert target != NO_SLOT
        return cmd

    def _ability_command(
        self, cmd: Command, frame: Frame, b: int, a: int, act: int, ability: int
    ) -> None:
        al = frame.allies
        slot = int(al.slot[b, a])
        k = act - self.n_no_attack - self.n_targets
        perform, code, _ = ABILITY_DICT[ability]
        if perform == 'self':
            cmd.set(b, slot, CommandKind.ABILITY, ability=code)
        elif perform == 'ally':
            if not al.health[b, k] > 0:
                return  # Upstream targets dead agents (engine rejects); contract forbids it
            cmd.set(b, slot, CommandKind.ABILITY, target_slot=int(al.slot[b, k]), ability=code)
        else:
            xy = point_target(ability, float(al.x[b, a]), float(al.y[b, a]), k)
            cmd.set(b, slot, CommandKind.ABILITY, target_xy=xy, ability=code)
