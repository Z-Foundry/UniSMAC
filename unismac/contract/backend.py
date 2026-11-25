"""Backend protocol and capability declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal, Protocol, runtime_checkable

from .types import Command, Scenario, StaticData, WorldState


class CapabilityError(RuntimeError):
    """Raised when a backend lacks a capability required by a scenario or preset."""


@dataclass(frozen=True)
class Capabilities:
    """Capabilities supported by a backend.

    Presets declare required capabilities and fail fast on construction rather than degrading
    silently.
    """

    name: str
    max_batch: int | None  # max concurrent envs per backend instance, or None if unbounded
    engine_ai: bool  # supports ComputerPlayer
    map_trigger: bool  # supports spawn_mode='map_trigger'
    pathing: Literal['none', 'grid', 'engine']
    collision: Literal['none', 'hard', 'engine']
    terrain_height: bool
    abilities: bool
    fog: bool  # visible reflects true fog of war; otherwise all alive units are visible
    deterministic: bool  # bit-reproducible for the same seed and command sequence
    ability_query: bool = False  # implements AbilityQuery (per-unit available ability queries)
    unit_overrides: bool = False  # guarantees Scenario.unit_overrides caps on reset (set/verify)
    mirror_spawn: bool = False  # supports Scenario.mirror (mirrors Player 2 spawn facing, v0.2)

    def require(self, **needs: object) -> None:
        """Check required capabilities, e.g. `caps.require(engine_ai=True, pathing='grid')`.

        Booleans require True; enum levels require >= target (`none < grid/hard < engine`).
        """
        order = {'none': 0, 'grid': 1, 'hard': 1, 'engine': 2}
        known = {f.name for f in fields(self)}
        for key, want in needs.items():
            if key not in known:
                raise KeyError(f'Unknown capability {key!r}')
            have = getattr(self, key)
            if isinstance(want, bool):
                ok = bool(have) or not want
            elif isinstance(want, str):
                ok = order[str(have)] >= order[want]
            else:
                raise TypeError(f'Requirement for capability {key!r} must be bool or str')
            if not ok:
                raise CapabilityError(f'Backend {self.name} has {key}={have!r}, requires {want!r}')


@runtime_checkable
class Backend(Protocol):
    """Protocol implemented by all simulation backends.

    * `reset` resets only the environments in `env_ids` (default: all) and returns the latest
      state for the full batch. All scenarios in one `reset` call must share the same slot and
      player counts; partial resets must match the current batch shape, whereas full-batch resets
      may change shape (otherwise `ContractError`).
    * `step` applies `command` across the batch and advances `n_loops` game loops; only slots
      owned by an `AgentPlayer` may issue non-NOOP commands.
    * Returned `WorldState` instances are owned by the caller and never mutated afterward.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def static_data(self) -> StaticData: ...

    def reset(
        self,
        scenarios: Sequence[Scenario],
        *,
        env_ids: Sequence[int] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> WorldState: ...

    def step(self, command: Command, n_loops: int) -> WorldState: ...

    def close(self) -> None: ...


@runtime_checkable
class AbilityQuery(Protocol):
    """Optional extension for querying currently available unit abilities.

    Implemented only by backends with `capabilities.ability_query == True`. Returns
    `result[i][slot]`: sorted backend-native ability IDs currently available to that slot in
    `env_ids[i]` (excluding basic commands such as attack/move); dead or empty slots return an
    empty list. Queries inspect the latest `reset`/`step` state without advancing game loops.
    """

    def query_abilities(self, env_ids: Sequence[int] | None = None) -> list[list[list[int]]]: ...
