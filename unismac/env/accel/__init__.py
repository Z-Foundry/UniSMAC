"""Acceleration tiers and kernel dispatch for the UniSMAC environment layer.

``accel`` controls how much of the environment layer executes in Rust via
``zgame._zgame.env``:

* ``'python'``: Pure-Python / NumPy reference implementation.
* ``'services'``: Rust system-layer kernels (command translation, unit view
  framing, row copying, settlement, unit catalog tables) and numerical services
  (``pairwise``, ``grid_lookup``, ``ordered_sum``); semantic layer stays in Python.
* ``'rust'``: Full Rust system and semantic layer for classic recipes
  (`unismac.env.accel.classic`), validated against `env/accel/rust_manifest.json`.
* ``'auto'`` (default): Selects ``'rust'`` when eligible and verified; else
  ``'services'`` when ``zgame._zgame.env`` is available; else ``'python'``.
  Emits a single ``RuntimeWarning`` if a classic recipe falls back to
  ``'services'`` due to a manifest mismatch.

All tiers produce bitwise-identical observations, states, action masks, rewards,
termination flags, and infos.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..semantics import features, generators, termination
from ..semantics import services as py_services
from ..semantics.actions import ActionSpace
from ..semantics.catalog import UnitCatalog, ally_tables_by_env
from ..semantics.reward import fill_max_rewards
from ..semantics.settle import settle, settle_enemy
from ..semantics.units import UnitView, advance, copy_rows
from . import classic

if TYPE_CHECKING:
    from ..config import EnvConfig

__all__ = ['TIERS', 'Kernels', 'kernels', 'native_available', 'resolve', 'rust_blockers']

TIERS = ('auto', 'python', 'services', 'rust')


@dataclass(frozen=True)
class Kernels:
    """Bundled system-layer, semantic-layer, and numerical service kernels."""

    tier: str
    services: py_services.Services
    to_command: Callable[..., Any]
    assign: Callable[..., Any]
    frame: Callable[..., Any]
    copy_rows: Callable[..., Any]
    advance: Callable[..., Any]
    ally_tables: Callable[..., Any]
    enemy_tables: Callable[..., Any]
    fill_max_rewards: Callable[..., Any]
    settle: Callable[..., Any]
    settle_enemy: Callable[..., Any]
    draw: Callable[..., Any]
    avail: Callable[..., Any]
    observe_spotter: Callable[..., Any]
    encode_obs: Callable[..., Any]
    encode_state: Callable[..., Any]
    game_end: Callable[..., Any]
    game_end_dual: Callable[..., Any]
    reward: Callable[..., Any]
    spotter_after_step: Callable[..., Any]
    opponent_reset: Callable[..., Any]
    opponent_issue: Callable[..., Any]


def avail(space: ActionSpace, *args: Any, **kw: Any) -> Any:
    """Delegate to ``space.avail(...)``."""
    return space.avail(*args, **kw)


def reward(rewarder: Any, prev: Any, cur: Any, active: Any) -> Any:
    """Delegate to ``rewarder.step(prev, cur, active)``."""
    return rewarder.step(prev, cur, active)


def spotter_after_step(spotter: Any, ally_dead: Any, enemy_dead: Any, rows: Any) -> None:
    """Delegate to ``spotter.after_step(ally_dead, enemy_dead, rows)``."""
    spotter.after_step(ally_dead, enemy_dead, rows)


def opponent_reset(
    opponent: Any, world: Any, env_ids: Sequence[int], names: Sequence[str | None] | None = None
) -> None:
    """Delegate to ``opponent.on_reset(...)``."""
    if names is None:
        opponent.on_reset(world, env_ids)
    else:
        opponent.on_reset(world, env_ids, names=names)


def opponent_issue(opponent: Any, world: Any, command: Any, active: Any) -> None:
    """Delegate to ``opponent.issue(world, command, active)``."""
    opponent.issue(world, command, active)


_SEMANTIC_PY = dict(
    draw=generators.draw_batch,
    avail=avail,
    observe_spotter=features.observe_spotter,
    encode_obs=features.encode_obs,
    encode_state=features.encode_state,
    game_end=termination.game_end,
    game_end_dual=termination.game_end_dual,
    reward=reward,
    spotter_after_step=spotter_after_step,
    opponent_reset=opponent_reset,
    opponent_issue=opponent_issue,
)

PYTHON = Kernels(
    tier='python',
    services=py_services.PYTHON,
    to_command=ActionSpace.to_command,
    assign=UnitView.assign,
    frame=UnitView.frame,
    copy_rows=copy_rows,
    advance=advance,
    ally_tables=ally_tables_by_env,
    enemy_tables=UnitCatalog.enemy_tables,
    fill_max_rewards=fill_max_rewards,
    settle=settle,
    settle_enemy=settle_enemy,
    **_SEMANTIC_PY,
)


def native_available() -> bool:
    """Return True if a compatible ``zgame._zgame.env`` module can be imported."""
    try:
        from . import services as rust_services  # noqa: F401
    except ImportError:
        return False
    return True


_NEED_NATIVE = (
    "requires the optional 'zgame' extension with a compatible 'env' submodule. "
    'Install it with `pip install zfoundry-zgame` or `pip install "unismac[zgame]"`.'
)
_warned: set[str] = set()


def rust_blockers(config: EnvConfig, spec: Any) -> tuple[list[str], list[str]]:
    """Return ``(recipe_blockers, manifest_mismatches)`` preventing ``accel='rust'``."""
    from . import manifest

    return classic.rust_blockers(config, spec), list(manifest.mismatches())


def resolve(accel: str, config: EnvConfig | None = None, spec: Any = None) -> str:
    """Resolve requested ``accel`` into an effective tier (``python``/``services``/``rust``)."""
    if accel not in TIERS:
        raise ValueError(f'accel must be one of {TIERS}, got {accel!r}')
    if accel == 'python':
        return accel
    if not native_available():
        if accel == 'auto':
            return 'python'
        raise ImportError(f'accel={accel!r} {_NEED_NATIVE}')
    if accel == 'services':
        return accel
    if config is None or spec is None:
        if accel == 'auto':
            return 'services'
        raise ValueError(
            "accel='rust' requires config and scenario metadata; construct via make_vec_env"
        )
    recipe, stale = rust_blockers(config, spec)
    if accel == 'rust':
        if recipe or stale:
            raise ValueError("accel='rust' is unavailable: " + '; '.join(recipe + stale))
        return 'rust'
    if recipe:
        return 'services'
    if stale:
        msg = '; '.join(stale)
        if msg not in _warned:
            _warned.add(msg)
            warnings.warn(
                f"accel='auto' falling back to 'services' (skipping 'rust'): {msg}",
                RuntimeWarning,
                stacklevel=3,
            )
        return 'services'
    return 'rust'


def kernels(tier: str) -> Kernels:
    """Instantiate the kernel bundle for a resolved acceleration tier."""
    if tier == 'python':
        return PYTHON
    if tier not in ('services', 'rust'):
        raise ValueError(f'Unknown acceleration tier {tier!r}')
    from . import services as rust_services

    nk = rust_services.NativeKernels()
    system = dict(
        services=rust_services.SERVICES,
        to_command=rust_services.to_command,
        assign=nk.assign,
        frame=rust_services.frame,
        copy_rows=rust_services.copy_rows,
        ally_tables=nk.ally_tables,
        enemy_tables=nk.enemy_tables,
        fill_max_rewards=rust_services.fill_max_rewards,
        settle=rust_services.settle,
        settle_enemy=rust_services.settle_enemy,
    )
    if tier == 'services':
        return Kernels(tier='services', advance=rust_services.advance, **system, **_SEMANTIC_PY)
    from . import semantics as rust_semantics

    rk = rust_semantics.RustKernels()
    return Kernels(
        tier='rust',
        advance=rust_semantics.advance,
        draw=rk.draw,
        avail=rust_semantics.avail,
        observe_spotter=rust_semantics.observe_spotter,
        encode_obs=rk.encode_obs,
        encode_state=rk.encode_state,
        game_end=rust_semantics.game_end,
        game_end_dual=rust_semantics.game_end_dual,
        reward=rust_semantics.reward,
        spotter_after_step=rust_semantics.spotter_after_step,
        opponent_reset=rk.opponent_reset,
        opponent_issue=rk.opponent_issue,
        **system,
    )
