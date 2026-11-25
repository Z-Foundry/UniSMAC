"""Shared helpers for UniSMAC ecosystem adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..env import VecEnv

__all__ = [
    'InvalidActionMode',
    'missing_extra_error',
    'require_single_env',
    'require_single_sided',
    'sanitize_actions',
    'validate_invalid_action_mode',
    'validate_render_mode',
]

InvalidActionMode = Literal['raise', 'fallback']


def missing_extra_error(
    module_name: str, extra: str, requirement: str, cause: ImportError
) -> ImportError:
    """Build an actionable ImportError when an optional adapter dependency is absent."""
    return ImportError(
        f'{module_name} requires {requirement}. '
        f'Install it with: pip install "unismac[{extra}]" '
        f'(original error: {cause})'
    )


def require_single_env(vec: VecEnv, adapter_name: str) -> None:
    """Ensure a single-environment adapter wraps a `batch_size=1` `VecEnv`."""
    if vec.batch_size != 1:
        raise ValueError(
            f'{adapter_name} only wraps a VecEnv with batch_size=1, got batch_size={vec.batch_size}'
        )


def require_single_sided(vec: VecEnv, adapter_name: str) -> None:
    """Ensure a single-sided adapter is not given a two-sided (`dual=True`) `VecEnv`."""
    if vec.dual:
        raise ValueError(
            f'{adapter_name} only supports single-sided environments (dual=False); '
            'use PettingZooParallelEnv or VecEnv directly for dual=True self-play'
        )


def validate_render_mode(render_mode: str | None, adapter_name: str) -> None:
    """Reject unsupported render modes up front (UniSMAC does not render)."""
    if render_mode is not None:
        raise ValueError(
            f'{adapter_name} does not support rendering (render_modes is empty); '
            f'got render_mode={render_mode!r}'
        )


def validate_invalid_action_mode(mode: str, adapter_name: str) -> InvalidActionMode:
    """Validate `invalid_action_mode` ('raise' or 'fallback')."""
    if mode == 'raise':
        return 'raise'
    if mode == 'fallback':
        return 'fallback'
    raise ValueError(
        f"{adapter_name} got unknown invalid_action_mode={mode!r}; expected 'raise' or 'fallback'"
    )


def sanitize_actions(
    actions: np.ndarray,
    avail: np.ndarray,
    active: np.ndarray,
    *,
    mode: InvalidActionMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate or remap `[B, A]` actions against `avail [B, A, K]` for `active [B]` slots.

    Inactive (already-done) slots are zeroed so they never trigger out-of-range checks in
    `VecEnv.step`. Under `mode='fallback'`, illegal active actions are remapped to a guaranteed
    legal fallback (`0` NOOP for dead units where `avail[..., 0] == 1`, `1` STOP for alive units
    where `avail[..., 1] == 1`, else the first available action).

    Returns `(safe_actions int64[B, A], remapped_counts int64[B])`.
    """
    b_size = actions.shape[0]
    n_actions = avail.shape[-1]
    safe = np.array(actions, dtype=np.int64, copy=True)
    if not active.all():
        safe[~active] = 0
    if mode == 'raise':
        return safe, np.zeros(b_size, dtype=np.int64)

    in_range = (safe >= 0) & (safe < n_actions)
    clipped = np.where(in_range, safe, 0)
    legal = np.take_along_axis(avail, clipped[..., None], axis=-1)[..., 0] == 1
    invalid = active[:, None] & (~in_range | ~legal)
    remapped = invalid.sum(axis=-1).astype(np.int64)
    if invalid.any():
        first_avail = np.argmax(avail, axis=-1).astype(np.int64)
        fallback = np.where(
            avail[..., 0] == 1,
            0,
            np.where(avail[..., 1] == 1, 1, first_avail),
        )
        safe = np.where(invalid, fallback, safe)
    return safe, remapped
