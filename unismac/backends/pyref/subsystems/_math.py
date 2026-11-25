"""Shared scalar math helpers for pyref subsystems."""

from __future__ import annotations

import math


def wrap_angle(a: float) -> float:
    """Wrap angle in radians into `[-pi, pi)`."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ['wrap_angle']
