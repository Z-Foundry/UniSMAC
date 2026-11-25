"""Shared pytest fixtures for the UniSMAC test suite."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import pytest

from unismac.assets import MapInfo, StaticData, load_map_info, load_static_data


@lru_cache(maxsize=1)
def _cached_static_data() -> StaticData:
    return load_static_data()


@lru_cache(maxsize=64)
def _cached_map_info(map_id: str) -> MapInfo:
    return load_map_info(map_id)


@pytest.fixture(scope='session')
def static_data() -> StaticData:
    """Return session-cached unit and weapon static asset data."""
    return _cached_static_data()


@pytest.fixture(scope='session')
def map_info() -> Callable[[str], MapInfo]:
    """Return a session-cached loader for ``MapInfo`` keyed by map ID."""
    return _cached_map_info
