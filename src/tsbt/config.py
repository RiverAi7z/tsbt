"""Paths and global constants.

Paths are resolved on every call (not at import time) so tests and scripts can
redirect them with environment variables or :func:`set_paths`.
"""

from __future__ import annotations

import os
from pathlib import Path

#: All minute arrays are indexed from this UTC instant (minute index 0).
ORIGIN_MS = 1_546_300_800_000  # 2019-01-01T00:00:00Z
MINUTE_MS = 60_000
DAY_MS = 86_400_000
#: 1-D chunk length of every zarr array (~1 year of minutes).
CHUNK = 524_288

DEFAULT_DATA_ROOT = "/share/binance_data/futures/um"
DEFAULT_CACHE_DIR = "~/.cache/tsbt"

_overrides: dict[str, str] = {}


def set_paths(
    data_root: str | os.PathLike | None = None, cache_dir: str | os.PathLike | None = None
) -> None:
    """Override data root / cache dir for this process (takes precedence over env vars)."""
    if data_root is not None:
        _overrides["data_root"] = str(data_root)
    if cache_dir is not None:
        _overrides["cache_dir"] = str(cache_dir)


def reset_paths() -> None:
    _overrides.clear()


def data_root() -> Path:
    raw = _overrides.get("data_root") or os.environ.get("TSBT_DATA_ROOT") or DEFAULT_DATA_ROOT
    return Path(raw).expanduser()


def cache_dir() -> Path:
    raw = _overrides.get("cache_dir") or os.environ.get("TSBT_CACHE_DIR") or DEFAULT_CACHE_DIR
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_path() -> Path:
    return cache_dir() / "store.zarr"


def lock_dir() -> Path:
    path = cache_dir() / "locks"
    path.mkdir(parents=True, exist_ok=True)
    return path
