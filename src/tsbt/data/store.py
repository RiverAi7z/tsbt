"""zarr v3 cache store.

Layout (all arrays 1-D float64, NaN = missing)::

    store.zarr/
      m1/<SYM>/<field>                 minute index i  <->  ORIGIN_MS + i*60s
      m1/<SYM>            attrs["builds"][<build key>] = manifest (see m1.py)
      bar/<tag>/<SYM>/<field>          bar index j     <->  ORIGIN_MS + offset + j*F
      bar/<tag>/<SYM>/user/<name>      user-saved bar fields

Every array is indexed from a global origin, so reading a time range never needs a
per-symbol offset and multi-symbol alignment is a plain slice.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import zarr
from filelock import FileLock

from tsbt import config

_COMPRESSOR = zarr.codecs.ZstdCodec(level=3)


_ROOTS: dict[str, zarr.Group] = {}


def root() -> zarr.Group:
    path = str(config.store_path())
    g = _ROOTS.get(path)
    if g is None:
        g = _ROOTS[path] = zarr.open_group(path, mode="a")
    return g


def forget() -> None:
    """Drop cached handles (after clearing the cache directory)."""
    _ROOTS.clear()


@contextlib.contextmanager
def symbol_lock(symbol: str, timeout: float = 3600) -> Iterator[None]:
    lock = FileLock(str(config.lock_dir() / f"{symbol}.lock"), timeout=timeout)
    with lock:
        yield


def group(path: str, create: bool = True) -> zarr.Group | None:
    r = root()
    if create:
        return r.require_group(path)
    try:
        node = r[path]
    except KeyError:
        return None
    return node if isinstance(node, zarr.Group) else None


def children(path: str) -> list[str]:
    """Sorted names of the sub-groups of ``path`` (empty when it does not exist)."""
    g = group(path, create=False)
    return sorted(g.group_keys()) if g is not None else []


def has_array(path: str) -> bool:
    try:
        node = root()[path]
    except KeyError:
        return False
    return isinstance(node, zarr.Array)


def read(path: str, lo: int, hi: int) -> np.ndarray:
    """Values at indices [lo, hi); NaN outside the stored extent or when absent."""
    out = np.full(hi - lo, np.nan)
    try:
        arr = root()[path]
    except KeyError:
        return out
    n = arr.shape[0]
    a, b = max(lo, 0), min(hi, n)
    if a < b:
        out[a - lo : b - lo] = arr[a:b]
    return out


def size(path: str) -> int:
    """Stored length of an array (0 when absent)."""
    try:
        return int(root()[path].shape[0])
    except KeyError:
        return 0


def write(path: str, lo: int, values: np.ndarray) -> None:
    """Write ``values`` at indices [lo, lo+len); creates / grows the array as needed."""
    if lo < 0:
        values = values[-lo:]
        lo = 0
    hi = lo + len(values)
    r = root()
    try:
        arr = r[path]
    except KeyError:
        arr = r.create_array(
            path,
            shape=(hi,),
            chunks=(config.CHUNK,),
            dtype="f8",
            fill_value=np.nan,
            compressors=_COMPRESSOR,
        )
    if arr.shape[0] < hi:
        arr.resize((hi,))
    arr[lo:hi] = np.asarray(values, dtype=np.float64)


def delete(path: str) -> None:
    r = root()
    with contextlib.suppress(KeyError):
        del r[path]


def get_attr(path: str, key: str, default=None):
    g = group(path, create=False)
    if g is None:
        return default
    return g.attrs.get(key, default)


def set_attr(path: str, key: str, value) -> None:
    g = group(path, create=True)
    g.attrs[key] = value
