from __future__ import annotations

import functools
import math

import numpy as np


def min_valid(d: int) -> int:
    """Minimum valid samples in a rolling window: min(d, max(2, ceil(d/2))); d=1 -> 1."""
    return min(d, max(2, math.ceil(0.5 * d)))


def check_d(d) -> int:
    if int(d) != d or d < 1:
        raise ValueError(f"window d must be a positive integer, got {d}")
    return int(d)


def purify(x):
    """Non-finite -> NaN."""
    out = np.asarray(x, dtype=np.float64)
    if out.ndim == 0:
        v = float(out)
        return v if math.isfinite(v) else math.nan
    bad = ~np.isfinite(out)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out


def ffill(x: np.ndarray, limit: int | None = None) -> np.ndarray:
    """Forward-fill non-finite entries along axis 0 with the latest finite value, at most
    ``limit`` rows old (``None`` = no limit). Leading gaps stay NaN."""
    valid = np.isfinite(x)
    rows = np.arange(x.shape[0]).reshape(-1, *([1] * (x.ndim - 1)))
    idx = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    got = np.take_along_axis(x, np.maximum(idx, 0), axis=0)
    keep = idx >= 0
    if limit is not None:
        keep &= rows - idx <= limit
    return np.where(keep, got, np.nan)


def as2d(x) -> tuple[np.ndarray, bool]:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a[:, None], True
    if a.ndim != 2:
        raise ValueError(f"time-series operators take (T,) or (T, N) arrays, got shape {a.shape}")
    return a, False


def ts_op(fn):
    """Lift a ``(T, N) -> (T, N)`` kernel to accept ``(T,)`` too."""

    @functools.wraps(fn)
    def run(x, *args, **kwargs):
        a, flat = as2d(x)
        out = fn(np.ascontiguousarray(a), *args, **kwargs)
        return out[:, 0] if flat else out

    return run


def ts_op2(fn):
    """Same for two-series operators; y is broadcast to x's shape."""

    @functools.wraps(fn)
    def run(x, y, *args, **kwargs):
        a, flat_a = as2d(x)
        b, flat_b = as2d(y)
        a, b = np.broadcast_arrays(a, b)
        out = fn(np.ascontiguousarray(a), np.ascontiguousarray(b), *args, **kwargs)
        return out[:, 0] if (flat_a and flat_b) else out

    return run
