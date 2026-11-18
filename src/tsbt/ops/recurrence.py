"""Path-dependent operators (numba kernels). Columns are independent."""

from __future__ import annotations

import numba as nb
import numpy as np

from tsbt.ops._util import as2d, check_d, ts_op


@nb.njit(cache=True)
def _ewma_nb(x, alpha):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        prev = np.nan
        for t in range(t_len):
            v = x[t, j]
            if not np.isfinite(v):
                continue  # NaN today -> NaN output, state not advanced
            prev = v if not np.isfinite(prev) else prev + alpha * (v - prev)
            out[t, j] = prev
    return out


def ts_ema(x, d: int):
    """Classic EMA with alpha = 2/(d+1); seeded by the first valid value."""
    d = check_d(d)
    return ts_op(lambda a: _ewma_nb(a, 2.0 / (d + 1.0)))(x)


def ts_sma_cn(x, n: int, m: int):
    """Chinese SMA: Y = (m*X + (n-m)*Y[-1]) / n, requires 0 < m < n."""
    if not 0 < m < n:
        raise ValueError(f"ts_sma_cn needs 0 < m < n, got n={n}, m={m}")
    return ts_op(lambda a: _ewma_nb(a, m / n))(x)


@nb.njit(cache=True)
def _hump_nb(x, th):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        prev = np.nan
        for t in range(t_len):
            v = x[t, j]
            if not np.isfinite(v):
                prev = np.nan
                continue
            if not np.isfinite(prev):
                cur = v
            else:
                dlt = v - prev
                if abs(dlt) <= th:
                    cur = prev
                else:
                    cur = prev + np.sign(dlt) * (abs(dlt) - th)
            out[t, j] = cur
            prev = cur
    return out


def hump(x, threshold: float = 0.01):
    """Move towards x only by the part of the change exceeding ``threshold`` (absolute).

    Note: unlike AlphaForge (threshold scaled by the cross-sectional mean |x|), columns
    are independent here.
    """
    return ts_op(lambda a: _hump_nb(a, float(threshold)))(x)


@nb.njit(cache=True)
def _hump_decay_nb(x, th):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        prev = np.nan
        for t in range(t_len):
            v = x[t, j]
            if not np.isfinite(v):
                prev = np.nan
                continue
            cur = v if (not np.isfinite(prev) or abs(v - prev) > th) else prev
            out[t, j] = cur
            prev = cur
    return out


def hump_decay(x, hump: float = 0.0):
    """Update to x only when |x - previous output| > hump, else hold."""
    return ts_op(lambda a: _hump_decay_nb(a, float(hump)))(x)


@nb.njit(cache=True)
def _jump_decay_nb(x, sigma, sens, force):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        prev = np.nan
        for t in range(t_len):
            v = x[t, j]
            if not np.isfinite(v):
                prev = np.nan
                continue
            s = sigma[t, j]
            if np.isfinite(prev) and np.isfinite(s):
                dlt = v - prev
                cur = prev + dlt * force if abs(dlt) > sens * s else v
            else:
                cur = v
            out[t, j] = cur
            prev = cur
    return out


def jump_decay(x, d: int, sensitivity: float = 0.5, force: float = 0.1):
    """Jumps larger than sensitivity * ts_std_dev(x, d) only advance by ``force``."""
    from tsbt.ops.rolling import ts_std_dev

    d = check_d(d)
    a, flat = as2d(x)
    sigma = ts_std_dev(a, d)
    out = _jump_decay_nb(
        np.ascontiguousarray(a), np.ascontiguousarray(sigma), float(sensitivity), float(force)
    )
    return out[:, 0] if flat else out


@nb.njit(cache=True)
def _trade_when_nb(trig, alpha, exit_):
    t_len, n = alpha.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        prev = np.nan
        for t in range(t_len):
            cur = alpha[t, j] if trig[t, j] > 0.0 else prev
            if exit_[t, j] > 0.0:
                cur = np.nan
            out[t, j] = cur
            prev = cur
    return out


def trade_when(trigger, alpha, exit):
    """trigger > 0 -> take alpha; exit > 0 -> NaN (wins); otherwise hold the previous value."""
    a, flat = as2d(alpha)
    tr, _ = as2d(trigger)
    ex, _ = as2d(exit)
    tr, a2, ex = np.broadcast_arrays(tr, a, ex)
    out = _trade_when_nb(*(np.ascontiguousarray(z) for z in (tr, a2, ex)))
    return out[:, 0] if flat else out


@nb.njit(cache=True)
def _keep_nb(x, f, period):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        cache = np.nan
        age = np.inf
        for t in range(t_len):
            if f[t, j] > 0.0:
                cache = x[t, j]
                age = 0.0
            else:
                age += 1.0
            if age <= period:
                out[t, j] = cache
    return out


def keep(x, f, period: float = 5):
    """When f > 0 cache x; output the cache for up to ``period`` further bars."""
    a, flat = as2d(x)
    ff, _ = as2d(f)
    a, ff = np.broadcast_arrays(a, ff)
    out = _keep_nb(np.ascontiguousarray(a), np.ascontiguousarray(ff), float(period))
    return out[:, 0] if flat else out


__all__ = ["hump", "hump_decay", "jump_decay", "keep", "trade_when", "ts_ema", "ts_sma_cn"]
