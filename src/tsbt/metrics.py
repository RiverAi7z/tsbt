"""Performance metrics.

Inputs are a NAV path with a base point: ``times[0]`` is the evaluation start (bar
open) with ``nav[0] == 1``; ``times[1:]`` are bar close times. Risk/return statistics
use the NAV sampled at UTC day ends; drawdown uses every bar.
"""

from __future__ import annotations

import math

import bottleneck as bn
import numpy as np

from tsbt.config import DAY_MS

ANN = 365.0


def daily_values(times: np.ndarray, nav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base point + last NAV of each UTC day.

    The base point (evaluation start) is always its own observation, so a partial
    first day and a partial last day each count once. A bar closing exactly at 00:00
    belongs to the day that just ended.
    """
    times = np.asarray(times, dtype=np.int64)
    nav = np.asarray(nav)
    key = (times[1:] - 1) // DAY_MS
    last = np.r_[key[1:] != key[:-1], True] if len(key) else np.zeros(0, bool)
    return np.r_[times[0], times[1:][last]], np.r_[nav[0], nav[1:][last]]


def period_returns(values: np.ndarray, compound: bool) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if compound:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = v[1:] / v[:-1] - 1.0
        return np.where(np.isfinite(r), r, 0.0)  # 0/0 after bankruptcy
    return np.diff(v)


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 2:
        return math.nan
    sd = r.std(ddof=1)
    if not sd > 0:
        return math.nan
    return float(r.mean() / sd * math.sqrt(ANN))


def drawdown_series(nav: np.ndarray, compound: bool) -> np.ndarray:
    """Drawdown from the running peak: relative (compound) or in principal units (simple)."""
    nav = np.asarray(nav, dtype=np.float64)
    peak = np.maximum.accumulate(nav)
    if compound:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(peak > 0, nav / peak - 1.0, -1.0)
    return nav - peak


def max_drawdown(nav: np.ndarray, compound: bool) -> float:
    dd = drawdown_series(nav, compound)
    return float(dd.min()) if len(dd) else math.nan


def apr(nav_end: float, days: float, compound: bool) -> float:
    if not days > 0:
        return math.nan
    if compound:
        if nav_end <= 0:
            return -1.0
        return float(nav_end ** (ANN / days) - 1.0)
    return float((nav_end - 1.0) * ANN / days)


def perf(times: np.ndarray, nav: np.ndarray, turnover: np.ndarray, compound: bool) -> dict:
    """Core metric set for one NAV path (with base point)."""
    times = np.asarray(times, dtype=np.int64)
    nav = np.asarray(nav, dtype=np.float64)
    days = (times[-1] - times[0]) / DAY_MS
    _, dv = daily_values(times, nav)
    r = period_returns(dv, compound)
    a = apr(nav[-1], days, compound)
    mdd = max_drawdown(nav, compound)
    return {
        "apr": a,
        "sharpe": sharpe(r),
        "calmar": a / abs(mdd) if mdd < 0 else math.nan,
        "turnover": float(np.nansum(turnover) / days) if days > 0 else math.nan,
        "mdd": mdd,
        "days": float(days),
        "total_return": float(nav[-1] - 1.0),
    }


def rolling_sharpe(times: np.ndarray, nav: np.ndarray, compound: bool, window_days: int = 90):
    """Daily rolling Sharpe; returns (day-end times, values)."""
    t, dv = daily_values(times, nav)
    r = period_returns(dv, compound)
    if len(r) < 2:
        return t[1:], np.full(len(r), np.nan)
    w = min(window_days, len(r))
    m = bn.move_mean(r, w, min_count=w)
    s = bn.move_std(r, w, min_count=w, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = m / s * math.sqrt(ANN)
    return t[1:], np.where(np.isfinite(out), out, np.nan)
