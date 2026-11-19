"""Shift / rolling-window operators along axis 0 (time); columns are independent.

Semantics follow AlphaForge: a window needs ``min_valid(d)`` finite samples or the output
is NaN; NaN samples inside the window are skipped; std/cov use ddof=1, skew/kurt use
population central moments; every non-finite result becomes NaN. ``d`` counts bars.
"""

from __future__ import annotations

import math

import bottleneck as bn
import numba as nb
import numpy as np

from tsbt.ops._util import check_d, ffill, min_valid, purify, ts_op, ts_op2

# --------------------------------------------------------------------------- shift / fill


@ts_op
def ts_delay(x, d: int = 1):
    d = int(d)
    if d < 0:
        raise ValueError("ts_delay needs d >= 0 (negative d looks into the future)")
    if d == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    out[d:] = x[:-d]
    return out


def ts_delta(x, d: int = 1):
    """x - x[t-d]."""
    return purify(np.asarray(x, dtype=np.float64) - ts_delay(x, d))


def ts_returns(x, d: int = 1):
    """(x - x[t-d]) / x[t-d]."""
    prev = ts_delay(x, d)
    with np.errstate(all="ignore"):
        return purify((np.asarray(x, dtype=np.float64) - prev) / prev)


def log_diff(x):
    """log(x) - log(x[t-1]); non-positive values give NaN."""
    from tsbt.ops.elementwise import log

    return purify(log(x) - log(ts_delay(x, 1)))


@ts_op
def ts_backfill(x, d: int):
    """Forward-fill NaN with the latest valid value at most ``d`` bars old."""
    return ffill(x, check_d(d))


# --------------------------------------------------------------------------- bottleneck


def _move(fn, a, d: int, **kw):
    """Call a bottleneck ``move_*`` on a window that may be longer than the sample.

    bottleneck refuses ``window > len``; head windows are simply truncated in our
    semantics, so pad the top with NaN rows and strip them again.
    """
    t_len = a.shape[0]
    if t_len == 0:
        return np.empty_like(a, dtype=np.float64)
    if d <= t_len:
        return fn(a, window=d, axis=0, **kw)
    pad = np.full((d - t_len, *a.shape[1:]), np.nan)
    return fn(np.concatenate([pad, a], axis=0), window=d, axis=0, **kw)[d - t_len :]


def _finite(a):
    """Non-finite samples count as missing (bottleneck only skips NaN)."""
    bad = ~np.isfinite(a)
    return np.where(bad, np.nan, a) if bad.any() else a


def _bn(name: str, doc: str, fn):
    def op(x, d: int):
        d = check_d(d)
        return ts_op(lambda a: fn(_finite(a), d, min_valid(d)))(x)

    op.__name__ = op.__qualname__ = name
    op.__doc__ = doc
    return op


ts_sum = _bn(
    "ts_sum",
    "Rolling sum of the valid samples.",
    lambda a, d, mc: _move(bn.move_sum, a, d, min_count=mc),
)
ts_mean = _bn(
    "ts_mean",
    "Rolling mean of the valid samples.",
    lambda a, d, mc: _move(bn.move_mean, a, d, min_count=mc),
)
ts_min = _bn(
    "ts_min",
    "Rolling minimum of the valid samples.",
    lambda a, d, mc: _move(bn.move_min, a, d, min_count=mc),
)
ts_max = _bn(
    "ts_max",
    "Rolling maximum of the valid samples.",
    lambda a, d, mc: _move(bn.move_max, a, d, min_count=mc),
)
ts_median = _bn(
    "ts_median",
    "Rolling median of the valid samples.",
    lambda a, d, mc: _move(bn.move_median, a, d, min_count=mc),
)

#: a window whose bottleneck std is below this fraction of its largest |value| is
#: recomputed exactly: the sliding update of bottleneck leaves a residual of about
#: sqrt(eps) * scale, which turns a constant window into a tiny non-zero std.
_STD_EXACT_BELOW = 1e-3


#: an exactly recomputed std below this fraction of the window's largest |value| is
#: pure rounding of a constant window -> exactly 0 (so zscore / ir become NaN).
_STD_ZERO_BELOW = 1e-12


@nb.njit(cache=True)
def _std_exact_rows(x, d, out, suspect, scale, zero_below):
    t_len, n = x.shape
    for j in range(n):
        for t in range(t_len):
            if not suspect[t, j]:
                continue
            lo = max(0, t - d + 1)
            c = 0
            s = 0.0
            for u in range(lo, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    c += 1
                    s += v
            mean = s / c
            ss = 0.0
            for u in range(lo, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    ss += (v - mean) ** 2
            sd = math.sqrt(ss / (c - 1))
            out[t, j] = sd if sd > zero_below * scale[t, j] else 0.0


def _std(a, d, mc):
    if d < 2:
        return np.full_like(a, np.nan)
    out = _move(bn.move_std, a, d, min_count=max(mc, 2), ddof=1)
    scale = _move(bn.move_max, np.abs(a), d, min_count=1)
    with np.errstate(invalid="ignore"):
        suspect = np.isfinite(out) & (out < _STD_EXACT_BELOW * scale)
    if suspect.any():
        _std_exact_rows(np.ascontiguousarray(a), d, out, suspect, scale, _STD_ZERO_BELOW)
    return out


ts_std_dev = _bn(
    "ts_std_dev", "Rolling sample standard deviation (ddof=1); needs >= 2 valid samples.", _std
)


def _valid_count(a, d):
    return _move(bn.move_sum, np.isfinite(a).astype(np.float64), d, min_count=1)


def _rank(a, d, mc):
    r = _move(bn.move_rank, a, d, min_count=1)
    out = (r + 1.0) / 2.0
    cnt = _valid_count(a, d)
    out[(cnt == 1) & np.isfinite(a)] = 0.5
    out[(cnt < mc) | ~np.isfinite(a)] = np.nan
    return out


ts_rank = _bn(
    "ts_rank", "Rank of today's value within the window, scaled to [0, 1] (ties averaged).", _rank
)


def ts_zscore(x, d: int):
    """(x - ts_mean) / ts_std_dev."""
    with np.errstate(all="ignore"):
        return purify((np.asarray(x, dtype=np.float64) - ts_mean(x, d)) / ts_std_dev(x, d))


def ts_av_diff(x, d: int):
    return purify(np.asarray(x, dtype=np.float64) - ts_mean(x, d))


def ts_ir(x, d: int):
    with np.errstate(all="ignore"):
        return purify(ts_mean(x, d) / ts_std_dev(x, d))


def ts_scale(x, d: int):
    """(x - ts_min) / (ts_max - ts_min)."""
    lo = ts_min(x, d)
    with np.errstate(all="ignore"):
        return purify((np.asarray(x, dtype=np.float64) - lo) / (ts_max(x, d) - lo))


def inst_tvr(x, d: int):
    """Rolling mean of |x - x[t-1]|."""
    return ts_mean(np.abs(ts_delta(x, 1)), d)


@ts_op
def ts_count_nans(x, d: int):
    d = check_d(d)
    c = np.isfinite(x).astype(np.int64).cumsum(axis=0)
    c[d:] = c[d:] - c[:-d]
    width = np.minimum(np.arange(1, x.shape[0] + 1), d)[:, None]
    return (width - c).astype(np.float64)


# --------------------------------------------------------------------------- numba windows


@nb.njit(cache=True)
def _moments_nb(x, d, k):
    """count and population central moment of order k per window."""
    t_len, n = x.shape
    cnt = np.zeros((t_len, n))
    m2 = np.full((t_len, n), np.nan)
    mk = np.full((t_len, n), np.nan)
    for j in range(n):
        for t in range(t_len):
            lo = max(0, t - d + 1)
            c = 0
            s = 0.0
            for u in range(lo, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    c += 1
                    s += v
            cnt[t, j] = c
            if c == 0:
                continue
            mean = s / c
            a2 = 0.0
            ak = 0.0
            for u in range(lo, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    dv = v - mean
                    a2 += dv * dv
                    p = dv
                    for _ in range(k - 1):
                        p *= dv
                    ak += p
            m2[t, j] = a2 / c
            mk[t, j] = ak / c
    return cnt, m2, mk


def _gate(out, cnt, mc):
    out = np.asarray(out, dtype=np.float64)
    out[(cnt < mc) | ~np.isfinite(out)] = np.nan
    return out


def _gated(x, kernel, mc: int):
    """Run a window ``kernel(a) -> (count, value)`` and gate it with :func:`_gate`."""

    def k(a):
        with np.errstate(all="ignore"):
            cnt, out = kernel(a)
        return _gate(out, cnt, mc)

    return ts_op(k)(x)


def ts_skewness(x, d: int):
    """Population skewness m3 / m2^1.5; needs >= max(min_valid, 3) samples."""
    d = check_d(d)

    def kern(a):
        cnt, m2, m3 = _moments_nb(a, d, 3)
        return cnt, m3 / m2**1.5

    return _gated(x, kern, max(min_valid(d), 3))


def ts_kurtosis(x, d: int):
    """Excess kurtosis m4 / m2^2 - 3; needs >= max(min_valid, 4) samples."""
    d = check_d(d)

    def kern(a):
        cnt, m2, m4 = _moments_nb(a, d, 4)
        return cnt, m4 / m2**2 - 3.0

    return _gated(x, kern, max(min_valid(d), 4))


def ts_moment(x, d: int, k: int):
    """k-th population central moment (not standardised)."""
    d = check_d(d)
    if int(k) != k or k < 1:
        raise ValueError(f"moment order k must be a positive integer, got {k}")
    k = int(k)

    def kern(a):
        cnt, _, mk = _moments_nb(a, d, k)
        return cnt, mk

    return _gated(x, kern, min_valid(d))


@nb.njit(cache=True)
def _product_nb(x, d):
    t_len, n = x.shape
    cnt = np.zeros((t_len, n))
    out = np.ones((t_len, n))
    for j in range(n):
        for t in range(t_len):
            c = 0
            p = 1.0
            for u in range(max(0, t - d + 1), t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    c += 1
                    p *= v
            cnt[t, j] = c
            out[t, j] = p
    return cnt, out


def ts_product(x, d: int):
    d = check_d(d)
    return _gated(x, lambda a: _product_nb(a, d), min_valid(d))


@nb.njit(cache=True)
def _arg_extreme_nb(x, d, want_max):
    t_len, n = x.shape
    cnt = np.zeros((t_len, n))
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        for t in range(t_len):
            best = np.nan
            bi = -1
            c = 0
            # scan from today backwards so ties keep the most recent
            for i in range(0, min(d, t + 1)):
                v = x[t - i, j]
                if not np.isfinite(v):
                    continue
                c += 1
                if bi < 0 or (want_max and v > best) or ((not want_max) and v < best):
                    best = v
                    bi = i
            cnt[t, j] = c
            if bi >= 0:
                out[t, j] = bi
    return cnt, out


def ts_arg_max(x, d: int):
    """Bars since the window maximum (0 = today; ties -> most recent)."""
    d = check_d(d)
    return _gated(x, lambda a: _arg_extreme_nb(a, d, True), min_valid(d))


def ts_arg_min(x, d: int):
    d = check_d(d)
    return _gated(x, lambda a: _arg_extreme_nb(a, d, False), min_valid(d))


@nb.njit(cache=True)
def _ndtri(p):
    """Inverse standard normal CDF (Acklam + one Halley step)."""
    if not (p > 0.0 and p < 1.0):
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        return np.nan
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    e = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1
        )
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        z = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1
        )
    err = 0.5 * math.erfc(-z / math.sqrt(2.0)) - p
    u = err * math.sqrt(2 * math.pi) * math.exp(z * z / 2)
    return z - u / (1 + z * u / 2)


@nb.njit(cache=True)
def _quantile_nb(x, d, driver):
    t_len, n = x.shape
    cnt = np.zeros((t_len, n))
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        for t in range(t_len):
            today = x[t, j]
            c = 0
            le = 0
            for u in range(max(0, t - d + 1), t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    c += 1
                    if np.isfinite(today) and v <= today:
                        le += 1
            cnt[t, j] = c
            if not np.isfinite(today):
                continue
            p = le / (c + 1.0)
            if driver == 0:
                out[t, j] = _ndtri(p)
            elif driver == 1:
                out[t, j] = p
            else:
                out[t, j] = math.tan(math.pi * (p - 0.5))
    return cnt, out


def ts_quantile(x, d: int, driver: str = "gaussian"):
    """p = #(window <= today) / (valid + 1), mapped by gaussian / uniform / cauchy."""
    d = check_d(d)
    codes = {"gaussian": 0, "uniform": 1, "cauchy": 2}
    if driver not in codes:
        raise ValueError(f"driver must be one of {sorted(codes)}, got {driver!r}")
    code = codes[driver]
    return _gated(x, lambda a: _quantile_nb(a, d, code), min_valid(d))


@nb.njit(cache=True)
def _entropy_nb(x, d, buckets, mc):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    counts = np.zeros(buckets)
    for j in range(n):
        for t in range(t_len):
            lo_t = max(0, t - d + 1)
            c = 0
            lo = np.inf
            hi = -np.inf
            for u in range(lo_t, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    c += 1
                    lo = min(lo, v)
                    hi = max(hi, v)
            span = hi - lo
            if c < max(mc, 2) or not span > 0:
                continue
            counts[:] = 0.0
            for u in range(lo_t, t + 1):
                v = x[u, j]
                if np.isfinite(v):
                    b = int((v - lo) / span * buckets)
                    b = min(max(b, 0), buckets - 1)
                    counts[b] += 1.0
            ent = 0.0
            for b in range(buckets):
                if counts[b] > 0:
                    p = counts[b] / c
                    ent -= p * math.log(p)
            out[t, j] = ent
    return out


def ts_entropy(x, d: int, buckets: int = 10):
    """Shannon entropy of a ``buckets``-bin histogram over the window range."""
    d = check_d(d)
    return ts_op(lambda a: _entropy_nb(a, d, int(buckets), min_valid(d)))(x)


@nb.njit(cache=True)
def _decay_linear_nb(x, d, dense):
    t_len, n = x.shape
    out = np.full((t_len, n), np.nan)
    full = d * (d + 1) / 2.0
    for j in range(n):
        for t in range(t_len):
            num = 0.0
            den = 0.0
            c = 0
            for i in range(min(d, t + 1)):  # i=0 today, weight d
                v = x[t - i, j]
                w = d - i
                if np.isfinite(v):
                    num += w * v
                    den += w
                    c += 1
            if c >= 1:
                out[t, j] = num / (full if dense else den)
    return out


def ts_decay_linear(x, d: int, dense: bool = False):
    """Linearly weighted mean (today d, oldest 1). ``dense=True``: NaN counts as 0."""
    d = check_d(d)
    return ts_op(lambda a: _decay_linear_nb(a, d, bool(dense)))(x)


@nb.njit(cache=True)
def _decay_exp_nb(x, d, f):
    t_len, n = x.shape
    cnt = np.zeros((t_len, n))
    out = np.full((t_len, n), np.nan)
    ws = np.empty(min(d, t_len))
    for i in range(ws.shape[0]):
        ws[i] = (1.0 - f) * f**i
    for j in range(n):
        for t in range(t_len):
            num = 0.0
            den = 0.0
            c = 0
            for i in range(min(d, t + 1)):
                v = x[t - i, j]
                if np.isfinite(v):
                    w = ws[i]
                    num += w * v
                    den += w
                    c += 1
            cnt[t, j] = c
            if c > 0 and den > 0:
                out[t, j] = num / den
    return cnt, out


def ts_decay_exp_window(x, d: int, factor: float = 0.5):
    """Weights (1-factor)*factor^i (i=0 today), renormalised over valid samples."""
    d = check_d(d)
    if not 0 < factor < 1:
        raise ValueError("factor must be in (0, 1)")
    return _gated(x, lambda a: _decay_exp_nb(a, d, float(factor)), min_valid(d))


@ts_op
def filter(x, h):
    """FIR filter: sum_k h[k] * x[t-k]; NaN propagates (incl. before the series start)."""
    if isinstance(h, str):
        h = [float(s) for s in h.split(",") if s.strip()]
    out = np.zeros_like(x)
    for k, coef in enumerate(h):
        out = out + coef * ts_delay(x, k)
    return out


# --------------------------------------------------------------------------- pairwise


def _pair_sums(x, y, d):
    v = np.isfinite(x) & np.isfinite(y)
    xv = np.where(v, x, 0.0)
    yv = np.where(v, y, 0.0)

    def ms(a):
        return _move(bn.move_sum, a, d, min_count=1)

    return {
        "n": ms(v.astype(np.float64)),
        "x": ms(xv),
        "y": ms(yv),
        "xx": ms(xv * xv),
        "yy": ms(yv * yv),
        "xy": ms(xv * yv),
    }


#: a variance below this fraction of the window's sum of squares is rounding noise of
#: the sliding sums (a constant window), not a real variance -> treated as zero.
_VAR_EPS = 1e-11


def _var_sum(sxx, sx, n):
    """Sum of squared deviations from sliding sums, with rounding noise clamped to 0."""
    v = sxx - sx**2 / n
    return np.where(v > _VAR_EPS * sxx, v, 0.0)


def ts_covariance(x, y, d: int):
    """Sample covariance (ddof=1) over pairwise-valid samples.

    Exactly 0 when either series does not move inside the window (the sliding sums
    would otherwise leave a tiny residual of random sign).
    """
    d = check_d(d)

    def k(a, b):
        s = _pair_sums(a, b, d)
        n = s["n"]
        nn = np.maximum(n, 1)
        with np.errstate(all="ignore"):
            cov = (s["xy"] - s["x"] * s["y"] / nn) / np.maximum(n - 1, 1)
            flat = (_var_sum(s["xx"], s["x"], nn) == 0) | (_var_sum(s["yy"], s["y"], nn) == 0)
        return _gate(np.where(flat, 0.0, cov), n, max(min_valid(d), 2))

    return ts_op2(k)(x, y)


def ts_corr(x, y, d: int):
    """Pearson correlation over pairwise-valid samples."""
    d = check_d(d)

    def k(a, b):
        s = _pair_sums(a, b, d)
        n = np.maximum(s["n"], 1)
        with np.errstate(all="ignore"):
            cov = s["xy"] - s["x"] * s["y"] / n
            den = np.sqrt(_var_sum(s["xx"], s["x"], n) * _var_sum(s["yy"], s["y"], n))
            rho = np.where(den > 0, np.clip(cov / den, -1.0, 1.0), np.nan)
        return _gate(rho, s["n"], max(min_valid(d), 2))

    return ts_op2(k)(x, y)


def _regression(y, x, d, rettype):
    s = _pair_sums(x, y, d)
    n = np.maximum(s["n"], 1)
    with np.errstate(all="ignore"):
        beta = (s["xy"] - s["x"] * s["y"] / n) / _var_sum(s["xx"], s["x"], n)
        alpha = s["y"] / n - beta * s["x"] / n
        if rettype == 1:
            out = alpha
        elif rettype == 2:
            out = beta
        else:
            fitted = alpha + beta * x
            out = y - fitted if rettype == 0 else fitted
    return _gate(out, s["n"], max(min_valid(d), 2))


def ts_regression(y, x, d: int, lag: int = 0, rettype: int = 0):
    """Rolling OLS y = a + b*delay(x, lag). rettype: 0 residual, 1 a, 2 b, 3 fitted."""
    d = check_d(d)
    if rettype not in (0, 1, 2, 3):
        raise ValueError("rettype must be 0..3")
    if lag:
        x = ts_delay(x, lag)
    return ts_op2(lambda a, b: _regression(a, b, d, rettype))(y, x)


def ts_seq_beta(x, d: int):
    """Rolling OLS slope of x on a time ramp."""
    d = check_d(d)

    def k(a):
        ramp = np.broadcast_to(np.arange(a.shape[0], dtype=np.float64)[:, None], a.shape)
        return _regression(a, np.ascontiguousarray(ramp), d, 2)

    return ts_op(k)(x)


__all__ = [
    "filter",
    "inst_tvr",
    "log_diff",
    "ts_arg_max",
    "ts_arg_min",
    "ts_av_diff",
    "ts_backfill",
    "ts_corr",
    "ts_count_nans",
    "ts_covariance",
    "ts_decay_exp_window",
    "ts_decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_entropy",
    "ts_ir",
    "ts_kurtosis",
    "ts_max",
    "ts_mean",
    "ts_median",
    "ts_min",
    "ts_moment",
    "ts_product",
    "ts_quantile",
    "ts_rank",
    "ts_regression",
    "ts_returns",
    "ts_scale",
    "ts_seq_beta",
    "ts_skewness",
    "ts_std_dev",
    "ts_sum",
    "ts_zscore",
]
