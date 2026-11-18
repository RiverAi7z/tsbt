"""Bar synthesis from the minute grid (reshape + NaN-aware reductions) and the L2 cache.

The minute grid is regular and hole-free (missing minutes are NaN), so the minutes of
bar-aligned range form a ``(T, k)`` block and every bar statistic is a reduction along
axis 1.  Fixed field rules and user ``BarData.agg`` share this code path.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import numpy as np

from tsbt import config
from tsbt.data import catalog, m1, store

MIN = config.MINUTE_MS

_FREQ_RE = re.compile(r"^\s*(\d+)\s*(m|min|h|d)\s*$", re.IGNORECASE)


def parse_freq(freq: str | int) -> int:
    """'4h' -> 240 minutes. Must divide one day."""
    if isinstance(freq, int):
        k = freq
    else:
        m = _FREQ_RE.match(freq)
        if not m:
            raise ValueError(f"bad freq {freq!r}; use e.g. '15m', '1h', '4h', '1D'")
        n, unit = int(m.group(1)), m.group(2).lower()
        k = n * {"m": 1, "min": 1, "h": 60, "d": 1440}[unit]
    if k <= 0 or 1440 % k:
        raise ValueError(f"freq must divide 1440 minutes, got {k} minutes")
    return k


def freq_label(k: int) -> str:
    if k % 1440 == 0:
        return f"{k // 1440}D"
    if k % 60 == 0:
        return f"{k // 60}h"
    return f"{k}m"


def tag(k: int, offset: int) -> str:
    return f"{k}m" if offset == 0 else f"{k}m+{offset}"


# --------------------------------------------------------------------------- reductions


def _first_valid(b: np.ndarray, last: bool = False) -> np.ndarray:
    v = np.isfinite(b)
    k = b.shape[1]
    if last:
        idx = k - 1 - np.argmax(v[:, ::-1], axis=1)
    else:
        idx = np.argmax(v, axis=1)
    out = np.take_along_axis(b, np.expand_dims(idx, 1), axis=1).squeeze(1)
    return np.where(v.any(axis=1), out, np.nan)


def _moments(b):
    v = np.isfinite(b)
    n = v.sum(axis=1)
    mean = np.where(v, b, 0.0).sum(axis=1) / np.maximum(n, 1)
    dev = np.where(v, b - np.expand_dims(mean, 1), 0.0)
    return n, mean, dev


def _red_sum(b):
    v = np.isfinite(b)
    return np.where(v.any(axis=1), np.where(v, b, 0.0).sum(axis=1), np.nan)


def _red_var(b, ddof=1):
    n, _, dev = _moments(b)
    out = (dev**2).sum(axis=1) / np.maximum(n - ddof, 1)
    return np.where(n > ddof, out, np.nan)


def _red_skew(b):
    n, _, dev = _moments(b)
    m2 = (dev**2).sum(axis=1) / np.maximum(n, 1)
    m3 = (dev**3).sum(axis=1) / np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = m3 / m2**1.5
    return np.where((n >= 3) & np.isfinite(out), out, np.nan)


def _red_kurt(b):
    n, _, dev = _moments(b)
    m2 = (dev**2).sum(axis=1) / np.maximum(n, 1)
    m4 = (dev**4).sum(axis=1) / np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = m4 / m2**2 - 3.0
    return np.where((n >= 4) & np.isfinite(out), out, np.nan)


def _quiet(fn):
    def run(b, *a, **kw):
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(b, *a, **kw)

    return run


REDUCERS: dict[str, Callable] = {
    "first": _first_valid,
    "last": lambda b: _first_valid(b, last=True),
    "sum": _red_sum,
    "mean": _quiet(lambda b: np.nanmean(b, axis=1)),
    "std": _quiet(lambda b: np.sqrt(_red_var(b))),
    "var": _quiet(_red_var),
    "max": _quiet(lambda b: np.nanmax(b, axis=1)),
    "min": _quiet(lambda b: np.nanmin(b, axis=1)),
    "median": _quiet(lambda b: np.nanmedian(b, axis=1)),
    "count": lambda b: np.isfinite(b).sum(axis=1).astype(np.float64),
    "skew": _quiet(_red_skew),
    "kurt": _quiet(_red_kurt),
}


@_quiet
def _pair(a, b, how):
    v = np.isfinite(a) & np.isfinite(b)
    n = v.sum(axis=1)
    if how == "wmean":
        w = np.where(v, b, 0.0)
        out = (np.where(v, a, 0.0) * w).sum(axis=1) / w.sum(axis=1)
        return np.where(np.isfinite(out), out, np.nan)
    ma = np.where(v, a, 0.0).sum(axis=1) / np.maximum(n, 1)
    mb = np.where(v, b, 0.0).sum(axis=1) / np.maximum(n, 1)
    da = np.where(v, a - np.expand_dims(ma, 1), 0.0)
    db = np.where(v, b - np.expand_dims(mb, 1), 0.0)
    cov = (da * db).sum(axis=1)
    if how == "cov":
        out = cov / np.maximum(n - 1, 1)
    elif how == "corr":
        out = np.clip(cov / np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1)), -1, 1)
    elif how == "beta":  # slope of a on b
        out = cov / (db**2).sum(axis=1)
    else:
        raise ValueError(f"unknown two-input reduction {how!r}; use corr/cov/beta/wmean")
    return np.where((n >= 2) & np.isfinite(out), out, np.nan)


def reduce(block, how, **kw) -> np.ndarray:
    """Reduce minute blocks along axis 1.

    block: ``(T, k[, N])`` or a 2-tuple of such blocks for ``corr / cov / beta / wmean``.
    how: a name in :data:`REDUCERS`, ``'quantile'`` (with ``q=``), a two-input name,
    or any callable ``f(block) -> array`` reducing axis 1.
    """
    if callable(how):
        return np.asarray(how(block), dtype=np.float64)
    if isinstance(block, tuple):
        a, b = (np.asarray(x, dtype=np.float64) for x in block)
        return _pair(a, b, how)
    block = np.asarray(block, dtype=np.float64)
    if how == "quantile":
        q = kw.get("q")
        if q is None:
            raise ValueError("how='quantile' needs q=")
        return _quiet(lambda b: np.nanquantile(b, q, axis=1))(block)
    if how in ("vwap", "twap"):
        raise ValueError(f"{how!r} is a field, not a reduction")
    try:
        fn = REDUCERS[how]
    except KeyError:
        raise ValueError(
            f"unknown reduction {how!r}; known: {sorted(REDUCERS)} + quantile, corr, cov, beta, wmean"
        ) from None
    return np.asarray(fn(block), dtype=np.float64)


# --------------------------------------------------------------------------- bar grid


def bar_open_ms(j, k: int, offset: int):
    return config.ORIGIN_MS + (offset + np.asarray(j, dtype=np.int64) * k) * MIN


def _m1_meta(symbol: str, meta: dict | None) -> dict:
    """Build versions + klines listing range of a symbol (memoised in ``meta``)."""
    key = ("m1", symbol)
    if meta is not None and key in meta:
        return meta[key]
    g = store.group(m1.m1_path(symbol), create=False)
    attrs = g.attrs.asdict() if g is not None else {}
    info = {
        "versions": {
            k[6:]: int(v.get("version", 0)) for k, v in attrs.items() if k.startswith("build:")
        },
        "range": (attrs.get("build:klines") or {}).get("range"),
    }
    if not info["range"]:
        raise RuntimeError(f"klines for {symbol} not built")
    if meta is not None:
        meta[key] = info
    return info


def symbol_bar_range(symbol: str, k: int, offset: int, meta: dict | None = None) -> tuple[int, int]:
    """[j0, j1) bar indices fully inside the symbol's klines listing range."""
    first_ms, last_ms = _m1_meta(symbol, meta)["range"]
    a = m1.ms_to_idx(first_ms) - offset
    b = m1.ms_to_idx(last_ms) + 1 - offset
    return -(-a // k), b // k


def compute_bar_field(symbol: str, name: str, j0: int, j1: int, k: int, offset: int) -> np.ndarray:
    if j1 <= j0:
        return np.empty(0)
    s_ms, e_ms = int(bar_open_ms(j0, k, offset)), int(bar_open_ms(j1, k, offset))
    base, window = catalog.parse_exec_field(name)
    w = k if window is None else window
    if not 1 <= w <= k:
        raise ValueError(f"exec window must be in [1, {k}] minutes, got {w}")
    nb = j1 - j0
    if base in ("vwap", "twap"):
        px = catalog.minute_values(symbol, "twap", s_ms, e_ms).reshape(nb, k)[:, :w]
        twap = REDUCERS["mean"](px)
        if base == "twap":
            return twap
        qv = m1.read_minutes(symbol, "quote_volume", s_ms, e_ms).reshape(nb, k)[:, :w]
        v = m1.read_minutes(symbol, "volume", s_ms, e_ms).reshape(nb, k)[:, :w]
        sq, sv = _red_sum(qv), _red_sum(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = sq / sv
        return np.where((sv > 0) & np.isfinite(vwap), vwap, twap)
    spec = catalog.spec_for(name)
    x = catalog.minute_values(symbol, name, s_ms, e_ms).reshape(nb, k)
    return reduce(x, spec.rule)


def _bar_path(k, offset, symbol, name=None) -> str:
    base = f"bar/{tag(k, offset)}/{symbol}"
    return base if name is None else f"{base}/{name}"


def _src_version(symbol: str, name: str, meta: dict | None = None) -> str:
    spec = catalog.spec_for(name)
    versions = _m1_meta(symbol, meta)["versions"]
    v = f"{versions.get('klines', 0)}"
    if spec.builder != "klines":
        v += f":{versions.get(spec.builder, 0)}"
    return f"{v}|{spec.rule}"


def ensure_bar_field(
    symbol: str, name: str, k: int, offset: int, meta: dict | None = None
) -> tuple[int, int]:
    """Make sure the L2 array is current; returns the symbol's [j0, j1) bar range.

    ``meta`` memoises store metadata within one load (pass the same dict for every field).
    """
    j0, j1 = symbol_bar_range(symbol, k, offset, meta)
    gpath = _bar_path(k, offset, symbol)
    want = {"src": _src_version(symbol, name, meta), "j0": j0, "j1": j1}
    akey = ("l2", gpath)
    if meta is not None and akey in meta:
        attrs = meta[akey]
    else:
        g = store.group(gpath, create=False)
        attrs = g.attrs.asdict() if g is not None else {}
        if meta is not None:
            meta[akey] = attrs
    if attrs.get(f"v:{name}") == want:
        return j0, j1
    with store.symbol_lock(symbol):
        vals = compute_bar_field(symbol, name, j0, j1, k, offset)
        store.delete(_bar_path(k, offset, symbol, name))
        store.write(_bar_path(k, offset, symbol, name), j0, vals)
        store.set_attr(gpath, f"v:{name}", want)
    attrs[f"v:{name}"] = want
    return j0, j1


def read_bar_field(symbol: str, name: str, k: int, offset: int, jlo: int, jhi: int) -> np.ndarray:
    return store.read(_bar_path(k, offset, symbol, name), jlo, jhi)


def user_path(k: int, offset: int, symbol: str, name: str) -> str:
    return _bar_path(k, offset, symbol, f"user/{name}")
