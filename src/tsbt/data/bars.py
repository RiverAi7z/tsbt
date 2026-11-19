"""Bar synthesis from the minute grid (reshape + NaN-aware reductions) and the L2 cache.

The minute grid is regular and hole-free (missing minutes are NaN), so the minutes of
bar-aligned range form a ``(T, k)`` block and every bar statistic is a reduction along
axis 1.  Fixed field rules and user ``BarData.agg`` share this code path.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable

import numpy as np

from tsbt import config
from tsbt.data import catalog, m1, store

MIN = config.MINUTE_MS
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MissingDataWarning(UserWarning):
    """Bars were synthesised from minute data with holes inside the listing range."""


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


def _red_std_moment(b, k: int, excess: float):
    """Standardised population moment m_k / m2^(k/2) - excess; needs >= k samples."""
    n, _, dev = _moments(b)
    m2 = (dev**2).sum(axis=1) / np.maximum(n, 1)
    mk = (dev**k).sum(axis=1) / np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = mk / m2 ** (k / 2) - excess
    return np.where((n >= k) & np.isfinite(out), out, np.nan)


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
    "skew": _quiet(lambda b: _red_std_moment(b, 3, 0.0)),
    "kurt": _quiet(lambda b: _red_std_moment(b, 4, 3.0)),
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


def _memo(meta: dict | None, key, fn):
    """``fn()`` memoised in ``meta`` (store metadata shared within one load)."""
    if meta is None:
        return fn()
    if key not in meta:
        meta[key] = fn()
    return meta[key]


def _read_m1_meta(symbol: str) -> dict:
    mans = m1.manifests(symbol)
    info = {
        "versions": {k: int(v.get("version", 0)) for k, v in mans.items()},
        "range": (mans.get("klines") or {}).get("range"),
    }
    if not info["range"]:
        raise RuntimeError(f"klines for {symbol} not built")
    return info


def _m1_meta(symbol: str, meta: dict | None) -> dict:
    """Build versions + klines listing range of a symbol (memoised in ``meta``)."""
    return _memo(meta, ("m1", symbol), lambda: _read_m1_meta(symbol))


def symbol_bar_range(symbol: str, k: int, offset: int, meta: dict | None = None) -> tuple[int, int]:
    """[j0, j1) bar indices fully inside the symbol's klines listing range."""
    first_ms, last_ms = _m1_meta(symbol, meta)["range"]
    a = m1.ms_to_idx(first_ms) - offset
    b = m1.ms_to_idx(last_ms) + 1 - offset
    return -(-a // k), b // k


def compute_bar_field(symbol: str, name: str, j0: int, j1: int, k: int, offset: int) -> np.ndarray:
    return _synthesise(symbol, name, j0, j1, k, offset)[0]


def _synthesise(symbol: str, name: str, j0: int, j1: int, k: int, offset: int):
    """(bar values, missing): ``missing`` marks the source holes behind the bars -
    ``(nb, w)`` absent minutes for a dense (klines) field, ``(nb,)`` bars without any
    value for a sparse one. Bars [j0, j1) all lie inside the symbol's listing range."""
    if j1 <= j0:
        return np.empty(0), np.zeros(0, dtype=bool)
    s_ms, e_ms = int(bar_open_ms(j0, k, offset)), int(bar_open_ms(j1, k, offset))
    base, window = catalog.parse_exec_field(name)
    w = k if window is None else window
    if not 1 <= w <= k:
        raise ValueError(f"exec window must be in [1, {k}] minutes, got {w}")
    nb = j1 - j0
    if base in ("vwap", "twap"):
        qv, v, c = (
            m1.read_minutes(symbol, f, s_ms, e_ms).reshape(nb, k)[:, :w]
            for f in ("quote_volume", "volume", "close")
        )
        twap = REDUCERS["mean"](catalog.minute_price(qv, v, c))
        if base == "twap":
            return twap, ~np.isfinite(c)
        sq, sv = _red_sum(qv), _red_sum(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = sq / sv
        return np.where((sv > 0) & np.isfinite(vwap), vwap, twap), ~np.isfinite(c)
    spec = catalog.spec_for(name)
    x = catalog.minute_values(symbol, name, s_ms, e_ms).reshape(nb, k)
    out = reduce(x, spec.rule)
    if spec.dense:
        return out, ~np.isfinite(x)
    if k < spec.m1_ffill:  # a bar shorter than the source's cadence may be empty
        return out, np.zeros(nb, dtype=bool)
    return out, ~np.isfinite(out)


def _warn_missing(symbol, name, k, offset, j0, missing, meta) -> None:
    """Report the holes found while synthesising ``name``: once per symbol and source
    for klines fields (they share their rows), once per field otherwise."""
    if not missing.any():
        return
    spec = catalog.spec_for(name)
    dense = missing.ndim == 2
    key = ("warned", symbol, spec.builder if dense else name, k, offset)
    if meta is not None:
        if key in meta:
            return
        meta[key] = True
    opens = bar_open_ms(j0 + np.arange(missing.shape[0]), k, offset)
    freq = freq_label(k)
    if dense:
        flat = missing.ravel()
        t = (opens[:, None] + np.arange(missing.shape[1]) * MIN).ravel()
        n_bars = int(missing.any(axis=1).sum())
        what = f"{spec.builder} source lacks {int(flat.sum())} minutes in {n_bars} {freq} bars"
    else:
        flat, t = missing, opens
        what = f"field {name!r} has no source data for {int(flat.sum())} {freq} bars"
    runs = np.flatnonzero(np.diff(np.r_[False, flat, False])).reshape(-1, 2)  # [a, z)
    spans = "; ".join(f"{_iso(t[a])} .. {_iso(t[z - 1])}" for a, z in runs[:3])
    if len(runs) > 3:
        spans += f"; +{len(runs) - 3} more"
    warnings.warn(f"{symbol}: {what} ({spans})", MissingDataWarning, skip_file_prefixes=(_PKG_DIR,))


def _iso(ms) -> str:
    t = np.datetime64(int(ms), "ms").astype("datetime64[m]")
    return str(t).replace("T", " ")


def tag_path(bar_tag: str, symbol: str, name: str | None = None) -> str:
    """Store path of a symbol's bar group (or one field) for a bar grid tag."""
    base = f"bar/{bar_tag}/{symbol}"
    return base if name is None else f"{base}/{name}"


def _bar_path(k, offset, symbol, name=None) -> str:
    return tag_path(tag(k, offset), symbol, name)


def field_attr(name: str) -> str:
    """Attribute of a bar group recording how its field ``name`` was built."""
    return f"v:{name}"


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
    attrs = _memo(meta, ("l2", gpath), lambda: _group_attrs(gpath))
    if attrs.get(field_attr(name)) == want:
        return j0, j1
    path = _bar_path(k, offset, symbol, name)
    with store.symbol_lock(symbol):
        vals, missing = _synthesise(symbol, name, j0, j1, k, offset)
        store.delete(path)
        store.write(path, j0, vals)
        store.set_attr(gpath, field_attr(name), want)
    attrs[field_attr(name)] = want
    _warn_missing(symbol, name, k, offset, j0, missing, meta)  # after the cache is written
    return j0, j1


def _group_attrs(path: str) -> dict:
    g = store.group(path, create=False)
    return g.attrs.asdict() if g is not None else {}


def read_bar_field(symbol: str, name: str, k: int, offset: int, jlo: int, jhi: int) -> np.ndarray:
    return store.read(_bar_path(k, offset, symbol, name), jlo, jhi)


def user_path(k: int, offset: int, symbol: str, name: str) -> str:
    return _bar_path(k, offset, symbol, f"user/{name}")
