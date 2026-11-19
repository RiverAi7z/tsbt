"""Public data API: ``load`` (bars), ``load_m1`` (minutes) and :class:`BarData`."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np

from tsbt import config
from tsbt.data import bars, catalog, m1, sources, store
from tsbt.data import features as feats
from tsbt.ops._util import ffill as _ffill

MIN = config.MINUTE_MS


def to_ms(t) -> int | None:
    if t is None:
        return None
    if isinstance(t, (int, np.integer)):
        return int(t)
    if isinstance(t, np.datetime64):
        return int(t.astype("datetime64[ms]").astype(np.int64))
    if isinstance(t, str):
        t = dt.datetime.fromisoformat(t)
    if isinstance(t, dt.date) and not isinstance(t, dt.datetime):
        t = dt.datetime(t.year, t.month, t.day)
    if isinstance(t, dt.datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.UTC)
        return int(t.timestamp() * 1000)
    raise TypeError(f"cannot interpret {t!r} as a time")


def _ms_to_dt64(ms) -> np.ndarray:
    return np.asarray(ms, dtype=np.int64).astype("datetime64[ms]")


class _Fields:
    """Lazy field container shared by bar and minute views."""

    def __init__(self, symbols: list[str], scalar: bool):
        self.symbols = symbols
        self._scalar = scalar
        self._cache: dict[str, np.ndarray] = {}

    def _out(self, arr: np.ndarray) -> np.ndarray:
        return arr[:, 0] if self._scalar else arr

    def _load(self, name: str) -> np.ndarray:  # (n, N)
        raise NotImplementedError

    def raw(self, name: str) -> np.ndarray:
        """Always 2-D ``(n, N)``."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def __getitem__(self, name: str) -> np.ndarray:
        return self._out(self.raw(name))

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None


def _ensure(sym: str, names, workers: int, done: set, meta: dict | None = None) -> None:
    todo = [k for k in catalog.builder_keys(names) if (sym, k) not in done]
    if todo:
        m1.ensure(sym, [catalog.builder(k) for k in todo], workers=workers)
        done.update((sym, k) for k in todo)
        if meta is not None:
            meta.pop(("m1", sym), None)  # versions may have changed


class M1View(_Fields):
    """Minute data over [start_ms, start_ms + n minutes)."""

    def __init__(self, symbols, scalar, start_ms: int, n: int, workers: int = 8):
        super().__init__(symbols, scalar)
        self.start_ms = start_ms
        self.n = n
        self._workers = workers
        self._done: set = set()
        self.ts = _ms_to_dt64(start_ms + np.arange(n, dtype=np.int64) * MIN)

    def _load(self, name: str) -> np.ndarray:
        spec = catalog.spec_for(name)
        if spec is None or "@" in name:
            raise KeyError(f"unknown minute field {name!r}")
        out = np.full((self.n, len(self.symbols)), np.nan)
        end = self.start_ms + self.n * MIN
        for i, sym in enumerate(self.symbols):
            _ensure(sym, [name], self._workers, self._done)
            out[:, i] = catalog.minute_values(sym, name, self.start_ms, end)
        return _ffill(out, spec.m1_ffill) if spec.m1_ffill > 0 else out


class BarData(_Fields):
    """Bar fields on a common UTC bar grid. ``d.close`` is ``(T,)`` for a single
    symbol and ``(T, N)`` for a list of symbols. Fields load lazily on access."""

    def __init__(self, symbols, scalar, k, offset, jlo, jhi, ranges, workers=8):
        super().__init__(symbols, scalar)
        self.k = k
        self.offset = offset
        self.freq = bars.freq_label(k)
        self._jlo, self._jhi = jlo, jhi
        self._workers = workers
        self._m1: M1View | None = None
        self._done: set = set()
        self._meta: dict = {}
        j = np.arange(jlo, jhi)
        self.open_ms = bars.bar_open_ms(j, k, offset)
        self.ts = _ms_to_dt64(self.open_ms)
        self.close_ts = _ms_to_dt64(self.open_ms + k * MIN)
        self.listed = np.stack([(j >= a) & (j < b) for a, b in ranges], axis=1)

    # ------------------------------------------------------------------ basics
    @property
    def T(self) -> int:
        return self._jhi - self._jlo

    @property
    def N(self) -> int:
        return len(self.symbols)

    @property
    def scalar(self) -> bool:
        return self._scalar

    def __len__(self) -> int:
        return self.T

    def __repr__(self) -> str:
        span = f"{self.ts[0]} .. {self.close_ts[-1]}" if self.T else "empty"
        return f"BarData({self.symbols if not self._scalar else self.symbols[0]}, freq={self.freq}, T={self.T}, {span})"

    def _load(self, name: str) -> np.ndarray:
        out = np.full((self.T, self.N), np.nan)
        spec = catalog.spec_for(name)
        for i, sym in enumerate(self.symbols):
            if spec is not None:
                _ensure(sym, [name], self._workers, self._done, self._meta)
                bars.ensure_bar_field(sym, name, self.k, self.offset, self._meta)
                out[:, i] = bars.read_bar_field(
                    sym, name, self.k, self.offset, self._jlo, self._jhi
                )
            else:
                path = bars.user_path(self.k, self.offset, sym, name)
                if not store.has_array(path):
                    raise KeyError(
                        f"unknown field {name!r} for {sym} at {self.freq}: not builtin "
                        f"({sorted(catalog.BUILTIN_FIELDS)}), not a feature "
                        f"({sorted(feats.feature_fields())}) and no saved user field"
                    )
                out[:, i] = store.read(path, self._jlo, self._jhi)
        return out

    def exec_price(self, price: str, window: int | None = None) -> np.ndarray:
        """(T, N) execution price inside each bar."""
        if price not in ("open", "close", "twap", "vwap"):
            raise ValueError(f"price must be open/twap/vwap/close, got {price!r}")
        if window is not None and price in ("twap", "vwap") and window != self.k:
            return self.raw(f"{price}@{int(window)}")
        return self.raw(price)

    # ------------------------------------------------------------------ minute view
    @property
    def m1(self) -> M1View:
        """Minute data exactly aligned with the bars: length ``T * k``."""
        if self._m1 is None:
            self._m1 = M1View(
                self.symbols,
                self._scalar,
                int(self.open_ms[0]) if self.T else 0,
                self.T * self.k,
                self._workers,
            )
            self._m1._done = self._done
        return self._m1

    def agg(self, x, how, window: int | None = None, **kw) -> np.ndarray:
        """Aggregate minute-level array(s) aligned with ``self.m1`` into bars.

        ``x``: array of length ``T*k`` (optionally ``(T*k, N)``) or a 2-tuple of such for
        corr/cov/beta/wmean. ``window=N`` uses only the first N minutes of each bar.
        """

        def blk(a):
            a = np.asarray(a, dtype=np.float64)
            if a.shape[0] != self.T * self.k:
                raise ValueError(
                    f"minute array length {a.shape[0]} != T*k = {self.T}*{self.k}; "
                    "derive it from d.m1 so it stays aligned"
                )
            b = a.reshape(self.T, self.k, *a.shape[1:])
            if window is not None:
                if not 1 <= window <= self.k:
                    raise ValueError(f"window must be in [1, {self.k}]")
                b = b[:, :window]
            return b

        block = tuple(blk(a) for a in x) if isinstance(x, tuple) else blk(x)
        return bars.reduce(block, how, **kw)

    # ------------------------------------------------------------------ user fields
    def save_field(self, name: str, values, overwrite: bool = False) -> None:
        """Cache a bar-level array under ``name`` for this frequency (per symbol).

        ``overwrite=True`` replaces the saved field entirely: values previously saved
        outside this object's time range are dropped, so two versions never mix.
        """
        if catalog.spec_for(name) is not None:
            raise ValueError(f"{name!r} is a builtin/feature field name")
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        if v.shape != (self.T, self.N):
            raise ValueError(f"values shape {v.shape} != {(self.T, self.N)}")
        for i, sym in enumerate(self.symbols):
            path = bars.user_path(self.k, self.offset, sym, name)
            if store.has_array(path):
                if not overwrite:
                    raise FileExistsError(
                        f"user field {name!r} exists for {sym}; pass overwrite=True"
                    )
                store.delete(path)
            store.write(path, self._jlo, v[:, i])
        self._cache[name] = v

    # ------------------------------------------------------------------ external
    def align(self, ts, values, ffill: bool = False) -> np.ndarray:
        """Align an external series to bars by the time each value became *known*.

        A value known at time tau lands on the bar with open < tau <= close; several
        values in one bar keep the last; bars without a value are NaN (or ffilled).
        """
        ts_arr = np.asarray(ts)
        if ts_arr.dtype.kind == "M":
            t = ts_arr.astype("datetime64[ms]").astype(np.int64)
        elif ts_arr.dtype.kind in "iu":
            t = ts_arr.astype(np.int64)  # epoch ms
        else:
            t = np.array([to_ms(x) for x in ts_arr.ravel()], dtype=np.int64)
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        if len(t) != v.shape[0]:
            raise ValueError("ts and values lengths differ")
        if v.shape[1] not in (1, self.N):
            raise ValueError(f"values must have 1 or {self.N} columns")
        F = self.k * MIN
        j = -(-(t - int(self.open_ms[0])) // F) - 1  # ceil((t-open0)/F) - 1
        order = np.argsort(t, kind="stable")
        j, v = j[order], v[order]
        out = np.full((self.T, v.shape[1]), np.nan)
        for c in range(v.shape[1]):
            ok = (j >= 0) & (j < self.T) & np.isfinite(v[:, c])
            jj, vv = j[ok][::-1], v[ok, c][::-1]
            uj, first = np.unique(jj, return_index=True)  # first in reversed = last in time
            out[uj, c] = vv[first]
        if ffill:
            out = _ffill(out)
        if v.shape[1] == 1 and self.N > 1:
            out = np.repeat(out, self.N, axis=1)
        return out[:, 0] if self._scalar else out


def _norm_symbols(symbols) -> tuple[list[str], bool]:
    if isinstance(symbols, str):
        return [symbols], True
    syms = list(symbols)
    if not syms:
        raise ValueError("no symbols")
    if len(set(syms)) != len(syms):
        raise ValueError("duplicate symbols")
    return syms, False


def load(
    symbols: str | Sequence[str],
    freq: str | int = "1h",
    fields: Sequence[str] | None = None,
    start=None,
    end=None,
    offset: int = 0,
    workers: int = 8,
) -> BarData:
    """Load bars. First use builds and caches the minute store; later calls only
    compare source-file listings.

    ``start``/``end``: keep bars whose open time is in ``[start, end)``.
    ``offset``: shift the bar phase by this many minutes (< bar length).
    """
    syms, scalar = _norm_symbols(symbols)
    k = bars.parse_freq(freq)
    if not 0 <= offset < k:
        raise ValueError(f"offset must be in [0, {k})")
    names = list(catalog.KLINE_DEFAULT_FIELDS if fields is None else fields)
    done: set = set()
    meta: dict = {}
    ranges = []
    for sym in syms:
        _ensure(sym, names, workers, done, meta)
        ranges.append(bars.symbol_bar_range(sym, k, offset, meta))
    live = [(a, b) for a, b in ranges if b > a]  # a symbol may not span one full bar
    if not live:
        raise ValueError("no complete bars in the requested range")
    jlo = min(a for a, _ in live)
    jhi = max(b for _, b in live)
    F = k * MIN
    base = config.ORIGIN_MS + offset * MIN
    if (s := to_ms(start)) is not None:
        jlo = max(jlo, -(-(s - base) // F))
    if (e := to_ms(end)) is not None:
        jhi = min(jhi, -(-(e - base) // F))
    if jhi <= jlo:
        raise ValueError("no complete bars in the requested range")
    d = BarData(syms, scalar, k, offset, jlo, jhi, ranges, workers)
    d._done = done
    d._meta = meta
    for name in names:
        d.raw(name)
    return d


def load_m1(
    symbols, fields: Sequence[str] = ("close",), start=None, end=None, workers: int = 8
) -> M1View:
    """Minute data over [start, end) (default: union of the symbols' klines ranges)."""
    syms, scalar = _norm_symbols(symbols)
    done: set = set()
    lo, hi = [], []
    for sym in syms:
        _ensure(sym, list(fields), workers, done)
        a, b = m1.listing_range(sym)
        lo.append(a)
        hi.append(b + MIN)
    s = to_ms(start) if start is not None else min(lo)
    e = to_ms(end) if end is not None else max(hi)
    s, e = (s // MIN) * MIN, -(-e // MIN) * MIN
    view = M1View(syms, scalar, s, (e - s) // MIN, workers)
    view._done = done
    for f in fields:
        view.raw(f)
    return view


def symbols(dataset: str = "klines") -> list[str]:
    return sources.list_symbols(dataset)


def fields() -> dict[str, str]:
    out = {name: spec.builder for name, spec in catalog.BUILTIN_FIELDS.items()}
    out.update({f: f"feat:{fs}" for f, fs in feats.feature_fields().items()})
    return out
