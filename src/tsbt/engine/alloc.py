"""Capital allocation across symbols in a pooled portfolio.

An allocation is a ``(T, N)`` weight matrix; row ``t`` counts as known at the close of
bar ``t`` and is delayed by the engine exactly like the signal, so
``position[t] = alloc[t - delay] * signal[t - delay]``.

Accepted forms for ``bt.run(..., alloc=...)``:

* ``None`` / :func:`equal` - static 1/N (default)
* :func:`equal_active` - 1/N among symbols listed at t
* :func:`fixed` or an ``(N,)`` array - static weights
* a ``(T, N)`` array - arbitrary time-varying weights (used as is)
* :func:`inv_vol` / :func:`rolling_sharpe` - driven by each symbol's stand-alone strategy
* any callable ``f(ctx) -> (T, N)`` (``ctx`` is :class:`AllocContext`)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tsbt.ops import ts_mean, ts_std_dev


@dataclass
class AllocContext:
    ts: np.ndarray  # (T,) bar open times
    symbols: list[str]
    signal: np.ndarray  # (T, N) raw (undelayed) signal
    data: object  # BarData
    listed: np.ndarray  # (T, N)
    _independent: Callable[[], tuple[np.ndarray, np.ndarray]] = field(repr=False, default=None)
    _cache: dict = field(default_factory=dict, repr=False)

    def _ind(self):
        if "ind" not in self._cache:
            self._cache["ind"] = self._independent()
        return self._cache["ind"]

    @property
    def ret(self) -> np.ndarray:
        """(T, N) per-bar net return of each symbol's stand-alone strategy; row t is
        realised at the close of bar t (NaN before the symbol's start)."""
        return self._ind()[0]

    @property
    def ret_gross(self) -> np.ndarray:
        return self._ind()[1]


class Alloc:
    """Base class: subclasses implement ``raw(ctx)``; options handled here."""

    #: static weights are used for every bar, listed or not (idle capital stays idle)
    static = False
    #: user callables are taken as they are: NaN means "no allocation", never a fallback
    as_is = False
    name = "alloc"

    def __init__(self, normalize: bool = True, rebalance_every: int | None = None):
        self.normalize = normalize
        self.rebalance_every = rebalance_every

    def raw(self, ctx: AllocContext) -> np.ndarray:  # (T, N), NaN = warm-up
        raise NotImplementedError

    def _params(self) -> dict:
        return {}

    def __repr__(self) -> str:
        args = {
            **self._params(),
            "normalize": self.normalize,
            "rebalance_every": self.rebalance_every,
        }
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
        return f"{self.name}({shown})"

    def weights(self, ctx: AllocContext) -> np.ndarray:
        w = np.asarray(self.raw(ctx), dtype=np.float64)
        T, N = ctx.listed.shape
        w = np.broadcast_to(w, (T, N)).copy()
        w[~np.isfinite(w)] = np.nan
        if self.static or self.as_is:
            w = np.nan_to_num(w, nan=0.0)
        else:
            w[~ctx.listed] = np.nan
            # NaN = not enough history. A row where no listed symbol has a weight yet
            # falls back to equal weights among listed symbols; otherwise NaN -> 0.
            warm = ~np.isfinite(w).any(axis=1)
            eq = ctx.listed / np.maximum(ctx.listed.sum(axis=1, keepdims=True), 1)
            w = np.where(warm[:, None], eq, np.nan_to_num(w, nan=0.0))
        if self.normalize:
            tot = np.abs(w).sum(axis=1, keepdims=True)
            w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
        if self.rebalance_every and self.rebalance_every > 1:
            k = int(self.rebalance_every)
            keep = (np.arange(T) % k) == 0
            idx = np.maximum.accumulate(np.where(keep, np.arange(T), 0))
            w = w[idx]
        return w


class _Static(Alloc):
    static = True

    def __init__(self, w=None, **kw):
        super().__init__(**kw)
        self.w = w
        self.name = "equal" if w is None else "fixed"

    def _params(self):
        return {} if self.w is None else {"w": np.asarray(self.w, dtype=np.float64).tolist()}

    def raw(self, ctx):
        N = ctx.listed.shape[1]
        if self.w is None:
            return np.full(N, 1.0 / N)
        w = np.asarray(self.w, dtype=np.float64)
        if w.shape != (N,):
            raise ValueError(f"fixed weights need shape ({N},), got {w.shape}")
        return w


def equal(**kw) -> Alloc:
    """Static 1/N (capital of not-yet-listed symbols stays idle)."""
    return _Static(None, **kw)


def fixed(w, normalize: bool = True, **kw) -> Alloc:
    """Static user weights (normalised to sum 1 by default)."""
    return _Static(w, normalize=normalize, **kw)


class _EqualActive(Alloc):
    name = "equal_active"

    def raw(self, ctx):
        return np.ones(ctx.listed.shape)


def equal_active(**kw) -> Alloc:
    """1 / (number of symbols listed at t)."""
    return _EqualActive(**kw)


class _InvVol(Alloc):
    name = "inv_vol"

    def __init__(self, window, **kw):
        super().__init__(**kw)
        self.window = int(window)

    def _params(self):
        return {"window": self.window}

    def raw(self, ctx):
        sd = ts_std_dev(ctx.ret, self.window)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / sd
        return np.where(np.isfinite(w), w, np.nan)


def inv_vol(window: int, **kw) -> Alloc:
    """Weights proportional to 1 / rolling std of each symbol's strategy returns."""
    return _InvVol(window, **kw)


class _Sharpe(Alloc):
    name = "rolling_sharpe"

    def __init__(self, window, floor=0.0, **kw):
        super().__init__(**kw)
        self.window = int(window)
        self.floor = float(floor)

    def _params(self):
        return {"window": self.window, "floor": self.floor}

    def raw(self, ctx):
        r = ctx.ret
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = ts_mean(r, self.window) / ts_std_dev(r, self.window)
        sr = np.where(np.isfinite(sr), sr, np.nan)
        return np.where(np.isnan(sr), np.nan, np.maximum(sr, self.floor))


def rolling_sharpe(window: int, floor: float = 0.0, **kw) -> Alloc:
    """Weights proportional to each symbol's rolling strategy Sharpe, floored at ``floor``.

    Rows where every weight is floored to 0 hold no positions.
    """
    return _Sharpe(window, floor, **kw)


class _Callable(Alloc):
    as_is = True
    name = "custom"

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn

    def _params(self):
        return {"fn": getattr(self.fn, "__name__", type(self.fn).__name__)}

    def raw(self, ctx):
        return self.fn(ctx)


def custom(fn: Callable[[AllocContext], np.ndarray], normalize: bool = False, **kw) -> Alloc:
    """Wrap a callable so ``normalize`` / ``rebalance_every`` can be applied to it.

    Its output is otherwise used as is, exactly like passing the callable directly:
    NaN (and weights of unlisted symbols) mean "no allocation".
    """
    return _Callable(fn, normalize=normalize, **kw)


def resolve(alloc, ctx: AllocContext) -> np.ndarray:
    T, N = ctx.listed.shape
    if alloc is None:
        alloc = equal()
    if isinstance(alloc, Alloc):
        w = alloc.weights(ctx)
    elif callable(alloc):
        w = np.asarray(alloc(ctx), dtype=np.float64)
    else:
        w = np.asarray(alloc, dtype=np.float64)
        if w.shape == (N,):
            w = np.broadcast_to(w, (T, N))
    if w.shape != (T, N):
        raise ValueError(f"allocation must be ({T}, {N}), got {w.shape}")
    return np.where(np.isfinite(w), w, 0.0)  # NaN / inf -> no allocation


def benchmark_weights(alloc, ctx: AllocContext) -> np.ndarray:
    """``(N,)`` initial weights of the buy-and-hold benchmark basket.

    A static allocation supplies its own weights. Anything time-varying (rolling
    Sharpe, inverse vol, arrays, callables) may depend on how the strategy performed,
    so the benchmark uses a plain 1/N basket instead.
    """
    N = ctx.listed.shape[1]
    if alloc is None:
        alloc = equal()
    if isinstance(alloc, Alloc) and alloc.static:
        return alloc.weights(ctx)[0]
    if not isinstance(alloc, Alloc) and not callable(alloc):
        w = np.asarray(alloc, dtype=np.float64)
        if w.shape == (N,):
            return np.where(np.isfinite(w), w, 0.0)
    return np.full(N, 1.0 / N)


def describe(alloc) -> str:
    """Short text recorded in the result config."""
    if alloc is None:
        return repr(equal())
    if isinstance(alloc, Alloc):
        return repr(alloc)
    if callable(alloc):
        return f"callable({getattr(alloc, '__name__', type(alloc).__name__)})"
    return f"array{np.shape(alloc)}"
