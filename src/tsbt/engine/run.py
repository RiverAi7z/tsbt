"""``bt.run``: signal -> positions -> P&L."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsbt.data.api import BarData, to_ms
from tsbt.engine import alloc as alloc_mod
from tsbt.engine.core import simulate
from tsbt.engine.result import BatchResult, PortfolioResult, Result


def _ffill(x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    rows = np.arange(x.shape[0])[:, None]
    idx = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    got = np.take_along_axis(x, np.maximum(idx, 0), axis=0)
    return np.where(idx >= 0, got, np.nan)


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d] if d else x
    return out


@dataclass
class _Prepared:
    s_raw: np.ndarray  # delayed signal, NaN kept
    target: np.ndarray  # delayed signal after nan policy, 0 when untradable
    R: np.ndarray
    b: np.ndarray
    close: np.ndarray  # forward-filled close (for plots)
    listed: np.ndarray
    hold: np.ndarray  # rows where the position must be held (price gap)
    flat: np.ndarray  # rows where the position must be 0 (unlisted / delisted)
    start: np.ndarray  # first row with a usable signal per column (-1 = none)
    last: np.ndarray  # last listed row per column (-1 = never)


def _prepare(
    s: np.ndarray, P: np.ndarray, C: np.ndarray, listed: np.ndarray, nan_policy: str
) -> _Prepared:
    n, m = s.shape
    rows = np.arange(n)[:, None]
    ok_px = np.isfinite(P) & np.isfinite(C) & (P > 0)
    any_listed = listed.any(axis=0)
    last = np.where(any_listed, n - 1 - np.argmax(listed[::-1], axis=0), -1)
    # A column whose data ends before the evaluation end is closed at its last bar.
    delisted = (rows >= last[None, :]) & (last[None, :] < n - 1)
    flat = ~listed | delisted
    hold = listed & ~ok_px & ~flat

    usable = np.isfinite(s) & listed
    start = np.where(usable.any(axis=0), np.argmax(usable, axis=0), -1)

    if nan_policy == "flat":
        tgt = np.nan_to_num(s, nan=0.0)
    elif nan_policy == "hold":
        tgt = np.nan_to_num(_ffill(s), nan=0.0)
    else:
        raise ValueError("nan_policy must be 'flat' or 'hold'")

    Pf, Cf = _ffill(np.where(ok_px, P, np.nan)), _ffill(np.where(ok_px, C, np.nan))
    R = np.zeros_like(Pf)
    with np.errstate(divide="ignore", invalid="ignore"):
        R[:-1] = Pf[1:] / Pf[:-1] - 1.0
        b = Cf / Pf - 1.0
    R = np.where(np.isfinite(R), R, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    return _Prepared(s, tgt, R, b, Cf, listed, hold, flat, start, last)


def _positions(target: np.ndarray, p: _Prepared) -> np.ndarray:
    th = np.where(p.flat, 0.0, target)
    th = np.where(p.hold, np.nan, th)  # no trade during a price gap: keep last target
    return np.nan_to_num(_ffill(th), nan=0.0)


def _independent_returns(p: _Prepared, fee: float, compound: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar returns of each column traded stand-alone (for allocators)."""
    th = _positions(p.target, p)
    out = []
    for f in (fee, 0.0):
        nav = simulate(th, p.R, p.b, f, compound, pooled=False).nav
        prev = np.vstack([np.ones((1, nav.shape[1])), nav[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = nav / prev - 1.0 if compound else nav - prev
        rows = np.arange(nav.shape[0])[:, None]
        r = np.where(np.isfinite(r), r, np.nan)
        out.append(np.where((rows >= p.start[None, :]) & (p.start[None, :] >= 0), r, np.nan))
    return out[0], out[1]


def run(
    signal,
    data: BarData,
    price: str = "vwap",
    exec_window: int | None = None,
    fee: float = 2e-4,
    compound: bool = True,
    delay: int = 1,
    mode: str | None = None,
    alloc=None,
    nan_policy: str = "flat",
    start=None,
    end=None,
    names: list[str] | None = None,
):
    """Backtest a position signal.

    ``signal[t]`` is the target position computed with information up to the close of
    bar t; it is executed inside bar ``t + delay`` at ``price`` (open / twap / vwap /
    close of that bar). The engine does the shift - never shift the signal yourself.
    NaN and +-inf both mean "no signal" (see ``nan_policy``).

    ``names`` labels the signal columns (the per-symbol accounts in portfolio mode).

    Returns :class:`Result` (one account), :class:`BatchResult` (independent columns)
    or :class:`PortfolioResult` (shared capital pool).
    """
    if int(delay) != delay or delay < 1:
        raise ValueError("delay must be an integer >= 1: a signal can only trade in a later bar")
    delay = int(delay)
    if fee < 0:
        raise ValueError("fee must be >= 0")
    sig = np.asarray(signal, dtype=np.float64)
    sig = np.where(np.isfinite(sig), sig, np.nan)  # +-inf is "no signal", not a huge position
    if sig.ndim == 1:
        sig = sig[:, None]
    if sig.ndim != 2 or sig.shape[0] != data.T:
        raise ValueError(
            f"signal must have {data.T} rows (one per bar), got shape {np.shape(signal)}"
        )
    T, N = data.T, data.N
    K = sig.shape[1]
    if N == 1:
        colmap = np.zeros(K, dtype=int)
        default_names = (
            [data.symbols[0]] if K == 1 else [f"{data.symbols[0]}#{i}" for i in range(K)]
        )
    elif K == N:
        colmap = np.arange(N)
        default_names = list(data.symbols)
    else:
        raise ValueError(f"signal has {K} columns but data has {N} symbols")
    names = list(names) if names is not None else default_names
    if len(names) != K:
        raise ValueError("names must match the signal columns")
    if mode is None:
        mode = "portfolio" if N > 1 else "independent"
    if mode not in ("portfolio", "independent"):
        raise ValueError("mode must be 'portfolio' or 'independent'")
    if mode == "portfolio" and N == 1:
        raise ValueError("portfolio mode needs several symbols")
    if alloc is not None and mode != "portfolio":
        raise ValueError("alloc only applies to mode='portfolio'")

    if price not in ("twap", "vwap") or exec_window == data.k:
        exec_window = None  # only twap / vwap have a window; the whole bar is the default
    P = data.exec_price(price, exec_window)[:, colmap]
    C = data.raw("close")[:, colmap]
    listed = data.listed[:, colmap]

    i0, i1 = 0, T
    if (s_ms := to_ms(start)) is not None:
        i0 = int(np.searchsorted(data.open_ms, s_ms, side="left"))
    if (e_ms := to_ms(end)) is not None:
        i1 = int(np.searchsorted(data.open_ms, e_ms, side="left"))
    if i1 - i0 < 2:
        raise ValueError("evaluation window has fewer than 2 bars")

    s_full = _shift(sig, delay)
    w_full = wb = None
    if mode == "portfolio":
        p_full = _prepare(s_full, P, C, listed, nan_policy)
        ctx = alloc_mod.AllocContext(
            ts=data.ts,
            symbols=list(data.symbols),
            signal=sig,
            data=data,
            listed=listed,
            _independent=lambda: _independent_returns(p_full, fee, compound),
        )
        w_full = np.nan_to_num(_shift(alloc_mod.resolve(alloc, ctx), delay), nan=0.0)
        wb = alloc_mod.benchmark_weights(alloc, ctx)

    sl = slice(i0, i1)
    p = _prepare(s_full[sl], P[sl], C[sl], listed[sl], nan_policy)
    if (p.start < 0).all():
        raise ValueError("signal has no usable value inside the evaluation window")
    open_ms = data.open_ms[sl]
    k_ms = data.k * 60_000
    cfg = dict(
        price=price,
        exec_window=exec_window,
        fee=fee,
        compound=compound,
        delay=delay,
        mode=mode,
        nan_policy=nan_policy,
        freq=data.freq,
        symbols=list(data.symbols),
        alloc=alloc_mod.describe(alloc) if mode == "portfolio" else None,
    )
    n = i1 - i0
    rows = np.arange(n)[:, None]

    if mode == "independent":
        theta = _positions(p.target, p)
        bench_t = _positions(np.ones_like(p.target), p) * (rows >= np.maximum(p.start, 0)[None, :])
        net = simulate(theta, p.R, p.b, fee, compound, pooled=False)
        gross = simulate(theta, p.R, p.b, 0.0, compound, pooled=False)
        bench = simulate(bench_t, p.R, p.b, 0.0, compound, pooled=False)
        books = []
        for j in range(K):
            if p.start[j] < 0:
                continue
            a, z = int(p.start[j]), int(p.last[j]) if p.last[j] >= 0 else n - 1
            r = slice(a, z + 1)
            tr = theta[r, j : j + 1] * p.R[r, j : j + 1]
            books.append(
                Result(
                    name=names[j],
                    symbols=[data.symbols[colmap[j]]],
                    open_ms=open_ms[r],
                    k_ms=k_ms,
                    nav=net.nav[r, j],
                    nav_gross=gross.nav[r, j],
                    bench=bench.nav[r, j],
                    turnover=net.turnover[r, j],
                    fee_paid=net.fee[r, j],
                    position=theta[r, j : j + 1],
                    signal=p.s_raw[r, j : j + 1],
                    close=p.close[r, j : j + 1],
                    long_pnl=np.where(theta[r, j : j + 1] > 0, tr, 0).sum(1),
                    short_pnl=np.where(theta[r, j : j + 1] < 0, tr, 0).sum(1),
                    bankrupt=bool(net.bankrupt_at[j] >= 0),
                    config=cfg,
                )
            )
        if not books:
            raise ValueError("no column has a usable signal")
        return books[0] if len(books) == 1 and K == 1 else BatchResult(books)

    # ---- pooled portfolio
    w = w_full[sl]
    a = int(p.start[p.start >= 0].min())
    after = rows >= a
    theta = _positions(w * p.target, p) * after
    net = simulate(theta, p.R, p.b, fee, compound, pooled=True)
    gross = simulate(theta, p.R, p.b, 0.0, compound, pooled=True)
    # Benchmark: buy each symbol once with its weight - at the start, or when it lists -
    # and never rebalance; capital waiting for a listing (or left over) stays in cash.
    # A column held at 1x in its own account is exactly "bought once and held".
    held = simulate(
        _positions(np.ones_like(p.target), p) * after, p.R, p.b, 0.0, True, pooled=False
    )
    bench_nav = (1.0 - wb.sum()) + held.nav @ wb
    r = slice(a, n)
    tr = theta[r] * p.R[r]
    port = Result(
        name="portfolio",
        symbols=list(data.symbols),
        open_ms=open_ms[r],
        k_ms=k_ms,
        nav=net.nav[r, 0],
        nav_gross=gross.nav[r, 0],
        bench=bench_nav[r],
        turnover=net.turnover[r, 0],
        fee_paid=net.fee[r, 0],
        position=theta[r],
        signal=p.s_raw[r],
        close=p.close[r],
        long_pnl=np.where(theta[r] > 0, tr, 0).sum(1),
        short_pnl=np.where(theta[r] < 0, tr, 0).sum(1),
        bankrupt=bool(net.bankrupt_at[0] >= 0),
        config=cfg,
        alloc=w[r],
        turnover_cols=net.turnover_cols[r],
        pnl_cols=tr,
    )
    components = run(
        sig,
        data,
        price=price,
        exec_window=exec_window,
        fee=fee,
        compound=compound,
        delay=delay,
        mode="independent",
        nan_policy=nan_policy,
        start=start,
        end=end,
        names=names,
    )
    if isinstance(components, Result):
        components = BatchResult([components])
    return PortfolioResult(port, components)
