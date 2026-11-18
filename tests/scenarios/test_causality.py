"""Changing anything at or after bar T must not change what happened before T."""

from __future__ import annotations

import numpy as np

import tsbt as bt
from tests.conftest import DAY, Market, bar_blocks, ms
from tsbt import config
from tsbt.data import store
from tsbt.engine import alloc as A

op = bt.ops


def _market(root, perturb: bool):
    mk = Market(root)
    t0 = ms("2024-01-01")
    for i, sym in enumerate(["AAAUSDT", "BBBUSDT"]):
        close, vol = bar_blocks(4 * 24, 60, seed=20 + i)
        if perturb:  # everything from day 3 on is different
            cut = 2 * 1440
            rng = np.random.default_rng(99 + i)
            close[cut:] *= np.exp(np.cumsum(rng.normal(0, 5e-3, len(close) - cut)))
            vol[cut:] = rng.uniform(5, 9, len(vol) - cut)
        mk.klines(sym, "2024-01-01", close, volume=vol)
        stamps = t0 + np.arange(0, 4 * DAY, 300_000)
        oi = 1000 + np.cumsum(np.random.default_rng(50 + i).normal(0, 1, len(stamps)))
        if perturb:
            oi[stamps >= t0 + 2 * DAY] *= 3
        mk.metrics(sym, stamps, oi)


def _pipeline():
    d = bt.load(["AAAUSDT", "BBBUSDT"], "1h", fields=["close", "volume", "oi"])
    ret = op.ts_returns(d.close, 1)
    rv = d.agg(op.ts_returns(d.m1.close, 1), "std")
    s = op.tanh(op.ts_zscore(ret, 12)) * 0.5
    s = s + 0.1 * op.ts_rank(d.volume, 8) - 0.05 * op.ts_corr(ret, op.ts_delta(d.oi, 1), 10)
    s = op.hump_decay(op.ts_ema(s, 3), 0.02) * op.if_else(
        op.greater(rv, op.ts_mean(rv, 6)), 0.5, 1.0
    )
    s = op.trade_when(op.is_not_nan(s), s, op.less(op.ts_quantile(ret, 10), -1.2))
    res = bt.run(
        s, d, price="vwap", exec_window=20, fee=3e-4, alloc=A.rolling_sharpe(6, rebalance_every=3)
    )
    return d, s, res


def test_future_data_cannot_change_the_past(tmp_path):
    out = {}
    for name, perturb in (("A", False), ("B", True)):
        root = tmp_path / name
        _market(root, perturb)
        config.set_paths(data_root=root, cache_dir=tmp_path / f"cache{name}")
        store.forget()
        try:
            out[name] = _pipeline()
        finally:
            config.reset_paths()
            store.forget()
    (da, sa, ra), (db, sb, rb) = out["A"], out["B"]
    T = 48  # first bar of day 3
    assert da.T == db.T
    np.testing.assert_array_equal(sa[:T], sb[:T])
    assert not np.array_equal(sa[T:], sb[T:], equal_nan=True)  # the perturbation is real
    pa, pb = ra.portfolio, rb.portfolio
    rows = pa.open_ms < da.open_ms[T]
    for attr in ("position", "turnover", "nav", "nav_gross", "bench", "alloc"):
        np.testing.assert_array_equal(
            getattr(pa, attr)[rows], getattr(pb, attr)[rows], err_msg=attr
        )
    for ca, cb in zip(ra.components, rb.components, strict=True):
        r = ca.open_ms < da.open_ms[T]
        np.testing.assert_array_equal(ca.nav[r], cb.nav[r])


def test_allocation_rules_and_listing(env):
    """equal_active spreads over listed symbols; late listings hold nothing before listing;
    a rolling-Sharpe weight row depends only on returns up to that row."""
    a, va = bar_blocks(72, 60, seed=1)
    b, vb = bar_blocks(48, 60, seed=2)
    env.klines("AAAUSDT", "2024-01-01", a, volume=va)
    env.klines("BBBUSDT", "2024-01-02", b, volume=vb)
    d = bt.load(["AAAUSDT", "BBBUSDT"], "1h")
    sig = np.full((d.T, 2), 0.8)
    r = bt.run(sig, d, alloc=A.equal_active(), price="open").portfolio
    listed = d.listed[d.T - len(r.open_ms) :]
    np.testing.assert_array_equal(r.position[~listed], 0.0)
    both = listed.all(axis=1)
    # alloc row t was decided at bar t-1: compare where both were listed at t-1 too
    np.testing.assert_allclose(r.alloc[1:][both[1:] & both[:-1]], 0.5)
    np.testing.assert_allclose(r.alloc[1:][~listed[:-1, 1] & ~listed[1:, 1], 0], 1.0)
    eq = bt.run(sig, d, price="open").portfolio  # static 1/N: half the capital idles before B lists
    np.testing.assert_allclose(eq.position[~listed[:, 1], 0], 0.4)
    # Portfolio benchmark = buy once and hold: each symbol is bought with its weight in
    # the first bar it can trade (B lists a day late; its half waits in cash), then
    # nothing is rebalanced - so the weights drift with prices.
    P, C = d.exec_price("open"), d.raw("close")
    a0 = int(np.searchsorted(d.open_ms, eq.open_ms[0]))
    b0 = int(np.argmax(d.listed[:, 1]))
    rows = np.arange(a0, d.T)

    def hold(w):
        leg_a = C[rows, 0] / P[a0, 0]
        leg_b = np.where(rows >= b0, C[rows, 1] / P[b0, 1], 1.0)
        return w[0] * leg_a + w[1] * leg_b

    np.testing.assert_allclose(eq.bench, hold([0.5, 0.5]), rtol=1e-12)
    tilted = bt.run(sig, d, alloc=A.fixed([0.9, 0.1]), price="open").portfolio
    np.testing.assert_allclose(tilted.bench, hold([0.9, 0.1]), rtol=1e-12)
    # it never depends on the strategy: a performance-driven allocation is measured
    # against the plain 1/N basket, whatever the signal did
    wiggle = sig * np.sin(np.arange(d.T))[:, None]
    for dyn in (A.rolling_sharpe(8), A.inv_vol(8)):
        np.testing.assert_array_equal(bt.run(wiggle, d, alloc=dyn, price="open").bench, eq.bench)
    # and it is the same number under simple interest (nothing is ever re-invested)
    np.testing.assert_allclose(
        bt.run(sig, d, price="open", compound=False).bench, eq.bench, rtol=1e-12
    )

    rng = np.random.default_rng(3)
    ret = rng.normal(0, 0.01, (200, 3))
    lst = np.ones((200, 3), bool)

    def ctx_for(rr):
        return A.AllocContext(
            np.arange(200),
            ["a", "b", "c"],
            np.zeros((200, 3)),
            None,
            lst,
            _independent=lambda: (rr, rr),
        )

    w1 = A.rolling_sharpe(20).weights(ctx_for(ret))
    ret2 = ret.copy()
    ret2[120:] = rng.normal(0.05, 0.01, (80, 3))
    w2 = A.rolling_sharpe(20).weights(ctx_for(ret2))
    np.testing.assert_array_equal(w1[:120], w2[:120])
    assert not np.allclose(w1[120:], w2[120:])
    np.testing.assert_allclose(w1[:9], 1 / 3)  # warm-up (< min_valid(20)=10 samples) -> equal
    tot = w1.sum(axis=1)
    assert np.all((np.abs(tot - 1) < 1e-12) | (tot == 0))
    # inverse volatility: a column with 4x the volatility gets a quarter of the weight
    scaled = ret * np.array([1.0, 2.0, 4.0])
    wv = A.inv_vol(30).weights(ctx_for(scaled))
    np.testing.assert_allclose(wv[150] * np.array([1.0, 2.0, 4.0]) / (wv[150, 0]), 1.0, rtol=0.35)
    assert wv[150, 0] > wv[150, 1] > wv[150, 2]
