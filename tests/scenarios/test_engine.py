"""Execution timing, price types, drift turnover, reference-simulator parity, identities."""

from __future__ import annotations

import numpy as np
import pytest

import tsbt as bt
from tests.conftest import bar_blocks
from tsbt import metrics as M

K = 60  # 1h bars


def _gapped(close):
    """OHLC minutes whose open differs from the previous close (a gap at every minute)."""
    o = close * np.exp(np.random.default_rng(99).normal(0, 3e-4, len(close)))
    return np.stack([o, np.maximum(o, close), np.minimum(o, close), close], axis=1)


def _minutes(close, vol, k=K, opens=None):
    """Hand-computed per-bar open / twap / vwap / close (and the @N window variants)."""
    c = close.reshape(-1, k)
    v = vol.reshape(-1, k)
    o = (np.r_[close[0], close[:-1]] if opens is None else opens).reshape(-1, k)
    return {
        "open": o[:, 0],
        "close": c[:, -1],
        "twap": c.mean(1),
        "vwap": (c * v).sum(1) / v.sum(1),
        "twap@15": c[:, :15].mean(1),
        "vwap@15": (c[:, :15] * v[:, :15]).sum(1) / v[:, :15].sum(1),
    }


def test_execution_happens_inside_the_next_bar_never_at_t_minus_1(env):
    """A signal formed at bar T-1 trades at bar T's open/twap/vwap/close and nowhere else."""
    n = 30
    close, vol = bar_blocks(n, K, seed=1)
    ohlc = _gapped(close)
    env.klines("BTCUSDT", "2024-01-01", ohlc, volume=vol)
    d = bt.load("BTCUSDT", "1h")
    px = _minutes(close, vol, opens=ohlc[:, 0])
    T = 10
    sig = np.zeros(n)
    sig[T - 1] = 1.0  # known at the close of bar T-1
    for price in ("open", "twap", "vwap", "close"):
        res = bt.run(sig, d, price=price, fee=0.0)
        pos = res.position[:, 0]
        # the only bar holding a position is bar T (compare bar open times)
        assert res.open_ms[np.flatnonzero(pos)].tolist() == [int(d.open_ms[T])]
        p = px[price]
        # one round trip: in at P[T], out at P[T+1]
        assert res.nav_gross[-1] == pytest.approx(p[T + 1] / p[T], rel=1e-12)
        # ...which is not what trading at any T-1 price would give
        for wrong in ("open", "twap", "vwap", "close"):
            assert res.nav_gross[-1] != pytest.approx(p[T + 1] / px[wrong][T - 1], rel=1e-9)
    res = bt.run(sig, d, price="vwap", exec_window=15, fee=0.0)
    assert res.nav_gross[-1] == pytest.approx(px["vwap@15"][T + 1] / px["vwap@15"][T], rel=1e-12)
    with pytest.raises(ValueError, match="delay"):
        bt.run(sig, d, delay=0)


def test_four_execution_prices_match_hand_computation(env):
    """Three bars of known minutes: NAV path for each price type equals the hand ledger."""
    n = 6
    close, vol = bar_blocks(n, K, seed=2)
    vol[K : K + 5] = 0.0  # some zero-volume minutes: vwap must still be defined
    env.klines("ETHUSDT", "2024-03-01", close, volume=vol)
    d = bt.load("ETHUSDT", "1h")
    sig = np.array([0.5, -0.3, 0.8, 0.8, 0.0, 0.0])
    px = _minutes(close, vol)
    for price in ("open", "twap", "vwap", "close", "twap@15", "vwap@15"):
        name, win = (
            (price.split("@")[0], int(price.split("@")[1])) if "@" in price else (price, None)
        )
        res = bt.run(sig, d, price=name, exec_window=win, fee=0.0, compound=False)
        P, C = px[price], px["close"]
        # simple-interest ledger by hand
        theta = np.r_[0.0, sig[:-1]]
        nav = []
        realized = 0.0
        for t in range(n):
            if t > 0:
                realized += theta[t - 1] * (P[t] / P[t - 1] - 1)
            nav.append(1 + realized + theta[t] * (C[t] / P[t] - 1))
        np.testing.assert_allclose(
            res.nav_gross, nav[1:], rtol=0, atol=1e-12
        )  # book starts at bar 1


def test_drift_turnover_constant_signal(env):
    """Signal 0.2 then 0.2 with a +5% move between executions still trades back to 0.2."""
    n = 5
    bars = 100.0 * 1.05 ** np.arange(n)
    close = np.repeat(bars, K)
    env.klines("BTCUSDT", "2024-01-01", close)
    d = bt.load("BTCUSDT", "1h")
    sig = np.full(n, 0.2)
    f = 2e-4
    rc = bt.run(sig, d, price="close", fee=f, compound=True)
    rs = bt.run(sig, d, price="close", fee=f, compound=False)
    # book row 0 (= bar 1) opens 0.2; every later row only rebalances the drift
    assert rc.turnover[0] == pytest.approx(0.2)
    assert rc.turnover[1] == pytest.approx(abs(0.2 - 0.2 * 1.05 / (1 + 0.2 * 0.05)), abs=1e-12)
    assert rc.turnover[1] == pytest.approx(0.007921, abs=1e-6)
    assert rs.turnover[1] == pytest.approx(0.01, abs=1e-12)
    np.testing.assert_allclose(rc.turnover[1:], rc.turnover[1], rtol=1e-12)
    # fees: simple -> nav_gross - nav = f * cumulative turnover; compound -> ratio = prod(1 - f tau)
    np.testing.assert_allclose(rs.nav_gross - rs.nav, f * np.cumsum(rs.turnover), atol=1e-14)
    np.testing.assert_allclose(rc.nav / rc.nav_gross, np.cumprod(1 - f * rc.turnover), rtol=1e-13)
    np.testing.assert_allclose(rs.fee_paid.sum(), f * rs.turnover.sum())


def _reference(theta, P, C, fee, compound):
    """Contracts + cash ledger, one bar at a time (independent derivation)."""
    T, N = theta.shape
    cash, q = 1.0, np.zeros(N)
    nav = np.empty(T)
    for t in range(T):
        px = P[t]
        live = np.isfinite(px)
        held = np.where(live, q * np.nan_to_num(px), 0.0)
        e_pre = cash + held.sum()
        if compound:
            w_pre = held / e_pre
            tau = np.abs(theta[t] - w_pre).sum()
            fee_t = fee * tau * e_pre
            target_notional = theta[t] * (e_pre - fee_t)
        else:
            tau = np.abs(theta[t] - held).sum()
            fee_t = fee * tau
            target_notional = theta[t].copy()
        q_new = np.where(live, target_notional / np.where(live, px, 1.0), 0.0)
        cash -= (np.where(live, (q_new - q) * np.nan_to_num(px), 0.0)).sum() + fee_t
        q = q_new
        nav[t] = cash + np.where(live, q * np.nan_to_num(C[t]), 0.0).sum()
    return nav


@pytest.mark.parametrize("compound", [True, False])
def test_vectorised_engine_matches_contract_ledger(env, compound):
    """Random prices/signals, 3 symbols listed at different times, all price types,
    single-account and pooled modes, against a naive contracts+cash simulator."""
    n = 200
    rng = np.random.default_rng(7)
    starts = {"AAAUSDT": 0, "BBBUSDT": 30, "CCCUSDT": 75}
    for i, (sym, s0) in enumerate(starts.items()):
        close, vol = bar_blocks(n - s0, K, seed=10 + i)
        start = np.datetime64("2024-01-01T00:00") + np.timedelta64(s0, "h")
        env.klines(sym, str(start), close, volume=vol)
    syms = list(starts)
    d = bt.load(syms, "1h")
    sig = rng.uniform(-1.5, 1.5, (d.T, 3))
    sig[rng.random((d.T, 3)) < 0.05] = np.nan
    w = np.array([0.5, 0.3, 0.2])
    for price in ("open", "twap", "vwap", "close"):
        P = d.exec_price(price)
        C = d.raw("close")
        theta = np.nan_to_num(np.vstack([np.zeros((1, 3)), sig[:-1]]), nan=0.0)
        theta[~d.listed] = 0.0
        # single accounts
        for j, sym in enumerate(syms):
            dj = bt.load(sym, "1h")
            sj = sig[d.listed[:, j], j]  # the union-axis column restricted to this symbol's bars
            res = bt.run(sj, dj, price=price, fee=5e-4, compound=compound)
            th = np.nan_to_num(np.r_[0.0, sj[:-1]], nan=0.0)[:, None]
            ref = _reference(th, dj.exec_price(price), dj.raw("close"), 5e-4, compound)
            a = len(ref) - len(res.nav)
            np.testing.assert_allclose(res.nav, ref[a:], rtol=1e-10, atol=1e-12)
        # pooled account with fixed weights
        pr = bt.run(sig, d, price=price, fee=5e-4, compound=compound, alloc=bt.alloc.fixed(w))
        ref = _reference(theta * w, P, C, 5e-4, compound)
        a = len(ref) - len(pr.portfolio.nav)
        np.testing.assert_allclose(pr.portfolio.nav, ref[a:], rtol=1e-10, atol=1e-12)


def test_identities_buy_and_hold_and_flat(env):
    n = 48
    close, vol = bar_blocks(n, K, seed=3)
    env.klines("BTCUSDT", "2024-01-01", close, volume=vol)
    d = bt.load("BTCUSDT", "1h")
    one = bt.run(np.ones(n), d, price="close", fee=0.0, compound=True)
    np.testing.assert_allclose(one.nav, one.bench, rtol=1e-13)
    c = d.close
    np.testing.assert_allclose(one.nav, c[1:] / c[1], rtol=1e-12)
    assert one.turnover[1:].max() < 1e-12  # holding 1x never needs rebalancing
    zero = bt.run(np.zeros(n), d, fee=2e-4)
    np.testing.assert_array_equal(zero.nav, 1.0)
    assert zero.metrics["turnover"] == 0


def test_daily_sharpe_independent_of_bar_frequency(env):
    """Holding 1x from the same instant gives the same daily NAV at 1h and 1D."""
    days = 40
    close, vol = bar_blocks(days * 24, K, seed=4)
    env.klines("BTCUSDT", "2024-01-01", close, volume=vol)
    dh = bt.load("BTCUSDT", "1h")
    dd = bt.load("BTCUSDT", "1D")
    sh = np.full(dh.T, np.nan)
    sh[23:] = 1.0  # first usable at 23:00 day 1 -> trades 00:00 day 2
    sd = np.full(dd.T, np.nan)
    sd[0:] = 1.0
    rh = bt.run(sh, dh, price="open", fee=0.0)
    rd = bt.run(sd, dd, price="open", fee=0.0)
    assert rh.metrics["start"] == rd.metrics["start"]
    assert rh.metrics["sharpe"] == pytest.approx(rd.metrics["sharpe"], rel=1e-10)
    assert rh.metrics["apr"] == pytest.approx(rd.metrics["apr"], rel=1e-10)
    np.testing.assert_allclose(
        rh.daily()["nav"].to_numpy(), rd.daily()["nav"].to_numpy(), rtol=1e-12
    )


def test_metrics_against_analytic_values():
    day = 86_400_000
    t = np.arange(366) * day
    # alternating +2% / -1% daily: known mean, std, compounding, drawdown
    r = np.where(np.arange(365) % 2 == 0, 0.02, -0.01)
    nav = np.r_[1.0, np.cumprod(1 + r)]
    m = M.perf(t, nav, np.full(365, 0.5), compound=True)
    assert m["days"] == 365
    assert m["apr"] == pytest.approx(nav[-1] - 1, rel=1e-12)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(365), rel=1e-12)
    assert m["mdd"] == pytest.approx(-0.01, rel=1e-12)
    assert m["calmar"] == pytest.approx(m["apr"] / 0.01, rel=1e-12)
    assert m["turnover"] == pytest.approx(0.5)
    # a 50% crash then flat: mdd -50%, simple-mode mdd in principal units
    nav2 = np.r_[1.0, 1.2, 0.6, 0.6, 0.9]
    assert M.max_drawdown(nav2, True) == pytest.approx(-0.5)
    assert M.max_drawdown(nav2, False) == pytest.approx(-0.6)
    # half-year CAGR
    assert M.apr(1.1, 182.5, True) == pytest.approx(1.1**2 - 1)
    assert M.apr(1.1, 182.5, False) == pytest.approx(0.2)


def test_delisting_closes_and_missing_bar_holds(env):
    """A symbol whose data stops is closed at its last bar (and charged); a bar with no
    minutes at all is not traded - the previous position is simply carried."""
    a, va = bar_blocks(48, 60, seed=40)
    b, vb = bar_blocks(30, 60, seed=41)
    env.klines(
        "AAAUSDT",
        "2024-01-01",
        a,
        volume=va,
        extra=lambda df: (
            (df["open_time"] < 1704067200000 + 20 * 3_600_000)
            | (df["open_time"] >= 1704067200000 + 21 * 3_600_000)
        ),
    )
    env.klines("BBBUSDT", "2024-01-01", b, volume=vb)
    d = bt.load(["AAAUSDT", "BBBUSDT"], "1h")
    assert np.isnan(d.close[20, 0]) and d.listed[20, 0]
    sig = np.column_stack([np.linspace(0.2, 0.9, d.T), np.full(d.T, 0.5)])
    res = bt.run(sig, d, price="open", fee=1e-3, alloc=bt.alloc.fixed([0.5, 0.5]))
    p = res.portfolio
    row = {int(t): i for i, t in enumerate(p.open_ms)}
    gap = row[int(d.open_ms[20])]
    assert p.position[gap, 0] == p.position[gap - 1, 0]  # held through the hole
    # only the pooled-equity drift is rebalanced (SPEC 8.8 approximation), no new target
    assert p.turnover_cols[gap, 0] < 0.01
    last_b = row[int(d.open_ms[29])]
    assert p.position[last_b, 1] == 0.0 and p.position[last_b - 1, 1] > 0  # closed at its last bar
    assert p.turnover_cols[last_b, 1] > 0.2  # the close is charged
    np.testing.assert_array_equal(p.position[last_b:, 1], 0.0)
    solo = res.components["BBBUSDT"]
    assert solo.position[-1, 0] == 0.0 and solo.end_ms == int(d.open_ms[29]) + 3_600_000
