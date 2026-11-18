"""Bar synthesis, caching, metrics alignment, user features, intra-bar aggregation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import tsbt as bt
from tests.conftest import DAY, MIN, REAL_ROOT, bar_blocks, ms
from tsbt.data import m1


@pytest.mark.skipif(
    not (REAL_ROOT / "monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-11.parquet").exists(),
    reason="no real data",
)
def test_real_month_resampling_matches_polars(env):
    """A real BTCUSDT month: reshape-built 1h..1D bars equal polars group_by_dynamic."""
    src = REAL_ROOT / "monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-11.parquet"
    dst = env.file("klines", "BTCUSDT", "2025-11", kind="monthly")
    dst.parent.mkdir(parents=True)
    dst.symlink_to(src)
    raw = pl.read_parquet(src).with_columns(t=pl.from_epoch("open_time", time_unit="ms")).sort("t")
    for freq, every in [
        ("1h", "1h"),
        ("2h", "2h"),
        ("4h", "4h"),
        ("6h", "6h"),
        ("12h", "12h"),
        ("1D", "1d"),
    ]:
        d = bt.load("BTCUSDT", freq)
        ref = raw.group_by_dynamic("t", every=every, closed="left", label="left").agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("quote_volume").sum(),
            pl.len().alias("n"),
        )
        k = bt.data.bars.parse_freq(freq)
        ref = ref.filter(pl.col("n") == k)
        assert d.T == ref.height
        np.testing.assert_array_equal(d.ts, ref["t"].to_numpy().astype("datetime64[ms]"))
        for c in ("open", "high", "low", "close"):
            np.testing.assert_array_equal(d[c], ref[c].to_numpy())
        for c in ("volume", "quote_volume"):
            np.testing.assert_allclose(d[c], ref[c].to_numpy(), rtol=1e-12)
        assert (d.open_ms % (k * MIN) == 0).all()  # aligned to UTC midnight


def test_partial_bars_dropped_and_gaps_are_nan(env):
    """Listing at 00:07 and ending 13:29: only complete 4h bars; a missing minute stays NaN."""
    n = 13 * 60 + 23
    close, vol = bar_blocks(1, n, seed=5)
    drop = ms("2024-01-01 05:30")
    env.klines(
        "NEWUSDT",
        "2024-01-01 00:07",
        close,
        volume=vol,
        extra=lambda df: pl.col("open_time") != drop,
    )
    d = bt.load("NEWUSDT", "4h")
    assert [str(t) for t in d.ts] == ["2024-01-01T04:00:00.000", "2024-01-01T08:00:00.000"]
    m = d.m1
    assert np.isnan(m.close[90])  # 05:30 is minute 90 of the 04:00 bar
    assert d.volume[0] == pytest.approx(np.nansum(m.volume[:240]))


def test_cache_reuse_incremental_append_and_monthly_precedence(env, monkeypatch):
    close, vol = bar_blocks(3, 1440, seed=6)
    env.klines("BTCUSDT", "2024-01-01", close, volume=vol)
    first = bt.load("BTCUSDT", "1h").close.copy()

    calls = []
    real = m1.BUILTIN_BUILDERS["klines"].compute

    def counting(f, sym):
        calls.append(f.name)
        return real(f, sym)

    monkeypatch.setattr(m1.BUILTIN_BUILDERS["klines"], "compute", counting)
    again = bt.load("BTCUSDT", "1h").close
    assert calls == []  # second load reads no source file
    np.testing.assert_array_equal(first, again)

    # a new daily file -> only that file is computed, history untouched
    more, mv = bar_blocks(1, 1440, seed=7)
    env.klines("BTCUSDT", "2024-01-04", more, volume=mv)
    d = bt.load("BTCUSDT", "1h")
    assert calls == ["BTCUSDT-1m-2024-01-04.parquet"]
    assert d.T == 96
    np.testing.assert_array_equal(d.close[:72], first)

    # the monthly file arrives: it wins over the dailies of that month, no duplicates
    calls.clear()
    full, fv = bar_blocks(31, 1440, seed=8)
    env.klines("BTCUSDT", "2024-01-01", full, volume=fv, kind="monthly")
    d = bt.load("BTCUSDT", "1h")
    assert calls == ["BTCUSDT-1m-2024-01.parquet"]
    assert d.T == 31 * 24
    np.testing.assert_array_equal(d.close, full.reshape(-1, 60)[:, -1])


def test_metrics_are_causally_aligned(env):
    """A record stamped exactly at a bar boundary belongs to the next bar."""
    close, vol = bar_blocks(6, 60, seed=9)
    env.klines("BTCUSDT", "2024-01-01", close, volume=vol)
    t0 = ms("2024-01-01")
    h = 3_600_000
    stamps = np.array(
        [
            t0 + h + 55 * MIN,
            t0 + 2 * h,
            t0 + h + 30 * MIN,
            t0 + 2 * h + 2 * MIN + 30_000,
            t0 + 2 * h + 2 * MIN + 30_000,
        ]
    )
    oi = np.array([1.0, 2.0, 0.5, 3.0, 4.0])  # deliberately unsorted; one duplicate stamp, off-grid
    env.metrics("BTCUSDT", stamps, oi)
    d = bt.load("BTCUSDT", "1h", fields=["close", "oi"])
    # bar 01:00-02:00 sees 01:30 and 01:55 -> last = 1.0; the 02:00:00 record is NOT in it
    assert d.oi[1] == 1.0
    # bar 02:00-03:00: 02:00 (2.0) then 02:02:30 twice -> last in file order = 4.0
    assert d.oi[2] == 4.0
    assert np.isnan(d.oi[0]) and np.isnan(d.oi[3])
    # minute view: 02:02:30 lands on minute 02:02, forward-filled for at most 15 minutes
    m = d.m1.oi
    assert m[2 * 60 + 2] == 4.0 and m[2 * 60 + 17] == 4.0 and np.isnan(m[2 * 60 + 18])


def _trades(env, days=2, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 3000
    t = np.sort(ms("2024-01-01") + rng.integers(0, days * DAY, n))
    price = 100 + np.cumsum(rng.normal(0, 0.01, n))
    qty = rng.exponential(1.0, n) * 10
    maker = rng.random(n) < 0.5
    env.agg_trades("BTCUSDT", t, price, qty, maker)
    close, vol = bar_blocks(days * 24, 60, seed=seed)
    env.klines("BTCUSDT", "2024-01-01", close, volume=vol)
    return pl.DataFrame({"t": t, "price": price, "qty": qty, "maker": maker}).with_columns(
        ts=(pl.col("t") // MIN) * MIN, amt=pl.col("price") * pl.col("qty")
    )


def test_user_features_three_styles_resume_and_invalidation(env, monkeypatch):
    tr = _trades(env)
    bt.feature_set(
        "flow",
        source="aggTrades",
        sql={
            "buy_amt": "sum(price*quantity) FILTER (WHERE NOT is_buyer_maker)",
            "n_tr": "count(*)",
        },
        bar_agg={"buy_amt": "sum", "n_tr": "sum"},
        fill={"buy_amt": 0.0, "n_tr": 0.0},
    )
    bt.feature_set(
        "micro",
        source="aggTrades",
        query="SELECT (transact_time // 60000) * 60000 AS ts, max(price*quantity) AS max_amt FROM {src} GROUP BY 1",
        bar_agg={"max_amt": "max"},
    )

    state = {"fail": True, "calls": 0}

    @bt.feature_set("pyfeat", source="aggTrades", outputs=["sell_qty"], bar_agg={"sell_qty": "sum"})
    def pyfeat(chunk):
        state["calls"] += 1
        if state["fail"] and "2024-01-02" in chunk.path:
            raise RuntimeError("boom")
        return (
            chunk.scan()
            .filter(pl.col("is_buyer_maker"))
            .group_by(ts=(pl.col("transact_time") // 60000) * 60000)
            .agg(sell_qty=pl.col("quantity").sum())
            .collect()
        )

    d = bt.load("BTCUSDT", "1h", fields=["close", "buy_amt", "n_tr", "max_amt"])
    by_min = (
        tr.group_by("ts")
        .agg(buy=pl.col("amt").filter(~pl.col("maker")).sum(), n=pl.len(), mx=pl.col("amt").max())
        .with_columns(bar=(pl.col("ts") - ms("2024-01-01")) // 3_600_000)
    )
    by_bar = (
        by_min.group_by("bar")
        .agg(pl.col("buy").sum(), pl.col("n").sum(), pl.col("mx").max())
        .sort("bar")
    )
    np.testing.assert_allclose(d.buy_amt, by_bar["buy"].to_numpy(), rtol=1e-9)
    np.testing.assert_array_equal(d.n_tr, by_bar["n"].to_numpy())
    np.testing.assert_allclose(d.max_amt, by_bar["mx"].to_numpy(), rtol=1e-12)

    # Python feature: crash on the 2nd file; the 1st is already persisted and not redone
    monkeypatch.setattr(m1, "FLUSH_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="boom"):
        bt.load("BTCUSDT", "1h", fields=["sell_qty"], workers=1)
    assert state["calls"] == 2
    state.update(fail=False, calls=0)
    d = bt.load("BTCUSDT", "1h", fields=["sell_qty"], workers=1)
    assert state["calls"] == 1
    sell = (
        tr.filter(pl.col("maker"))
        .with_columns(bar=(pl.col("ts") - ms("2024-01-01")) // 3_600_000)
        .group_by("bar")
        .agg(pl.col("qty").sum())
        .sort("bar")
    )
    np.testing.assert_allclose(d.sell_qty, sell["qty"].to_numpy(), rtol=1e-9)

    # changing one SQL definition rebuilds that set only
    v_micro = m1.version("BTCUSDT", "feat:micro")
    bt.feature_set(
        "flow",
        source="aggTrades",
        sql={"buy_amt": "sum(quantity) FILTER (WHERE NOT is_buyer_maker)", "n_tr": "count(*)"},
        bar_agg={"buy_amt": "sum", "n_tr": "sum"},
    )
    d = bt.load("BTCUSDT", "1h", fields=["buy_amt", "max_amt"])
    buy_qty = (
        tr.filter(~pl.col("maker"))
        .with_columns(bar=(pl.col("ts") - ms("2024-01-01")) // 3_600_000)
        .group_by("bar")
        .agg(pl.col("qty").sum())
        .sort("bar")
    )
    np.testing.assert_allclose(d.buy_amt, buy_qty["qty"].to_numpy(), rtol=1e-9)
    assert m1.version("BTCUSDT", "feat:micro") == v_micro

    # bad output is rejected loudly
    bt.feature_set(
        "bad", source="aggTrades", query="SELECT transact_time AS ts, price AS px FROM {src}"
    )
    with pytest.raises(ValueError, match="multiple of 60000|one row per minute"):
        bt.load("BTCUSDT", "1h", fields=["px"])


def test_intrabar_aggregation_matches_naive_loop(env):
    close, vol = bar_blocks(10, 240, seed=11)
    env.klines("ETHUSDT", "2024-02-01", close, volume=vol)
    d = bt.load("ETHUSDT", "4h")
    m = d.m1
    ret = bt.ops.ts_returns(m.close, 1)
    rv = d.agg(ret, "std")
    pvc = d.agg((ret, m.volume), "corr")
    rng = d.agg(m.high - m.low, lambda b: np.nanmax(b, axis=1))
    early = d.agg(m.volume, "sum", window=30)
    q = d.agg(m.close, "quantile", q=0.9)
    for j in range(d.T):
        sl = slice(j * 240, (j + 1) * 240)
        r = ret[sl]
        ok = np.isfinite(r)
        assert rv[j] == pytest.approx(np.std(r[ok], ddof=1), rel=1e-10)
        assert pvc[j] == pytest.approx(np.corrcoef(r[ok], m.volume[sl][ok])[0, 1], rel=1e-9)
        assert rng[j] == pytest.approx(np.max(m.high[sl] - m.low[sl]))
        assert early[j] == pytest.approx(vol[j * 240 : j * 240 + 30].sum(), rel=1e-12)
        assert q[j] == pytest.approx(np.quantile(close[sl], 0.9), rel=1e-12)
    with pytest.raises(ValueError, match="T\\*k"):
        d.agg(ret[:-1], "std")
    d.save_field("rv", rv)
    again = bt.load("ETHUSDT", "4h", fields=["rv"])
    np.testing.assert_array_equal(again.rv, rv)
    with pytest.raises(FileExistsError):
        d.save_field("rv", rv * 2)
