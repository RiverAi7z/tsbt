"""Synthetic Binance-layout data written into a temp data root, plus isolated caches."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tsbt import config
from tsbt.data import features, store

MIN = 60_000
DAY = 86_400_000
REAL_ROOT = Path("/share/binance_data/futures/um")


def ms(s: str) -> int:
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return int(t.timestamp() * 1000)


class Market:
    """Writes klines / metrics / aggTrades parquet files in the real directory layout."""

    def __init__(self, root: Path):
        self.root = root

    # ------------------------------------------------------------------ klines
    def klines(
        self, sym: str, start: str, prices, volume=None, quote=None, kind: str = "daily", extra=None
    ):
        """One row per minute from ``start``. ``prices``: (n,) close prices, or (n, 4) OHLC."""
        p = np.asarray(prices, dtype=np.float64)
        n = p.shape[0]
        if p.ndim == 1:
            o = np.r_[p[0], p[:-1]]
            ohlc = np.stack([o, np.maximum(o, p), np.minimum(o, p), p], axis=1)
        else:
            ohlc = p
        v = np.ones(n) if volume is None else np.asarray(volume, dtype=np.float64)
        qv = v * ohlc[:, 3] if quote is None else np.asarray(quote, dtype=np.float64)
        t = ms(start) + np.arange(n, dtype=np.int64) * MIN
        df = pl.DataFrame(
            {
                "symbol": [sym] * n,
                "open_time": t,
                "open": ohlc[:, 0],
                "high": ohlc[:, 1],
                "low": ohlc[:, 2],
                "close": ohlc[:, 3],
                "volume": v,
                "close_time": t + MIN - 1,
                "quote_volume": qv,
                "count": np.ones(n, dtype=np.int64),
                "taker_buy_volume": v / 2,
                "taker_buy_quote_volume": qv / 2,
                "ignore": np.zeros(n, dtype=np.int64),
            }
        )
        if extra is not None:
            df = df.filter(extra(df))
        self._split(df, "open_time", "klines", sym, kind, sub="1m", label="1m")
        return t

    # ------------------------------------------------------------------ metrics
    def metrics(self, sym: str, ts_ms, oi, kind: str = "daily"):
        ts_ms = np.asarray(ts_ms, dtype=np.int64)
        oi = np.asarray(oi, dtype=np.float64)
        df = pl.DataFrame(
            {
                "symbol": [sym] * len(ts_ms),
                "create_time": [
                    dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
                    for t in ts_ms
                ],
                "sum_open_interest": oi,
                "sum_open_interest_value": oi * 10,
                "count_toptrader_long_short_ratio": oi * 0 + 1.5,
                "sum_toptrader_long_short_ratio": oi * 0 + 2.0,
                "count_long_short_ratio": oi * 0 + 1.2,
                "sum_taker_long_short_vol_ratio": oi * 0 + 0.9,
                "timestamp": ts_ms,
            }
        )
        self._split(df, "timestamp", "metrics", sym, kind, keep_order=True)

    # ------------------------------------------------------------------ aggTrades
    def agg_trades(self, sym: str, ts_ms, price, qty, buyer_maker, kind: str = "daily"):
        ts_ms = np.asarray(ts_ms, dtype=np.int64)
        n = len(ts_ms)
        df = pl.DataFrame(
            {
                "symbol": [sym] * n,
                "agg_trade_id": np.arange(n, dtype=np.int64),
                "price": np.asarray(price, dtype=np.float64),
                "quantity": np.asarray(qty, dtype=np.float64),
                "first_trade_id": np.arange(n, dtype=np.int64),
                "last_trade_id": np.arange(n, dtype=np.int64),
                "transact_time": ts_ms,
                "is_buyer_maker": np.asarray(buyer_maker, dtype=bool),
            }
        )
        self._split(df, "transact_time", "aggTrades", sym, kind)

    # ------------------------------------------------------------------ helpers
    def _split(self, df, tcol, dataset, sym, kind, sub=None, label=None, keep_order=False):
        label = label or dataset
        fmt = "%Y-%m-%d" if kind == "daily" else "%Y-%m"
        period = pl.from_epoch(pl.col(tcol), time_unit="ms").dt.strftime(fmt)
        base = self.root / kind / dataset / sym
        if sub:
            base = base / sub
        base.mkdir(parents=True, exist_ok=True)
        for (per,), part in df.with_columns(_p=period).group_by("_p", maintain_order=True):
            part.drop("_p").write_parquet(base / f"{sym}-{label}-{per}.parquet")

    def file(self, dataset, sym, period, kind="daily", sub=None) -> Path:
        label = "1m" if dataset == "klines" else dataset
        base = self.root / kind / dataset / sym
        if dataset == "klines":
            base = base / "1m"
        return base / f"{sym}-{label}-{period}.parquet"


@pytest.fixture
def env(tmp_path):
    """Fresh data root + cache dir for one test."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    data.mkdir()
    config.set_paths(data_root=data, cache_dir=cache)
    store.forget()
    saved = dict(features.FEATURES)
    features.FEATURES.clear()
    yield Market(data)
    features.FEATURES.clear()
    features.FEATURES.update(saved)
    config.reset_paths()
    store.forget()


def use_cache(path: Path):
    config.set_paths(cache_dir=path)
    store.forget()


def bar_blocks(n_bars: int, k: int, seed: int = 0, base: float = 100.0):
    """Random minute prices + volumes for ``n_bars`` bars of ``k`` minutes."""
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0, 1e-3, n_bars * k)))
    vol = rng.uniform(0.5, 2.0, n_bars * k)
    return close, vol
