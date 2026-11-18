"""Saving a backtest produces every required metric and all five charts."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

import tsbt as bt
from tests.conftest import bar_blocks

REQUIRED = {"apr", "sharpe", "calmar", "turnover", "mdd", "start", "end", "days"}
PNGS = {"nav.png", "position.png", "long_short.png", "periodic.png", "rolling_sharpe.png"}


def test_save_single_and_portfolio_reports(env, tmp_path, monkeypatch):
    for i, sym in enumerate(["AAAUSDT", "BBBUSDT"]):
        close, vol = bar_blocks(24 * 70, 60, seed=30 + i)
        env.klines(sym, "2024-01-01", close, volume=vol)
    d = bt.load("AAAUSDT", "4h")
    sig = bt.ops.tanh(bt.ops.ts_zscore(bt.ops.ts_returns(d.close, 3), 30))
    out = bt.run(sig, d).save(tmp_path / "single")
    m = json.loads((out / "metrics.json").read_text())
    assert REQUIRED <= set(m) and all(m[k] is not None for k in REQUIRED - {"calmar"})
    for p in PNGS:
        assert (out / p).stat().st_size > 10_000, p
    # in a notebook: one figure renders by itself, all five render through the container
    import IPython.display
    import xy.pyplot as plt

    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)
    figs = bt.run(sig, d).plot()
    assert list(figs) == ["nav", "position", "long_short", "periodic", "rolling_sharpe"]
    figs._ipython_display_()
    assert shown == list(figs.values()) and "<" in figs["nav"]._repr_html_()
    assert plt.get_fignums() == []  # nothing accumulates in pyplot's registry
    s = pl.read_parquet(out / "series.parquet")
    assert {"nav", "nav_gross", "benchmark", "turnover", "fee"} <= set(s.columns)

    dd = bt.load(["AAAUSDT", "BBBUSDT"], "4h")
    sig2 = bt.ops.tanh(bt.ops.ts_zscore(bt.ops.ts_returns(dd.close, 3), 30))
    res = bt.run(sig2, dd, compound=False)
    out2 = res.save(tmp_path / "pf")
    summ = pl.read_parquet(out2 / "summary.parquet")
    assert summ["name"].to_list() == ["portfolio", "AAAUSDT", "BBBUSDT"]
    assert REQUIRED <= set(summ.columns)
    for sym in ("AAAUSDT", "BBBUSDT"):
        assert (out2 / "symbols" / sym / "nav.png").exists()
    assert np.isclose(pl.read_parquet(out2 / "contrib.parquet")["pnl_share"].sum(), 1.0)
