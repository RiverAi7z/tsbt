"""Backtest result containers."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from tsbt import metrics as M
from tsbt.config import DAY_MS


def _iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.UTC).strftime("%Y-%m-%d %H:%M")


def _year_month(ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UTC (year, month) of the day a time stamp *ends* (00:00 belongs to the day before)."""
    d = ((np.asarray(ms, dtype=np.int64) - 1) // DAY_MS).astype("datetime64[D]")
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return y, m


@dataclass
class Result:
    """One account: a single symbol (or parameter column) or a pooled portfolio."""

    name: str
    symbols: list[str]
    open_ms: np.ndarray
    k_ms: int
    nav: np.ndarray
    nav_gross: np.ndarray
    bench: np.ndarray
    turnover: np.ndarray
    fee_paid: np.ndarray
    position: np.ndarray  # (n, M)
    signal: np.ndarray  # (n, M) delayed signal driving each row
    close: np.ndarray  # (n, M)
    long_pnl: np.ndarray
    short_pnl: np.ndarray
    bankrupt: bool
    config: dict
    alloc: np.ndarray | None = None
    turnover_cols: np.ndarray | None = None
    pnl_cols: np.ndarray | None = None

    # ------------------------------------------------------------------ time axes
    @property
    def close_ms(self) -> np.ndarray:
        return self.open_ms + self.k_ms

    @property
    def ts(self) -> np.ndarray:
        """Bar close times (datetime64[ms]); NAV values are marked at these instants."""
        return self.close_ms.astype("datetime64[ms]")

    @property
    def compound(self) -> bool:
        return bool(self.config["compound"])

    @property
    def fee(self) -> float:
        return float(self.config["fee"])

    def _path(self, values) -> tuple[np.ndarray, np.ndarray]:
        """(times, values) including the base point (start, 1.0)."""
        times = np.r_[self.open_ms[0], self.close_ms]
        return times, np.r_[1.0, np.asarray(values, dtype=np.float64)]

    @property
    def start_ms(self) -> int:
        return int(self.open_ms[0])

    @property
    def end_ms(self) -> int:
        return int(self.close_ms[-1])

    # ------------------------------------------------------------------ metrics
    @cached_property
    def metrics(self) -> dict:
        t, nav = self._path(self.nav)
        out = M.perf(t, nav, self.turnover, self.compound)
        g = M.perf(*self._path(self.nav_gross), self.turnover, self.compound)
        bm = M.perf(*self._path(self.bench), np.zeros_like(self.turnover), self.compound)
        net = self.position.sum(axis=1)
        res = {
            "apr": out["apr"],
            "sharpe": out["sharpe"],
            "calmar": out["calmar"],
            "turnover": out["turnover"],
            "mdd": out["mdd"],
            "start": _iso(self.start_ms),
            "end": _iso(self.end_ms),
            "days": out["days"],
            "total_return": out["total_return"],
            "apr_gross": g["apr"],
            "sharpe_gross": g["sharpe"],
            "fee_drag": out["turnover"] * 365.0 * self.fee,
            "avg_abs_pos": float(np.abs(self.position).sum(axis=1).mean()),
            "long_ratio": float((net > 0).mean()),
            "bench_apr": bm["apr"],
            "bench_sharpe": bm["sharpe"],
            "bench_mdd": bm["mdd"],
            "bankrupt": self.bankrupt,
        }
        return res

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame([{"name": self.name, **self.metrics}])

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"Result({self.name}, {m['start']} -> {m['end']}, apr={m['apr']:.2%}, sharpe={m['sharpe']:.2f}, "
            f"mdd={m['mdd']:.2%}, calmar={m['calmar']:.2f}, turnover={m['turnover']:.3f}/day)"
        )

    # ------------------------------------------------------------------ tables
    def series(self) -> pl.DataFrame:
        data = {
            "ts": self.ts,
            "nav": self.nav,
            "nav_gross": self.nav_gross,
            "benchmark": self.bench,
            "ret": M.period_returns(self._path(self.nav)[1], self.compound),
            "turnover": self.turnover,
            "fee": self.fee_paid,
            "long_pnl": self.long_pnl,
            "short_pnl": self.short_pnl,
        }
        for i, sym in enumerate(
            self.symbols if self.position.shape[1] == len(self.symbols) else [self.name]
        ):
            data[f"pos_{sym}"] = self.position[:, i]
        return pl.DataFrame(data)

    def daily(self) -> pl.DataFrame:
        """Day-end values; ``ret`` is the return of each (possibly partial) UTC day."""
        t, nav = self._path(self.nav)
        _, g = M.daily_values(t, self._path(self.nav_gross)[1])
        _, bm = M.daily_values(t, self._path(self.bench)[1])
        dt_, dv = M.daily_values(t, nav)
        key = (t[1:] - 1) // DAY_MS
        day_turn = np.bincount(np.unique(key, return_inverse=True)[1], weights=self.turnover)
        return pl.DataFrame(
            {
                "date": ((dt_[1:] - 1) // DAY_MS).astype("datetime64[D]"),
                "nav": dv[1:],
                "nav_gross": g[1:],
                "benchmark": bm[1:],
                "ret": M.period_returns(dv, self.compound),
                "turnover": day_turn,
            }
        )

    def _segments(self, labels: np.ndarray):
        """Yield (label, times, nav, turnover) for consecutive groups of NAV points."""
        t, nav = self._path(self.nav)
        lab = labels
        bounds = np.r_[0, np.nonzero(lab[1:] != lab[:-1])[0] + 1, len(lab)]
        for a, z in zip(bounds[:-1], bounds[1:], strict=True):
            base = a  # point index a in `lab` is point a+1 in the path
            seg_t = t[base : z + 1]
            seg_v = nav[base : z + 1]
            if self.compound:
                with np.errstate(divide="ignore", invalid="ignore"):
                    seg_v = seg_v / seg_v[0] if seg_v[0] > 0 else np.full_like(seg_v, np.nan)
            else:
                seg_v = seg_v - seg_v[0] + 1.0
            yield lab[a], seg_t, seg_v, self.turnover[a:z]

    def yearly(self) -> pl.DataFrame:
        y, _ = _year_month(self.close_ms)
        rows = []
        for year, t, v, tv in self._segments(y):
            p = M.perf(t, v, tv, self.compound) if np.isfinite(v).all() else {}
            rows.append(
                {
                    "year": int(year),
                    "return": v[-1] - 1.0 if len(v) else math.nan,
                    **{
                        k: p.get(k, math.nan)
                        for k in ("apr", "sharpe", "calmar", "turnover", "mdd", "days")
                    },
                }
            )
        return pl.DataFrame(rows)

    def monthly(self) -> pl.DataFrame:
        y, m = _year_month(self.close_ms)
        rows = [
            {"year": int(lab // 100), "month": int(lab % 100), "return": float(v[-1] - 1.0)}
            for lab, _, v, _ in self._segments(y * 100 + m)
        ]
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------ output
    def plot(self, kind: str | None = None, **kw):
        """All figures as a dict (kind=None) or one of nav / position / long_short /
        periodic / rolling_sharpe."""
        from tsbt import plot as P

        return P.plot(self, kind, **kw)

    def save(self, path, plots: bool = True) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(_jsonable(self.metrics), indent=2))
        (out / "config.json").write_text(
            json.dumps(
                _jsonable(
                    {
                        **self.config,
                        "name": self.name,
                        "symbols": self.symbols,
                        "start": _iso(self.start_ms),
                        "end": _iso(self.end_ms),
                        "version": _version(),
                    }
                ),
                indent=2,
            )
        )
        self.series().write_parquet(out / "series.parquet")
        self.yearly().write_parquet(out / "yearly.parquet")
        if plots:
            from tsbt import plot as P

            P.save_all(self, out)
        return out


def _version() -> str:
    from importlib.metadata import version

    try:
        return version("tsbt")
    except Exception:  # pragma: no cover
        return "dev"


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return None if not math.isfinite(float(d)) else float(d)
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    return d


class BatchResult:
    """Independent accounts (several symbols, or several signal columns on one symbol)."""

    def __init__(self, books: list[Result]):
        self.books = books

    def __len__(self) -> int:
        return len(self.books)

    def __iter__(self):
        return iter(self.books)

    def __getitem__(self, key) -> Result:
        if isinstance(key, str):
            for b in self.books:
                if b.name == key:
                    return b
            raise KeyError(key)
        return self.books[key]

    def __repr__(self) -> str:
        return f"BatchResult({[b.name for b in self.books]})"

    def summary(self) -> pl.DataFrame:
        return pl.concat([b.summary() for b in self.books], how="vertical_relaxed")

    def plot(self, key=0, kind: str | None = None, **kw):
        return self[key].plot(kind, **kw)

    def save(self, path, plots: bool = True) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.summary().write_parquet(out / "summary.parquet")
        (out / "metrics.json").write_text(
            json.dumps(_jsonable({b.name: b.metrics for b in self.books}), indent=2)
        )
        for b in self.books:
            b.save(out / b.name.replace("/", "_"), plots=plots)
        return out


class PortfolioResult:
    """A pooled account plus each symbol's stand-alone result."""

    def __init__(self, portfolio: Result, components: BatchResult):
        self.portfolio = portfolio
        self.components = components

    def __getattr__(self, name):
        # delegate nav / metrics / series / ... to the portfolio account
        if name in ("portfolio", "components") or name.startswith("__"):
            raise AttributeError(name)  # not set yet (unpickling / copying): no recursion
        return getattr(self.portfolio, name)

    def __repr__(self) -> str:
        return (
            f"PortfolioResult({self.portfolio!r}, components={[b.name for b in self.components]})"
        )

    def summary(self) -> pl.DataFrame:
        port = self.portfolio.summary().with_columns(pl.lit("portfolio").alias("kind"))
        comp = self.components.summary().with_columns(pl.lit("symbol").alias("kind"))
        return pl.concat([port, comp], how="vertical_relaxed").select("kind", pl.exclude("kind"))

    def contrib(self) -> pl.DataFrame:
        """Each symbol's share of the pooled P&L (gross, sum of position x return) and turnover."""
        p = self.portfolio
        days = p.metrics["days"]
        pnl = p.pnl_cols.sum(axis=0)
        tot = pnl.sum()
        return pl.DataFrame(
            {
                "symbol": p.symbols,
                "pnl": pnl,
                "pnl_share": pnl / tot if tot != 0 else np.full_like(pnl, np.nan),
                "turnover": p.turnover_cols.sum(axis=0) / days,
                "avg_alloc": p.alloc.mean(axis=0),
                "avg_abs_pos": np.abs(p.position).mean(axis=0),
            }
        )

    def plot(self, kind: str | None = None, **kw):
        return self.portfolio.plot(kind, **kw)

    def save(self, path, plots: bool = True) -> Path:
        out = self.portfolio.save(path, plots=plots)
        self.summary().write_parquet(out / "summary.parquet")
        self.contrib().write_parquet(out / "contrib.parquet")
        for b in self.components:
            b.save(out / "symbols" / b.name, plots=plots)
        return out
