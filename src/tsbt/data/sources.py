"""Raw dataset registry and source-file discovery.

Layout: ``<root>/{monthly,daily}/<dataset>/<SYMBOL>[/<sub>]/<SYMBOL>-...-<period>.parquet``
where ``period`` is ``YYYY-MM`` (monthly) or ``YYYY-MM-DD`` (daily).

When a monthly file exists for a month, every daily file of that month is ignored.
"""

from __future__ import annotations

import calendar
import datetime as dt
import os
import re
from dataclasses import dataclass, field

from tsbt import config

_PERIOD_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?\.parquet$")


@dataclass(frozen=True)
class Dataset:
    name: str
    #: directory below ``<root>/<kind>/`` with ``{sym}`` placeholder
    subdir: str
    #: column holding the event time (ms since epoch)
    time_col: str
    #: column -> duckdb type, used to type-check user SQL without touching data
    schema: dict[str, str] = field(default_factory=dict)


DATASETS: dict[str, Dataset] = {}


def register_dataset(ds: Dataset) -> Dataset:
    DATASETS[ds.name] = ds
    return ds


register_dataset(
    Dataset(
        "klines",
        "klines/{sym}/1m",
        "open_time",
        {
            "symbol": "VARCHAR",
            "open_time": "BIGINT",
            "open": "DOUBLE",
            "high": "DOUBLE",
            "low": "DOUBLE",
            "close": "DOUBLE",
            "volume": "DOUBLE",
            "close_time": "BIGINT",
            "quote_volume": "DOUBLE",
            "count": "BIGINT",
            "taker_buy_volume": "DOUBLE",
            "taker_buy_quote_volume": "DOUBLE",
            "ignore": "BIGINT",
        },
    )
)
register_dataset(
    Dataset(
        "metrics",
        "metrics/{sym}",
        "timestamp",
        {
            "symbol": "VARCHAR",
            "create_time": "VARCHAR",
            "sum_open_interest": "DOUBLE",
            "sum_open_interest_value": "DOUBLE",
            "count_toptrader_long_short_ratio": "DOUBLE",
            "sum_toptrader_long_short_ratio": "DOUBLE",
            "count_long_short_ratio": "DOUBLE",
            "sum_taker_long_short_vol_ratio": "DOUBLE",
            "timestamp": "BIGINT",
        },
    )
)
register_dataset(
    Dataset(
        "aggTrades",
        "aggTrades/{sym}",
        "transact_time",
        {
            "symbol": "VARCHAR",
            "agg_trade_id": "BIGINT",
            "price": "DOUBLE",
            "quantity": "DOUBLE",
            "first_trade_id": "BIGINT",
            "last_trade_id": "BIGINT",
            "transact_time": "BIGINT",
            "is_buyer_maker": "BOOLEAN",
        },
    )
)
register_dataset(
    Dataset(
        "trades",
        "trades/{sym}",
        "time",
        {
            "symbol": "VARCHAR",
            "id": "BIGINT",
            "price": "DOUBLE",
            "qty": "DOUBLE",
            "quote_qty": "DOUBLE",
            "time": "BIGINT",
            "is_buyer_maker": "BOOLEAN",
        },
    )
)


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}") from None


@dataclass(frozen=True)
class SourceFile:
    path: str
    name: str
    kind: str  # "monthly" | "daily"
    period: str
    start_ms: int
    end_ms: int  # exclusive
    size: int
    mtime_ns: int

    @property
    def stamp(self) -> list[int]:
        return [self.size, self.mtime_ns]


def _period_range(y: int, m: int, d: int | None) -> tuple[int, int]:
    if d is None:
        start = dt.datetime(y, m, 1, tzinfo=dt.UTC)
        days = calendar.monthrange(y, m)[1]
        end = start + dt.timedelta(days=days)
    else:
        start = dt.datetime(y, m, d, tzinfo=dt.UTC)
        end = start + dt.timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def file_period(name: str) -> tuple[str, int, int] | None:
    """(period, start_ms, end_ms) encoded in a source file name, or None."""
    m = _PERIOD_RE.search(name)
    if m is None:
        return None
    d = m.group(3)
    start, end = _period_range(int(m.group(1)), int(m.group(2)), int(d) if d else None)
    return m.group(0)[: -len(".parquet")], start, end


def _scan(kind: str, ds: Dataset, symbol: str) -> list[SourceFile]:
    base = config.data_root() / kind / ds.subdir.format(sym=symbol)
    if not base.is_dir():
        return []
    out = []
    with os.scandir(base) as it:
        for entry in it:
            if not entry.name.endswith(".parquet"):
                continue
            parsed = file_period(entry.name)
            if parsed is None:
                continue
            period, start, end = parsed
            if (kind == "monthly") != (len(period) == 7):  # YYYY-MM vs YYYY-MM-DD
                continue
            st = entry.stat()
            out.append(
                SourceFile(
                    path=entry.path,
                    name=entry.name,
                    kind=kind,
                    period=period,
                    start_ms=start,
                    end_ms=end,
                    size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                )
            )
    return out


def list_files(dataset: str, symbol: str) -> list[SourceFile]:
    """Selected source files (monthly preferred over daily), sorted by time."""
    ds = get_dataset(dataset)
    monthly = _scan("monthly", ds, symbol)
    months = {f.period for f in monthly}
    daily = [f for f in _scan("daily", ds, symbol) if f.period[:7] not in months]
    return sorted(monthly + daily, key=lambda f: f.start_ms)


def list_symbols(dataset: str = "klines") -> list[str]:
    ds = get_dataset(dataset)
    top = ds.subdir.split("/")[0]
    syms: set[str] = set()
    for kind in ("monthly", "daily"):
        base = config.data_root() / kind / top
        if base.is_dir():
            syms.update(p.name for p in base.iterdir() if p.is_dir())
    return sorted(syms)
