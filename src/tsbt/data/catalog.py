"""Field catalog: where each field comes from and how it aggregates into a bar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsbt.data import m1


@dataclass(frozen=True)
class FieldSpec:
    name: str
    builder: str  # manifest key of the L1 builder providing the inputs
    rule: str  # bar aggregation rule
    #: forward-fill limit (minutes) for the minute view; sparse sources only
    m1_ffill: int = 0


BUILTIN_FIELDS: dict[str, FieldSpec] = {
    "open": FieldSpec("open", "klines", "first"),
    "high": FieldSpec("high", "klines", "max"),
    "low": FieldSpec("low", "klines", "min"),
    "close": FieldSpec("close", "klines", "last"),
    "volume": FieldSpec("volume", "klines", "sum"),
    "quote_volume": FieldSpec("quote_volume", "klines", "sum"),
    "count": FieldSpec("count", "klines", "sum"),
    "taker_buy_volume": FieldSpec("taker_buy_volume", "klines", "sum"),
    "taker_buy_quote_volume": FieldSpec("taker_buy_quote_volume", "klines", "sum"),
    # derived from klines (see minute_values / bars.bar_values)
    "vwap": FieldSpec("vwap", "klines", "vwap"),
    "twap": FieldSpec("twap", "klines", "twap"),
    **{name: FieldSpec(name, "metrics", "last", m1_ffill=15) for name in m1.METRIC_MAP},
}

KLINE_DEFAULT_FIELDS = [*m1.KLINE_FIELDS, "vwap", "twap"]


def parse_exec_field(name: str) -> tuple[str, int | None]:
    """'twap@15' -> ('twap', 15); 'close' -> ('close', None)."""
    if "@" in name:
        base, n = name.split("@", 1)
        if base not in ("twap", "vwap"):
            raise KeyError(f"only twap@N / vwap@N take a window, got {name!r}")
        return base, int(n)
    return name, None


def spec_for(name: str) -> FieldSpec | None:
    base, _ = parse_exec_field(name)
    if base in BUILTIN_FIELDS:
        return BUILTIN_FIELDS[base]
    from tsbt.data import features

    fs = features.feature_for_field(name)
    if fs is not None:
        return FieldSpec(name, fs.key, fs.bar_rule(name))
    return None


def builder_keys(names) -> list[str]:
    """Keys of the minute builders the fields need (klines first, each once)."""
    keys = {"klines": None}
    for name in names:
        spec = spec_for(name)
        if spec is not None:  # None: user bar field, resolved at load time
            keys[spec.builder] = None
    return list(keys)


def builder(key: str):
    if key in m1.BUILTIN_BUILDERS:
        return m1.BUILTIN_BUILDERS[key]
    from tsbt.data import features

    return features.FEATURES[key.split(":", 1)[1]].builder()


def builders_for(names) -> list:
    return [builder(k) for k in builder_keys(names)]


def minute_price(qv: np.ndarray, v: np.ndarray, close: np.ndarray) -> np.ndarray:
    """A minute's traded price: its VWAP, or its close when nothing traded."""
    with np.errstate(invalid="ignore", divide="ignore"):
        px = qv / v
    return np.where((v > 0) & np.isfinite(px), px, close)


def minute_values(symbol: str, name: str, start_ms: int, end_ms: int) -> np.ndarray:
    """Minute-level values of a builtin / feature field over [start_ms, end_ms)."""
    if name in ("vwap", "twap"):
        qv = m1.read_minutes(symbol, "quote_volume", start_ms, end_ms)
        v = m1.read_minutes(symbol, "volume", start_ms, end_ms)
        c = m1.read_minutes(symbol, "close", start_ms, end_ms)
        return minute_price(qv, v, c)
    return m1.read_minutes(symbol, name, start_ms, end_ms)
