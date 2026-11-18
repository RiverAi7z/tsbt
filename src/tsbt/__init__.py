"""tsbt - vectorised time-series signal backtests on Binance USD-M perpetuals."""

from tsbt import ops
from tsbt.config import set_paths
from tsbt.data.api import BarData, M1View, fields, load, load_m1, symbols
from tsbt.data.features import SourceChunk, feature_set
from tsbt.engine import alloc
from tsbt.engine.result import BatchResult, PortfolioResult, Result
from tsbt.engine.run import run

__all__ = [
    "BarData",
    "BatchResult",
    "M1View",
    "PortfolioResult",
    "Result",
    "SourceChunk",
    "alloc",
    "feature_set",
    "fields",
    "load",
    "load_m1",
    "ops",
    "run",
    "set_paths",
    "symbols",
]
