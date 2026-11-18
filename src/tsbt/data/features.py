"""User-defined minute features built from raw datasets (aggTrades, trades, ...).

Three ways to define a feature set, from most convenient to most flexible::

    # 1) aggregate expressions; the framework wraps them in GROUP BY minute
    bt.feature_set("flow", source="aggTrades",
                   sql={"buy_amt": "sum(price*quantity) FILTER (WHERE NOT is_buyer_maker)"},
                   bar_agg={"buy_amt": "sum"}, fill={"buy_amt": 0.0})

    # 2) a full DuckDB query; {src} is the current source file, must return a `ts` column
    bt.feature_set("micro", source="aggTrades", query="SELECT ... AS ts, ... FROM {src} GROUP BY 1")

    # 3) a Python callable receiving one source file, returning a polars DataFrame
    @bt.feature_set("custom", source="aggTrades", outputs=["x"], version=1)
    def custom(chunk: bt.SourceChunk) -> pl.DataFrame: ...

Each source file (a month or a day) is computed independently and cached per file; a
changed definition (SQL text / function source + ``version``) rebuilds only that set.

Cache invalidation caveat for Python features: only the decorated function's *own*
source code and ``version`` are hashed. Changing a helper it calls, or a module-level
constant it reads (e.g. ``BIG = 1e5``), is NOT detected - the old cached values keep
being used silently. After changing such outside logic, bump ``version`` (or keep the
logic/thresholds inside the function). ``sql=`` / ``query=`` features are not affected:
their full SQL text is hashed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import duckdb
import numpy as np
import polars as pl

from tsbt.data import sources
from tsbt.data.sources import SourceFile

MIN = 60_000


@dataclass
class SourceChunk:
    """One raw source file handed to a Python feature function."""

    path: str
    dataset: str
    symbol: str
    start_ms: int
    end_ms: int
    threads: int = 1

    def scan(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)

    def sql(self, query: str) -> pl.DataFrame:
        """Run a DuckDB query; ``{src}`` is replaced by this file."""
        return _to_polars(_duck(self.threads).sql(query.replace("{src}", _src(self.path))))


def _to_polars(rel) -> pl.DataFrame:
    """DuckDB relation -> polars without pyarrow (NULL -> NaN for numeric columns)."""
    cols = {}
    for name, arr in rel.fetchnumpy().items():
        if isinstance(arr, np.ma.MaskedArray):
            if arr.mask is not np.ma.nomask and arr.mask.any():
                arr = np.ma.filled(arr.astype(np.float64), np.nan)
            else:
                arr = np.asarray(arr.data)
        cols[name] = np.asarray(arr)
    return pl.DataFrame(cols)


def _src(path: str) -> str:
    return "read_parquet('" + path.replace("'", "''") + "')"


def _duck(threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={max(1, threads)}")
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        con.execute(
            f"SET memory_limit='{max(2, int(total * 0.5 / 2**30 / max(1, 24 // threads)))}GB'"
        )
    except (ValueError, OSError):
        pass
    con.execute("SET preserve_insertion_order=false")
    return con


@dataclass
class FeatureSet:
    name: str
    source: str
    outputs: list[str]
    sql: dict[str, str] | None = None
    query: str | None = None
    fn: Callable[[SourceChunk], pl.DataFrame | pl.LazyFrame] | None = None
    bar_agg: dict[str, str] = field(default_factory=dict)
    fill: dict[str, float] = field(default_factory=dict)
    version: int | str = 0
    threads: int = 4

    @property
    def key(self) -> str:
        return f"feat:{self.name}"

    @property
    def def_hash(self) -> str:
        body = {
            "source": self.source,
            "outputs": self.outputs,
            "sql": self.sql,
            "query": self.query,
            "fn": _fn_source(self.fn) if self.fn else None,
            "fill": self.fill,
            "version": self.version,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    def bar_rule(self, fld: str) -> str:
        return self.bar_agg.get(fld, "last")

    def compute(self, f: SourceFile, symbol: str) -> pl.DataFrame:
        if self.fn is not None:
            out = self.fn(
                SourceChunk(f.path, self.source, symbol, f.start_ms, f.end_ms, self.threads)
            )
            if isinstance(out, pl.LazyFrame):
                out = out.collect()
            if not isinstance(out, pl.DataFrame):
                raise TypeError(
                    f"feature set {self.name!r}: function must return a polars DataFrame"
                )
            return out
        return _to_polars(_duck(self.threads).sql(self._query_for(f.path)))

    def _query_for(self, path: str) -> str:
        if self.query is not None:
            return self.query.replace("{src}", _src(path))
        tcol = sources.get_dataset(self.source).time_col
        cols = ", ".join(f'({expr}) AS "{name}"' for name, expr in self.sql.items())
        return f"SELECT ({tcol} // {MIN}) * {MIN} AS ts, {cols} FROM {_src(path)} GROUP BY 1"

    def builder(self):
        from tsbt.data.m1 import Builder

        return Builder(
            key=self.key,
            dataset=self.source,
            fields=list(self.outputs),
            compute=self.compute,
            def_hash=self.def_hash,
            fill=dict(self.fill),
            strict=True,
        )


def _fn_source(fn) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return getattr(fn, "__qualname__", repr(fn))


FEATURES: dict[str, FeatureSet] = {}


def _infer_query_outputs(source: str, query: str) -> list[str]:
    ds = sources.get_dataset(source)
    if not ds.schema:
        raise ValueError(f"cannot infer outputs for dataset {source!r}; pass outputs=[...]")
    con = duckdb.connect()
    cols = ", ".join(f'"{c}" {t}' for c, t in ds.schema.items())
    con.execute(f"CREATE TABLE __src ({cols})")
    desc = con.sql(f"DESCRIBE {query.replace('{src}', '__src')}").fetchall()
    names = [row[0] for row in desc]
    if "ts" not in names:
        raise ValueError("query must return a 'ts' column (minute start, ms)")
    return [n for n in names if n != "ts"]


def feature_set(
    name: str,
    source: str,
    *,
    sql: dict[str, str] | None = None,
    query: str | None = None,
    outputs: list[str] | None = None,
    bar_agg: dict[str, str] | None = None,
    fill: dict[str, float] | None = None,
    version: int | str = 0,
    threads: int = 4,
):
    """Register a feature set. Without ``sql``/``query`` it is a decorator for a function.

    For a Python function only its own source and ``version`` decide whether cached
    values are recomputed: bump ``version`` after changing helpers or constants it uses.
    """
    sources.get_dataset(source)
    if sql is not None and query is not None:
        raise ValueError("pass either sql= or query=, not both")

    def _register(fn=None) -> FeatureSet:
        if sql is not None:
            outs = list(sql)
        elif query is not None:
            outs = list(outputs) if outputs else _infer_query_outputs(source, query)
        else:
            outs = list(outputs) if outputs else list(bar_agg or {})
            if not outs:
                raise ValueError(
                    f"feature set {name!r}: a Python feature needs outputs=[...] (or bar_agg keys)"
                )
        fs = FeatureSet(
            name=name,
            source=source,
            outputs=outs,
            sql=dict(sql) if sql else None,
            query=query,
            fn=fn,
            bar_agg=dict(bar_agg or {}),
            fill=dict(fill or {}),
            version=version,
            threads=threads,
        )
        _check_names(fs)
        FEATURES[name] = fs
        return fs

    if sql is None and query is None:
        return _register
    return _register()


def _check_names(fs: FeatureSet) -> None:
    from tsbt.data.catalog import BUILTIN_FIELDS

    for out in fs.outputs:
        if out in BUILTIN_FIELDS or out == "ts":
            raise ValueError(f"feature {out!r} clashes with a builtin field")
        for other in FEATURES.values():
            if other.name != fs.name and out in other.outputs:
                raise ValueError(f"feature {out!r} already defined by feature set {other.name!r}")


def feature_for_field(fld: str) -> FeatureSet | None:
    for fs in FEATURES.values():
        if fld in fs.outputs:
            return fs
    return None


def feature_fields() -> dict[str, str]:
    return {o: fs.name for fs in FEATURES.values() for o in fs.outputs}


def unregister(name: str) -> None:
    FEATURES.pop(name, None)
