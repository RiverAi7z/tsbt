"""L1 minute store: per-source-file incremental builds.

A *builder* turns one raw source file into a minute-level frame (``ts`` + fields, at most
one row per minute).  Every source file covers a disjoint time range, so files are
computed independently and each one overwrites exactly its own range in the minute
arrays.  The manifest (per symbol x builder) records which files were built and their
(size, mtime) stamps; on each load only the directory listing is compared:

* unchanged -> nothing is read;
* new / changed files -> only those files are (re)computed;
* removed files -> their range is blanked (unless a selected file re-covers it);
* definition hash changed -> full rebuild of that builder's fields only.

Results are flushed in batches and the manifest is updated after every flush, so an
interrupted build resumes where it stopped.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from tsbt import config
from tsbt.data import sources, store
from tsbt.data.sources import SourceFile

MIN = config.MINUTE_MS
#: flush computed files to the store at least this often (seconds) so builds can resume
FLUSH_SECONDS = 30.0


def m1_path(symbol: str, fld: str | None = None) -> str:
    return f"m1/{symbol}" if fld is None else f"m1/{symbol}/{fld}"


def ms_to_idx(ms: int) -> int:
    return (int(ms) - config.ORIGIN_MS) // MIN


@dataclass
class Builder:
    key: str
    dataset: str
    fields: list[str]
    compute: Callable[[SourceFile, str], pl.DataFrame]
    def_hash: str = "builtin-v1"
    fill: dict[str, float] = field(default_factory=dict)
    #: built-ins silently drop rows outside a file's range; user features raise
    strict: bool = False
    #: threads one ``compute`` call uses itself (DuckDB); bounds the file concurrency
    threads: int = 1


# --------------------------------------------------------------------------- builtins


KLINE_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]


def _normalize_ms(col: pl.Expr) -> pl.Expr:
    # Guard against files stored in microseconds.
    return pl.when(col > 10**14).then(col // 1000).otherwise(col)


def _compute_klines(f: SourceFile, symbol: str) -> pl.DataFrame:
    df = pl.read_parquet(f.path, columns=["open_time", *KLINE_FIELDS])
    return (
        df.select(
            _normalize_ms(pl.col("open_time").cast(pl.Int64)).alias("ts"),
            *[pl.col(c).cast(pl.Float64) for c in KLINE_FIELDS],
        )
        .filter(pl.col("ts") % MIN == 0)
        .unique("ts", keep="last", maintain_order=True)
        .sort("ts")
    )


METRIC_MAP = {
    "oi": "sum_open_interest",
    "oi_value": "sum_open_interest_value",
    "top_ls_acct": "count_toptrader_long_short_ratio",
    "top_ls_pos": "sum_toptrader_long_short_ratio",
    "ls_acct": "count_long_short_ratio",
    "taker_ls_vol": "sum_taker_long_short_vol_ratio",
}


def _compute_metrics(f: SourceFile, symbol: str) -> pl.DataFrame:
    df = pl.read_parquet(f.path)
    if "timestamp" in df.columns:
        t = _normalize_ms(pl.col("timestamp").cast(pl.Int64))
    else:
        t = (
            pl.col("create_time")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
            .dt.epoch("ms")
        )
    cols = [
        (pl.col(src).cast(pl.Float64) if src in df.columns else pl.lit(None, pl.Float64)).alias(dst)
        for dst, src in METRIC_MAP.items()
    ]
    # A record stamped tau is known only from tau on -> it belongs to minute floor(tau).
    return (
        df.select(t.alias("t"), *cols)
        .sort("t", maintain_order=True)
        .with_columns(ts=(pl.col("t") // MIN) * MIN)
        .unique("ts", keep="last", maintain_order=True)
        .drop("t")
        .sort("ts")
    )


BUILTIN_BUILDERS: dict[str, Builder] = {
    "klines": Builder("klines", "klines", KLINE_FIELDS, _compute_klines),
    "metrics": Builder("metrics", "metrics", list(METRIC_MAP), _compute_metrics),
}


# --------------------------------------------------------------------------- sync


def manifest(symbol: str, key: str) -> dict | None:
    return store.get_attr(m1_path(symbol), f"build:{key}")


def _validate(df: pl.DataFrame, b: Builder, f: SourceFile) -> pl.DataFrame:
    if "ts" not in df.columns:
        raise ValueError(
            f"[{b.key}] {f.name}: output must contain a 'ts' column (minute start, ms)"
        )
    missing = [c for c in b.fields if c not in df.columns]
    if missing:
        raise ValueError(f"[{b.key}] {f.name}: output lacks columns {missing}")
    df = df.select(pl.col("ts").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in b.fields])
    if df.height == 0:
        return df
    ts = df["ts"]
    if ts.null_count():
        raise ValueError(f"[{b.key}] {f.name}: 'ts' contains nulls")
    if (ts % MIN != 0).any():
        raise ValueError(f"[{b.key}] {f.name}: 'ts' must be a multiple of 60000 ms (minute start)")
    if ts.n_unique() != df.height:
        raise ValueError(f"[{b.key}] {f.name}: at most one row per minute is allowed")
    inside = (ts >= f.start_ms) & (ts < f.end_ms)
    if not inside.all():
        if b.strict:
            bad = ts.filter(~inside).head(3).to_list()
            raise ValueError(
                f"[{b.key}] {f.name}: 'ts' outside the file range "
                f"[{f.start_ms}, {f.end_ms}): e.g. {bad}"
            )
        df = df.filter(inside)
    return df


@dataclass
class _Job:
    start_ms: int
    end_ms: int
    df: pl.DataFrame | None  # None = blank the range
    name: str | None = None
    stamp: list[int] | None = None


def _flush(symbol: str, b: Builder, jobs: list[_Job], man: dict) -> None:
    lo = ms_to_idx(min(j.start_ms for j in jobs))
    hi = ms_to_idx(max(j.end_ms for j in jobs))
    span = {fl: store.read(m1_path(symbol, fl), lo, hi) for fl in b.fields}
    rng = man.get("range")
    # blanks first so a file job over the same range wins
    for j in sorted(jobs, key=lambda j: j.df is not None):
        a, z = ms_to_idx(j.start_ms) - lo, ms_to_idx(j.end_ms) - lo
        for fl in b.fields:
            span[fl][a:z] = b.fill.get(fl, np.nan) if j.df is not None else np.nan
        if j.df is None or j.df.height == 0:
            continue
        idx = (j.df["ts"].to_numpy() - config.ORIGIN_MS) // MIN - lo
        for fl in b.fields:
            span[fl][idx] = j.df[fl].to_numpy()
        t0, t1 = int(j.df["ts"].min()), int(j.df["ts"].max())
        rng = [t0, t1] if rng is None else [min(rng[0], t0), max(rng[1], t1)]
    for fl in b.fields:
        store.write(m1_path(symbol, fl), lo, span[fl])
    if any(j.df is None for j in jobs):
        # Source files were withdrawn: the remembered range may now be too wide, and a
        # blanked stretch must not keep counting as "listed". Re-derive it from the data.
        rng = _stored_range(symbol, b)
    files = man.setdefault("files", {})
    for j in jobs:
        if j.name is not None:
            if j.stamp is None:
                files.pop(j.name, None)
            else:
                files[j.name] = j.stamp
    man["range"] = rng
    man["version"] = time.time_ns()
    store.set_attr(m1_path(symbol), f"build:{b.key}", man)


def _stored_range(symbol: str, b: Builder) -> list[int] | None:
    """[first, last] minute (ms) holding any value of the builder's fields."""
    first = last = None
    for fl in b.fields:
        path = m1_path(symbol, fl)
        idx = np.flatnonzero(np.isfinite(store.read(path, 0, store.size(path))))
        if idx.size:
            first = int(idx[0]) if first is None else min(first, int(idx[0]))
            last = int(idx[-1]) if last is None else max(last, int(idx[-1]))
    if first is None:
        return None
    return [config.ORIGIN_MS + first * MIN, config.ORIGIN_MS + last * MIN]


def sync(
    symbol: str,
    b: Builder,
    workers: int = 8,
    flush_seconds: float | None = None,
    progress: Callable[[str], None] | None = None,
    files: list[SourceFile] | None = None,
) -> bool:
    """Bring the minute arrays of builder ``b`` for ``symbol`` up to date.

    Returns True if anything was (re)computed. ``files`` is the current source listing
    (looked up when omitted).
    """
    flush_seconds = FLUSH_SECONDS if flush_seconds is None else flush_seconds
    if files is None:
        files = sources.list_files(b.dataset, symbol)
    with store.symbol_lock(symbol):
        man = manifest(symbol, b.key)
        if man is None or man.get("def_hash") != b.def_hash or man.get("fields") != b.fields:
            if man is not None:
                for fl in set(man.get("fields", [])) | set(b.fields):
                    store.delete(m1_path(symbol, fl))
            man = {"def_hash": b.def_hash, "fields": list(b.fields), "files": {}, "range": None}
            store.set_attr(m1_path(symbol), f"build:{b.key}", man)
        built = man.get("files", {})
        current = {f.name: f for f in files}
        pending = [f for f in files if built.get(f.name) != f.stamp]
        removed = [n for n in built if n not in current]
        if not pending and not removed:
            return False

        blanks: list[_Job] = []
        for name in removed:
            parsed = sources.file_period(name)
            if parsed is not None:
                blanks.append(_Job(parsed[1], parsed[2], None, name=name, stamp=None))
        if blanks and not pending:
            _flush(symbol, b, blanks, man)
            return True

        # files in parallel x threads per file must not exceed the machine
        cpus = os.cpu_count() or 1
        workers = max(1, min(workers, cpus // min(max(1, b.threads), cpus)))
        if progress:
            progress(f"{symbol}/{b.key}: computing {len(pending)} file(s)")
        batch: list[_Job] = list(blanks)
        last = time.monotonic()
        rows = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(b.compute, f, symbol): f for f in pending}
            try:
                for done, fut in enumerate(as_completed(futs), 1):
                    f = futs[fut]
                    df = _validate(fut.result(), b, f)
                    batch.append(_Job(f.start_ms, f.end_ms, df, name=f.name, stamp=f.stamp))
                    rows += (f.end_ms - f.start_ms) // MIN
                    if time.monotonic() - last > flush_seconds or rows > 3_000_000:
                        _flush(symbol, b, batch, man)
                        if progress:
                            progress(f"{symbol}/{b.key}: {done}/{len(pending)} files")
                        batch, rows, last = [], 0, time.monotonic()
            finally:
                # On an error or Ctrl-C: do not start the queued files, and keep every
                # file that did finish so the next run resumes instead of starting over.
                for fut in futs:
                    fut.cancel()
                if batch:
                    _flush(symbol, b, batch, man)
        return True


def ensure(symbol: str, builders: list[Builder], workers: int = 8, progress=None) -> None:
    """Sync every builder once (klines first: it defines the listing range)."""
    seen = set()
    for b in sorted(builders, key=lambda b: b.key != "klines"):
        if b.key in seen:
            continue
        seen.add(b.key)
        files = sources.list_files(b.dataset, symbol)
        if not files:
            if b.key == "klines":
                raise FileNotFoundError(f"no klines files for {symbol} under {config.data_root()}")
            raise FileNotFoundError(
                f"{symbol} has no {b.dataset!r} data under {config.data_root()} "
                f"(needed for fields {b.fields})"
            )
        sync(symbol, b, workers=workers, progress=progress, files=files)


def listing_range(symbol: str) -> tuple[int, int]:
    """(first, last) minute start (ms) with klines data."""
    man = manifest(symbol, "klines")
    if not man or not man.get("range"):
        raise RuntimeError(f"klines for {symbol} not built")
    return int(man["range"][0]), int(man["range"][1])


def version(symbol: str, key: str) -> int:
    man = manifest(symbol, key)
    return int(man["version"]) if man and "version" in man else 0


def read_minutes(symbol: str, fld: str, start_ms: int, end_ms: int) -> np.ndarray:
    return store.read(m1_path(symbol, fld), ms_to_idx(start_ms), ms_to_idx(end_ms))
