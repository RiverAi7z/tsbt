"""``tsbt cache build | status | clear``."""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from tsbt import config


def _build(args) -> int:
    from tsbt.data import api, bars, catalog, m1

    fields = args.fields or catalog.KLINE_DEFAULT_FIELDS
    unknown = [f for f in fields if catalog.spec_for(f) is None]
    if unknown:
        print(f"unknown field(s) {unknown}; known: {sorted(api.fields())}", file=sys.stderr)
        return 2
    syms = api.symbols("klines") if args.all else args.symbols
    for sym in syms:
        t = time.time()
        try:
            m1.ensure(sym, catalog.builders_for(fields), workers=args.workers, progress=print)
            for fq in args.freq:
                k = bars.parse_freq(fq)
                meta: dict = {}
                for f in fields:
                    bars.ensure_bar_field(sym, f, k, 0, meta)
        except FileNotFoundError as e:
            print(f"{sym}: skipped ({e})")
            continue
        print(f"{sym}: ok ({time.time() - t:.1f}s)")
    return 0


def _status(args) -> int:
    from tsbt.data import m1, store

    print(f"cache: {config.store_path()}")
    print(f"data : {config.data_root()}")
    if not config.store_path().exists():  # do not create an empty store just to look
        print("  (empty)")
        return 0
    root = store.root()
    syms = sorted(root["m1"].group_keys()) if "m1" in root else []
    for sym in syms:
        attrs = dict(store.group(m1.m1_path(sym)).attrs)
        parts = []
        for key, man in sorted(attrs.items()):
            if not key.startswith("build:"):
                continue
            rng = man.get("range")
            span = ""
            if rng:
                a, b = (time.strftime("%Y-%m-%d", time.gmtime(x / 1000)) for x in rng)
                span = f" {a}..{b}"
            parts.append(f"{key[6:]}[{len(man.get('files', {}))} files{span}]")
        print(f"  {sym}: " + ", ".join(parts))
    return 0


def _clear(args) -> int:
    from tsbt.data import store

    if not args.symbol and not args.field:
        path = config.store_path()
        if path.exists():
            shutil.rmtree(path)
        store.forget()
        print(f"removed {path}")
        return 0
    root = store.root()
    for sym in args.symbol or (sorted(root["m1"].group_keys()) if "m1" in root else []):
        if args.field:
            for f in args.field:
                store.delete(f"m1/{sym}/{f}")
            g = store.group(f"m1/{sym}", create=False)
            if g is not None:
                for key, man in dict(g.attrs).items():
                    if key.startswith("build:") and set(args.field) & set(man.get("fields", [])):
                        del g.attrs[key]
        else:
            store.delete(f"m1/{sym}")
        for tag in sorted(root["bar"].group_keys()) if "bar" in root else []:
            if args.field:
                for f in args.field:
                    store.delete(f"bar/{tag}/{sym}/{f}")
                g = store.group(f"bar/{tag}/{sym}", create=False)
                if g is not None:
                    for f in args.field:
                        g.attrs.pop(f"v:{f}", None)
            else:
                store.delete(f"bar/{tag}/{sym}")
        print(f"cleared {sym} {args.field or ''}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tsbt")
    sub = p.add_subparsers(dest="cmd", required=True)
    cache = sub.add_parser("cache", help="manage the minute/bar cache")
    csub = cache.add_subparsers(dest="action", required=True)
    b = csub.add_parser("build", help="build / update the cache")
    which = b.add_mutually_exclusive_group(required=True)
    which.add_argument("--symbols", nargs="+", help="symbols to build")
    which.add_argument("--all", action="store_true", help="every klines symbol (hundreds)")
    b.add_argument("--freq", nargs="*", default=[], help="also pre-build bars, e.g. 1h 4h 1D")
    b.add_argument("--fields", nargs="*", help="default: all klines fields")
    b.add_argument("--workers", type=int, default=8)
    b.set_defaults(fn=_build)
    s = csub.add_parser("status", help="show what is cached")
    s.set_defaults(fn=_status)
    c = csub.add_parser("clear", help="delete cached data")
    c.add_argument("--symbol", nargs="*")
    c.add_argument("--field", nargs="*")
    c.set_defaults(fn=_clear)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
