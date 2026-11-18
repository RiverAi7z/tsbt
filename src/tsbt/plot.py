"""Report figures (xy.pyplot, native mode). The only module that imports xy.

Figures: nav, position, long_short, periodic, rolling_sharpe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsbt import metrics as M

# Palette (light surface): categorical slots in fixed order + neutral benchmark ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # net / long / positive
ORANGE = "#eb6834"  # gross / short / gross exposure
RED = "#e34948"  # drawdown / negative
NEUTRAL = "#f0efec"  # diverging midpoint

KINDS = ("nav", "position", "long_short", "periodic", "rolling_sharpe")
_DPI = 110


def _plt():
    import xy.pyplot as plt

    return plt


def _style(ax, title: str | None = None):
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)


def _zero(ax, t0, t1, y=0.0):
    ax.plot([t0, t1], [y, y], color=AXIS, linewidth=1)


def _headroom(ax, values, log: bool = False, top: float = 0.18):
    """Leave room above the data for an in-plot legend."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return
    lo, hi = float(v.min()), float(v.max())
    if log and lo > 0:
        span = np.log(hi / lo) or 1.0
        ax.set_ylim(lo / np.exp(0.06 * span), hi * np.exp(top * span))
    else:
        span = (hi - lo) or abs(hi) or 1.0
        ax.set_ylim(lo - 0.06 * span, hi + top * span)


def _daily_mean(res, x):
    """Mean of a per-bar series within each UTC day (at day-end times)."""
    key = (res.close_ms - 1) // 86_400_000
    uk, inv = np.unique(key, return_inverse=True)
    mean = np.bincount(inv, weights=x) / np.bincount(inv)
    return ((uk + 1) * 86_400_000).astype("datetime64[ms]"), mean


def _pct(ax):
    plt = _plt()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.0%}"))


def _fig(n_rows: int, ratios, title: str, height: float = 7.5):
    plt = _plt()
    fig, axs = plt.subplots(
        n_rows, 1, figsize=(12, height), sharex=True, gridspec_kw={"height_ratios": list(ratios)}
    )
    if n_rows == 1:
        axs = [axs]
    fig.set_facecolor(SURFACE)
    fig.suptitle(title, fontsize=13, color=INK)
    return fig, list(axs)


def _title(res) -> str:
    m = res.metrics
    cfg = res.config
    unit = "" if res.compound else " (simple)"
    return (
        f"{res.name} · {cfg['freq']} · exec {cfg['price']}"
        f"{'@' + str(cfg['exec_window']) if cfg.get('exec_window') else ''} · fee {cfg['fee'] * 1e4:g}bp{unit}"
    ) + f"\n{m['start']} → {m['end']}"


def _stats(res) -> str:
    m = res.metrics
    mdd = f"{m['mdd']:.1%}" if res.compound else f"{m['mdd']:.3f}"
    return (
        f"apr {m['apr']:.1%}   sharpe {m['sharpe']:.2f}   calmar {m['calmar']:.2f}   "
        f"mdd {mdd}   turnover {m['turnover']:.2f}/day   days {m['days']:.0f}"
    )


def _times(res):
    t, _ = res._path(res.nav)
    return t.astype("datetime64[ms]")


# --------------------------------------------------------------------------- figures


def fig_nav(res, log: bool | None = None):
    """Net / gross NAV with the traded instrument(s) as benchmark; drawdown below."""
    t = _times(res)
    _, nav = res._path(res.nav)
    _, gross = res._path(res.nav_gross)
    _, bench = res._path(res.bench)
    fig, (ax, axd) = _fig(2, (3, 1), _title(res))
    bench_label = f"{res.symbols[0]} hold" if len(res.symbols) == 1 else "buy & hold basket"
    ax.plot(t, bench, color=MUTED, linewidth=1.3, label=bench_label)
    ax.plot(t, gross, color=ORANGE, linewidth=1.3, label="gross (before fee)")
    ax.plot(t, nav, color=BLUE, linewidth=1.8, label="net (after fee)")
    if log is None:
        hi = np.nanmax(np.r_[nav, gross, bench])
        lo = np.nanmin(np.r_[nav, gross, bench])
        log = res.compound and lo > 0 and hi / lo > 8
    if log:
        ax.set_yscale("log")
        lo, hi = (
            float(np.nanmin(np.r_[nav, gross, bench])),
            float(np.nanmax(np.r_[nav, gross, bench])),
        )
        ticks = [
            m * 10.0**e
            for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
            for m in (1, 2, 5)
        ]
        ticks = [v for v in ticks if lo / 1.2 <= v <= hi * 1.2]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:g}" for v in ticks])
    _headroom(ax, np.r_[nav, gross, bench], log)
    _style(ax, "NAV" + (" (log scale)" if log else ""))
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    ax.text(0.01, 0.02, _stats(res), transform=ax.transAxes, fontsize=9, color=INK2)
    dd = M.drawdown_series(nav, res.compound)
    axd.fill_between(t, dd, 0, color=RED, alpha=0.35)
    axd.plot(t, dd, color=RED, linewidth=1)
    _style(axd, "drawdown (net)")
    if res.compound:
        _pct(axd)
    return fig


def fig_position(res):
    """Instrument price, position and daily turnover."""
    t = res.ts
    single = res.position.shape[1] == 1
    fig, (axp, axq, axt) = _fig(3, (2, 1.4, 1), _title(res), height=8.5)
    if single:
        axp.plot(t, res.close[:, 0], color=MUTED, linewidth=1.2)
        _style(axp, f"{res.symbols[0]} close")
    else:
        tt, bench = res._path(res.bench)
        axp.plot(tt.astype("datetime64[ms]"), bench, color=MUTED, linewidth=1.2)
        _style(axp, "buy & hold basket NAV")
    net = res.position.sum(axis=1)
    dts, dnet = _daily_mean(res, net)
    unit = "equity" if res.compound else "principal"
    if single:
        axq.fill_between(t, net, 0, step="post", color=BLUE, alpha=0.18)
        axq.plot(dts, dnet, color=BLUE, linewidth=1.4, label="daily mean")
        axq.legend(frameon=False, loc="upper left", fontsize=9)
        _headroom(axq, net, top=0.3)
        _style(axq, f"position (fraction of {unit}; shaded = per bar)")
    else:
        gross = np.abs(res.position).sum(axis=1)
        _, dgross = _daily_mean(res, gross)
        axq.plot(dts, dgross, color=ORANGE, linewidth=1.3, label="gross exposure")
        axq.plot(dts, dnet, color=BLUE, linewidth=1.3, label="net exposure")
        axq.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
        _headroom(axq, np.r_[dgross, dnet], top=0.35)
        _style(axq, f"exposure (daily mean, fraction of {unit})")
    _zero(axq, t[0], t[-1])
    daily = res.daily()
    dts = daily["date"].to_numpy().astype("datetime64[ms]")
    axt.fill_between(dts, daily["turnover"].to_numpy(), 0, step="post", color=BLUE, alpha=0.6)
    _style(axt, "daily turnover (two-sided)")
    return fig


def fig_long_short(res):
    """Cumulative gross P&L split by long / short positions."""
    t = res.ts
    lp, sp = np.cumsum(res.long_pnl), np.cumsum(res.short_pnl)
    fig, (ax,) = _fig(1, (1,), _title(res), height=5.5)
    ax.plot(t, lp, color=BLUE, linewidth=1.5, label=f"long  {lp[-1]:+.2f}")
    ax.plot(t, sp, color=ORANGE, linewidth=1.5, label=f"short {sp[-1]:+.2f}")
    ax.plot(t, lp + sp, color=INK, linewidth=1.2, label=f"total {lp[-1] + sp[-1]:+.2f}")
    _zero(ax, t[0], t[-1])
    _headroom(ax, np.r_[lp, sp, lp + sp])
    _style(ax, "cumulative P&L by side (Σ position × return, before fee)")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    return fig


def _diverging(v: np.ndarray, vmax: float) -> np.ndarray:
    """red (neg) - neutral gray - blue (pos); NaN -> surface."""

    def rgb(h):
        return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255

    neg, mid, pos, bg = rgb(RED), rgb(NEUTRAL), rgb(BLUE), rgb(SURFACE)
    x = np.clip(np.nan_to_num(v / vmax), -1, 1)[..., None]
    out = np.where(x < 0, mid + (neg - mid) * -x, mid + (pos - mid) * x)
    out[np.isnan(v)] = bg
    return out


def fig_periodic(res):
    """Monthly return heatmap + yearly returns (with Sharpe / MDD)."""
    plt = _plt()
    mon = res.monthly()
    yr = res.yearly()
    years = sorted(set(mon["year"].to_list()))
    grid = np.full((len(years), 12), np.nan)
    for y, m, r in mon.iter_rows():
        grid[years.index(y), m - 1] = r
    fig = plt.figure(figsize=(12, 3.2 + 0.42 * len(years) + 3))
    fig.set_facecolor(SURFACE)
    fig.suptitle(_title(res), fontsize=13, color=INK)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.6 * len(years) + 0.8, 3])
    ax = fig.add_subplot(gs[0])
    finite = np.abs(grid[np.isfinite(grid)])
    vmax = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    ax.imshow(_diverging(grid, vmax), aspect="auto")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    fmt = "{:+.1%}" if res.compound else "{:+.3f}"
    for i in range(len(years)):
        for j in range(12):
            v = grid[i, j]
            if np.isfinite(v):
                strong = abs(v) / vmax > 0.6
                ax.text(
                    j,
                    i,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#ffffff" if strong else INK,
                )
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("monthly return (net)", loc="left", fontsize=11, color=INK)
    axb = fig.add_subplot(gs[1])
    ys = yr["year"].to_numpy()
    rets = yr["return"].to_numpy()
    axb.bar(ys, rets, color=[BLUE if r >= 0 else RED for r in rets], width=0.6)
    span = np.nanmax(np.abs(rets)) if len(rets) else 1.0
    for y, r, s, d in zip(ys, rets, yr["sharpe"].to_numpy(), yr["mdd"].to_numpy(), strict=True):
        dd = f"{d:.0%}" if res.compound else f"{d:.2f}"
        label = (f"{r:+.1%}" if res.compound else f"{r:+.3f}") + f"\nS {s:.2f} · DD {dd}"
        if r >= 0:
            axb.text(y, r + 0.03 * span, label, ha="center", va="bottom", fontsize=8, color=INK2)
        else:
            axb.text(y, r - 0.03 * span, label, ha="center", va="top", fontsize=8, color=INK2)
    axb.set_xticks(list(ys))
    axb.set_xticklabels([str(y) for y in ys])
    axb.set_ylim(min(0, np.nanmin(rets)) - 0.4 * span, max(0, np.nanmax(rets)) + 0.3 * span)
    _style(axb, "yearly return (net) · S = Sharpe · DD = max drawdown")
    if res.compound:
        _pct(axb)
    _zero(axb, ys.min() - 0.5, ys.max() + 0.5)
    return fig


def fig_rolling_sharpe(res, window_days: int = 90):
    t0 = _times(res)
    _, nav = res._path(res.nav)
    _, bench = res._path(res.bench)
    ts, s = M.rolling_sharpe(res._path(res.nav)[0], nav, res.compound, window_days)
    _, sb = M.rolling_sharpe(res._path(res.nav)[0], bench, res.compound, window_days)
    t = ts.astype("datetime64[ms]")
    fig, (ax,) = _fig(1, (1,), _title(res), height=5.5)
    ax.plot(t, sb, color=MUTED, linewidth=1.2, label="benchmark")
    ax.plot(t, s, color=BLUE, linewidth=1.6, label="strategy (net)")
    _zero(ax, t0[0], t0[-1])
    _headroom(ax, np.r_[s, sb])
    _style(ax, f"rolling {window_days}-day Sharpe (daily returns, ×√365)")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    return fig


_FIGS = {
    "nav": fig_nav,
    "position": fig_position,
    "long_short": fig_long_short,
    "periodic": fig_periodic,
    "rolling_sharpe": fig_rolling_sharpe,
}


class Figures(dict):
    """The report figures by name. Ending a notebook cell with it shows all of them
    (a plain dict would only print its repr)."""

    def _ipython_display_(self):
        from IPython.display import display

        for fig in self.values():
            display(fig)


def _build(res, kind: str, **kw):
    fig = _FIGS[kind](res, **kw)
    # Take it out of pyplot's global registry so repeated calls do not pile up figures;
    # the object itself still renders in a notebook and can be saved.
    _plt().close(fig)
    return fig


def plot(res, kind: str | None = None, **kw):
    if kind is None:
        return Figures((k, _build(res, k)) for k in KINDS)
    if kind not in _FIGS:
        raise ValueError(f"kind must be one of {KINDS}")
    return _build(res, kind, **kw)


def save_all(res, out: Path) -> list[Path]:
    paths = []
    for k in KINDS:
        p = Path(out) / f"{k}.png"
        _build(res, k).savefig(str(p), dpi=_DPI, facecolor=SURFACE)
        paths.append(p)
    return paths
