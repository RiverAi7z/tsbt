# tsbt agent notes

## Data scope

Use only what [docs/DATA.md](docs/DATA.md) lists: 10 symbols × 6 datasets, starting 2023-01-01. Leave out the other symbols and the `trades` / `liquidationSnapshot` datasets.

A `funding_rate` row stamped `t` is the rate settled at `t` for the 8h period that ended at `t`. It is knowable only from `t` onward.

## Sample split

Pass the window to `bt.run` with `start` / `end`. `end` is exclusive.

Current split (set 2026-11-19). IS covers both bull and bear regimes:

| Segment | start | end | use |
|---|---|---|---|
| IS | 2023-01-01 | 2026-07-01 | research; model choice by time-block cross-validation inside IS |
| Holdout | 2026-07-01 | (data end) | one look, final signal only, then paper trading |

- **Never evaluate on the holdout unless the user explicitly asks for it in that conversation.** Calling `bt.run` without `end` evaluates through the end of the data, holdout included. While iterating, always pass `end="2026-07-01"` or earlier.
- Choose parameters and models with time-block cross-validation inside IS (purged, with an embargo), not by repeatedly checking one fixed window.
- When reporting a Sharpe ratio, state the window and give its standard error (about 1/√years).

## Research log (outside the package)

Trial logging is a research convention. Keep it out of `src/tsbt`.

- Each idea gets `research/<idea>/NOTES.md`, written before any backtest. It records:
  - the hypothesis and its mechanism;
  - the free parameters (at most 3) and the grid to try;
  - the validation pass criteria.
- Save every evaluated configuration with `res.save("research/<idea>/runs/<name>")`, using a path relative to the project root. Rejected configurations count too. `series.parquet` keeps the return series for later trial-count and noise-threshold analysis.
- `research/` and `.cache/` both live in the project root and are gitignored. The bar cache defaults to `<project>/.cache`, whatever the working directory.
- Append one row per evaluation to the trial table in `NOTES.md`: date, configuration, window, Sharpe, standard error. Record the final decision there as well.
