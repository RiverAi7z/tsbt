# tsbt agent notes

## Data scope

Use only what [docs/DATA.md](docs/DATA.md) lists: 10 symbols × 6 datasets, starting 2023-01-01. Leave out the other symbols and the `trades` / `liquidationSnapshot` datasets.

A `funding_rate` row stamped `t` is the rate settled at `t` for the 8h period that ended at `t`. It is knowable only from `t` onward.

## Sample split

Pass the window to `bt.run` with `start` / `end`. `end` is exclusive.

| Segment | start | end |
|---|---|---|
| Train (IS) | 2023-01-01 | 2025-01-01 |
| Validation | 2025-01-01 | 2026-01-01 |
| Test (OOS) | 2026-01-01 | (data end) |

- **Never evaluate on the test segment unless the user explicitly asks for it in that conversation.** Calling `bt.run` without `end` evaluates through the end of the data, test segment included. While iterating, always pass `end="2026-01-01"` or earlier.
- Do not use validation results to tune parameters step by step. Validation is a go / no-go check against criteria written down before running it.
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
