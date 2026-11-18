"""Operators: no look-ahead, and the same numbers as AlphaForge's reference kernels."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tsbt import ops as op

AF_SRC = Path("/home/river/PythonProject/AlphaForge/src")


def _panel(seed=0, T=400, N=4):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(T, N)), axis=0) + 50
    x[rng.random((T, N)) < 0.08] = np.nan
    x[100:115, 1] = np.nan  # a long gap
    x[:, 3] = np.round(x[:, 3])  # ties
    return x


def _cases(x, y):
    def yy(a):  # y with a's length and dimensionality
        return y[: len(a)] if a.ndim == 2 else y[: len(a), 0]

    return {
        "ts_sum": lambda a: op.ts_sum(a, 10),
        "ts_mean": lambda a: op.ts_mean(a, 10),
        "ts_std_dev": lambda a: op.ts_std_dev(a, 10),
        "ts_min": lambda a: op.ts_min(a, 7),
        "ts_max": lambda a: op.ts_max(a, 7),
        "ts_median": lambda a: op.ts_median(a, 9),
        "ts_rank": lambda a: op.ts_rank(a, 12),
        "ts_zscore": lambda a: op.ts_zscore(a, 12),
        "ts_scale": lambda a: op.ts_scale(a, 12),
        "ts_av_diff": lambda a: op.ts_av_diff(a, 5),
        "ts_ir": lambda a: op.ts_ir(a, 15),
        "ts_arg_max": lambda a: op.ts_arg_max(a, 9),
        "ts_arg_min": lambda a: op.ts_arg_min(a, 9),
        "ts_product": lambda a: op.ts_product(a / 50, 6),
        "ts_skewness": lambda a: op.ts_skewness(a, 20),
        "ts_kurtosis": lambda a: op.ts_kurtosis(a, 20),
        "ts_moment": lambda a: op.ts_moment(a, 20, 3),
        "ts_quantile": lambda a: op.ts_quantile(a, 15),
        "ts_entropy": lambda a: op.ts_entropy(a, 20, 5),
        "ts_count_nans": lambda a: op.ts_count_nans(a, 10),
        "inst_tvr": lambda a: op.inst_tvr(a, 10),
        "ts_decay_linear": lambda a: op.ts_decay_linear(a, 8),
        "ts_decay_exp_window": lambda a: op.ts_decay_exp_window(a, 8, 0.7),
        "filter": lambda a: op.filter(a, "0.5,0.3,0.2"),
        "ts_delay": lambda a: op.ts_delay(a, 3),
        "ts_delta": lambda a: op.ts_delta(a, 3),
        "ts_returns": lambda a: op.ts_returns(a, 2),
        "log_diff": op.log_diff,
        "ts_backfill": lambda a: op.ts_backfill(a, 4),
        "ts_corr": lambda a: op.ts_corr(a, yy(a), 15),
        "ts_covariance": lambda a: op.ts_covariance(a, yy(a), 15),
        "ts_regression": lambda a: op.ts_regression(a, yy(a), 15, lag=1, rettype=0),
        "ts_seq_beta": lambda a: op.ts_seq_beta(a, 12),
        "ts_ema": lambda a: op.ts_ema(a, 5),
        "ts_sma_cn": lambda a: op.ts_sma_cn(a, 5, 2),
        "hump": lambda a: op.hump(a, 0.8),
        "hump_decay": lambda a: op.hump_decay(a, 0.8),
        "jump_decay": lambda a: op.jump_decay(a, 10, 0.5, 0.2),
        "trade_when": lambda a: op.trade_when(a > 50, a, a < 45),
        "keep": lambda a: op.keep(a, a > 52, 3),
    }


def test_operators_never_look_ahead():
    """op(x)[:k] == op(x[:k]) for every operator and several cut points."""
    x = _panel()
    y = _panel(seed=1)
    for name, f in _cases(x, y).items():
        full = f(x)
        assert full.shape == x.shape, name
        for k in (37, 150, 399):
            np.testing.assert_array_equal(full[:k], f(x[:k]), err_msg=name)
        np.testing.assert_array_equal(f(x[:, 0]), full[:, 0], err_msg=f"{name} 1-D")


def _alphaforge():
    """Import AlphaForge's kernel modules without its package __init__ (CLI deps)."""
    if not AF_SRC.exists():
        pytest.skip("AlphaForge not available")
    for pkg, path in (
        ("alphaforge", AF_SRC / "alphaforge"),
        ("alphaforge.ops", AF_SRC / "alphaforge/ops"),
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    try:
        return {
            n: importlib.import_module(f"alphaforge.ops.{n}")
            for n in ("k2_cumsum", "k3_strided", "k5_recurrence", "k6_shift_fill", "nanops")
        }
    except Exception as e:  # pragma: no cover
        pytest.skip(f"AlphaForge kernels not importable: {e}")


def test_operators_match_alphaforge_reference_kernels():
    af = _alphaforge()
    k2, k3, k5, k6, nv = (
        af["k2_cumsum"],
        af["k3_strided"],
        af["k5_recurrence"],
        af["k6_shift_fill"],
        af["nanops"],
    )
    mv = nv.min_valid
    x = _panel(seed=4)
    y = _panel(seed=5)
    ref = {
        "ts_sum": (op.ts_sum(x, 10), k2.ts_sum_ref(x, 10, mv(10))),
        "ts_mean": (op.ts_mean(x, 10), k2.ts_mean_ref(x, 10, mv(10))),
        "ts_std_dev": (op.ts_std_dev(x, 10), k3._ts_std_numpy(x, 10, mv(10))),
        "ts_min": (op.ts_min(x, 7), k3.ts_min_ref(x, 7, mv(7))),
        "ts_max": (op.ts_max(x, 7), k3.ts_max_ref(x, 7, mv(7))),
        "ts_median": (op.ts_median(x, 9), k3.ts_median_ref(x, 9, mv(9))),
        "ts_rank": (op.ts_rank(x, 12), k3.ts_rank_ref(x, 12, mv(12))),
        "ts_arg_max": (op.ts_arg_max(x, 9), k3.ts_arg_max_ref(x, 9, mv(9))),
        "ts_arg_min": (op.ts_arg_min(x, 9), k3.ts_arg_min_ref(x, 9, mv(9))),
        "ts_product": (op.ts_product(x / 50, 6), k3._ts_product_numpy(x / 50, 6, mv(6))),
        "ts_skewness": (op.ts_skewness(x, 20), k3._ts_skewness_numpy(x, 20, mv(20))),
        "ts_kurtosis": (op.ts_kurtosis(x, 20), k3._ts_kurtosis_numpy(x, 20, mv(20))),
        "ts_moment": (op.ts_moment(x, 20, 3), k3._ts_moment_numpy(x, 20, 3, mv(20))),
        "ts_decay_linear": (op.ts_decay_linear(x, 8), k3._ts_decay_linear_numpy(x, 8)),
        "ts_decay_exp_window": (
            op.ts_decay_exp_window(x, 8, 0.7),
            k3._ts_decay_exp_numpy(x, 8, 0.7, mv(8)),
        ),
        "ts_entropy": (op.ts_entropy(x, 20, 5), k3._ts_entropy_numpy(x, 20, 5, mv(20))),
        "ts_corr": (op.ts_corr(x, y, 15), k2.ts_corr_ref(x, y, 15, mv(15))),
        "ts_covariance": (op.ts_covariance(x, y, 15), k2.ts_covariance_ref(x, y, 15, mv(15))),
        "ts_regression": (
            op.ts_regression(x, y, 15, rettype=2),
            k2.ts_regression_ref(x, y, 15, mv(15), 2),
        ),
        "ts_count_nans": (op.ts_count_nans(x, 10), k2.ts_count_nans_ref(x, 10)),
        "filter": (op.filter(x, "0.5,0.3,0.2"), k2.fir_filter_ref(x, (0.5, 0.3, 0.2))),
        "ts_ema": (op.ts_ema(x, 5), k5._ewma_alpha_py(x, 2 / 6)),
        "hump_decay": (op.hump_decay(x, 0.8), k5._hump_decay_py(x, 0.8)),
        "trade_when": (
            op.trade_when(x > 50, x, x < 45),
            k5._trade_when_py((x > 50).astype(float), x, (x < 45).astype(float)),
        ),
        "keep": (op.keep(x, x > 52, 3), k5._keep_py(x, (x > 52).astype(float), 3.0)),
        "jump_decay": (
            op.jump_decay(x, 10, 0.5, 0.2),
            k5._jump_decay_py(x, k3._ts_std_numpy(x, 10, mv(10)), 0.5, 0.2),
        ),
        "ts_backfill": (op.ts_backfill(x, 4), k6.k_ts_backfill((x,), {"d": 4}, None)),
    }
    try:
        import scipy  # noqa: F401

        ref["ts_quantile"] = (
            op.ts_quantile(x, 15),
            k3._ts_quantile_numpy(x, 15, "gaussian", mv(15)),
        )
    except ImportError:
        pass
    for name, (ours, theirs) in ref.items():
        np.testing.assert_allclose(
            ours, theirs, rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=name
        )
