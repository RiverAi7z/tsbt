"""Element-wise operators. NaN propagates unless stated; non-finite results become NaN."""

from __future__ import annotations

import functools

import numpy as np

from tsbt.ops._util import purify


def _f(x):
    return np.asarray(x, dtype=np.float64)


def _quiet(fn):
    @functools.wraps(fn)
    def run(*a, **kw):
        with np.errstate(all="ignore"):
            return purify(fn(*a, **kw))

    return run


def _zf(x, filter):
    x = _f(x)
    return np.where(np.isnan(x), 0.0, x) if filter else x


@_quiet
def add(*xs, filter: bool = False):
    """Sum of two or more inputs; ``filter=True`` treats NaN as 0."""
    return functools.reduce(np.add, (_zf(x, filter) for x in xs))


@_quiet
def subtract(x, y, filter: bool = False):
    return _zf(x, filter) - _zf(y, filter)


@_quiet
def multiply(*xs, filter: bool = False):
    """Product of two or more inputs; ``filter=True`` treats NaN as 0."""
    return functools.reduce(np.multiply, (_zf(x, filter) for x in xs))


@_quiet
def divide(x, y):
    return _f(x) / _f(y)


@_quiet
def inverse(x):
    return 1.0 / _f(x)


@_quiet
def reverse(x):
    return -_f(x)


@_quiet
def abs(x):
    return np.abs(_f(x))


@_quiet
def sign(x):
    return np.sign(_f(x))


@_quiet
def log(x):
    x = _f(x)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


@_quiet
def exp(x):
    return np.exp(_f(x))


@_quiet
def sqrt(x):
    return np.sqrt(_f(x))


@_quiet
def power(x, y):
    return np.power(_f(x), _f(y))


@_quiet
def signed_power(x, y):
    x = _f(x)
    return np.sign(x) * np.power(np.abs(x), _f(y))


@_quiet
def tanh(x):
    return np.tanh(_f(x))


@_quiet
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-_f(x)))


@_quiet
def clamp(x, lower=-np.inf, upper=np.inf):
    x = _f(x)
    return np.where(np.isnan(x), np.nan, np.clip(x, lower, upper))


def _variadic(reducer, xs, filter):
    arrs = [_f(x) for x in xs]
    if filter:
        return functools.reduce(reducer[1], arrs)  # fmax/fmin skip one-sided NaN
    return functools.reduce(reducer[0], arrs)


@_quiet
def max(*xs, filter: bool = False):
    """Element-wise maximum of two or more inputs (``filter=True`` skips one-sided NaN)."""
    return _variadic((np.maximum, np.fmax), xs, filter)


@_quiet
def min(*xs, filter: bool = False):
    return _variadic((np.minimum, np.fmin), xs, filter)


@_quiet
def round(x):
    return np.round(_f(x))


@_quiet
def floor(x):
    return np.floor(_f(x))


@_quiet
def ceiling(x):
    return np.ceil(_f(x))


@_quiet
def round_down(x, n: int = 0):
    s = 10.0**n
    return np.floor(_f(x) * s) / s


@_quiet
def fraction(x):
    x = _f(x)
    return x - np.trunc(x)


@_quiet
def if_else(cond, a, b):
    c = _f(cond)
    out = np.where(c != 0, _f(a), _f(b))
    return np.where(np.isnan(c), np.nan, out)


def _cmp(npfn):
    @_quiet
    def run(x, y):
        x, y = _f(x), _f(y)
        out = npfn(x, y).astype(np.float64)
        return np.where(np.isnan(x) | np.isnan(y), np.nan, out)

    run.__name__ = npfn.__name__
    return run


greater = _cmp(np.greater)
greater_equal = _cmp(np.greater_equal)
less = _cmp(np.less)
less_equal = _cmp(np.less_equal)
equal = _cmp(np.equal)
not_equal = _cmp(np.not_equal)


@_quiet
def and_(x, y):
    x, y = _f(x), _f(y)
    return np.where(np.isnan(x) | np.isnan(y), np.nan, ((x != 0) & (y != 0)).astype(np.float64))


@_quiet
def or_(x, y):
    x, y = _f(x), _f(y)
    return np.where(np.isnan(x) | np.isnan(y), np.nan, ((x != 0) | (y != 0)).astype(np.float64))


@_quiet
def not_(x):
    x = _f(x)
    return np.where(np.isnan(x), np.nan, (x == 0).astype(np.float64))


def is_nan(x):
    return np.isnan(_f(x)).astype(np.float64)


def is_not_nan(x):
    return (~np.isnan(_f(x))).astype(np.float64)


def is_finite(x):
    return np.isfinite(_f(x)).astype(np.float64)


def nan_mask(x, y):
    """NaN where y is non-zero and not NaN."""
    x, y = _f(x), _f(y)
    return np.where((y != 0) & ~np.isnan(y), np.nan, x)


def nan_out(x, lower=-np.inf, upper=np.inf):
    """NaN where x lies strictly outside [lower, upper]."""
    x = _f(x)
    with np.errstate(invalid="ignore"):
        return np.where((x < lower) | (x > upper), np.nan, x)


def to_nan(x, value=0.0):
    x = _f(x)
    return np.where(x == value, np.nan, x)


def left_tail(x, maximum):
    x = _f(x)
    with np.errstate(invalid="ignore"):
        return np.where(x > maximum, np.nan, x)


def right_tail(x, minimum):
    x = _f(x)
    with np.errstate(invalid="ignore"):
        return np.where(x < minimum, np.nan, x)


__all__ = [
    n for n in dir() if not n.startswith("_") and n not in ("annotations", "functools", "np")
]
