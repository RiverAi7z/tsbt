"""Closed-form vectorised P&L kernels (no loop over time).

Notation per column i and bar t (see SPEC section 8):

    theta[t]  target position held after trading inside bar t
    R[t]      P[t+1]/P[t] - 1   (return between consecutive executions)
    b[t]      C[t]/P[t] - 1     (execution -> bar close, for mark-to-market)

Compound (theta = notional / current equity)::

    G[t]   = 1 + S(theta[t] R[t])
    w-[t]  = theta[t-1] (1 + R[t-1]) / G[t-1],  w-[0] = 0
    tau[t] = S|theta[t] - w-[t]|
    E+[t]  = prod_{u<=t}(1 - f tau[u]) * prod_{u<t} G[u]
    NAV[t] = E+[t] * (1 + S(theta[t] b[t]))

Simple (theta = notional / fixed principal 1)::

    h-[t]  = theta[t-1] (1 + R[t-1])
    tau[t] = S|theta[t] - h-[t]|
    NAV[t] = 1 + sum_{u<t} S(theta R)[u] - f sum_{u<=t} tau[u] + S(theta[t] b[t])

``S`` sums over columns when ``pooled`` (one shared account) and is the identity
otherwise (one account per column).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimOut:
    nav: np.ndarray  # (T, M)  M = 1 if pooled else N
    turnover: np.ndarray  # (T, M)  two-sided, fraction of pre-trade equity / principal
    turnover_cols: np.ndarray  # (T, N) per-column share of the turnover
    fee: np.ndarray  # (T, M)  fee paid, in NAV units
    bankrupt_at: np.ndarray  # (M,) first bankrupt row or -1


def simulate(
    theta: np.ndarray, R: np.ndarray, b: np.ndarray, fee: float, compound: bool, pooled: bool
) -> SimOut:
    return simulate_fees(theta, R, b, (fee,), compound, pooled)[0]


def simulate_fees(
    theta: np.ndarray,
    R: np.ndarray,
    b: np.ndarray,
    fees: tuple[float, ...],
    compound: bool,
    pooled: bool,
) -> list[SimOut]:
    """One :class:`SimOut` per fee; the fee-independent terms are computed once."""
    theta = np.asarray(theta, dtype=np.float64)

    def S(x):
        return x.sum(axis=1, keepdims=True) if pooled else x

    tR = theta * R
    mark = S(theta * b)
    prev = np.zeros_like(theta)
    if not compound:
        prev[1:] = theta[:-1] * (1.0 + R[:-1])
        tau_cols = np.abs(theta - prev)
        tau = S(tau_cols)
        pnl = S(tR)
        realized = np.zeros_like(pnl)
        realized[1:] = np.cumsum(pnl[:-1], axis=0)
        cum_tau = np.cumsum(tau, axis=0)
        none = np.full(tau.shape[1], -1)
        return [
            SimOut(
                nav=1.0 + realized - fee * cum_tau + mark,
                turnover=tau,
                turnover_cols=tau_cols,
                fee=fee * tau,
                bankrupt_at=none,
            )
            for fee in fees
        ]

    G = 1.0 + S(tR)
    # Rows after a bankruptcy can overflow; they are overwritten below.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        prev[1:] = theta[:-1] * (1.0 + R[:-1]) / G[:-1]
        tau_cols0 = np.abs(theta - prev)
        tau0 = S(tau_cols0)
        growth = np.ones_like(G)
        growth[1:] = np.cumprod(G[:-1], axis=0)
    # bankruptcy between two trades or at a bar close does not depend on the fee
    bad0 = 1.0 + mark <= 0
    bad0[1:] |= G[:-1] <= 0
    out = []
    for fee in fees:
        tau, tau_cols = tau0, tau_cols0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            c = 1.0 - fee * tau
            e_plus = np.cumprod(c, axis=0) * growth
            nav = e_plus * (1.0 + mark)
            fee_paid = fee * tau * np.divide(e_plus, c, out=np.zeros_like(e_plus), where=c > 0)
        # bankruptcy: equity <= 0 at a trade, at a bar close, or between two trades
        bad = (c <= 0) | bad0
        any_bad = bad.any(axis=0)
        first = np.where(any_bad, bad.argmax(axis=0), -1)
        if any_bad.any():
            tau, tau_cols = tau.copy(), tau_cols.copy()
        for m in np.nonzero(any_bad)[0]:
            t0 = first[m]
            nav[t0:, m] = 0.0
            tau[t0:, m] = 0.0
            if pooled:
                tau_cols[t0:] = 0.0
            else:
                tau_cols[t0:, m] = 0.0
            fee_paid[t0:, m] = 0.0
        out.append(
            SimOut(nav=nav, turnover=tau, turnover_cols=tau_cols, fee=fee_paid, bankrupt_at=first)
        )
    return out
