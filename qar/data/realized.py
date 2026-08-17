r"""Realised-variance estimators — the target every volatility forecast is scored against.

Volatility is never observed. Every evaluation in part 3 compares a forecast to
a *proxy* built from returns, and the choice of proxy has a first-order effect
on which model appears to win. Four estimators are provided, in increasing order
of statistical efficiency:

===================== ============================= =====================
Estimator             Uses                          Variance ratio
===================== ============================= =====================
Close-to-close        Closes only                   1.00 (baseline)
Parkinson (1980)      High, low                     ~0.20
Garman-Klass (1980)   Open, high, low, close        ~0.14
Rogers-Satchell (1991) Open, high, low, close       ~0.17, drift-robust
===================== ============================= =====================

The ratio is the estimator's variance relative to close-to-close, so Parkinson
extracts about five times more information from the same bar. That matters:
a noisier proxy inflates every model's measured error and compresses the
differences between them, which is exactly what makes a Diebold-Mariano test
fail to reject when it should.

Two cautions the docstrings expand on: Parkinson and Garman-Klass assume zero
drift and will understate variance in a trending market, which is why
Rogers-Satchell is included; and squared returns, though a wildly noisy proxy,
are *unbiased*, which is what licenses the QLIKE loss in
:mod:`qar.evaluation.loss`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "trailing_mean",
    "log_returns",
    "close_to_close",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "realized_variance",
    "annualize",
    "ESTIMATORS",
    "TRADING_DAYS",
]

#: Crypto trades continuously, so every calendar day is a trading day. Set this
#: to 252 for equities — using the wrong constant rescales every volatility
#: number by 20% and is a classic silent error.
TRADING_DAYS = 365.0


def trailing_mean(series: np.ndarray, window: int) -> np.ndarray:
    r"""Rolling mean over ``[i-window, i)`` — strictly excluding the current point.

    The half-open interval is the entire point of this function. A rolling mean
    that includes observation :math:`i` is a perfectly good smoother, and a
    catastrophic *signal*: any strategy that trades on it at time :math:`i` has
    already seen part of the outcome it is betting on. Because volatility is
    strongly persistent, that leak does not stay confined to one day — it
    propagates into the following days as well, and inflates a backtest by
    enough to look like a discovery.

    Leading positions with no history fall back to ``series[0]``, so the output
    is the same length as the input and contains no ``nan``.
    """
    series = np.asarray(series, dtype=float)
    if window < 1:
        raise ValueError("window must be at least 1")

    cumulative = np.concatenate([[0.0], np.cumsum(series)])
    indices = np.arange(series.size)
    starts = np.maximum(0, indices - window)
    counts = indices - starts

    with np.errstate(invalid="ignore", divide="ignore"):
        means = (cumulative[indices] - cumulative[starts]) / counts
    return np.where(counts > 0, means, series[0] if series.size else 0.0)


def log_returns(close: np.ndarray) -> np.ndarray:
    r"""Continuously compounded returns :math:`r_t = \ln(P_t/P_{t-1})`.

    Log rather than simple returns because the model is geometric Brownian
    motion, whose increments are additive in logs — and because they aggregate
    over time by addition, which is what makes multi-period variance scaling
    work.
    """
    close = np.asarray(close, dtype=float)
    return np.diff(np.log(close))


def close_to_close(close: np.ndarray) -> np.ndarray:
    r"""Squared log returns, :math:`r_t^2`.

    The noisiest proxy by a wide margin — its own variance is twice the square
    of the quantity it estimates — but unbiased for the daily variance, which
    is the property the QLIKE loss function needs.
    """
    returns = log_returns(close)
    return returns**2


def parkinson(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    r"""Parkinson (1980) range estimator.

    .. math:: \hat\sigma^2_t = \frac{1}{4\ln 2}\left[\ln(H_t/L_t)\right]^2

    The intraday range carries far more information about volatility than the
    close alone: a day that swings 5% and closes flat has a close-to-close
    estimate of zero and a Parkinson estimate that reflects what happened.

    **Assumes zero drift.** Under a strong trend the range is inflated by the
    drift and the estimator reads high.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    return (np.log(high / low) ** 2) / (4.0 * np.log(2.0))


def garman_klass(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    r"""Garman-Klass (1980) estimator — the most efficient of the four.

    .. math::
        \hat\sigma^2_t = \tfrac{1}{2}\left[\ln(H_t/L_t)\right]^2
            - (2\ln 2 - 1)\left[\ln(C_t/O_t)\right]^2

    Combines the range with the open-to-close move. Also assumes zero drift,
    and can return small negative values on bars where the close-to-open term
    dominates; those are clipped to zero, since a negative variance estimate is
    meaningless as a forecast target.
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    range_term = 0.5 * np.log(high / low) ** 2
    body_term = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    return np.maximum(range_term - body_term, 0.0)


def rogers_satchell(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    r"""Rogers-Satchell (1991) estimator — efficient *and* drift-robust.

    .. math::
        \hat\sigma^2_t = \ln\frac{H_t}{C_t}\ln\frac{H_t}{O_t}
                       + \ln\frac{L_t}{C_t}\ln\frac{L_t}{O_t}

    Unlike Parkinson and Garman-Klass this remains unbiased under a non-zero
    drift, which makes it the right default on an asset in a sustained trend —
    a description that fits crypto more often than not.
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    return (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )


#: Estimators keyed by name, for sweeping an evaluation across proxies.
ESTIMATORS = {
    "close_to_close": lambda bars: close_to_close(bars["close"]),
    "parkinson": lambda bars: parkinson(bars["high"], bars["low"])[1:],
    "garman_klass": lambda bars: garman_klass(
        bars["open"], bars["high"], bars["low"], bars["close"]
    )[1:],
    "rogers_satchell": lambda bars: rogers_satchell(
        bars["open"], bars["high"], bars["low"], bars["close"]
    )[1:],
}


def realized_variance(bars: dict[str, np.ndarray], estimator: str = "garman_klass") -> np.ndarray:
    """Daily realised variance from OHLC bars, using the named estimator.

    Range-based estimators are sliced to drop their first bar so that every
    estimator returns a series aligned with ``log_returns(close)``.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {sorted(ESTIMATORS)}")
    return ESTIMATORS[estimator](bars)


def annualize(daily_variance: np.ndarray | float, periods: float = TRADING_DAYS) -> np.ndarray | float:
    r"""Convert daily variance to annualised volatility, :math:`\sqrt{252\,\sigma^2_d}`.

    Variance scales linearly in time under independent increments; volatility
    therefore scales as the square root. Applying the square root to the
    *annualised variance* rather than annualising the volatility directly is
    the order that avoids a subtle bias when averaging.
    """
    return np.sqrt(np.asarray(daily_variance, dtype=float) * periods)
