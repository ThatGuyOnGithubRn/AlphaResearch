r"""Statistical tests that decide whether a forecasting improvement is real.

A lower mean loss is not evidence on its own. Forecast losses are serially
correlated and heavy-tailed, so the difference between two models routinely
looks large while being well inside sampling noise. These are the tests that
convert a ranking into a claim.

*   :func:`diebold_mariano` — is the loss difference significantly different
    from zero, accounting for autocorrelation and small samples?
*   :func:`mincer_zarnowitz` — is the forecast *unbiased*, or does it
    systematically over- or under-shoot?

Both report a p-value, and both are one-liners to misuse, so the docstrings
spell out the failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "DieboldMarianoResult",
    "MincerZarnowitzResult",
    "diebold_mariano",
    "mincer_zarnowitz",
    "newey_west_variance",
]


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of a Diebold-Mariano test of equal predictive accuracy."""

    statistic: float
    p_value: float
    mean_loss_difference: float
    n_observations: int
    horizon: int
    harvey_corrected: bool

    @property
    def favours(self) -> str:
        """Which model the point estimate favours (ignoring significance)."""
        if self.mean_loss_difference < 0:
            return "first"
        if self.mean_loss_difference > 0:
            return "second"
        return "neither"

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value > alpha:
            return "no significant difference"
        return f"{self.favours} model is better (p={self.p_value:.4g})"


def newey_west_variance(series: np.ndarray, lags: int | None = None) -> float:
    r"""Long-run variance of a series, robust to autocorrelation.

    .. math::
        \hat\gamma_0 + 2\sum_{k=1}^{L}
            \left(1 - \frac{k}{L+1}\right)\hat\gamma_k

    The Bartlett weights :math:`1 - k/(L+1)` are what guarantee a non-negative
    estimate; the naive unweighted sum can come out negative and produce a
    complex-valued test statistic.

    Loss differences are almost always autocorrelated — volatility arrives in
    clusters, so consecutive forecast errors are related — which makes the
    plain sample variance too small and the test too eager to reject.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    if lags is None:
        # Newey-West's rule of thumb.
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    centred = series - series.mean()
    variance = float(np.dot(centred, centred) / n)
    for k in range(1, lags + 1):
        covariance = float(np.dot(centred[k:], centred[:-k]) / n)
        variance += 2.0 * (1.0 - k / (lags + 1.0)) * covariance

    return max(variance, 1e-300)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    harvey_correction: bool = True,
) -> DieboldMarianoResult:
    r"""Test whether two forecasts have equal expected loss.

    The statistic is the mean loss difference :math:`\bar d` scaled by its
    long-run standard error:

    .. math:: DM = \frac{\bar d}{\sqrt{\hat\sigma^2_{LR}/n}}

    A **negative** statistic favours the first model, since
    :math:`d_t = L_a - L_b` is then negative on average.

    Parameters
    ----------
    loss_a, loss_b:
        Per-observation losses from two models, same length, same target.
    horizon:
        Forecast horizon. An :math:`h`-step forecast has errors that are
        MA(h-1) by construction, which sets the minimum lag truncation.
    harvey_correction:
        Apply the Harvey-Leybourne-Newbold (1997) small-sample correction and
        use a :math:`t_{n-1}` reference distribution rather than the normal.
        The uncorrected test over-rejects badly below a few hundred
        observations — and a walk-forward evaluation on two years of daily data
        has a few hundred observations.

    Notes
    -----
    **This test does not apply to nested models.** Comparing GARCH(1,1) against
    a restricted version of itself violates its assumptions; use a Clark-West
    or Giacomini-White test there. Comparing GARCH against HAR-RV, or against a
    neural network, is fine — those are non-nested.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss series must have the same shape")
    if loss_a.size < 8:
        raise ValueError("need at least 8 observations for a meaningful test")

    difference = loss_a - loss_b
    n = difference.size
    mean_difference = float(difference.mean())

    # An h-step forecast error is MA(h-1) by construction, so the truncation
    # must cover at least h-1 lags; beyond that, defer to the rule of thumb.
    rule_of_thumb = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    long_run_variance = newey_west_variance(
        difference, lags=max(horizon - 1, rule_of_thumb)
    )
    statistic = mean_difference / np.sqrt(long_run_variance / n)

    if harvey_correction:
        # Harvey, Leybourne and Newbold (1997), equation 9.
        factor = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
        statistic *= np.sqrt(max(factor, 1e-12))
        p_value = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    else:
        p_value = 2.0 * stats.norm.sf(abs(statistic))

    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_loss_difference=mean_difference,
        n_observations=n,
        horizon=horizon,
        harvey_corrected=harvey_correction,
    )


@dataclass(frozen=True)
class MincerZarnowitzResult:
    """Outcome of a Mincer-Zarnowitz forecast-efficiency regression."""

    intercept: float
    slope: float
    r_squared: float
    joint_p_value: float
    n_observations: int

    @property
    def unbiased(self) -> bool:
        """True when (intercept, slope) = (0, 1) cannot be rejected at 5%."""
        return self.joint_p_value > 0.05

    def verdict(self) -> str:
        if self.unbiased:
            return f"unbiased (joint p={self.joint_p_value:.4g}, R2={self.r_squared:.3f})"
        direction = "over-reacts" if self.slope < 1 else "under-reacts"
        return (
            f"biased: {direction} (slope={self.slope:.3f}, "
            f"joint p={self.joint_p_value:.4g})"
        )


def mincer_zarnowitz(forecast: np.ndarray, realized: np.ndarray) -> MincerZarnowitzResult:
    r"""Regress the realised outcome on the forecast and test for unbiasedness.

    .. math:: RV_t = a + b\,\hat\sigma^2_t + \varepsilon_t

    A perfectly calibrated forecast gives :math:`a = 0` and :math:`b = 1`. The
    joint F-test of that restriction is what this reports.

    The slope carries the interpretation:

    *   :math:`b < 1` — the forecast **over-reacts**: it moves more than the
        outcome does, so shrinking it toward the mean would improve it. This is
        the usual finding for GARCH after a volatility spike.
    *   :math:`b > 1` — the forecast **under-reacts** and should be amplified.

    A high :math:`R^2` with a slope far from 1 is a common and encouraging
    result: the forecast has real information that is simply mis-scaled, and
    the fix is a calibration step rather than a new model.
    """
    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if forecast.shape != realized.shape:
        raise ValueError("forecast and realized must have the same shape")

    n = forecast.size
    if n < 10:
        raise ValueError("need at least 10 observations")

    design = np.column_stack([np.ones(n), forecast])
    coefficients, *_ = np.linalg.lstsq(design, realized, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])

    fitted = design @ coefficients
    residuals = realized - fitted
    ss_residual = float(residuals @ residuals)
    ss_total = float(((realized - realized.mean()) ** 2).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    # Joint F-test of (a, b) = (0, 1): compare against the restricted model
    # RV = forecast, which has no free parameters.
    restricted_residuals = realized - forecast
    ss_restricted = float(restricted_residuals @ restricted_residuals)

    degrees_of_freedom = n - 2
    if degrees_of_freedom <= 0 or ss_residual <= 0:
        joint_p_value = float("nan")
    else:
        f_statistic = ((ss_restricted - ss_residual) / 2.0) / (
            ss_residual / degrees_of_freedom
        )
        joint_p_value = float(stats.f.sf(max(f_statistic, 0.0), 2, degrees_of_freedom))

    return MincerZarnowitzResult(
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        joint_p_value=joint_p_value,
        n_observations=n,
    )
