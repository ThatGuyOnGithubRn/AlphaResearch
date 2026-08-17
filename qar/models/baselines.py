r"""Baseline volatility forecasters — the bar every proposed model must clear.

Choosing weak baselines is the easiest way to manufacture an impressive result,
so these are the ones that are genuinely hard to beat:

*   **Random walk on realised variance.** Tomorrow equals today. Beating it
    requires the model to add something beyond persistence.
*   **EWMA / RiskMetrics** with :math:`\lambda = 0.94`. One fixed parameter, no
    estimation, and competitive with fitted GARCH at short horizons — the
    reason it survived as an industry standard for thirty years.
*   **GARCH(1,1)**, fitted by maximum likelihood. Andersen and Bollerslev's
    result that GARCH(1,1) is hard to beat out of sample is the benchmark the
    rest of part 3 is measured against.

The GARCH likelihood is written out and optimised directly rather than taken
from the ``arch`` package, matching the same first-principles rule that governs
the pricing code. SciPy supplies the optimiser, not the model.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, signal

from qar.models.base import FitResult, Forecaster

__all__ = ["RandomWalkVariance", "EWMAVariance", "GARCH11", "BASELINES"]


class RandomWalkVariance(Forecaster):
    r"""Tomorrow's variance equals today's realised variance.

    Deceptively strong. Realised variance is highly persistent, so the naive
    carry-forward captures most of the predictable component at a one-day
    horizon and leaves very little for a model to add.
    """

    name = "random-walk"

    def __init__(self) -> None:
        super().__init__()
        self.last_: float = float("nan")

    def fit(self, returns, realized_variance=None):
        series = returns**2 if realized_variance is None else np.asarray(realized_variance)
        series = np.asarray(series, dtype=float)
        if series.size == 0:
            raise ValueError("need at least one observation")
        self.last_ = float(series[-1])
        self.fitted_ = True
        self.diagnostics_ = FitResult(last_observation=self.last_)
        return self

    def forecast_variance(self, horizon: int = 1) -> float:
        self._require_fit()
        return self.last_


class EWMAVariance(Forecaster):
    r"""Exponentially weighted moving average — RiskMetrics.

    .. math:: \sigma^2_{t+1} = \lambda\sigma^2_t + (1-\lambda) r_t^2

    An IGARCH(1,1) with :math:`\omega = 0` and the persistence pinned at one,
    so shocks never decay to a long-run mean. That is a genuine weakness over
    long horizons and barely matters over one day, which is why it competes
    with fitted models at the daily frequency.

    :math:`\lambda = 0.94` is the RiskMetrics daily value, not a fitted one —
    keeping it fixed is what makes this a baseline rather than a rival model.
    """

    name = "ewma"

    def __init__(self, lam: float = 0.94) -> None:
        super().__init__()
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must lie in (0, 1)")
        self.lam = lam
        self.variance_: float = float("nan")

    def fit(self, returns, realized_variance=None):
        returns = np.asarray(returns, dtype=float)
        if returns.size < 2:
            raise ValueError("need at least two returns")

        # Seed with the sample variance of the first tenth of the window, then
        # recurse. Seeding from the whole window would leak future information.
        seed_length = max(2, returns.size // 10)
        variance = float(np.var(returns[:seed_length]))
        for r in returns[seed_length:]:
            variance = self.lam * variance + (1.0 - self.lam) * r * r

        self.variance_ = variance
        self.fitted_ = True
        self.diagnostics_ = FitResult(lam=self.lam, seed_length=seed_length)
        return self

    def forecast_variance(self, horizon: int = 1) -> float:
        self._require_fit()
        # Persistence is exactly 1, so the forecast is flat across horizons.
        return self.variance_


class GARCH11(Forecaster):
    r"""GARCH(1,1) fitted by maximum likelihood.

    .. math:: \sigma^2_t = \omega + \alpha r_{t-1}^2 + \beta\sigma^2_{t-1}

    With Gaussian innovations the log-likelihood, dropping constants, is

    .. math::
        \ell = -\tfrac{1}{2}\sum_t
            \left[\ln\sigma^2_t + \frac{r_t^2}{\sigma^2_t}\right]

    Three implementation points that decide whether the fit is trustworthy:

    *   **Parameters are optimised in an unconstrained space** and mapped back
        through a softplus/logistic transform. Constrained optimisers on the
        boundary :math:`\alpha + \beta < 1` routinely stall; reparameterising
        removes the boundary entirely.
    *   **Returns are demeaned** using the window mean. Leaving a drift in
        inflates :math:`\omega` and biases the long-run variance upward.
    *   **The recursion is seeded with the sample variance**, the standard
        choice, and the first observation's contribution is dropped so the seed
        does not enter the likelihood twice.

    Forecasts mean-revert toward the long-run variance
    :math:`\omega/(1-\alpha-\beta)` at rate :math:`(\alpha+\beta)^h`, which is
    the behaviour the random walk and EWMA baselines both lack.
    """

    name = "garch(1,1)"

    def __init__(self, max_iter: int = 500) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.omega_ = self.alpha_ = self.beta_ = float("nan")
        self.last_variance_ = self.last_return_ = float("nan")
        self.mean_ = 0.0

    # -- parameter transform ---------------------------------------------

    @staticmethod
    def _unpack(theta: np.ndarray) -> tuple[float, float, float]:
        r"""Map :math:`\mathbb{R}^3` to the admissible parameter region.

        ``omega`` through a softplus to keep it positive; ``alpha`` and
        ``beta`` as fractions of a logistic-bounded persistence, which
        guarantees :math:`\alpha + \beta < 1` by construction rather than by
        constraint.
        """
        omega = np.logaddexp(0.0, theta[0])            # softplus > 0
        persistence = 1.0 / (1.0 + np.exp(-theta[1]))  # in (0, 1)
        alpha_share = 1.0 / (1.0 + np.exp(-theta[2]))  # in (0, 1)
        alpha = persistence * alpha_share
        beta = persistence - alpha
        return float(omega), float(alpha), float(beta)

    @staticmethod
    def _variance_path(
        omega: float, alpha: float, beta: float, squared: np.ndarray, seed: float
    ) -> np.ndarray:
        r"""Conditional variances for the whole sample, without a Python loop.

        The GARCH recursion

        .. math:: \sigma^2_t = \underbrace{\omega + \alpha r^2_{t-1}}_{u_t}
                               + \beta\sigma^2_{t-1}

        is a first-order linear IIR filter driven by :math:`u_t`, so it can be
        evaluated by ``scipy.signal.lfilter`` in compiled code rather than
        iterated in Python. That matters a great deal here: a walk-forward
        evaluation refits several hundred times, and each fit costs the
        optimiser tens of likelihood evaluations. The loop version made the
        test suite take minutes; this makes it seconds, with identical output.

        ``zi`` carries the initial delay state, set to :math:`\beta\sigma^2_0`
        so that the first filtered value is :math:`u_1 + \beta\sigma^2_0`.
        """
        driver = omega + alpha * squared[:-1]
        tail, _ = signal.lfilter([1.0], [1.0, -beta], driver, zi=np.array([beta * seed]))
        return np.concatenate([[seed], tail])

    def _negative_log_likelihood(self, theta: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = self._unpack(theta)
        squared = returns * returns
        seed = float(np.var(returns))
        if seed <= 0 or not np.isfinite(seed):
            return 1e10

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            variances = self._variance_path(omega, alpha, beta, squared, seed)
            if not np.all(np.isfinite(variances)) or np.any(variances <= 1e-300):
                return 1e10
            total = float(np.sum(np.log(variances) + squared / variances))

        if not np.isfinite(total):
            return 1e10
        return 0.5 * total

    def fit(self, returns, realized_variance=None):
        returns = np.asarray(returns, dtype=float)
        if returns.size < 50:
            raise ValueError("GARCH(1,1) needs at least 50 observations to be meaningful")

        self.mean_ = float(np.mean(returns))
        centred = returns - self.mean_

        # Seed near the textbook (0.05, 0.90) region, in transformed space.
        start = np.array([np.log(np.expm1(max(np.var(centred) * 0.05, 1e-12))), 2.2, -2.9])
        result = optimize.minimize(
            self._negative_log_likelihood,
            start,
            args=(centred,),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        if not result.success:
            # Quasi-Newton can stall on the flatter parts of this surface;
            # Nelder-Mead is slower but derivative-free and usually recovers.
            result = optimize.minimize(
                self._negative_log_likelihood,
                start,
                args=(centred,),
                method="Nelder-Mead",
                options={"maxiter": self.max_iter * 2, "xatol": 1e-8, "fatol": 1e-8},
            )

        self.omega_, self.alpha_, self.beta_ = self._unpack(result.x)

        # One more pass to recover the terminal conditional variance, i.e. the
        # one-step-ahead forecast.
        squared = centred * centred
        variances = self._variance_path(
            self.omega_, self.alpha_, self.beta_, squared, float(np.var(centred))
        )
        self.last_variance_ = float(
            self.omega_ + self.alpha_ * squared[-1] + self.beta_ * variances[-1]
        )
        self.last_return_ = float(centred[-1])

        self.fitted_ = True
        self.diagnostics_ = FitResult(
            omega=self.omega_,
            alpha=self.alpha_,
            beta=self.beta_,
            persistence=self.alpha_ + self.beta_,
            long_run_variance=self.long_run_variance,
            log_likelihood=-float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
        )
        return self

    @property
    def long_run_variance(self) -> float:
        r"""Unconditional variance :math:`\omega/(1 - \alpha - \beta)`."""
        persistence = self.alpha_ + self.beta_
        if persistence >= 1.0:
            return float("inf")
        return self.omega_ / (1.0 - persistence)

    def forecast_variance(self, horizon: int = 1) -> float:
        r"""Forecast at ``horizon`` days.

        .. math::
            E[\sigma^2_{t+h}] = \bar\sigma^2
                + (\alpha+\beta)^{h-1}\left(\sigma^2_{t+1} - \bar\sigma^2\right)
        """
        self._require_fit()
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if horizon == 1:
            return self.last_variance_

        persistence = self.alpha_ + self.beta_
        long_run = self.long_run_variance
        if not np.isfinite(long_run):
            return self.last_variance_
        return long_run + persistence ** (horizon - 1) * (self.last_variance_ - long_run)


#: Every baseline, ready to hand to the walk-forward evaluator.
BASELINES: dict[str, type[Forecaster]] = {
    "random-walk": RandomWalkVariance,
    "ewma": EWMAVariance,
    "garch(1,1)": GARCH11,
}
