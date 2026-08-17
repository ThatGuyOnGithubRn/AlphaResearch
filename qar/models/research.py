r"""Models under construction — the part-3 research agenda.

Each class below is a stub with its specification written out: the equation, why
it is expected to beat the baselines, and the specific thing that usually goes
wrong when implementing it. They deliberately raise :class:`NotImplementedError`
rather than returning something plausible, so a half-finished model can never
quietly contribute a number to a results table.

They already conform to :class:`~qar.models.base.Forecaster`, so filling one in
is enough to make it appear in the walk-forward comparison — no plumbing
required.
"""

from __future__ import annotations

import numpy as np

from qar.models.base import Forecaster

__all__ = ["GJRGarch", "MarkovSwitchingVariance", "HARRV", "NeuralVolatility"]


class GJRGarch(Forecaster):
    r"""GJR-GARCH(1,1,1) — asymmetric volatility response.

    .. math::
        \sigma^2_t = \omega + (\alpha + \gamma I_{t-1}) r_{t-1}^2
                     + \beta\sigma^2_{t-1},
        \qquad I_{t-1} = \mathbb{1}\{r_{t-1} < 0\}

    **Why it should help.** Symmetric GARCH treats a 5% drop and a 5% rally as
    equally informative. Equity markets show a strong leverage effect
    (:math:`\gamma > 0`); crypto's is weaker and sometimes inverts during
    melt-ups, which makes it a genuine empirical question here rather than a
    foregone conclusion.

    **The pitfall.** Stationarity now requires
    :math:`\alpha + \gamma/2 + \beta < 1`, not :math:`\alpha + \beta < 1`.
    Reuse the reparameterisation trick from :class:`~qar.models.baselines.GARCH11`
    with the persistence redefined accordingly, or the optimiser will wander
    into a non-stationary region and report a fit that forecasts to infinity.
    """

    name = "gjr-garch"

    def fit(self, returns, realized_variance=None):
        raise NotImplementedError("GJR-GARCH: see class docstring for the specification")

    def forecast_variance(self, horizon: int = 1) -> float:
        raise NotImplementedError


class MarkovSwitchingVariance(Forecaster):
    r"""Two-state Markov-switching variance.

    Variance takes one of :math:`K` values according to a latent state
    :math:`S_t` following a Markov chain with transition matrix :math:`P`.
    Fitted by the Hamilton filter for the likelihood and, if smoothed state
    probabilities are wanted, the Kim smoother.

    **Why it should help.** GARCH models a single slowly mean-reverting
    process, so it adapts to a regime break only gradually — it consistently
    over-forecasts for weeks after a crisis subsides. A switching model can
    jump. Crypto's alternation between long quiet stretches and violent
    repricings is close to the textbook motivation.

    **The pitfalls.** The likelihood is multi-modal, so a single optimisation
    from one starting point is not a fit — use several starts and keep the best.
    The label-switching symmetry (states 1 and 2 are exchangeable) must be
    broken by ordering the variances, or the parameters are unidentified.
    """

    name = "markov-switching"

    def __init__(self, n_states: int = 2) -> None:
        super().__init__()
        self.n_states = n_states

    def fit(self, returns, realized_variance=None):
        raise NotImplementedError(
            "Markov-switching: Hamilton filter for the likelihood, multi-start "
            "optimisation, order the state variances to break label switching"
        )

    def forecast_variance(self, horizon: int = 1) -> float:
        raise NotImplementedError


class HARRV(Forecaster):
    r"""Heterogeneous Autoregressive model of realised volatility (Corsi, 2009).

    .. math::
        RV_{t+1} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)}
                   + \beta_m RV_t^{(m)} + \varepsilon_{t+1}

    where the regressors are realised variance averaged over the last day,
    week (5 days) and month (22 days).

    **Why it should help.** It reproduces the long-memory decay of realised
    volatility with three OLS coefficients, and it consumes the realised-variance
    proxy directly rather than inferring variance from squared returns — a
    strictly more informative input. It is the model to beat in the modern
    forecasting literature, and the honest benchmark for anything fancier.

    **The pitfall.** Fit on :math:`\log RV` and transform back, or the
    right-skewed residuals will let a handful of crisis days dominate the OLS
    fit. Remember the Jensen correction on the way back:
    :math:`E[RV] = \exp(\mu + \sigma^2/2)`.
    """

    name = "har-rv"

    def __init__(self, lags: tuple[int, int, int] = (1, 5, 22)) -> None:
        super().__init__()
        self.lags = lags

    def fit(self, returns, realized_variance=None):
        raise NotImplementedError(
            "HAR-RV: OLS on log realised variance over (1, 5, 22)-day averages, "
            "with the Jensen correction applied when transforming back"
        )

    def forecast_variance(self, horizon: int = 1) -> float:
        raise NotImplementedError


class NeuralVolatility(Forecaster):
    r"""Neural network forecaster (MLP or LSTM over a return/RV window).

    **Why it might help.** It can represent interactions the linear models
    cannot — asymmetry, volatility-of-volatility, and the joint behaviour of
    realised and implied vol — without those being specified in advance.

    **Why it usually does not, at first.** Daily volatility data is small: two
    years of history is roughly 700 observations, which is nothing for a model
    with thousands of parameters. Expect it to lose to HAR-RV until the
    following are all true:

    *   the target is :math:`\log RV`, not :math:`RV` (the raw scale is too
        skewed for a squared-error objective to behave);
    *   inputs are standardised **using training-window statistics only** —
        standardising over the full sample is the most common leak in this
        literature, and it inflates results dramatically;
    *   the network is small (one or two layers, tens of units) with dropout or
        early stopping on a held-out slice of the *training* window;
    *   the comparison is against HAR-RV, not against a random walk.

    Adding ``torch`` to the dependencies is acceptable here — a neural network
    is not a pricing formula, and reimplementing backpropagation would prove
    nothing this project has not already proven.
    """

    name = "neural"

    def __init__(self, window: int = 22, hidden: tuple[int, ...] = (32,)) -> None:
        super().__init__()
        self.window = window
        self.hidden = hidden

    def fit(self, returns, realized_variance=None):
        raise NotImplementedError(
            "Neural forecaster: target log RV, standardise on training statistics "
            "only, keep the network small, and benchmark against HAR-RV"
        )

    def forecast_variance(self, horizon: int = 1) -> float:
        raise NotImplementedError
