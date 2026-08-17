r"""Implied volatility: invert the BSM price for :math:`\sigma`.

There is no closed form, so this is a root-find on

.. math:: f(\sigma) = V_{\text{BSM}}(\sigma) - V_{\text{market}} = 0

Three things make the naive version fail on real quotes, and each is handled:

1.  **Newton stalls in the wings.** Vega collapses to ~0 for deep in- or
    out-of-the-money options, so the Newton step :math:`f/\mathcal{V}` explodes.
    The solver falls back to bisection whenever a step leaves the bracket.
2.  **No solution exists for arbitrageable quotes.** :math:`f` is strictly
    increasing from the intrinsic (at :math:`\sigma\to 0`) to the spot bound
    (at :math:`\sigma\to\infty`). A price outside that range has no implied vol
    at all — the correct response is to say so, not to return a number.
3.  **A bad seed costs iterations.** Brenner-Subrahmanyam gives a good one in
    closed form for near-the-money options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qar.core.bsm import CALL, BSMInputs, OptionKind, price
from qar.greeks._common import resolve_kind

__all__ = ["ImpliedVolResult", "implied_vol", "NoImpliedVolError"]

_MAX_SIGMA = 10.0      # 1000% annualised; nothing real implies more
_MIN_SIGMA = 1e-9
_DEFAULT_TOL = 1e-10
_MAX_ITER = 100


class NoImpliedVolError(ValueError):
    """Raised when the quoted price admits no Black-Scholes implied volatility.

    Always means the quote violates a no-arbitrage bound — it sits below the
    intrinsic value or above the underlying. :mod:`qar.arb.bounds` will say
    which.
    """


@dataclass(frozen=True)
class ImpliedVolResult:
    """Outcome of an implied-vol solve, including how it got there."""

    sigma: float
    iterations: int
    converged: bool
    residual: float
    method: str  # "newton", "bisection", or "mixed"

    def __float__(self) -> float:
        return self.sigma


def _seed(S: float, K: float, T: float, r: float, b: float, target: float) -> float:
    r"""Brenner-Subrahmanyam (1988) initial guess.

    For an at-the-forward option the BSM price is very nearly linear in
    :math:`\sigma`:

    .. math:: V \approx 0.3989\, S e^{(b-r)T} \sigma\sqrt{T}

    Inverting gives the seed below. Away from the forward it degrades, but it
    only has to land in the right order of magnitude — bisection handles the
    rest.
    """
    if T <= 0.0:
        return 0.2
    forward = S * math.exp(b * T)
    discount = math.exp(-r * T)
    atm_scale = math.sqrt(2.0 * math.pi / T) / (forward * discount)
    guess = atm_scale * target
    return min(max(guess, 0.01), 3.0)


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    b: float | None = None,
    kind: Any = CALL,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _MAX_ITER,
) -> ImpliedVolResult:
    r"""Solve for the volatility that reproduces ``target_price``.

    Newton's method on Vega, with a maintained bracket and automatic bisection
    fallback whenever a Newton step would leave it. That combination is
    unconditionally convergent (from the bracket) while keeping Newton's
    quadratic rate where Vega is healthy.

    Parameters
    ----------
    target_price:
        The observed option price to invert.
    S, K, T, r, b, kind:
        As in :class:`~qar.core.bsm.BSMInputs`. ``b`` defaults to ``r``.
    tol:
        Absolute price tolerance for convergence.

    Raises
    ------
    NoImpliedVolError
        If ``target_price`` lies outside the attainable range, which means the
        quote is not arbitrage-free.

    Notes
    -----
    Vega is strictly positive for :math:`0 < \sigma < \infty`, so :math:`f` is
    strictly monotone and the root — when it exists — is unique. That is what
    licenses the bracket.
    """
    option: OptionKind = resolve_kind(kind)
    carry = r if b is None else b

    def model(sigma: float) -> float:
        return float(
            price(BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, b=carry), option)
        )

    lower, upper = _MIN_SIGMA, _MAX_SIGMA
    price_low, price_high = model(lower), model(upper)

    if target_price < price_low - tol:
        raise NoImpliedVolError(
            f"price {target_price:.6g} is below the zero-volatility bound "
            f"{price_low:.6g}; the quote is below intrinsic and admits arbitrage"
        )
    if target_price > price_high + tol:
        raise NoImpliedVolError(
            f"price {target_price:.6g} exceeds the price at sigma={_MAX_SIGMA:.0f} "
            f"({price_high:.6g}); no finite implied volatility exists"
        )

    sigma = _seed(S, K, T, r, carry, target_price)
    used_bisection = False
    used_newton = False

    for iteration in range(1, max_iter + 1):
        value = model(sigma)
        residual = value - target_price

        if abs(residual) < tol:
            method = "mixed" if (used_newton and used_bisection) else (
                "bisection" if used_bisection else "newton"
            )
            return ImpliedVolResult(sigma, iteration, True, residual, method)

        # Maintain the bracket using monotonicity of price in sigma.
        if residual > 0:
            upper = sigma
        else:
            lower = sigma

        # Vega, inlined to avoid a circular import with qar.greeks.
        inputs = BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, b=carry)
        d1 = inputs.d1
        vega_value = float(
            np.asarray(S * inputs.df_carry * inputs.pdf_d1 * inputs.sqrt_T)
        )

        if vega_value > 1e-12:
            step = sigma - residual / vega_value
            if lower < step < upper:
                sigma = step
                used_newton = True
                continue

        sigma = 0.5 * (lower + upper)
        used_bisection = True

    return ImpliedVolResult(
        sigma, max_iter, False, model(sigma) - target_price, "mixed"
    )
