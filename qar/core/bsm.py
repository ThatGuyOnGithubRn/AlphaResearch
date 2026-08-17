r"""Black-Scholes-Merton pricing in generalised cost-of-carry form.

Rather than implement the 1973 equation and then bolt on dividends, futures and
FX as special cases, everything is written once against a carry rate ``b``. The
option is priced as if the underlying grows at ``b`` and is discounted at ``r``:

.. math::
    d_1 = \frac{\ln(S/K) + (b + \sigma^2/2)T}{\sigma\sqrt{T}},
    \qquad d_2 = d_1 - \sigma\sqrt{T}

.. math::
    c = S e^{(b-r)T} N(d_1) - K e^{-rT} N(d_2)

.. math::
    p = K e^{-rT} N(-d_2) - S e^{(b-r)T} N(-d_1)

Choosing ``b`` selects the market:

======================  =========  ==========================================
``b``                   Model      Underlying
======================  =========  ==========================================
``b = r``               BSM 1973   Non-dividend-paying stock
``b = r - q``           Merton 73  Stock with continuous dividend yield ``q``
``b = 0``               Black 76   Futures / forwards
``b = r - r_f``         Garman-    FX, ``r_f`` the foreign rate
                        Kohlhagen
======================  =========  ==========================================

Degenerate limits
-----------------
As :math:`\sigma\sqrt{T} \to 0` the option stops being an option: ``d1`` and
``d2`` diverge to :math:`\pm\infty` depending on whether the forward is above or
below the strike, and the price collapses to the discounted forward intrinsic
:math:`e^{-rT}(F - K)^+`. Rather than let that surface as a divide-by-zero or a
``nan``, the degenerate branch is detected and the limit returned directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from qar.core._numeric import scalarize
from qar.core.distributions import norm_cdf, norm_pdf

__all__ = [
    "OptionKind",
    "CALL",
    "PUT",
    "BSMInputs",
    "carry_bsm",
    "carry_dividend",
    "carry_futures",
    "carry_fx",
    "d1_d2",
    "forward",
    "price",
    "call_price",
    "put_price",
    "intrinsic",
]

# Below this, sigma*sqrt(T) is treated as zero and the limiting payoff is used.
# 1e-12 sits far above double-precision noise in the log-moneyness term but far
# below any volatility/maturity combination that could arise from real data.
_DEGENERATE = 1e-12


class OptionKind(str, Enum):
    """Call or put. Subclasses ``str`` so ``kind == "call"`` also works."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """+1 for a call, -1 for a put.

        Nearly every BSM formula is the same expression with the sign of the
        ``d`` arguments flipped, so carrying this around collapses what would
        otherwise be two parallel implementations into one.
        """
        return 1.0 if self is OptionKind.CALL else -1.0


CALL = OptionKind.CALL
PUT = OptionKind.PUT


def _as_kind(kind: Any) -> OptionKind:
    if isinstance(kind, OptionKind):
        return kind
    try:
        return OptionKind(str(kind).lower())
    except ValueError:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}") from None


# --------------------------------------------------------------------------
# Carry helpers
# --------------------------------------------------------------------------


def carry_bsm(r: Any) -> Any:
    """``b = r`` — classic Black-Scholes-Merton, non-dividend-paying stock."""
    return r


def carry_dividend(r: Any, q: Any) -> Any:
    """``b = r - q`` — Merton (1973), continuous dividend yield ``q``."""
    return np.subtract(r, q)


def carry_futures() -> float:
    """``b = 0`` — Black (1976), options on futures.

    The future costs nothing to carry, so it drifts at zero under the
    risk-neutral measure while the option is still discounted at ``r``.
    """
    return 0.0


def carry_fx(r_domestic: Any, r_foreign: Any) -> Any:
    """``b = r_d - r_f`` — Garman-Kohlhagen (1983), FX options."""
    return np.subtract(r_domestic, r_foreign)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _is_complex(*values: Any) -> bool:
    """True if any input carries an imaginary part.

    The complex-step validator perturbs one argument into the complex plane.
    Ordering comparisons are undefined there, so the degenerate-case guards are
    skipped when this is true — complex-step is only ever applied at interior,
    non-degenerate parameter points, where those guards would not fire anyway.
    """
    return any(np.iscomplexobj(v) for v in values)


@dataclass(frozen=True)
class BSMInputs:
    """The six parameters every pricing and Greek function needs.

    Bundling them keeps one signature across ~20 functions and gives a single
    place to validate. ``b`` defaults to ``r``, i.e. textbook BSM.

    Attributes
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility, as a decimal (0.20 means 20%).
    b : float
        Cost of carry. See the module docstring for the standard choices.
    carry_depends_on_r : bool
        Whether ``b`` moves one-for-one with ``r``. This only affects Rho; see
        :func:`qar.greeks.first.rho`. True for BSM and Merton, False for
        Black-76 on futures.
    """

    S: Any
    K: Any
    T: Any
    r: Any
    sigma: Any
    b: Any = None
    carry_depends_on_r: bool = True

    def __post_init__(self) -> None:
        if self.b is None:
            object.__setattr__(self, "b", self.r)
        if not _is_complex(self.S, self.K, self.T, self.sigma):
            if np.any(np.asarray(self.S) <= 0):
                raise ValueError("S must be positive")
            if np.any(np.asarray(self.K) <= 0):
                raise ValueError("K must be positive")
            if np.any(np.asarray(self.T) < 0):
                raise ValueError("T must be non-negative")
            if np.any(np.asarray(self.sigma) < 0):
                raise ValueError("sigma must be non-negative")

    # -- derived quantities, computed on demand ---------------------------

    @property
    def sqrt_T(self) -> Any:
        return np.sqrt(self.T)

    @property
    def vol_sqrt_T(self) -> Any:
        r"""":math:`\sigma\sqrt{T}` — the total volatility over the option's life."""
        return self.sigma * self.sqrt_T

    @property
    def df_r(self) -> Any:
        r""":math:`e^{-rT}` — the discount factor applied to the strike."""
        return np.exp(-self.r * self.T)

    @property
    def df_carry(self) -> Any:
        r""":math:`e^{(b-r)T}` — carry-adjusted discount applied to spot.

        Equals 1 under BSM (``b = r``), :math:`e^{-qT}` with a dividend yield,
        and :math:`e^{-rT}` for futures (``b = 0``).
        """
        return np.exp((self.b - self.r) * self.T)

    @property
    def forward(self) -> Any:
        r""":math:`F = S e^{bT}` — the forward price to expiry."""
        return self.S * np.exp(self.b * self.T)

    @property
    def is_degenerate(self) -> Any:
        r"""True where :math:`\sigma\sqrt{T}` is numerically zero."""
        if _is_complex(self.S, self.K, self.T, self.sigma, self.r, self.b):
            return np.asarray(False)
        return np.asarray(self.vol_sqrt_T) <= _DEGENERATE

    @property
    def d1(self) -> Any:
        return d1_d2(self)[0]

    @property
    def d2(self) -> Any:
        return d1_d2(self)[1]

    @property
    def pdf_d1(self) -> Any:
        r""":math:`n(d_1)` — appears in almost every Greek."""
        return norm_pdf(self.d1)

    def replace(self, **changes: Any) -> "BSMInputs":
        """Copy with some fields changed. Used heavily by the FD validator."""
        fields = {
            "S": self.S,
            "K": self.K,
            "T": self.T,
            "r": self.r,
            "sigma": self.sigma,
            "b": self.b,
            "carry_depends_on_r": self.carry_depends_on_r,
        }
        fields.update(changes)
        return BSMInputs(**fields)


# --------------------------------------------------------------------------
# Core quantities
# --------------------------------------------------------------------------


def d1_d2(inputs: BSMInputs) -> tuple[Any, Any]:
    r"""Return :math:`(d_1, d_2)`.

    .. math::
        d_1 = \frac{\ln(S/K) + (b + \sigma^2/2)T}{\sigma\sqrt{T}},
        \qquad d_2 = d_1 - \sigma\sqrt{T}

    In the degenerate limit :math:`\sigma\sqrt{T} \to 0` both tend to
    :math:`+\infty` when the forward exceeds the strike and :math:`-\infty`
    when it does not, which drives ``N(d)`` to 1 or 0 and recovers the
    discounted intrinsic. At-the-forward they are set to 0, the symmetric
    convention that makes the price continuous there.
    """
    S, K, T, sigma, b = inputs.S, inputs.K, inputs.T, inputs.sigma, inputs.b
    vol_sqrt_T = inputs.vol_sqrt_T

    if _is_complex(S, K, T, sigma, b):
        # No ordering in the complex plane; use the plain formula. Safe because
        # complex-step is only applied away from the degenerate boundary.
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / vol_sqrt_T
        return d1, d1 - vol_sqrt_T

    log_moneyness = np.log(np.asarray(S, dtype=float) / np.asarray(K, dtype=float))
    drift = log_moneyness + np.asarray(b, dtype=float) * np.asarray(T, dtype=float)
    degenerate = inputs.is_degenerate

    # Guard the denominator so the non-degenerate branch never divides by zero;
    # np.where evaluates both branches regardless of the condition.
    safe_denominator = np.where(degenerate, 1.0, vol_sqrt_T)
    regular = (log_moneyness + (np.asarray(b) + 0.5 * np.asarray(sigma) ** 2) * T) / safe_denominator

    limit = np.where(drift > 0, np.inf, np.where(drift < 0, -np.inf, 0.0))

    d1 = np.where(degenerate, limit, regular)
    d2 = np.where(degenerate, limit, regular - vol_sqrt_T)
    return d1, d2


def forward(inputs: BSMInputs) -> Any:
    r"""Forward price :math:`F = S e^{bT}`."""
    return inputs.forward


def intrinsic(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Undiscounted intrinsic value :math:`(\phi(S - K))^+`."""
    phi = _as_kind(kind).sign
    return np.maximum(phi * (inputs.S - inputs.K), 0.0)


def price(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Generalised Black-Scholes-Merton price.

    Both branches collapse into one expression using :math:`\phi = \pm 1`:

    .. math::
        V = \phi\left[S e^{(b-r)T} N(\phi d_1) - K e^{-rT} N(\phi d_2)\right]

    which is worth doing because it halves the surface area for sign errors —
    the single most common bug in a hand-derived option library.
    """
    phi = _as_kind(kind).sign
    d1, d2 = d1_d2(inputs)
    return scalarize(
        phi
        * (
            inputs.S * inputs.df_carry * norm_cdf(phi * d1)
            - inputs.K * inputs.df_r * norm_cdf(phi * d2)
        )
    )


def call_price(inputs: BSMInputs) -> Any:
    """Price of a European call."""
    return price(inputs, CALL)


def put_price(inputs: BSMInputs) -> Any:
    """Price of a European put."""
    return price(inputs, PUT)
