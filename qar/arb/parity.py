r"""Put-call parity: the model-free backbone of the arbitrage checks.

A long call and a short put with the same strike and expiry replicate a forward,
whatever the underlying does:

.. math:: \max(S_T - K, 0) - \max(K - S_T, 0) = S_T - K

Two portfolios with identical payoffs must cost the same today, or one can be
bought and the other sold for a certain profit. Discounting each side gives

.. math:: c - p = S e^{(b-r)T} - K e^{-rT}

**No model is involved.** Not Black-Scholes, not any volatility, not any
distributional assumption. The relation holds for every European option on
every underlying, which is what makes it the right first screen on market data:
a violation is either free money or bad data, and never a modelling
disagreement.

Two consequences the rest of the package leans on:

*   Any Greek taken purely with respect to volatility is identical for the call
    and the put, since :math:`\sigma` does not appear on the right-hand side.
    :mod:`qar.greeks` states this and the test suite asserts it.
*   Given any three of :math:`\{c, p, S, K\}` the fourth follows. That is what
    :func:`imply_call`, :func:`imply_put` and :func:`imply_forward` do — useful
    for filling an illiquid leg from its liquid counterpart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qar.arb.violation import (
    ArbitrageViolation,
    Trade,
    ViolationKind,
    cash_leg,
    option_leg,
    underlying_leg,
)

__all__ = [
    "ParityResult",
    "parity_residual",
    "check_parity",
    "imply_call",
    "imply_put",
    "imply_forward",
]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a put-call parity check."""

    residual: float
    call_price: float
    put_price: float
    synthetic_forward: float
    violated: bool
    violation: ArbitrageViolation | None

    def __bool__(self) -> bool:
        """True when parity holds."""
        return not self.violated


def parity_residual(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    b: float | None = None,
) -> float:
    r"""Return :math:`(c - p) - (S e^{(b-r)T} - K e^{-rT})`.

    Positive means calls are rich relative to puts; negative means the reverse.
    Zero to within transaction costs means the quotes are mutually consistent.
    """
    carry = r if b is None else b
    left = call_price - put_price
    right = S * math.exp((carry - r) * T) - K * math.exp(-r * T)
    return left - right


def check_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    b: float | None = None,
    tol: float = 1e-8,
) -> ParityResult:
    r"""Test put-call parity and, if it fails, build the trade that exploits it.

    Parameters
    ----------
    tol:
        Absolute tolerance in price units. On live quotes set this to at least
        the bid-ask spread — a "violation" smaller than the cost of crossing
        the spread is not tradeable, and reporting it is noise.

    Returns
    -------
    ParityResult
        Falsy when parity is violated, so ``if not check_parity(...)`` reads
        naturally.

    Notes
    -----
    Two trades, depending on the sign of the residual:

    *Conversion* (residual > 0, calls rich). Sell the call, buy the put, buy
    :math:`e^{(b-r)T}` units of the underlying — which grow to exactly one unit
    by expiry under continuous carry — and borrow :math:`K e^{-rT}`. At expiry
    the option legs pay :math:`-(S_T - K)`, the underlying delivers
    :math:`S_T`, and the loan takes :math:`K`. Everything cancels; the residual
    was banked on day one.

    *Reversal* (residual < 0, puts rich). Every leg flipped.
    """
    carry = r if b is None else b
    residual = parity_residual(call_price, put_price, S, K, T, r, carry)
    synthetic_forward = call_price - put_price + K * math.exp(-r * T)

    if abs(residual) <= tol:
        return ParityResult(
            residual=residual,
            call_price=call_price,
            put_price=put_price,
            synthetic_forward=synthetic_forward,
            violated=False,
            violation=None,
        )

    calls_rich = residual > 0
    underlying_units = math.exp((carry - r) * T)
    # Income reinvestment grows the holding back to exactly one unit by expiry.
    underlying_growth = math.exp((r - carry) * T)
    loan_pv = K * math.exp(-r * T)

    if calls_rich:
        legs = (
            option_leg(buy=False, quantity=1.0, kind="call", strike=K, premium=call_price),
            option_leg(buy=True, quantity=1.0, kind="put", strike=K, premium=put_price),
            underlying_leg(
                buy=True, quantity=underlying_units, spot=S, growth=underlying_growth
            ),
            cash_leg(lend=False, present_value=loan_pv, rate=r, maturity=T),
        )
        rationale = (
            "Conversion: the call is rich against the put. Sell the call, buy the "
            "put, hold the underlying, fund it by borrowing the discounted strike."
        )
    else:
        legs = (
            option_leg(buy=True, quantity=1.0, kind="call", strike=K, premium=call_price),
            option_leg(buy=False, quantity=1.0, kind="put", strike=K, premium=put_price),
            underlying_leg(
                buy=False, quantity=underlying_units, spot=S, growth=underlying_growth
            ),
            cash_leg(lend=True, present_value=loan_pv, rate=r, maturity=T),
        )
        rationale = (
            "Reversal: the put is rich against the call. Buy the call, sell the "
            "put, short the underlying, lend the proceeds."
        )

    trade = Trade(legs=legs, rationale=rationale)
    violation = ArbitrageViolation(
        kind=ViolationKind.PARITY,
        magnitude=abs(residual),
        detail=(
            f"c - p = {call_price - put_price:.6g} but "
            f"S*e^((b-r)T) - K*e^(-rT) = "
            f"{S * underlying_units - loan_pv:.6g}"
        ),
        trade=trade,
    )

    return ParityResult(
        residual=residual,
        call_price=call_price,
        put_price=put_price,
        synthetic_forward=synthetic_forward,
        violated=True,
        violation=violation,
    )


def imply_call(
    put_price: float, S: float, K: float, T: float, r: float, b: float | None = None
) -> float:
    """Call price implied by parity from an observed put."""
    carry = r if b is None else b
    return put_price + S * math.exp((carry - r) * T) - K * math.exp(-r * T)


def imply_put(
    call_price: float, S: float, K: float, T: float, r: float, b: float | None = None
) -> float:
    """Put price implied by parity from an observed call."""
    carry = r if b is None else b
    return call_price - S * math.exp((carry - r) * T) + K * math.exp(-r * T)


def imply_forward(call_price: float, put_price: float, K: float, T: float, r: float) -> float:
    r"""Forward price implied by a call and a put at the same strike.

    .. math:: F = K + e^{rT}(c - p)

    The market's own view of the forward, extracted without any dividend or
    borrow assumption. Comparing it against :math:`S e^{bT}` is how a desk backs
    out the implied carry — and on crypto venues, the funding rate.
    """
    return K + math.exp(r * T) * (call_price - put_price)
